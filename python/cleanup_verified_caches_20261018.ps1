param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$prefix = $workspace.TrimEnd('\') + '\'
$auditFolder = Join-Path $workspace 'data/project_cleanup_audit_20261018'
$planPath = Join-Path $auditFolder 'verified_cache_plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.workspace -ne $workspace -or $plan.status -ne 'prepared_not_deleted') { throw 'Unexpected plan identity.' }

function Assert-SafePath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    $cursor = $absolute
    while ($cursor -and $cursor -ne $workspace) {
        if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing linked path: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

function Assert-Hash([string]$Path, [string]$Expected) {
    $checked = Assert-SafePath $Path
    if ((Get-FileHash -LiteralPath $checked -Algorithm SHA256).Hash -ne $Expected) { throw "Hash changed: $checked" }
}

function Assert-TasksIdle {
    $active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='vivado.exe' OR Name='xsim.exe' OR Name='xelab.exe' OR Name='xvlog.exe'")
    foreach ($task in $active) {
        if ($task.Name -ne 'python.exe') { throw "Hardware build/simulation is running: PID $($task.ProcessId)" }
        if ([string]::IsNullOrWhiteSpace($task.CommandLine)) { throw "Cannot inspect Python task: PID $($task.ProcessId)" }
        if ($task.CommandLine -match 'lung_prune|lung_candidate|prepare_verified_cleanup|ct_basic_shared_vectors') {
            throw "Cache-producing/consuming task is running: PID $($task.ProcessId)"
        }
    }
}

$allowedReasons = @('reconstructable_27_case_test_input', 'lossless_archived_old_candidate_payload',
    'duplicate_tool_cache', 'regenerable_tool_cache_metadata', 'regenerable_xsim_binary',
    'python_bytecode_cache', 'regenerable_text_simulation_vectors')
$unique = @{}
foreach ($item in $plan.files) {
    $path = Assert-SafePath $item.path
    $relative = $path.Substring($prefix.Length).Replace('\','/')
    if ($unique.ContainsKey($path)) { throw "Duplicate target: $path" }
    $unique[$path] = $true
    if ($item.reason -notin $allowedReasons) { throw "Unknown deletion reason: $($item.reason)" }
    $valid = switch ($item.reason) {
        'reconstructable_27_case_test_input' { $relative -match '^data/lung_prune_test_cache_20261017/cache/[^/]+/(images|reference_native|hu_native)\.npy$' }
        'lossless_archived_old_candidate_payload' { $relative -match '^data/lung_candidates_pilot_20261018_v[12]/[^/]+/(candidates\.json|all_structure_features\.json|masks_packed\.npz)$' }
        'duplicate_tool_cache' { $relative -match '^(data|fpga)/.+/(?:[^/]+\.cache|\.Xil)/' }
        'regenerable_tool_cache_metadata' { $relative -match '^(data|fpga)/.+/(?:[^/]+\.cache|\.Xil)/' }
        'regenerable_xsim_binary' { $relative -match '^(data|fpga)/.+/xsim\.dir/' }
        'python_bytecode_cache' { $relative -match '^(data|fpga|python|pynq)/.+/__pycache__/[^/]+\.pyc$' -or $relative -match '^(python|pynq)/__pycache__/[^/]+\.pyc$' }
        'regenerable_text_simulation_vectors' { $relative -eq 'data/ct_basic_native_sim_20261017/vectors.txt' }
    }
    if (-not $valid) { throw "Target does not match its allowed rule: $path" }
    if ($path -match '\\repo\\|\\Lab_9\\|\\\.git\\|\\\.venv') { throw "Excluded tree: $path" }
}
Write-Output "Planned files: $($plan.files.Count); bytes: $($plan.gross_bytes); archive retained: $($plan.archive.bytes) bytes."
if (-not $Execute) { return }
Assert-TasksIdle
Assert-Hash $plan.archive.path $plan.archive.sha256
Assert-Hash (Join-Path $auditFolder 'source_reconstruction_preflight.json') $plan.source_preflight_sha256
$protected = @{}
foreach ($item in $plan.retained) {
    if ($unique.ContainsKey($item.path)) { throw "Protected file selected for deletion: $($item.path)" }
    $protected[$item.path] = $item.sha256
}
& (Join-Path $workspace '.venv-lung/Scripts/python.exe') (Join-Path $auditFolder 'check_retained_hashes.py')
if ($LASTEXITCODE -ne 0) { throw 'Retained-file preflight failed.' }
foreach ($item in $plan.files) {
    Assert-Hash $item.path $item.sha256
    if ((Get-Item -LiteralPath $item.path).Length -ne $item.bytes) { throw "Size changed: $($item.path)" }
    if ($item.reason -eq 'duplicate_tool_cache') {
        if (-not $protected.ContainsKey($item.retained_duplicate)) { throw 'Retained duplicate is not protected.' }
        Assert-Hash $item.retained_duplicate $item.sha256
    }
}
$manifestPath = Join-Path $auditFolder ('verified_cleanup_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.json')
$record = [ordered]@{
    status='prepared'; workspace=$workspace; started_at_utc=[DateTime]::UtcNow.ToString('o')
    plan_path=$planPath; plan_sha256=(Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash
    archive=$plan.archive; groups=$plan.groups; removed=@(); removed_bytes=[long]0
    protected_files=$plan.retained.Count; protected_verified=$false; empty_directories_removed=@()
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
try {
    Assert-TasksIdle
    foreach ($item in $plan.files) {
        $checked = Assert-SafePath $item.path
        # Recheck each exact file immediately before permanent deletion.
        Assert-Hash $checked $item.sha256
        Remove-Item -LiteralPath $checked -Force
        $record.removed += $checked
        $record.removed_bytes += $item.bytes
        if (($record.removed.Count % 100) -eq 0) {
            $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
        }
    }
    # Only remove empty parents of explicitly selected files, never whole trees.
    $parents = @{}
    foreach ($item in $plan.files) {
        $parent = [IO.Path]::GetDirectoryName($item.path)
        while ($parent -and $parent -ne $workspace -and $parent -notin @((Join-Path $workspace 'data'),(Join-Path $workspace 'fpga'),(Join-Path $workspace 'python'),(Join-Path $workspace 'pynq'))) {
            $parents[$parent] = $true
            $parent = [IO.Path]::GetDirectoryName($parent)
        }
    }
    foreach ($directory in ($parents.Keys | Sort-Object Length -Descending)) {
        if (-not (Test-Path -LiteralPath $directory)) { continue }
        $checked = Assert-SafePath $directory
        if (@(Get-ChildItem -LiteralPath $checked -Force).Count -eq 0) {
            Remove-Item -LiteralPath $checked -Force
            $record.empty_directories_removed += $checked
        }
    }
    & (Join-Path $workspace '.venv-lung/Scripts/python.exe') (Join-Path $auditFolder 'check_retained_hashes.py')
    if ($LASTEXITCODE -ne 0) { throw 'Retained-file final verification failed.' }
    Assert-Hash $plan.archive.path $plan.archive.sha256
    $record.protected_verified = $true
    $record.status = 'complete'
} catch {
    $record.status = 'incomplete'
    $record['error'] = $_.Exception.Message
    throw
} finally {
    $record['finished_at_utc'] = [DateTime]::UtcNow.ToString('o')
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Output "Status: $($record.status); removed: $($record.removed.Count); bytes: $($record.removed_bytes); protected verified: $($record.protected_verified); audit: $manifestPath"
}
