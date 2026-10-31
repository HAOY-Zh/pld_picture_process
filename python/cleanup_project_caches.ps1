param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$prefix = $workspace + [IO.Path]::DirectorySeparatorChar

function Assert-SafePath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside workspace: $absolute"
    }
    $cursor = $absolute
    while ($cursor -and $cursor -ne $workspace) {
        if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing linked path: $cursor"
        }
        $cursor = Split-Path $cursor -Parent
    }
}

$rules = @(
    @{ root = 'data/lung_outer_contour_20261013/paired_screen'; kind = 'arrays' },
    @{ root = 'data/lung_distillation_20261013/teacher_cache'; kind = 'arrays' },
    @{ root = 'data/lung_reference_20261012/r231_hu_comparison'; kind = 'arrays' },
    @{ root = 'data/lung_case_browser_20261014'; kind = 'old_html' },
    @{ root = 'data/lung_3d_viewer_20261013'; kind = 'old_exports' },
    @{ root = 'data/lung_external_20261012/prepared/cache'; kind = 'arrays' },
    @{ root = 'data/lung_training_20261011/heldout_2d_evaluation/cache'; kind = 'arrays' }
)
$targets = @{}
foreach ($rule in $rules) {
    $root = Join-Path $workspace $rule.root
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Assert-SafePath $root
    foreach ($entry in (Get-ChildItem -LiteralPath $root -Recurse -Force)) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked entry: $($entry.FullName)" }
        if ($entry.PSIsContainer) { continue }
        $selected = switch ($rule.kind) {
            'arrays' { $entry.Extension -eq '.npy' }
            'old_html' { $entry.Extension -eq '.html' }
            'old_exports' { $entry.Extension -in @('.html', '.npz') -or $entry.Name.EndsWith('.nii.gz') }
        }
        if ($selected) { $targets[$entry.FullName] = $entry }
    }
}

$fpga = Join-Path $workspace 'fpga'
$excluded = (Join-Path $fpga 'Lab_9') + '\'
$cacheDirs = @(Get-ChildItem -LiteralPath $fpga -Directory -Recurse -Force | Where-Object {
    ($_.Name -like '*.cache' -or $_.Name -eq '.Xil') -and
    -not $_.FullName.StartsWith($excluded, [StringComparison]::OrdinalIgnoreCase)
})
foreach ($directory in $cacheDirs) {
    Assert-SafePath $directory.FullName
    $entries = @(Get-ChildItem -LiteralPath $directory.FullName -Recurse -Force)
    if ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw "Linked cache: $directory" }
    $files = @($entries | Where-Object { -not $_.PSIsContainer })
    foreach ($file in $files) {
        if ($file.Extension -in @('.bit', '.hwh', '.v', '.sv', '.vhd', '.xdc', '.tcl', '.xpr', '.bd', '.xci')) { continue }
        $targets[$file.FullName] = $file
    }
}

# Guard exact file references in current software and hardware evidence.
$references = @{}
$guardRoots = @('lung_body_gate_20261014', 'lung_fixed_network_20261014', 'lung_fpga_readiness_20261014',
    'lung_tile8_20261014', 'lung_rtl_20261014', 'lung_weight_reuse_20261015', 'lung_window_20261015', 'lung_cache_axi_20261015')
foreach ($name in $guardRoots) {
    $root = Join-Path $workspace "data/$name"
    foreach ($file in (Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.json')) {
        $raw = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        # Match JSON strings in keys or values without traversing huge numeric arrays.
        foreach ($match in [regex]::Matches($raw, '"[A-Za-z]:(?:\\\\|/)(?:\\.|[^"\\])*"')) {
            $path = ConvertFrom-Json -InputObject $match.Value
            $references[[IO.Path]::GetFullPath($path)] = $true
        }
    }
}
$planned = @($targets.Values | Sort-Object FullName)
foreach ($file in $planned) {
    Assert-SafePath $file.FullName
    if ($references.ContainsKey($file.FullName)) { throw "Current evidence references cleanup target: $($file.FullName)" }
}
$bytes = ($planned | Measure-Object Length -Sum).Sum
Write-Output "Planned: $($planned.Count) files, $bytes bytes ($([math]::Round($bytes / 1GB, 3)) GiB)."
$planned | Group-Object { if ($_.FullName.StartsWith($fpga)) { 'fpga caches' } else { $_.FullName.Substring($prefix.Length).Split('\')[1] } } |
    ForEach-Object { Write-Output "$($_.Name): $($_.Count) files, $(($_.Group | Measure-Object Length -Sum).Sum) bytes" }
if (-not $Execute) { return }

$manifest = Join-Path $workspace ('data/cleanup_project_caches_20261015_' + [DateTime]::UtcNow.ToString('HHmmssfff') + '.json')
if (Test-Path -LiteralPath $manifest) { throw 'Manifest already exists.' }
# Hash every retained project artifact (exclude installed environment and Git internals).
$retained = @(Get-ChildItem -LiteralPath $workspace -Force | Where-Object {
    $_.Name -notin @('.venv-lung', '.git')
} | ForEach-Object {
    if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force } else { $_ }
} | Where-Object { -not $targets.ContainsKey($_.FullName) })
Write-Output "Hashing $($retained.Count) retained files before deletion."
$hashes = @($retained | ForEach-Object {
    [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
$record = [ordered]@{
    status = 'planned'; utc = [DateTime]::UtcNow.ToString('o'); workspace = $workspace
    authorization = 'User approved both retired artifacts and regenerable evaluation caches on 2026-10-15.'
    recovery = 'Permanently deleted, not in recycle bin. Rebuild derived caches using preserved code and original CT in D:\CT2\lctsc and D:\CT3. Old HTML exports must be regenerated.'
    bytes = $bytes; files = @($planned | ForEach-Object {
        [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; lastWriteUtc = $_.LastWriteTimeUtc.ToString('o') }
    }); retained = $hashes; removed = @(); retained_verified = $false
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
try {
    foreach ($item in $record.files) {
        Assert-SafePath $item.path
        $current = Get-Item -LiteralPath $item.path -Force
        if ($current.Length -ne $item.bytes -or $current.LastWriteTimeUtc.ToString('o') -ne $item.lastWriteUtc) {
            throw "Cleanup target changed: $($item.path)"
        }
        Remove-Item -LiteralPath $item.path -Force
        $record.removed += $item.path
    }
    Write-Output 'Deletion finished; verifying retained artifact hashes.'
    foreach ($item in $hashes) {
        if (-not (Test-Path -LiteralPath $item.path -PathType Leaf)) { throw "Retained file missing: $($item.path)" }
        if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Retained file changed: $($item.path)" }
    }
    $record.retained_verified = $true
    $record.status = 'complete'
} catch {
    $record.status = 'incomplete'
    $record['error'] = $_.Exception.Message
    throw
} finally {
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
    Write-Output "Manifest: $manifest; status: $($record.status); removed: $($record.removed.Count)"
}
