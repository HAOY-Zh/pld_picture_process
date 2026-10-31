param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$workspacePrefix = $workspace.TrimEnd('\') + '\'
$relativeTargets = @(
    'data/lung_live_checks_20261016/edge_profile',
    'data/lung_live_partial_checks_20261016/edge_profile',
    'data/lung_live_local3d_checks_20261016/edge_profile',
    'data/lung_live_expert_checks_20261016/edge_profile',
    'data/ct_platform_fast_checks_20261017/edge-profile',
    'data/ct_platform_checks_20261017/browser_profile',
    'data/ct_workspace_checks_20261017_v3/edge-profile',
    'data/ct_workspace_checks_20261017/browser_profile'
)

function Assert-SafePath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside workspace: $resolved"
    }
    $cursor = $resolved
    while ($cursor -and $cursor -ne $workspace) {
        $entry = Get-Item -LiteralPath $cursor -Force
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing linked path: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $resolved
}

$targets = @()
foreach ($relative in $relativeTargets) {
    $path = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $path = Assert-SafePath $path
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Not directory: $path" }
    $entries = @(Get-ChildItem -LiteralPath $path -Recurse -Force)
    if ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
        throw "Linked entry in target: $path"
    }
    $files = @($entries | Where-Object { -not $_.PSIsContainer })
    $targets += [pscustomobject]@{
        path = $path
        bytes = [long](($files | Measure-Object Length -Sum).Sum)
        files = @($files | ForEach-Object {
            [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; ticks = $_.LastWriteTimeUtc.Ticks }
        })
    }
}

function Assert-ProfilesUnused {
    $processes = @(Get-CimInstance Win32_Process -Filter "Name='msedge.exe'")
    foreach ($process in $processes) {
        if ([string]::IsNullOrWhiteSpace($process.CommandLine)) {
            throw "Cannot inspect Edge process $($process.ProcessId); no deletion."
        }
        $command = $process.CommandLine.Replace('/', '\')
        foreach ($target in $targets) {
            if ($command.IndexOf($target.path, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "Target used by Edge process $($process.ProcessId): $($target.path)"
            }
        }
    }
}
Assert-ProfilesUnused
$bytes = [long](($targets | Measure-Object bytes -Sum).Sum)
Write-Output "Browser test cache targets: $($targets.Count), bytes: $bytes, MiB: $([math]::Round($bytes / 1MB, 2))"
if (-not $Execute -or $targets.Count -eq 0) { return }

# Protect all sibling test evidence, plus project sources, tests and docs.
$protected = @{}
foreach ($target in $targets) {
    $parent = [IO.Path]::GetDirectoryName($target.path)
    foreach ($file in (Get-ChildItem -LiteralPath $parent -Recurse -File -Force)) {
        if ($file.FullName.StartsWith($target.path + '\', [StringComparison]::OrdinalIgnoreCase)) { continue }
        $protected[$file.FullName] = $file
    }
}
$extensions = @('.py', '.ps1', '.v', '.sv', '.vhd', '.xdc', '.tcl', '.html', '.css', '.js', '.md', '.h', '.cpp')
foreach ($name in @('python', 'pynq', 'fpga', 'docs')) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $workspace $name) -Recurse -File -Force)) {
        if ($file.Extension -in $extensions) { $protected[$file.FullName] = $file }
    }
}
$hashes = @($protected.Values | ForEach-Object {
    [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
$record = [ordered]@{
    status = 'planned'
    scope = 'Eight explicit regenerable Edge test profiles only; no code, medical arrays, models, release files, test reports, screenshots or logs removed.'
    workspace = $workspace
    bytes = $bytes
    targets = $targets
    protected = $hashes
    removed = @()
    removed_bytes = 0
    protected_verified = $false
}
$auditFolder = Join-Path $workspace 'data/project_cleanup_audit_20261018'
if (-not (Test-Path -LiteralPath $auditFolder)) { New-Item -ItemType Directory -Path $auditFolder | Out-Null }
$manifest = Join-Path $auditFolder ('browser_cleanup_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.json')
if (Test-Path -LiteralPath $manifest) { throw 'Cleanup manifest exists.' }
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
try {
    foreach ($target in $targets) {
        $checked = Assert-SafePath $target.path
        Assert-ProfilesUnused
        $entries = @(Get-ChildItem -LiteralPath $checked -Recurse -Force)
        if ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw "Linked target: $checked" }
        $currentFiles = @($entries | Where-Object { -not $_.PSIsContainer })
        if ($currentFiles.Count -ne $target.files.Count) { throw "Target changed: $checked" }
        foreach ($item in $target.files) {
            $current = Get-Item -LiteralPath $item.path -Force
            if ($current.Length -ne $item.bytes -or $current.LastWriteTimeUtc.Ticks -ne $item.ticks) {
                throw "Target file changed: $($item.path)"
            }
        }
        # The absolute target and every descendant were checked within this workspace.
        Remove-Item -LiteralPath $checked -Recurse -Force
        if (Test-Path -LiteralPath $checked) { throw "Target remains: $checked" }
        $record.removed += $checked
        $record.removed_bytes += $target.bytes
    }
    foreach ($item in $hashes) {
        if (-not (Test-Path -LiteralPath $item.path -PathType Leaf)) { throw "Protected file missing: $($item.path)" }
        if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash -ne $item.sha256) {
            throw "Protected file changed: $($item.path)"
        }
    }
    $record.protected_verified = $true
    $record.status = 'complete'
} catch {
    $record.status = 'incomplete'
    $record['error'] = $_.Exception.Message
    throw
} finally {
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
    Write-Output "Audit: $manifest; status: $($record.status); removed directories: $($record.removed.Count); removed bytes: $($record.removed_bytes); protected files: $($hashes.Count)"
}
