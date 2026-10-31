param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$prefix = $workspace + '\'
function Assert-SafePath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    $cursor = $absolute
    while ($cursor -ne $workspace) {
        if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked path: $cursor" }
        $cursor = Split-Path $cursor -Parent
    }
}
$retired = @('lung_refine_20261012', 'lung_body_crop_20261013', 'lung_distillation_20261013',
    'lung_outer_contour_20261013', 'lung_layer_context_20261013')
$targets = @{}
foreach ($name in $retired) {
    $root = Join-Path $workspace "data/$name"
    Assert-SafePath $root
    foreach ($entry in (Get-ChildItem -LiteralPath $root -Recurse -Force)) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked entry: $entry" }
        if (-not $entry.PSIsContainer -and $entry.Name -in @('evaluation.json', 'history.json')) {
            Assert-SafePath $entry.FullName
            $targets[$entry.FullName] = $entry
        }
    }
}
$guardRoots = @('lung_body_gate_20261014', 'lung_island_cleanup_20261014', 'lung_fixed_network_20261014',
    'lung_fpga_readiness_20261014', 'lung_tile8_20261014', 'lung_rtl_20261014',
    'lung_weight_reuse_20261015', 'lung_window_20261015', 'lung_cache_axi_20261015')
$protected = @{}
$referenceCount = 0
foreach ($name in $guardRoots) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $workspace "data/$name") -Recurse -File -Filter '*.json')) {
        $protected[$file.FullName] = $file
        $raw = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        foreach ($match in [regex]::Matches($raw, '"(?:[A-Za-z]:(?:\\\\|/)|data/)(?:\\.|[^"\\])*"')) {
            $referenceCount++
            $path = ConvertFrom-Json -InputObject $match.Value
            if (-not [IO.Path]::IsPathRooted($path)) { $path = Join-Path $workspace $path }
            $path = [IO.Path]::GetFullPath($path)
            if ($targets.ContainsKey($path)) { throw "Active evidence references retired report: $path" }
        }
    }
}
if ($referenceCount -eq 0) { throw 'Reference guard unexpectedly found no paths.' }
foreach ($path in @('data/lung_training_20261011/early_stop_run/best.pt',
    'data/lung_learning_20261009/dataset.json', 'data/lung_body_gate_20261014/browser/index.html')) {
    $file = Get-Item -LiteralPath (Join-Path $workspace $path)
    $protected[$file.FullName] = $file
}
$planned = @($targets.Values | Sort-Object FullName)
$bytes = ($planned | Measure-Object Length -Sum).Sum
Write-Output "Planned reports: $($planned.Count); bytes: $bytes; MiB: $([math]::Round($bytes / 1MB, 3)); checked references: $referenceCount"
$planned | ForEach-Object { Write-Output $_.FullName.Substring($prefix.Length) }
if (-not $Execute) { return }
$manifest = Join-Path $workspace ('data/cleanup_retired_reports_20261015_' + [DateTime]::UtcNow.ToString('HHmmssfff') + '.json')
if (Test-Path -LiteralPath $manifest) { throw 'Manifest already exists.' }
$retained = @(Get-ChildItem -LiteralPath $workspace -Force | Where-Object { $_.Name -notin @('.venv-lung', '.git') } | ForEach-Object {
    if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force } else { $_ }
} | Where-Object { -not $targets.ContainsKey($_.FullName) } | ForEach-Object {
    [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; ticks = $_.LastWriteTimeUtc.Ticks }
})
$hashes = @($protected.Values | ForEach-Object {
    [pscustomobject]@{ path = $_.FullName; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
$record = [ordered]@{
    status = 'planned'; utc = [DateTime]::UtcNow.ToString('o'); workspace = $workspace
    authorization = 'User approved deletion of rejected-experiment detailed reports and training histories, preserving written conclusions and configuration.'
    recovery = 'Permanent deletion, not recycle bin. Deleted report bytes are not backed up; rerunning the experiment may regenerate reports, but stochastic training need not reproduce the exact old results.'
    bytes = $bytes; files = @($planned | ForEach-Object {
        [pscustomobject]@{ path = $_.FullName; bytes = $_.Length; ticks = $_.LastWriteTimeUtc.Ticks; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    }); retained_metadata = $retained; protected_hashes = $hashes; removed = @(); verified = $false
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
try {
    foreach ($item in $record.files) {
        Assert-SafePath $item.path
        $current = Get-Item -LiteralPath $item.path
        if ($current.Length -ne $item.bytes -or $current.LastWriteTimeUtc.Ticks -ne $item.ticks -or
            (Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Report changed: $($item.path)" }
        Remove-Item -LiteralPath $item.path
        $record.removed += $item.path
    }
    $after = @{}
    foreach ($file in (Get-ChildItem -LiteralPath $workspace -Force | Where-Object { $_.Name -notin @('.venv-lung', '.git') } | ForEach-Object {
        if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force } else { $_ }
    })) { $after[$file.FullName] = $file }
    foreach ($item in $retained) {
        $file = $after[$item.path]
        if ($null -eq $file -or $file.Length -ne $item.bytes -or $file.LastWriteTimeUtc.Ticks -ne $item.ticks) { throw "Retained file changed: $($item.path)" }
    }
    foreach ($item in $hashes) {
        if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Protected evidence changed: $($item.path)" }
    }
    $record.verified = $true
    $record.status = 'complete'
} catch {
    $record.status = 'incomplete'; $record['error'] = $_.Exception.Message; throw
} finally {
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
    Write-Output "Status: $($record.status); removed: $($record.removed.Count); manifest: $manifest"
}
