param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$retired = @(
    'data/lung_learning_20261011_512',
    'data/lung_body_crop_20261013',
    'data/lung_refine_20261012',
    'data/lung_outer_contour_20261013/distance_cache',
    'data/lung_distillation_20261013/paired_screen',
    'data/lung_training_20261011/resolution_512_evaluation',
    'data/lung_training_20261011/small_sampling_evaluation',
    'data/lung_training_20261011/context_25d_evaluation',
    'data/lung_training_20261011/short_run_evaluation'
)
$targets = @()
foreach ($relative in $retired) {
    $directory = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not $directory.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside workspace: $directory"
    }
    if (-not (Test-Path -LiteralPath $directory)) { continue }
    $entries = @(Get-Item -LiteralPath $directory) + @(Get-ChildItem -LiteralPath $directory -Recurse -Force)
    if ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
        throw "Refusing linked directories/files: $directory"
    }
    foreach ($file in ($entries | Where-Object { -not $_.PSIsContainer -and $_.Extension -eq '.npy' })) {
        $resolved = [IO.Path]::GetFullPath($file.FullName)
        if (-not $resolved.StartsWith($directory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Target escapes retired directory: $resolved"
        }
        $targets += [pscustomobject]@{ path = $resolved; bytes = $file.Length }
    }
}
$total = ($targets | Measure-Object bytes -Sum).Sum
Write-Output "Retired regenerable arrays: $($targets.Count); bytes: $total; GiB: $([math]::Round($total / 1GB,3))"
if (-not $Execute) { $targets | Group-Object { Split-Path $_.path -Parent } | Select-Object Name,Count; return }
$manifest = Join-Path $workspace 'data/cleanup_retired_arrays_20261013.json'
if (Test-Path -LiteralPath $manifest) { throw 'Cleanup manifest already exists; inspect it before repeating.' }
$record = [ordered]@{
    status = 'planned'; utc = [DateTime]::UtcNow.ToString('o'); workspace = $workspace
    reason = 'User authorized removal of failed-experiment regenerable arrays before layer-aware experiment.'
    recovery = 'Deleted arrays are not in recycle bin. Regenerate using retained source code and original data; reports, figures and checkpoints retained.'
    kept = @('Original 256 cache/GT/split', 'Original best.pt and last.pt', 'Held-out/external evaluations', 'INT8 v2', 'R231 weights and teacher cache', 'Latest outer-contour predictions', 'FPGA and board files', 'All source/tests/reports/figures')
    bytes = $total; files = $targets; removed = @()
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
foreach ($item in $targets) {
    $file = Get-Item -LiteralPath $item.path
    if ($file.Length -ne $item.bytes) { throw "File changed during cleanup: $($item.path)" }
    Remove-Item -LiteralPath $item.path
    $record.removed += $item.path
}
$record.status = 'complete'
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Output "Completed: $($record.removed.Count) arrays; evidence manifest: $manifest"
