# Rebuild only the removed test inputs. Never reruns training or evaluation and
# never overwrites historical dataset/protocol/verification metadata.
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$prefix = $workspace.TrimEnd('\') + '\'
$audit = Join-Path $workspace 'data/project_cleanup_audit_20261018'
$plan = Get-Content -LiteralPath (Join-Path $audit 'verified_cache_plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$items = @($plan.files | Where-Object { $_.reason -eq 'reconstructable_27_case_test_input' })
if ($items.Count -ne 81 -or $plan.workspace -ne $workspace) { throw 'Unexpected cache identity.' }
$cache = Join-Path $workspace 'data/lung_prune_test_cache_20261017'

function Assert-SafePath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    $cursor = $absolute
    while ($cursor -and $cursor -ne $workspace) {
        if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Refusing linked path: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

$missing = @()
foreach ($item in $items) {
    $path = Assert-SafePath $item.path
    if (Test-Path -LiteralPath $path) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Conflicting existing cache: $path" }
    } else { $missing += $item }
}
if ($missing.Count -eq 0) { Write-Output 'All 81 original arrays already match; nothing to restore.'; return }
foreach ($item in $plan.retained | Where-Object { $_.path.StartsWith($cache + '\', [StringComparison]::OrdinalIgnoreCase) }) {
    $path = Assert-SafePath $item.path
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Historical cache metadata changed: $path" }
}
$staging = Assert-SafePath (Join-Path $audit ('cache_rebuild_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')))
if (Test-Path -LiteralPath $staging) { throw 'Staging path already exists.' }
Push-Location -LiteralPath $workspace
try {
    & (Join-Path $workspace '.venv-lung/Scripts/python.exe') (Join-Path $workspace 'python/lung_prune_test_cache.py') --outdir $staging
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed; preserved diagnostics in $staging" }
} finally { Pop-Location }
# Check all regenerated array bytes before moving any into the historical cache.
foreach ($item in $items) {
    $relative = $item.path.Substring($cache.Length + 1)
    $source = Assert-SafePath (Join-Path $staging $relative)
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $item.sha256) { throw "Restored byte identity differs: $source" }
}
foreach ($item in $missing) {
    $destination = Assert-SafePath $item.path
    if (Test-Path -LiteralPath $destination) { throw "Destination appeared during restore: $destination" }
    $parent = Assert-SafePath ([IO.Path]::GetDirectoryName($destination))
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw "Missing original metadata directory: $parent" }
    $source = Assert-SafePath (Join-Path $staging $item.path.Substring($cache.Length + 1))
    Move-Item -LiteralPath $source -Destination $destination
}
foreach ($item in $items) {
    $path = Assert-SafePath $item.path
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256) { throw "Final verification failed: $path" }
}
@{ status='complete'; restored_arrays=$missing.Count; verified_arrays=81; historical_metadata_overwritten=$false;
    utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $staging 'restore_result.json') -Encoding UTF8
Write-Output "Restored $($missing.Count) arrays; all 81 original SHA256 values match. Reconstruction metadata: $staging"
