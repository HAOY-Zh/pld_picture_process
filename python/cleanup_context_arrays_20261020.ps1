param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$target = [IO.Path]::GetFullPath((Join-Path $workspace 'data/lung_nodule_context_20261020_v1/patches'))
$expected = $workspace.TrimEnd('\') + '\data\lung_nodule_context_20261020_v1\patches'
if ($target -ne $expected) { throw 'Unexpected absolute deletion target.' }
foreach ($folder in @($workspace, (Split-Path $target), $target)) {
    if ((Get-Item -LiteralPath $folder -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing linked directory.' }
}
$run = Split-Path $target
$scope = Get-Content -LiteralPath (Join-Path $run 'scope_completion.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($scope.status -ne 'COMPLETE_FOR_USER_REQUESTED_SCOPE') { throw 'Training/evaluation not complete.' }
$catalogue = Get-Content -LiteralPath (Join-Path $run 'training_catalogue.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$files = @(Get-ChildItem -LiteralPath $target -File -Filter '*.npy')
$preserved = @{}
foreach ($relative in @('plan.json','training_catalogue.json','scope_completion.json','final_comparison.json',
    'near_far5/model.pt','near_far5/model_freeze.json','near_far5/audit.json','near_far5/evaluation.json',
    'context7/model.pt','context7/model_freeze.json','context7/audit.json','context7/evaluation.json',
    'standalone_timing/results.json')) {
    $p = Join-Path $run $relative
    $preserved[$p] = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash
}
$bytes = ($files | Measure-Object Length -Sum).Sum
Write-Output "Exact target: $target; files: $($files.Count); bytes: $bytes"
if (-not $Execute) { return }
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")
foreach ($task in $active) {
    if ([string]::IsNullOrWhiteSpace($task.CommandLine)) { throw 'Cannot inspect running Python process.' }
    if ($task.CommandLine -match 'lung_nodule_context|lung_nodule_expanded') { throw 'Related training/evaluation still active.' }
}
$audit = Join-Path $workspace 'data/project_cleanup_20261020'
New-Item -ItemType Directory -Path $audit -Force | Out-Null
$journal = Join-Path $audit 'context_arrays.json'
if (Test-Path -LiteralPath $journal) { throw 'Preserve existing cleanup journal.' }
$entries = @()
foreach ($file in $files) {
    if ($file.DirectoryName -ne $target -or $file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unsafe exact file.' }
    $property = $catalogue.cache_sha256.PSObject.Properties[$file.FullName]
    if ($null -eq $property -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne $property.Value) { throw "Cache identity mismatch: $($file.FullName)" }
    $entries += [ordered]@{path=$file.FullName; bytes=$file.Length; sha256=$property.Value}
}
$record = [ordered]@{status='verified_before_deletion'; workspace=$workspace; target=$target;
    started_utc=[DateTime]::UtcNow.ToString('o'); files=$entries; removed_count=0; removed_bytes=[long]0;
    preserved_sha256=$preserved; reason='Completed shared nine-channel context cache; models, metadata, metrics, masks, source ZIPs retained';
    recovery='python/lung_nodule_context_cache_restore.py; rebuild exact arrays from retained original five-channel caches and original CT archives'}
$record | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $journal -Encoding UTF8
foreach ($entry in $entries) {
    # Only delete a verified, direct child file. No recursive removal.
    Remove-Item -LiteralPath $entry.path -Force
    $record.removed_count++
    $record.removed_bytes += $entry.bytes
}
foreach ($p in $preserved.Keys) {
    if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash -ne $preserved[$p]) { throw 'Preserved output changed.' }
}
$record.status='COMPLETE'
$record.completed_utc=[DateTime]::UtcNow.ToString('o')
$record | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $journal -Encoding UTF8
Write-Output "Removed $($record.removed_count) arrays; freed $($record.removed_bytes) bytes; preserved outputs verified."
