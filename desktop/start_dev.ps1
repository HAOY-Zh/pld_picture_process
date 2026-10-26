param(
    [string]$DesktopPackage,
    [switch]$Diagnose,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
try {
    $sourceRoot = Split-Path -Parent $PSScriptRoot
    $settingsPath = Join-Path $sourceRoot 'data\ct_studio_dev\settings.json'
    if (-not $DesktopPackage -and (Test-Path -LiteralPath $settingsPath)) {
        $DesktopPackage = (Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json).desktop_package
    }
    if (-not $DesktopPackage -or -not (Test-Path -LiteralPath (Join-Path $DesktopPackage 'runtime\python.exe'))) {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = 'Select the extracted private CT Studio package (contains app and runtime).'
        $dialog.ShowNewFolderButton = $false
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 1 }
        $DesktopPackage = $dialog.SelectedPath
        $dialog.Dispose()
    }
    $pythonPath = Join-Path $DesktopPackage 'runtime\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The selected folder does not contain runtime\python.exe. Extract the complete private package first.' }
    $launchArgs = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'run_source.py'), '--desktop-package', $DesktopPackage)
    if ($Diagnose) { $launchArgs += '--diagnose' }
    if ($NoBrowser) { $launchArgs += '--no-browser' }
    & $pythonPath @launchArgs
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
