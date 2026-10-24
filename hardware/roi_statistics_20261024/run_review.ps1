param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$VivadoBin='D:\Xilinx\Vivado\2018.2\bin',
    [string]$Python='D:\PLD\integration\ct-studio-runtime-20261018\.venv-web\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
$sourceRoot=$PSScriptRoot
$reviewRoot=[IO.Path]::GetFullPath($OutDir)
if(Test-Path -LiteralPath $reviewRoot){throw 'Use a fresh review output directory'}
New-Item -ItemType Directory -Path $reviewRoot | Out-Null
foreach($relative in @('fpga/pynq_z2_roi.v','fpga/pynq_z2_roi_statistics.v','tests/tb_roi_statistics.sv','scripts/generate_statistics_vectors.py','scripts/synth_statistics.tcl')){
    $destination=Join-Path $reviewRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath (Join-Path $sourceRoot $relative) -Destination $destination
}
New-Item -ItemType Directory -Force -Path (Join-Path $reviewRoot 'build/sim'),(Join-Path $reviewRoot 'results/fpga_statistics'),(Join-Path $reviewRoot 'vectors') | Out-Null
& $Python -X utf8 (Join-Path $reviewRoot 'scripts/generate_statistics_vectors.py')
if($LASTEXITCODE -ne 0){throw 'Independent vector generation failed'}
$previousArchitecture=$env:PROCESSOR_ARCHITECTURE
$previousTemp=$env:TEMP
$previousTmp=$env:TMP
try{
    if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
    New-Item -ItemType Directory -Path (Join-Path $reviewRoot 'tmp') | Out-Null
    $env:TEMP=Join-Path $reviewRoot 'tmp'
    $env:TMP=$env:TEMP
    Push-Location (Join-Path $reviewRoot 'build/sim')
    try{
        & (Join-Path $VivadoBin 'xvlog.bat') (Join-Path $reviewRoot 'fpga/pynq_z2_roi.v') (Join-Path $reviewRoot 'fpga/pynq_z2_roi_statistics.v')
        if($LASTEXITCODE -ne 0){throw 'Verilog compile failed'}
        & (Join-Path $VivadoBin 'xvlog.bat') -sv -i (Join-Path $reviewRoot 'vectors') (Join-Path $reviewRoot 'tests/tb_roi_statistics.sv')
        if($LASTEXITCODE -ne 0){throw 'Testbench compile failed'}
        & (Join-Path $VivadoBin 'xelab.bat') tb_roi_statistics -s roi_stats_sim -debug typical -mt 2
        if($LASTEXITCODE -ne 0){throw 'Elaboration failed'}
        & (Join-Path $VivadoBin 'xsim.bat') roi_stats_sim -runall
        if($LASTEXITCODE -ne 0){throw 'Simulation failed'}
        $simulation=Get-Content -LiteralPath xsim.log -Raw
        if($simulation -notmatch 'ROI_STATISTICS_XSIM_ALL_PASS frames=9 pixels=803180' -or $simulation -match 'Fatal:|ERROR:'){throw 'Simulation did not finish all cases cleanly'}
    }finally{Pop-Location}
    $synthScript=Join-Path $reviewRoot 'scripts/synth_statistics.tcl'
    # Use the source path as a Tcl literal; no shell-built command or evaluator.
    $wrapper="set_param general.maxThreads 2`nsource {$($synthScript -replace '\\','/')}"
    $wrapperPath=Join-Path $reviewRoot 'review_synth.tcl'
    Set-Content -LiteralPath $wrapperPath -Value $wrapper -Encoding ASCII
    Push-Location $reviewRoot
    try{
        & (Join-Path $VivadoBin 'vivado.bat') -mode batch -nolog -nojournal -source $wrapperPath
        if($LASTEXITCODE -ne 0){throw 'OOC implementation failed'}
    }finally{Pop-Location}
}finally{$env:PROCESSOR_ARCHITECTURE=$previousArchitecture;$env:TEMP=$previousTemp;$env:TMP=$previousTmp}
& $Python -X utf8 (Join-Path $sourceRoot 'verify_review.py') --source $sourceRoot --result $reviewRoot
if($LASTEXITCODE -ne 0){throw 'Independent returned-statistics verification failed'}
