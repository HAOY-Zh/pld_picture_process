param([string]$VivadoRoot='D:\vivado\Vivado\2018.2',[switch]$Synthesize)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskBuild=Join-Path $taskRoot 'build\sim'
$taskLogs=Join-Path $taskRoot 'results\fpga_statistics'
New-Item -ItemType Directory -Force -Path $taskBuild,$taskLogs | Out-Null
function Invoke-StatsTool([string]$Tool,[string[]]$ToolArgs,[string]$Log){
    & (Join-Path $VivadoRoot ('bin\'+$Tool+'.bat')) @ToolArgs 2>&1 | Tee-Object -FilePath (Join-Path $taskLogs $Log)
    if($LASTEXITCODE -ne 0){throw "$Tool failed: $LASTEXITCODE"}
}
Push-Location $taskBuild
try{
    Invoke-StatsTool 'xvlog' @((Join-Path $taskRoot 'fpga\pynq_z2_roi.v'),(Join-Path $taskRoot 'fpga\pynq_z2_roi_statistics.v')) 'xvlog_verilog.log'
    Invoke-StatsTool 'xvlog' @('-sv','-i',(Join-Path $taskRoot 'vectors'),(Join-Path $taskRoot 'tests\tb_roi_statistics.sv')) 'xvlog_tb.log'
    Invoke-StatsTool 'xelab' @('tb_roi_statistics','-s','roi_stats_sim','-debug','typical') 'xelab.log'
    Invoke-StatsTool 'xsim' @('roi_stats_sim','-runall') 'xsim.log'
    if(-not(Select-String -LiteralPath (Join-Path $taskLogs 'xsim.log') -Pattern 'ROI_STATISTICS_XSIM_ALL_PASS' -Quiet)){throw 'Missing statistics simulation PASS'}
    if($Synthesize){
        Invoke-StatsTool 'vivado' @('-mode','batch','-nolog','-nojournal','-source',(Join-Path $PSScriptRoot 'synth_statistics.tcl')) 'implementation_console.log'
        if(-not(Select-String -LiteralPath (Join-Path $taskLogs 'implementation_console.log') -Pattern 'ROI_STATISTICS_OOC_PASS' -Quiet)){throw 'Missing OOC implementation PASS'}
    }
}finally{Pop-Location}
