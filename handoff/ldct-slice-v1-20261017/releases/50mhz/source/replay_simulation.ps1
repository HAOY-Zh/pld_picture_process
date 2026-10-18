param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$VivadoBin='D:\Xilinx\Vivado\2018.2\bin'
)
$ErrorActionPreference='Stop'
$bundle=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($OutDir)
if(Test-Path -LiteralPath $out){throw 'Use a new output directory'}
New-Item -ItemType Directory -Path $out | Out-Null
Copy-Item -LiteralPath (Join-Path $bundle 'verification/vectors') -Destination (Join-Path $out 'vectors') -Recurse
Get-ChildItem -LiteralPath (Join-Path $out 'vectors') -File | Copy-Item -Destination $out
$env:PROCESSOR_ARCHITECTURE='AMD64'
Push-Location $out
try{
    & (Join-Path $VivadoBin 'xvlog.bat') --sv -i (Join-Path $PSScriptRoot 'fpga/ct_denoise_model') -i $out (Join-Path $PSScriptRoot 'fpga/ct_unsigned_divider.v') (Join-Path $PSScriptRoot 'fpga/ct_denoise_tile_axis.v') (Join-Path $PSScriptRoot 'fpga/tb_ct_denoise_tile_axis.sv')
    if($LASTEXITCODE -ne 0){throw 'xvlog failed'}
    & (Join-Path $VivadoBin 'xelab.bat') tb_ct_denoise_tile_axis -s tile_sim
    if($LASTEXITCODE -ne 0){throw 'xelab failed'}
    & (Join-Path $VivadoBin 'xsim.bat') tile_sim -runall
    if($LASTEXITCODE -ne 0 -or -not (Select-String -LiteralPath xsim.log -Pattern 'CT_TILE_PASS' -Quiet) -or (Select-String -LiteralPath xsim.log -Pattern 'Fatal:|ERROR:' -Quiet)){throw 'Simulation failed'}
}finally{Pop-Location}
