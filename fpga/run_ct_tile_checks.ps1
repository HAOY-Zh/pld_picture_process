param(
    [string]$VivadoBin='D:\Xilinx\Vivado\2018.2\bin',
    [string]$Python='D:\anaconda3\envs\PytorchTest\python.exe',
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$QuantizedDir,
    [string]$RomDir,
    [switch]$Stress
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$primary=Join-Path $root 'configs\ldct_primary_model.json'
if(-not $QuantizedDir){
    if(Test-Path -LiteralPath $primary){$QuantizedDir=Join-Path $root ((Get-Content -LiteralPath $primary -Raw -Encoding UTF8 | ConvertFrom-Json).model_dir)}
    else{$QuantizedDir=Join-Path $root 'data\denoise\ct_quant_diagnosis_20261017_01\range_safe_quantized'}
}
if(-not $RomDir){$RomDir=Join-Path $PSScriptRoot 'ct_denoise_model_ldctv1'}
$RomDir=[IO.Path]::GetFullPath($RomDir)
$out=[IO.Path]::GetFullPath($OutDir)
if(Test-Path -LiteralPath $out){throw "Use new output: $out"}
$extra=@();if($Stress){$extra=@('--stress')}
$cache=Join-Path $root 'data\denoise\tinycnn_cache_12train'
$sampleArgs=@()
if(Test-Path -LiteralPath (Join-Path $cache 'C067\low.npy')){$sampleArgs=@('--cache',$cache)}
elseif(Test-Path -LiteralPath $primary){
    $release=(Get-Content -LiteralPath $primary -Raw -Encoding UTF8 | ConvertFrom-Json).preferred_release
    $sample=Join-Path (Join-Path $root $release) 'samples\C067_input_hu.npy'
    if(Test-Path -LiteralPath $sample){$sampleArgs=@('--sample-input',$sample)}
}
if(-not $sampleArgs){throw 'Provide the local patient cache or the primary handoff sample before replaying all 78 cases'}
& $Python (Join-Path $root 'python\make_denoise_tile_vectors.py') --quantized-dir $QuantizedDir --romdir $RomDir --outdir $out @sampleArgs @extra
if($LASTEXITCODE -ne 0){throw 'Vector export failed'}
$previousArchitecture=$env:PROCESSOR_ARCHITECTURE
try{
    if(-not $env:PROCESSOR_ARCHITECTURE -and [Environment]::Is64BitOperatingSystem){$env:PROCESSOR_ARCHITECTURE='AMD64'}
    Push-Location $out
    try{
        & (Join-Path $VivadoBin 'xvlog.bat') --sv -i $RomDir -i $out (Join-Path $PSScriptRoot 'ct_unsigned_divider.v') (Join-Path $PSScriptRoot 'ct_denoise_tile_axis.v') (Join-Path $PSScriptRoot 'tb_ct_denoise_tile_axis.sv')
        if($LASTEXITCODE -ne 0){throw 'xvlog failed'}
        & (Join-Path $VivadoBin 'xelab.bat') tb_ct_denoise_tile_axis -s tile_sim
        if($LASTEXITCODE -ne 0){throw 'xelab failed'}
        & (Join-Path $VivadoBin 'xsim.bat') tile_sim -runall
        if($LASTEXITCODE -ne 0 -or -not (Select-String -LiteralPath xsim.log -Pattern 'CT_TILE_PASS' -Quiet) -or (Select-String -LiteralPath xsim.log -Pattern 'Fatal:|ERROR:' -Quiet)){throw 'RTL mismatch; inspect xsim.log'}
        $hashes=@{}
        foreach($name in @('ct_denoise_tile_axis.v','ct_unsigned_divider.v')){
            $hashes[$name]=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $name) -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $hashes['ct_denoise_roms.vh']=(Get-FileHash -LiteralPath (Join-Path $RomDir 'ct_denoise_roms.vh') -Algorithm SHA256).Hash.ToLowerInvariant()
        $simText=Get-Content -LiteralPath xsim.log -Raw
        $simMatch=[regex]::Match($simText,'CT_TILE_PASS frames=(\d+) pixels=(\d+)')
        @{rtl_sha256=$hashes;stress=[bool]$Stress;result='PASS';frames=[int]$simMatch.Groups[1].Value;output_pixels=[int]$simMatch.Groups[2].Value;cnn_compact_protocol=[bool]$Stress} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'simulation_manifest.json') -Encoding UTF8
    }finally{Pop-Location}
}finally{$env:PROCESSOR_ARCHITECTURE=$previousArchitecture}
