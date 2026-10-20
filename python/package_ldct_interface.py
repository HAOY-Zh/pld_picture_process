"""Extend an immutable, verified tile bundle with ldct_slice_v1 integration files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from ctdenoise.cnn_fixed import infer_fixed, load_quantized
from ctdenoise.pl_reference import fixed_filter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pynq'))
from ldct_processor import interface_contract


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main(a):
    base, out = a.base_bundle.resolve(), a.outdir.resolve()
    if out.exists():
        raise ValueError('Use a new release directory')
    manifest = json.loads((base/'manifest.json').read_text())
    for relative, sha in manifest['files_sha256'].items():
        p = (base/relative).resolve()
        if base not in p.parents or digest(p) != sha:
            raise ValueError('Base release is not intact: '+relative)
    evidence = json.loads((base/'verification/simulation_manifest.json').read_text(encoding='utf-8-sig'))
    if evidence['result'] != 'PASS' or evidence['frames'] < 78:
        raise ValueError('Expanded RTL simulation required')
    for name, sha in evidence['rtl_sha256'].items():
        if digest(a.build/'rtl'/name) != sha:
            raise ValueError('Source, build and simulation identity differs: '+name)
    out.mkdir(parents=True)
    for relative in manifest['files_sha256']:
        destination = out/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/relative, destination)
    for name in ['ldct_processor.py', 'benchmark_ldct_interface.py', 'ct_denoise_tile_pl_test.py']:
        shutil.copy2(ROOT/'pynq'/name, out/'pynq'/name)
    src = out/'source'
    (src/'fpga/ct_denoise_model').mkdir(parents=True)
    for name in ['ct_denoise_tile_axis.v', 'ct_unsigned_divider.v', 'ct_denoise_roms.vh']:
        destination = src/'fpga'/('ct_denoise_model/'+name if name.endswith('.vh') else name)
        shutil.copy2(a.build/'rtl'/name, destination)
    shutil.copy2(ROOT/'fpga/build_ct_denoise_tile.tcl', src/'fpga/build_ct_denoise_tile.tcl')
    shutil.copy2(ROOT/'fpga/tb_ct_denoise_tile_axis.sv', src/'fpga/tb_ct_denoise_tile_axis.sv')
    shutil.copy2(base/'model/rtl_model_manifest.json', src/'fpga/ct_denoise_model/model_manifest.json')
    shutil.copytree(ROOT/'board_files/pynq-z2', src/'board_files/pynq-z2')
    (src/'tests').mkdir()
    shutil.copy2(ROOT/'python/tests/test_ldct_processor.py', src/'tests/test_ldct_processor.py')
    vectors = out/'verification/vectors'
    vectors.mkdir()
    for p in a.simulation.iterdir():
        if p.suffix == '.hex' or p.name in ('vectors.vh', 'cases.json'):
            shutil.copy2(p, vectors/p.name)
    (src/'replay_simulation.ps1').write_text('''param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$VivadoBin='D:\\Xilinx\\Vivado\\2018.2\\bin'
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
''', encoding='utf-8')
    data = ROOT/'data/denoise'
    if digest(a.float_checkpoint) != manifest['model']['checkpoint_sha256']:
        raise ValueError('Equivalent floating checkpoint identity differs from integer model')
    if digest(a.source_checkpoint) != manifest['model']['source_checkpoint_sha256']:
        raise ValueError('Original trained checkpoint identity differs')
    shutil.copy2(a.float_checkpoint, out/'model/equivalent_model.pt')
    shutil.copy2(a.source_checkpoint, out/'model/source_trained_best.pt')
    (out/'model/FLOAT_INFERENCE.txt').write_text(
        'Optional PC FP32 reference: three replicate-padded Conv3x3 layers 1->8->8->1, ReLU after first two.\n'
        'Normalized input HU/1024; output=(input-predicted_noise)*1024. No display clipping/resizing.\n'
        'equivalent_model.pt is the compensated hidden-channel parameterization; source_trained_best.pt is original best epoch118.\n'
        'Both store schema=1/model state_dict/channels/hu_scale. Use model.float() after loading for FP32.\n'
        'Torch is optional on PC only; board interface and integer reference require NumPy and PYNQ, no Torch.\n', encoding='utf-8')
    arrays, spec = load_quantized(out/'model')
    fixtures = out/'fixtures'
    fixtures.mkdir()
    selections = []
    def add(name, image, method):
        mode = interface_contract(manifest['clock_mhz'])['methods'][method]
        rounded = np.rint(image).astype(np.int16)
        expected = infer_fixed(rounded, arrays, spec)[0] if mode == 6 else fixed_filter(rounded, mode)
        np.save(fixtures/(name+'_input.npy'), image, allow_pickle=False)
        np.save(fixtures/(name+'_expected.npy'), expected, allow_pickle=False)
        selections.append(dict(name=name, method=method, input='fixtures/'+name+'_input.npy',
                               expected='fixtures/'+name+'_expected.npy', shape=list(image.shape)))
    for patient in ['C067', 'C077']:
        image = np.load(out/f'samples/{patient}_input_hu.npy')
        add(patient, image, 'cnn')
    a0 = np.load(out/'samples/C067_input_hu.npy')
    for name, image in [('edge67x69', a0[200:267,110:179]), ('row1x69', a0[200:201,110:179]),
                        ('column67x1', a0[200:267,110:111]), ('single', a0[200:201,110:111])]:
        add(name, image, 'cnn')
    add('float_ties', np.array([[-32768.,-2.5,-1.5,-.5,.5,1.5,2.5,32767.4]],np.float32), 'cnn')
    for method in interface_contract(manifest['clock_mhz'])['methods']:
        add('method_'+method, a0[208:227,112:135], method)
    write_json(fixtures/'cases.json', selections)
    clock = float(manifest['clock_mhz'])
    manifest.update(interface_contract(clock))
    manifest['release_id'] = a.release_id
    manifest['primary_model'] = 'new_epoch118_equalized_channel5_qsafe'
    manifest['model_id'] = spec['checkpoint_sha256']
    manifest['source_checkpoint_sha256'] = spec['source_checkpoint_sha256']
    manifest['scope'] = 'Timing- and simulation-verified release; board acceptance records stored separately by caller'
    manifest['data_protocol'] = dict(source='IDC LDCT-and-Projection-data Siemens chest reconstruction',
        train=['C002','C004','C012','C016','C021','C027','C081','C095','C099','C107','C166','C179'],
        validation=['C050','C052'], test=['C067','C077'], sample_slices={'C067':180,'C077':190},
        sample_patients_used_for_training=False, test_previously_inspected=True,
        metrics='Unclipped HU RMSE; fixed [-1000,1000] HU window PSNR/SSIM with data_range=2000; display [-1000,400] only')
    (out/'README.txt').write_text(
        'LDCT slice v1 release: '+a.release_id+'\n\n'
        'PRIMARY MODEL: latest epoch118 + validated channel equalization + range-safe INT8/INT16 quantization.\n'
        'The service owns its hardware lock, Overlay loading, FCLK0 setting, and CMA buffers.\n'
        'After verified load and clock setup, pass borrowed DMA/GPIO/TX/RX to LDCTProcessor.\n'
        'The processor never resets, frees, reloads, retries, changes clocks or returns a CPU fallback.\n\n'
        'import sys, json\n'
        'from pathlib import Path\n'
        'bundle=Path("/absolute/path/to/this/release")\n'
        'sys.path.insert(0,str(bundle/"pynq"))\n'
        'from ldct_processor import LDCTProcessor\n'
        'from pynq import Clocks\n'
        'manifest=json.loads((bundle/"manifest.json").read_text())\n'
        'processor=LDCTProcessor(dma=dma,gpio=gpio,tx_buffer=tx_buffer,rx_buffer=rx_buffer,\n'
        '    model_dir=bundle/"model",read_clock_mhz=lambda:float(Clocks.fclk0_mhz),\n'
        '    expected_clock_mhz=manifest["clock_mhz"],timeout_s=10.0)\n'
        'result=processor.process_slice(hu,method="cnn")\n'
        'processor.assert_idle()\n'
        'enhanced_hu=result["hu"]\n\n'
        'Input dimensions: each 1..512, int16/float32 raw HU; output independent int16, same shape/coordinates.\n'
        'The service retains spacing/origin/direction. No display window or 0..1 normalization enters this API.\n'
        'If an exception leaves faulted/dma_may_access true, caller must recover before freeing memory or replacing bit.\n'
        'Reset/recovery is deliberately absent from LDCTProcessor. Reconstruct it after verified external recovery.\n\n'
        'Replay frozen RTL vectors without patient cache:\n'
        'powershell -ExecutionPolicy Bypass -File source/replay_simulation.ps1 -OutDir ABS_NEW_SIM_DIR\n'
        'Rebuild this frozen source (new folder, board preset included):\n'
        'vivado -mode batch -source source/fpga/build_ct_denoise_tile.tcl -tclargs ABS_NEW_BUILD_DIR ABS_RELEASE/source/board_files/pynq-z2 build ABS_RELEASE/source/fpga/ct_denoise_model '+str(int(clock))+'\n\n'
        'Host contract tests (NumPy only): python -m unittest discover -s source/tests -p test_ldct_processor.py -v\n'
        'Owned-resource board benchmark: see pynq/benchmark_ldct_interface.py --help; does not implement an HTTP service.\n'
        'Record actual readback frequency before/after each call. Never overclock a 50MHz bit to claim a 100MHz release.\n', encoding='utf-8')
    manifest['files_sha256'] = {str(p.relative_to(out)).replace('\\','/'):digest(p)
                               for p in out.rglob('*') if p.is_file() and p != out/'manifest.json'}
    write_json(out/'manifest.json', manifest)
    archive = shutil.make_archive(str(out), 'zip', out.parent, out.name)
    print(json.dumps(dict(release=str(out), zip=archive, files=len(manifest['files_sha256']), fixtures=len(selections)),indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--simulation', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--release-id', required=True)
    p.add_argument('--float-checkpoint',type=Path,default=ROOT/'data/denoise/ct_quant_diagnosis_20261017_01/range_safe_quantized/equivalent_model.pt')
    p.add_argument('--source-checkpoint',type=Path,default=ROOT/'data/denoise/tinycnn_12train_detail_long01/best.pt')
    main(p.parse_args())
