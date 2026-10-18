"""Package only a successfully routed and timing-checked PL implementation."""
import argparse
import hashlib
import json
import shutil
import re
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root=Path(__file__).resolve().parents[1];build=Path(args.build).resolve();out=Path(args.outdir).resolve()
    log=Path(str(build)+'.log')
    if not log.is_file() or 'CT_TILE_OVERLAY_READY:' not in log.read_text(errors='replace'):
        raise ValueError('No successful bitstream+timing completion marker; refuse to package')
    if out.exists():raise ValueError('Choose a new bundle directory')
    primary=root/'configs/ldct_primary_model.json'
    default_model=root/json.loads(primary.read_text(encoding='utf-8'))['model_dir'] if primary.is_file() else root/'data/denoise/ct_quant_diagnosis_20261017_01/range_safe_quantized'
    model=Path(args.quantized_dir).resolve() if args.quantized_dir else default_model
    rom=Path(args.romdir).resolve() if args.romdir else root/'fpga/ct_denoise_model_ldctv1'
    out.mkdir(parents=True)
    for folder in ('overlay','model','samples','pynq','python/ctdenoise','verification'):(out/folder).mkdir(parents=True)
    for ext in ('bit','hwh'):
        shutil.copy2(build/f'overlay/ct_denoise_tile.{ext}',out/f'overlay/ct_denoise_tile.{ext}')
    for name in ('weights_integer.npz','quantization.json'):shutil.copy2(model/name,out/'model'/name)
    shutil.copy2(rom/'model_manifest.json',out/'model/rtl_model_manifest.json')
    shutil.copy2(root/'pynq/ct_denoise_tile_pl_test.py',out/'pynq/ct_denoise_tile_pl_test.py')
    for name in ('__init__.py','filters.py','cnn_fixed.py','pl_reference.py'):
        shutil.copy2(root/'python/ctdenoise'/name,out/'python/ctdenoise'/name)
    for name in ('timing_routed.rpt','utilization_routed.rpt','drc.rpt','clock_mhz.txt'):
        shutil.copy2(build/name,out/'verification'/name)
    sim=Path(args.simulation).resolve()
    simlog=(sim/'xsim.log').read_text(errors='replace')
    passed=re.search(r'CT_TILE_PASS frames=(\d+) pixels=(\d+)',simlog)
    if not passed or int(passed[1])<42 or 'Fatal:' in simlog or 'ERROR:' in simlog:
        raise ValueError('Complete large-tile simulation evidence missing')
    evidence=json.loads((sim/'simulation_manifest.json').read_text(encoding='utf-8-sig'))
    for name in ('ct_denoise_tile_axis.v','ct_unsigned_divider.v','ct_denoise_roms.vh'):
        if sha(build/'rtl'/name)!=evidence['rtl_sha256'][name]:
            raise ValueError('Build/simulation RTL differs: '+name)
    rtl_manifest=json.loads((rom/'model_manifest.json').read_text())
    quantization=json.loads((model/'quantization.json').read_text())
    if rtl_manifest['rom_sha256']!=sha(build/'rtl/ct_denoise_roms.vh') or rtl_manifest['quantization']!=quantization:
        raise ValueError('Packaged model/ROM differs from built frozen weights')
    compact='incoming_crop_valid' in (build/'rtl/ct_denoise_tile_axis.v').read_text()
    if compact and int(passed[1])<78:raise ValueError('Compact protocol requires the expanded 78-frame simulation')
    shutil.copy2(sim/'xsim.log',out/'verification/rtl_simulation.log')
    shutil.copy2(sim/'simulation_manifest.json',out/'verification/simulation_manifest.json')
    for patient,z in [('C067',180),('C077',190)]:
        a=np.load(root/f'data/denoise/tinycnn_cache_12train/{patient}/low.npy',mmap_mode='r')[z]
        np.save(out/f'samples/{patient}_input_hu.npy',a,allow_pickle=False)
        reference=np.load(root/f'data/denoise/tinycnn_cache_12train/{patient}/reference.npy',mmap_mode='r')[z]
        np.save(out/f'samples/{patient}_reference_hu.npy',reference,allow_pickle=False)
    cells=[dict(cell_type='markdown',metadata={},source=['# CT denoising on PYNQ-Z2\n','CPU controls tiles/DMA. Denoising runs in PL. Outputs must match the integer reference.']),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['import pynq, platform\n','print("PYNQ", pynq.__version__, "Python", platform.python_version())']),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run pynq/ct_denoise_tile_pl_test.py --outdir board_results_001 --modes 0 1 2 3 4 5 6 7 --repeats 3']),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['from IPython.display import Image, display\n','display(Image(filename="board_results_001/actual_board_outputs.png"))'])]
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=4)
    (out/'run_board_test.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    if compact:
        cells=[dict(cell_type='markdown',metadata={},source=['# PYNQ-Z2 CNN speed comparison\n',
            'Same frozen weights and integer arithmetic. Each stage must match the integer reference.\n',
            'Run the entire directory after checking manifest hashes. Both samples are measured.']),
            dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
                '%run pynq/benchmark_ct_tile_speed.py --outdir board_speed_001 --repeats 5\n']),
            dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
                'import json\n','print(json.dumps(json.load(open("board_speed_001/comparison.json")), indent=2))\n'])]
        notebook['cells']=cells
        (out/'run_speed_test.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
        shutil.copy2(root/'pynq/benchmark_ct_tile_speed.py',out/'pynq/benchmark_ct_tile_speed.py')
    (out/'README.txt').write_text('PYNQ-Z2 CT denoising bundle\n\n'
        '1. Upload the entire directory to the board, preserving subdirectories.\n'
        '2. Open run_board_test.ipynb and run the cells in order.\n'
        '3. First test bypass and Gaussian: --modes 0 1. Then run all modes.\n'
        '4. Every output must match the CPU integer reference pixel-for-pixel.\n'
        '5. board_result.json records actual PL execution and complete DMA+PL+PS time.\n'
        '6. Keep board_results_001; use another output name for repeat experiments.\n\n'
        'CPU: file I/O, tile overlap/packing, DMA control, result checking and display.\n'
        'PL: Gaussian, bilateral, joint bilateral, guided filter and integer CNN.\n'
        'CNN: fixed audited weights; int8 weights/int16 activations/int32 accumulator.\n'
        'CPU reference timing is for correctness; not an optimized ARM baseline.\n'
        'PC simulation and implementation do not establish board speed or reliability.\n',encoding='utf-8')
    if compact:
        with (out/'README.txt').open('a',encoding='utf-8') as f:
            f.write('\nSpeed test: open run_speed_test.ipynb, or run:\n'
                    'python3 pynq/benchmark_ct_tile_speed.py --outdir board_speed_001 --repeats 5\n'
                    'This compares legacy48, fast48, fast58, compact58 on BOTH real samples.\n'
                    'Every output must match the SAME integer model identified in manifest.json.\n'
                    'Results include PS packing, DMA/cache, PL and reassembly.\n')
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()}
    meta=dict(board='PYNQ-Z2',part='xc7z020clg400-1',clock_mhz=float((build/'clock_mhz.txt').read_text()),tool='Vivado 2018.2',
        numerical_contract='Same integer RTL/reference, no float inference on PL',
        model=json.loads((model/'quantization.json').read_text()),files_sha256=files,
        cnn_compact_protocol=compact,simulation_frames=int(passed[1]),simulation_output_pixels=int(passed[2]),
        scope='Simulation and timed routed bitstream ready; board execution must produce board_result.json')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    archive=shutil.make_archive(str(out),'zip',out.parent,out.name)
    print(json.dumps(dict(bundle=str(out),archive=archive,files=len(files)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',required=True);p.add_argument('--outdir',required=True)
    p.add_argument('--simulation',required=True)
    p.add_argument('--quantized-dir');p.add_argument('--romdir')
    main(p.parse_args())
