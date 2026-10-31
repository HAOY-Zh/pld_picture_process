"""Seal the additive exact-input/batched-weight optimization for board testing."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import load_package,sha


def package(out):
    out=Path(out).resolve();pack=out/'nodule_fast'
    if pack.exists():raise FileExistsError('Preserve existing optimization package')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board'
    load_package(parent)
    old=json.loads((parent.parent/'release.json').read_text(encoding='utf-8'))
    if sha(parent/'nodule_manifest.json')!=old['manifest_sha256']:raise ValueError('Parent package changed')
    sim=out/'rtl_sim';proof=json.loads((sim/'verification.json').read_text(encoding='utf-8'))
    if proof['status']!='actual_xsim_two_tile_resident_weight_reuse_pass' or sha(sim/'rtl_run.log')!=proof['log_sha256']:
        raise ValueError('Need actual batch RTL proof')
    for p,h in proof['source_sha256'].items():
        if sha(p)!=h:raise ValueError('RTL source changed after simulation')
    for p,h in proof['vectors_sha256'].items():
        if sha(sim/'vectors'/p)!=h:raise ValueError('RTL vectors changed')
    tests=out/'python_tests.log';text=tests.read_text(encoding='utf-8')
    if not re.search(r'Ran 8 tests.*?\n\s*\nOK\s*$',text,re.S) or 'skipped=' in text:raise ValueError('Need eight complete input/runtime tests')
    pack.mkdir();sources={}
    runtime=['nodule_fast_input.py','nodule_fast_input.cpp','nodule_fast_batch.py','nodule_fast_benchmark.py','nodule_fast_detect.py']
    for name in runtime+['nodule_slice_benchmark.py']:
        src=ROOT/'pynq'/name;shutil.copy2(src,pack/name);sources[str(src)]=sha(src)
    oldslice=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
    for name in ['slice_case.npz','slice_case.json']:shutil.copy2(oldslice/name,pack/name)
    if sha(pack/'nodule_slice_benchmark.py')!=sha(oldslice/'nodule_slice_benchmark.py'):
        raise ValueError('Baseline complete-plane benchmark changed')
    shutil.copy2(ROOT/'docs/肺结节采样与批量加速_20261023.md',pack/'README.md')
    evidence=pack/'evidence';evidence.mkdir()
    for src in [tests,sim/'verification.json',sim/'rtl_run.log']:shutil.copy2(src,evidence/src.name)
    benches=pack/'simulation';benches.mkdir();(benches/'benches').mkdir()
    for src in [sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv',ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv']:
        shutil.copy2(src,benches/'benches'/src.name)
    shutil.copytree(sim/'vectors',benches/'vectors');shutil.copy2(sim/'run.tcl',benches/'run.tcl')
    replay='''import argparse\nfrom pathlib import Path\nimport subprocess\nimport re\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);p.add_argument('--parent-package',required=True);a=p.parse_args()\nr=Path(__file__).resolve().parent;b=Path(a.vivado_bin);old=Path(a.parent_package).resolve()\nsrc=list((old/'rtl').glob('*.v'))+list((r/'benches').glob('*.sv'))\nfor c in [[str(b/'xvlog.bat'),'--sv',*map(str,src)],[str(b/'xelab.bat'),'tb_nodule_network','-s','fast_batch_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'fast_batch_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(c,cwd=r,check=True)\nt=(r/'replay.log').read_text(encoding='utf-8')\nassert 'Nodule batch reuse RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t)\nprint('Batch reuse actual RTL replay PASS')\n'''
    (benches/'run_rtl_sim.py').write_text(replay,encoding='utf-8')
    dev=pack/'development';dev.mkdir()
    for src in [ROOT/'python/nodule_fast_package.py',ROOT/'python/nodule_fast_rtl.py',ROOT/'python/verify_nodule_fast.py',ROOT/'python/tests/test_nodule_fast.py']:
        shutil.copy2(src,dev/src.name);sources[str(src)]=sha(src)
    manifest=dict(schema='nodule_fast_addon_v1',board_tested=False,weights_bit_config_changed=False,
        parent_manifest_sha256=sha(parent/'nodule_manifest.json'),local_tests=8,rtl_status=proof['status'],
        default_batch_size=8,expected_cma_bytes_batch8=4157440,
        expected_weight_loads_32_tiles=dict(baseline=352,optimized=44),
        files_sha256={p.relative_to(pack).as_posix():sha(p) for p in pack.rglob('*') if p.is_file()})
    (pack/'fast_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Isolated imports against the exact sealed parent, not repository aliases.
    code="import sys;sys.path[:0]=sys.argv[1:3];import nodule_fast_input,nodule_fast_batch,nodule_fast_benchmark,nodule_fast_detect;from nodule_plan import load_package;load_package(sys.argv[2]);assert 'torch' not in sys.modules;assert 'sklearn' not in sys.modules;print('flat addon imports PASS')"
    smoke=subprocess.run([sys.executable,'-c',code,str(pack),str(parent)],cwd=out,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    archive=out/'nodule_fast_20261023.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in pack.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Bad ZIP')
        for n,h in manifest['files_sha256'].items():
            if __import__('hashlib').sha256(z.read('nodule_fast/'+n)).hexdigest()!=h:raise ValueError('ZIP payload mismatch')
    release=dict(status='ready_for_board_same_plane_acceleration_comparison',board_tested=False,
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        addon_manifest_sha256=sha(pack/'fast_manifest.json'),parent_manifest_unchanged=True,
        local_tests=8,actual_rtl_pass=True,source_sha256=sources,standalone_imports=smoke.stdout.strip())
    (out/'release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    print(json.dumps({k:release[k] for k in ['status','archive_bytes','local_tests','actual_rtl_pass','standalone_imports']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();package(a.out)
