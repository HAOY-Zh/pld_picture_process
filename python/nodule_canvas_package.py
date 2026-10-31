"""Seal the spatial-reuse benchmark only after complete quality and RTL QA."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import load_package,sha


def package(out):
    out=Path(out).resolve();pack=out/'nodule_canvas'
    if pack.exists():raise FileExistsError('Preserve sealed addon')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    quality=ROOT/'data/nodule_canvas_quality_20261023_v1/final_report.json'
    q=json.loads(quality.read_text())
    if (q['status']!='complete' or q['completed']!=75 or q['baseline_int8']!=q['canvas_int8']
            or not q['all_required_values_and_candidates_exact']):raise ValueError('Incomplete exact quality regression')
    for name,digest in q['case_files_sha256'].items():
        if sha(quality.parent/name)!=digest:raise ValueError('Changed quality case')
    protocol=json.loads((quality.parent/'protocol.json').read_text())
    for name,digest in protocol['sources_sha256'].items():
        if Path(name).name!='nodule_canvas_quality.py' and sha(name)!=digest:raise ValueError('Numerical QA source changed')
    if protocol['parent_sha256']!=sha(parent/'nodule_manifest.json'):raise ValueError('Quality parent changed')
    sim=out/'rtl_sim';proof=json.loads((sim/'verification.json').read_text())
    if proof['status']!='actual_xsim_full_448_canvas_pass' or proof['log_sha256']!=sha(sim/'rtl_run.log'):raise ValueError('Need full canvas RTL pass')
    for name,digest in proof['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed simulated source')
    for name,digest in proof['vectors_sha256'].items():
        if sha(sim/'vectors'/name)!=digest:raise ValueError('Changed simulated vector')
    tests=out/'python_tests_final.log';log=tests.read_text(encoding='utf-8')
    if not re.search(r'Ran 9 tests.*?\n\s*\nOK\s*$',log,re.S) or 'skipped=' in log:raise ValueError('Need nine complete tests')
    pack.mkdir();sources={}
    names=['nodule_canvas.py','nodule_canvas_pl.py','nodule_canvas_cpu.py','nodule_canvas_benchmark.py',
        'nodule_fast_input.py','nodule_fast_input.cpp','nodule_fast_batch.py','nodule_fast_benchmark.py']
    for name in names:
        src=ROOT/'pynq'/name;shutil.copy2(src,pack/name);sources[str(src)]=sha(src)
    old=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
    for name in ['slice_case.json','slice_case.npz']:shutil.copy2(old/name,pack/name)
    doc=ROOT/'docs/肺结节重叠卷积复用加速_20261023.md';shutil.copy2(doc,pack/'README.md')
    evidence=pack/'evidence';evidence.mkdir()
    for src in [quality,quality.parent/'protocol.json',sim/'verification.json',sim/'rtl_run.log',tests]:shutil.copy2(src,evidence/src.name)
    dev=pack/'development';dev.mkdir()
    for src in [Path(__file__),ROOT/'python/nodule_canvas_quality.py',ROOT/'python/nodule_canvas_quality_report.py',
        ROOT/'python/nodule_canvas_rtl.py',ROOT/'python/nodule_int8_cuda_reference.py',ROOT/'python/verify_nodule_canvas.py',
        ROOT/'python/tests/test_nodule_canvas.py',ROOT/'python/tests/test_nodule_int8_reference.py']:
        shutil.copy2(src,dev/src.name);sources[str(src)]=sha(src)
    replay=pack/'simulation';replay.mkdir();shutil.copytree(sim/'vectors',replay/'vectors')
    benches=replay/'benches';benches.mkdir()
    for src in [sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv',ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv']:
        shutil.copy2(src,benches/src.name)
    shutil.copy2(sim/'run.tcl',replay/'run.tcl')
    code='''import argparse\nfrom pathlib import Path\nimport subprocess,re\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);p.add_argument('--parent-package',required=True);a=p.parse_args()\nr=Path(__file__).resolve().parent;b=Path(a.vivado_bin);old=Path(a.parent_package)\nsources=list((old/'rtl').glob('*.v'))+list((r/'benches').glob('*.sv'))\nfor cmd in [[str(b/'xvlog.bat'),'--sv',*map(str,sources)],[str(b/'xelab.bat'),'tb_nodule_network','-s','canvas_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'canvas_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(cmd,cwd=r,check=True)\nt=(r/'replay.log').read_text();assert 'Nodule full canvas RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t);print('Full canvas actual RTL replay PASS')\n'''
    (replay/'run_rtl_sim.py').write_text(code,encoding='utf-8')
    m=dict(schema='nodule_canvas_addon_v1',parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
        board_tested=False,quality_cohort='calibration75',quality_unchanged=True,local_tests=9,
        rtl_status=proof['status'],canvas_hw=[448,448],cma_resident_bytes=6348800,
        files_sha256={p.relative_to(pack).as_posix():sha(p) for p in pack.rglob('*') if p.is_file()})
    (pack/'canvas_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    code="import sys;sys.path[:0]=sys.argv[1:3];import nodule_canvas,nodule_canvas_pl,nodule_canvas_cpu,nodule_canvas_benchmark;assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('flat no-Torch imports PASS')"
    smoke=subprocess.run([sys.executable,'-B','-c',code,str(pack),str(parent)],cwd=out,capture_output=True,text=True,check=True)
    archive=out/'nodule_canvas_20261023.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in pack.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Bad ZIP')
        for name,digest in m['files_sha256'].items():
            if hashlib.sha256(z.read('nodule_canvas/'+name)).hexdigest()!=digest:raise ValueError('ZIP receipt differs')
    release=dict(status='ready_for_board_canvas_benchmark',board_tested=False,archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive),manifest_sha256=sha(pack/'canvas_manifest.json'),sources_sha256=sources,
        quality_sha256=sha(quality),local_tests=9,rtl_pass=True,imports=smoke.stdout.strip())
    (out/'release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    print(json.dumps({k:release[k] for k in ['status','archive_bytes','local_tests','rtl_pass','imports']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();package(a.out)
