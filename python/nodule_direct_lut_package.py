"""Seal native-input plus exact-LUT ablation after exactness proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from verify_nodule_canvas import verify


def package(out):
    out=Path(out).resolve();target=out/'nodule_direct_lut'
    if target.exists():raise FileExistsError('Preserve sealed direct/LUT package')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    verify(ROOT/'data/nodule_canvas_speed_01')
    prep=json.loads((out/'preparation.json').read_text())
    for p,h in prep['input_files_sha256'].items():
        if sha(p)!=h:raise ValueError('Changed preparation input')
    if sha(out/'lung_network.bit')!=prep['bit_sha256'] or sha(out/'lung_network.hwh')!=prep['hwh_sha256']:raise ValueError('Changed LUT bit/HWH')
    for folder,status in [('rtl_sim','actual_xsim_nodule_448_exact_lut_pass'),('exhaustive','actual_xsim_all_nodule_concat_codes_exact')]:
        p=json.loads((out/folder/'verification.json').read_text())
        if p['status']!=status or sha(out/folder/'rtl_run.log')!=p['log_sha256']:raise ValueError('Incomplete actual LUT verification')
        for n,h in p['source_sha256'].items():
            if sha(n)!=h:raise ValueError('Changed simulated source')
    tests=out/'python_tests.log'
    if not re.search(r'Ran 7 tests.*?\n\s*\nOK\s*$',tests.read_text(encoding='utf-8'),re.S):raise ValueError('Incomplete direct-input tests')
    old=ROOT/'data/nodule_canvas_20261023_v1/nodule_canvas'
    manifest=json.loads((old/'canvas_manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if sha(old/n)!=h:raise ValueError('Changed original canvas package')
    if prep['parent_manifest_sha256']!=sha(parent/'nodule_manifest.json'):raise ValueError('Parent changed')
    target.mkdir();sources={}
    for name in ['nodule_canvas_input.py','nodule_canvas_input.cpp','nodule_canvas_direct.py','nodule_direct_lut_benchmark.py']:
        src=ROOT/'pynq'/name;shutil.copy2(src,target/name);sources[str(src)]=sha(src)
    for name in ['nodule_canvas.py','nodule_canvas_pl.py','nodule_canvas_cpu.py','nodule_canvas_benchmark.py','nodule_fast_input.py','slice_case.json','slice_case.npz']:
        shutil.copy2(old/name,target/name)
    for name in ['lung_fast_io.cpp','lung_fast_io.h']:shutil.copy2(parent/name,target/name)
    for name in ['lung_network.bit','lung_network.hwh']:shutil.copy2(out/name,target/name)
    shutil.copytree(out/'rtl',target/'rtl')
    doc=ROOT/'docs/肺结节直接输入与精确查表加速_20261023.md';shutil.copy2(doc,target/'README.md')
    evidence=target/'evidence';evidence.mkdir()
    for src in [out/'preparation.json',tests,ROOT/'data/nodule_canvas_quality_20261023_v1/final_report.json']:
        shutil.copy2(src,evidence/src.name)
    for folder in ['rtl_sim','exhaustive']:
        sub=evidence/folder;sub.mkdir()
        for name in ['verification.json','rtl_run.log']:shutil.copy2(out/folder/name,sub/name)
    dev=target/'development';dev.mkdir()
    for src in [Path(__file__),ROOT/'python/nodule_direct_lut_prepare.py',ROOT/'python/verify_nodule_direct_lut.py',ROOT/'python/tests/test_nodule_canvas_direct.py']:
        shutil.copy2(src,dev/src.name);sources[str(src)]=sha(src)
    replay=target/'simulation';replay.mkdir();shutil.copytree(out/'rtl_sim/vectors',replay/'vectors')
    for src in (out/'rtl_sim').glob('*.sv'):shutil.copy2(src,replay/src.name)
    shutil.copy2(out/'rtl_sim/run.tcl',replay/'run.tcl')
    code='''import argparse\nfrom pathlib import Path\nimport subprocess,re\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);a=p.parse_args()\nr=Path(__file__).resolve().parent;b=Path(a.vivado_bin);sources=list((r.parent/'rtl').glob('*.v'))+list(r.glob('*.sv'))\nfor cmd in [[str(b/'xvlog.bat'),'--sv',*map(str,sources)],[str(b/'xelab.bat'),'tb_nodule_network','-s','nodule_lut_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'nodule_lut_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(cmd,cwd=r,check=True)\nt=(r/'replay.log').read_text();assert 'Nodule full canvas RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t);print('Full nodule LUT replay PASS')\n'''
    (replay/'run_rtl_sim.py').write_text(code,encoding='utf-8')
    m=dict(schema='nodule_direct_lut_addon_v1',parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
        flow_bit_sha256=sha(parent/'lung_network.bit'),lut_bit_sha256=sha(target/'lung_network.bit'),local_tests=7,
        model_quantization_threshold_changed=False,sampling_changed=False,board_tested_for_nodule=False,
        files_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
    (target/'direct_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    code="import sys;sys.path[:0]=sys.argv[1:3];import nodule_canvas_input,nodule_canvas_direct,nodule_direct_lut_benchmark;assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('flat direct/LUT imports PASS')"
    smoke=subprocess.run([sys.executable,'-B','-c',code,str(target),str(parent)],cwd=out,text=True,capture_output=True,check=True)
    archive=out/'nodule_direct_lut_20261023.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in target.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Bad package ZIP')
        for n,h in m['files_sha256'].items():
            if hashlib.sha256(z.read('nodule_direct_lut/'+n)).hexdigest()!=h:raise ValueError('ZIP manifest mismatch')
    release=dict(status='ready_for_board_direct_input_and_exact_lut',board_tested_for_nodule=False,
        archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(target/'direct_manifest.json'),
        sources_sha256=sources,local_tests=7,full_rtl_pass=True,exhaustive_rtl_codes=5120,imports=smoke.stdout.strip())
    (out/'release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    print(json.dumps({k:release[k] for k in ['status','archive_bytes','local_tests','full_rtl_pass','exhaustive_rtl_codes','imports']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();package(a.out)
