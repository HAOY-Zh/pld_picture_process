"""Seal a tested nodule board delivery without rerunning model inference."""
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
from nodule_plan import load_package,sha,validate

RUN_SIM='''"""Run this package's shared-DDR network bench using locally installed XSim."""
import argparse
from pathlib import Path
import subprocess
import re
p=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;v=Path(a.vivado_bin)
sources=list((root.parent/'rtl').glob('*.v'))+list((root/'benches').glob('*.sv'))
calls=[[str(v/'xvlog.bat'),'--sv',*map(str,sources)],
       [str(v/'xelab.bat'),'tb_nodule_network','-s','nodule_package_snapshot','--debug','typical','--mt','2'],
       [str(v/'xsim.bat'),'nodule_package_snapshot','-tclbatch','run.tcl','-log','replay.log']]
for cmd in calls:subprocess.run(cmd,cwd=root,check=True)
text=(root/'replay.log').read_text(encoding='utf-8')
if 'Nodule continuous network RTL: PASS' not in text or re.search(r'Fatal:|ERROR:',text):raise RuntimeError('RTL replay failed')
print('Actual XSim full nodule network replay PASS')
'''


def finalize(out):
    out=Path(out).resolve();pack,manifest,model,params=load_package(out/'nodule_board')
    release=out/'release.json'
    if release.exists():raise FileExistsError('Release already sealed')
    sim=out/'rtl_sim';proof=json.loads((sim/'verification.json').read_text(encoding='utf-8'))
    if proof['status']!='actual_xsim_full_nodule_graph_pass' or sha(sim/proof['log_name'])!=proof['log_sha256']:
        raise ValueError('Actual XSim proof missing or changed')
    for p,h in proof['source_sha256'].items():
        if sha(p)!=h:raise ValueError('RTL changed after simulation: '+p)
    for p,h in proof['vectors_sha256'].items():
        if sha(sim/'vectors'/p)!=h:raise ValueError('Simulation vectors changed')
    log=out/'python_tests_accepted.log';text=log.read_text(encoding='utf-8')
    if not re.search(r'Ran 8 tests.*?\n\s*\nOK\s*$',text,re.S) or 'skipped=' in text:
        raise ValueError('Need all eight local tests')
    protected={n:sha(pack/n) for n in ['nodule_model.json','nodule_parameters.npz','nodule_reference.npz',
        'nodule_ranker.json','lung_network.bit','lung_network.hwh']}
    source_receipts={}
    # Refresh only source snapshots; integer parameters, bit and references stay frozen.
    for n in manifest['files_sha256']:
        if '/' in n or Path(n).suffix not in ['.py','.cpp','.h']:continue
        src=ROOT/'pynq'/n
        if not src.exists():src=ROOT/'python'/n
        shutil.copy2(src,pack/n);source_receipts[str(src)]=sha(src)
    evidence=pack/'evidence';evidence.mkdir()
    for src in [sim/'verification.json',sim/proof['log_name'],log,out/'calibration.json']:
        shutil.copy2(src,evidence/src.name)
    doc=ROOT/'docs/肺结节板端部署与测速_20261023.md';shutil.copy2(doc,pack/'README.md')
    simulation=pack/'simulation';simulation.mkdir();(simulation/'benches').mkdir()
    for src in [sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv',
            ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv']:
        shutil.copy2(src,simulation/'benches'/src.name)
    shutil.copytree(sim/'vectors',simulation/'vectors')
    shutil.copy2(sim/'run.tcl',simulation/'run.tcl')
    (simulation/'run_rtl_sim.py').write_text(RUN_SIM,encoding='utf-8')
    # Developer-side reproducibility sources are optional on the board.
    tools=pack/'development';tools.mkdir()
    for n in ['nodule_board_prepare.py','nodule_board_finalize.py','nodule_integer_reference.py',
              'nodule_rtl_sim.py','nodule_board_case.py','verify_nodule_board.py']:
        shutil.copy2(ROOT/'python'/n,tools/n);source_receipts[str(ROOT/'python'/n)]=sha(ROOT/'python'/n)
    shutil.copy2(ROOT/'python/tests/test_nodule_board.py',tools/'test_nodule_board.py')
    source_receipts[str(ROOT/'python/tests/test_nodule_board.py')]=sha(ROOT/'python/tests/test_nodule_board.py')
    for n,h in protected.items():
        if sha(pack/n)!=h:raise ValueError('Frozen model/reference/bit changed')
    manifest.update(release='20261023_v1',local_tests=8,rtl_simulation='actual shared-DDR complete network PASS',
        board_tested=False,int8_detection_quality_accepted=False,
        files_sha256={p.relative_to(pack).as_posix():sha(p) for p in pack.rglob('*') if p.is_file() and p.name!='nodule_manifest.json'})
    (pack/'nodule_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Test the actual flat package in a fresh interpreter, away from source import paths.
    code="import sys;import nodule_detect;from nodule_plan import load_package;load_package('.');assert 'torch' not in sys.modules;assert 'sklearn' not in sys.modules;print('Standalone board package imports and file identity PASS')"
    result=subprocess.run([sys.executable,'-c',code],cwd=pack,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    archive=out/'nodule_board_20261023.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in pack.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failed')
    report=dict(status='sealed_local_verified_not_yet_board_tested',package=str(pack),archive=str(archive),
        archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        manifest_sha256=sha(pack/'nodule_manifest.json'),board_tested=False,int8_detection_quality_accepted=False,
        local_tests=8,rtl_status=proof['status'],source_sha256=source_receipts,
        flat_package_smoke=result.stdout.strip(),protected_payload_sha256=protected)
    release.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','archive','archive_bytes','flat_package_smoke']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();finalize(a.out)
