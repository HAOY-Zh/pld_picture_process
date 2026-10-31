"""Seal single-bit shared-engine acceptance package without changing the bit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
from lung_nodule_shared_prepare import ROOT,checked
from nodule_plan import sha


def package(root):
    root,prep=checked(root);target=root/'lung_nodule_shared'
    if (target/'shared_manifest.json').exists():raise FileExistsError('Preserve sealed shared package')
    accepted=ROOT/'data/nodule_overlap_release_20261024/nodule_overlap'
    old=json.loads((accepted/'overlap_manifest.json').read_text())
    if sha(target/'lung_network.bit')!=old['overlap_bit_sha256']:raise ValueError('Wrong accepted bit')
    for f in (target/'rtl').glob('*.v'):
        if sha(f)!=sha(accepted/'rtl'/f.name):raise ValueError('Unexpected shared RTL change')
    oracle=json.loads((root/'oracle/verification.json').read_text());sim=json.loads((root/'mixed_sim/verification.json').read_text())
    if oracle['status']!='frozen_full_models_and_small_dense_integer_oracles_pass' or sim['status']!='actual_xsim_shared_models_no_reset_exact':raise ValueError('Missing numerical proof')
    for proof in [oracle,sim]:
        for f,h in proof['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed tested source '+f)
        for f,h in proof.get('outputs_sha256',{}).items():
            if sha(f)!=h:raise ValueError('Changed independent gold')
    if len(sim['command_results'])!=176 or len(sim['stages'])!=5 or sim['full256_unet_rtl']:raise ValueError('Wrong mixed simulation scope')
    for n,h in sim['vectors_sha256'].items():
        if sha(root/'mixed_sim/vectors'/n)!=h:raise ValueError('Changed mixed oracle')
    if sha(root/'mixed_sim/rtl_run.log')!=sim['log_sha256']:raise ValueError('Changed RTL log')
    for name,count in [('shared_tests_pass.log',5),('driver_tests_fixed.log',1)]:
        text=(root/name).read_text()
        if not re.search(r'Ran '+str(count)+r' tests?.*?\n\s*\nOK\s*$',text,re.S):raise ValueError('Missing tests '+name)
    # Copy new service/acceptance code only after independent checking. Model,
    # bit, original runtime and oracle payloads stay frozen from preparation.
    for n in ['lung_nodule_shared.py','lung_nodule_shared_benchmark.py']:shutil.copy2(ROOT/'pynq'/n,target/n)
    ev=target/'evidence';ev.mkdir()
    for name in ['preparation.json','shared_tests_pass.log','driver_tests_fixed.log']:
        shutil.copy2(root/name,ev/name)
    for sub in ['oracle','mixed_sim']:
        (ev/sub).mkdir();shutil.copy2(root/sub/'verification.json',ev/sub/'verification.json')
    shutil.copy2(root/'mixed_sim/rtl_run.log',ev/'mixed_sim/rtl_run.log')
    shutil.copy2(accepted/'overlap_manifest.json',ev/'overlap_manifest.json')
    physical=ROOT/'data/nodule_overlap_speed_01/independent_verification.json';shutil.copy2(physical,ev/'nodule_physical_verification.json')
    for n in ['timing_routed.rpt','utilization_routed.rpt','drc.rpt']:
        shutil.copy2(accepted/'evidence'/n,ev/n)
    for label,src in prep['sources'].items():
        f=Path(src)/('nodule_manifest.json' if label=='nodule' else 'fast_manifest.json')
        shutil.copy2(f,ev/(label+'_parent_manifest.json'))
    dev=target/'development';dev.mkdir()
    files=[ROOT/'pynq/lung_nodule_shared.py',ROOT/'pynq/lung_nodule_shared_benchmark.py',ROOT/'python/lung_nodule_shared_prepare.py',
        ROOT/'python/lung_nodule_shared_sim.py',Path(__file__),ROOT/'python/verify_lung_nodule_shared.py',
        ROOT/'python/tests/test_lung_nodule_shared.py',ROOT/'python/tests/test_lung_nodule_shared_driver.py']
    for f in files:shutil.copy2(f,dev/f.name)
    replay=target/'simulation';replay.mkdir();shutil.copytree(root/'mixed_sim/vectors',replay/'vectors')
    for f in (root/'mixed_sim').glob('*.sv'):shutil.copy2(f,replay/f.name)
    shutil.copy2(root/'mixed_sim/run.tcl',replay/'run.tcl')
    (replay/'run_rtl_sim.py').write_text('''import argparse,re,subprocess\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);a=p.parse_args();r=Path(__file__).resolve().parent;b=Path(a.vivado_bin);s=list((r.parent/'rtl').glob('*.v'))+list(r.glob('*.sv'))\nfor c in [[str(b/'xvlog.bat'),'--sv',*map(str,s)],[str(b/'xelab.bat'),'tb_nodule_network','-s','shared_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'shared_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(c,cwd=r,check=True)\nt=(r/'replay.log').read_text();assert 'Shared U-Net/nodule mixed RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t);print('Mixed real-weight full-topology RTL replay PASS; U-Net32 and nodule128, not full256 U-Net')\n''')
    shutil.copy2(ROOT/'docs/肺分割与结节共用单bit验证_20261024.md',target/'README.md')
    manifest=dict(schema='lung_nodule_single_engine_v1',board_tested=False,clock_mhz=100,
        bit_sha256=sha(target/'lung_network.bit'),hwh_sha256=sha(target/'lung_network.hwh'),
        rtl_changed=False,model_parameters_quantization_changed=False,overlay_downloads_per_session=1,
        compatibility=prep['compatibility'],all_models_resident_cma_bytes=sum(v['cma_bytes'] for v in prep['compatibility'].values()),
        route=old['route'],utilization=old['utilization'],frozen_model_sources=prep['sources'],
        simulation=dict(commands=176,no_inter_model_reset=True,unet_spatial=32,nodule_raw_patch_spatial=128,full256_unet_on_new_bit_pending=True),
        files_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
    (target/'shared_manifest.json').write_text(json.dumps(manifest,indent=2))
    smoke="import sys;sys.path.insert(0,sys.argv[1]);import lung_nodule_shared,lung_nodule_shared_benchmark;from lung_nodule_shared import checked_package,load_model;p,m=checked_package(sys.argv[1]);[load_model(p,n) for n in ['unet_original','unet_pruned','nodule']];assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('Self-contained portable shared package imports and all hashes PASS')"
    r=subprocess.run([sys.executable,'-B','-c',smoke,str(target)],cwd=root,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    archive=root/'lung_nodule_shared_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in target.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Archive CRC failed')
        for n,h in manifest['files_sha256'].items():
            if hashlib.sha256(z.read('lung_nodule_shared/'+n)).hexdigest()!=h:raise ValueError('Archive file mismatch')
    release=dict(status='one_bit_shared_models_ready_for_full_size_board_acceptance',board_tested=False,
        bit_changed=False,model_changed=False,compatibility=prep['compatibility'],total_cma_bytes=manifest['all_models_resident_cma_bytes'],
        resource_utilization=old['utilization'],manifest_sha256=sha(target/'shared_manifest.json'),archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size,imports=r.stdout.strip(),source_sha256={str(p):sha(p) for p in files},
        numerical_proof_sha256=sha(root/'oracle/verification.json'),mixed_rtl_proof_sha256=sha(root/'mixed_sim/verification.json'))
    (root/'release.json').write_text(json.dumps(release,indent=2));(root/'progress.json').write_text(json.dumps(release,indent=2))
    print(json.dumps(release,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data/lung_nodule_shared_20261024_v1'));a=p.parse_args();package(a.root)
