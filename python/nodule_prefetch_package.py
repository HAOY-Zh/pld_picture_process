"""Timing/source gated packaging of the independent window-prefetch overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from nodule_prefetch_build import ROOT,checked
from nodule_plan import sha,load_package
from verify_lung_axil import timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware


def package(root):
    root,prep=checked(root);target=root/'nodule_prefetch'
    if target.exists():raise FileExistsError('Preserve sealed package')
    evidence=[];proofs={}
    for folder,status in [('sim_128','actual_xsim_prefetch_exact_128'),('sim_448','actual_xsim_prefetch_exact_448'),('protocol','actual_xsim_prefetch_protocol_pass')]:
        p=root/folder/'verification.json';d=json.loads(p.read_text())
        if d['status']!=status:raise ValueError('Incomplete RTL verification')
        for f,h in d['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed simulated RTL/bench')
        for n,h in d.get('vectors_sha256',{}).items():
            if sha(root/folder/'vectors'/n)!=h:raise ValueError('Changed simulation vectors')
        logs=[root/folder/'rtl_run.log'] if folder!='protocol' else [root/folder/(n+'_run.log') for n in d['results']]
        for f in logs:
            if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',f.read_text(),re.I):raise ValueError('Failed simulation log')
        if folder=='protocol':
            if len(d['results'])!=3:raise ValueError('Missing protocol proof')
            for n,v in d['results'].items():
                if sha(root/folder/(n+'_run.log'))!=v['log_sha256']:raise ValueError('Changed protocol log')
        elif sha(logs[0])!=d['log_sha256']:raise ValueError('Changed graph log')
        proofs[folder]=d;evidence+=[p]+logs
    # Compare the 128 patch to the SAME exact-LUT version, not to original flow.
    old128=ROOT/'data/nodule_rtl_stall_profile_20261023_v1/profile.json'
    b=json.loads(old128.read_text());a=proofs['sim_128']
    if b['status']!='actual_xsim_exact_lut_stall_profile_pass':raise ValueError('Missing matched patch baseline')
    for p,h in b['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Changed matched baseline')
    patch_before=sum(int(x[2]) for x in b['all_commands']);patch_after=sum(x['cycles'] for x in a['commands'])
    full=proofs['sim_448'];before=sum(x['before'] for x in full['gains']);after=sum(x['after'] for x in full['gains'])
    if after>=before or any(x['after']>=x['before'] for x in full['gains'] if x['kind']==0):raise ValueError('Missing full-plane convolution gains')
    comparison=dict(baseline='same exact-LUT RTL plus FIFO64; tensor/MAC/quantization unchanged',
        patch128=dict(before_cycles=patch_before,after_cycles=patch_after,ratio=patch_before/patch_after),
        canvas448=dict(before_cycles=before,after_cycles=after,ratio=before/after,reduction_percent=100*(before-after)/before),
        raw_128_report_uses_original_flow_baseline=True,board_tested=False)
    (root/'matched_comparison.json').write_text(json.dumps(comparison,indent=2));evidence+=[root/'matched_comparison.json',old128]
    baseline=ROOT/'data/nodule_direct_lut_20261023_v1'
    diffs=[p.name for p in (root/'rtl').glob('*.v') if sha(p)!=sha(baseline/'rtl'/p.name)]
    if diffs!=['lung_conv8_window.v']:raise ValueError('Unexpected functional RTL difference')
    for n in ['lung_axil_config_timing.xdc','validate_lung_window_step.tcl']:
        old=json.loads((ROOT/'data/lung_flow_20261017/preparation.json').read_text())['source_sha256']
        expected=[h for p,h in old.items() if Path(p).name==n]
        if len(expected)!=1 or sha(ROOT/'fpga'/n)!=expected[0]:raise ValueError('Changed baseline constraints/BD checks')
        evidence.append(ROOT/'fpga'/n)
    raw=(root/'timing_routed.rpt').read_text();wns=timing(root/'timing_routed.rpt')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',raw,re.S)
    if not match or float(match[1].split()[4])<0:raise ValueError('Hold failed')
    for label in ['no_clock','unconstrained_internal_endpoints','loops','latch_loops']:
        if not re.search(r'checking '+label+r' \(0\)',raw):raise ValueError('Incomplete timing constraints: '+label)
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text()):raise ValueError('DRC failed')
    resources=utilization(root/'utilization_routed.rpt')
    hw=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    physlog=root/'physopt_console.log'
    if physlog.exists():
        if 'Unchanged 100MHz constraints: post-route physical optimization and bit generation PASS.' not in physlog.read_text():
            raise ValueError('Incomplete current-design physical optimization/bit generation')
        repair=root/'resume_lung_lut_physopt.tcl'
        if sha(repair)!=sha(ROOT/'fpga/resume_lung_lut_physopt.tcl'):raise ValueError('Changed physical repair script')
        evidence.append(repair)
    elif 'Resumed route and timing-gated bit generation complete;' not in (root/'build_console.log').read_text():
        raise ValueError('Incomplete timing-gated bit generation')
    imported=list((root/'ip_repo').rglob('lung_*.v'))
    for f in imported:
        if sha(f)!=sha(root/'rtl'/f.name):raise ValueError('Imported IP differs from simulated RTL')
    required={'lung_conv8_axil.v','lung_tensor_axil.v','lung_conv8_window.v','lung_mac_tile8.v','lung_tensor_burst_engine.v'}
    if not required.issubset({p.name for p in imported}):raise ValueError('Missing packaged IP source')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    tests=root/'python_driver_tests_cpu_clock.log'
    if not re.search(r'Ran 1 test.*?\n\s*\nOK\s*$',tests.read_text(),re.S):raise ValueError('Missing driver/verifier integration check')
    evidence.append(tests)
    oldaddon=baseline/'nodule_direct_lut';m=json.loads((oldaddon/'direct_manifest.json').read_text())
    for n,h in m['files_sha256'].items():
        if sha(oldaddon/n)!=h:raise ValueError('Changed frozen host runtime')
    if (root/'deploy/lung_network.bit').stat().st_size<1000000:raise ValueError('Missing bit')
    target.mkdir()
    names=['nodule_canvas.py','nodule_canvas_pl.py','nodule_canvas_cpu.py','nodule_canvas_benchmark.py','nodule_fast_input.py',
        'nodule_canvas_input.py','nodule_canvas_input.cpp','nodule_canvas_direct.py','slice_case.json','slice_case.npz','lung_fast_io.cpp','lung_fast_io.h']
    for n in names:shutil.copy2(oldaddon/n,target/n)
    shutil.copy2(ROOT/'pynq/nodule_prefetch_benchmark.py',target/'nodule_prefetch_benchmark.py')
    for n in ['lung_network.bit','lung_network.hwh']:shutil.copy2(root/'deploy'/n,target/n)
    sub=target/'baseline';sub.mkdir()
    for n in ['lung_network.bit','lung_network.hwh','lung_fast_io.cpp','lung_fast_io.h']:shutil.copy2(oldaddon/n,sub/n)
    shutil.copytree(root/'rtl',target/'rtl')
    proofdir=target/'evidence';proofdir.mkdir()
    evidence += [root/n for n in ['preparation.json','timing_routed.rpt','utilization_routed.rpt','drc.rpt','configuration_exceptions.rpt','build_flow.tcl','ps_settings.tcl','build_console.log']]
    for f in evidence:
        rel=f.relative_to(root) if root in f.parents else Path('baseline_proofs')/f.name
        dest=proofdir/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for n in ['route_initial_failed','route_status_physopt.rpt','physopt_console.log']:
        f=root/n
        if f.is_dir():shutil.copytree(f,proofdir/n)
        elif f.exists():shutil.copy2(f,proofdir/n)
    dev=target/'development';dev.mkdir()
    for f in [Path(__file__),ROOT/'python/nodule_prefetch_build.py',ROOT/'python/nodule_prefetch_regression.py',ROOT/'python/verify_nodule_prefetch.py',ROOT/'python/tests/test_nodule_prefetch_driver.py']:
        shutil.copy2(f,dev/f.name)
    shutil.copy2(ROOT/'docs/肺结节窗口预取加速_20261024.md',target/'README.md')
    replay=target/'simulation';replay.mkdir();shutil.copytree(root/'sim_448/vectors',replay/'vectors')
    for f in (root/'sim_448').glob('*.sv'):shutil.copy2(f,replay/f.name)
    shutil.copy2(root/'sim_448/run.tcl',replay/'run.tcl')
    (replay/'run_rtl_sim.py').write_text('''import argparse,re,subprocess\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);a=p.parse_args()\nr=Path(__file__).resolve().parent;b=Path(a.vivado_bin);s=list((r.parent/'rtl').glob('*.v'))+list(r.glob('*.sv'))\nfor c in [[str(b/'xvlog.bat'),'--sv',*map(str,s)],[str(b/'xelab.bat'),'tb_nodule_network','-s','prefetch_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'prefetch_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(c,cwd=r,check=True)\nt=(r/'replay.log').read_text();assert 'RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t);print('Actual full prefetch replay PASS')\n''')
    manifest=dict(schema='nodule_window_prefetch64_v1',board_tested=False,clock_mhz=100,
        parent_manifest_sha256=sha(parent/'nodule_manifest.json'),baseline_bit_sha256=sha(sub/'lung_network.bit'),
        prefetch_bit_sha256=sha(target/'lung_network.bit'),route=dict(wns_ns=wns,whs_ns=float(match[1].split()[4])),
        hardware=hw,utilization=resources,simulation=comparison,model_arithmetic_threshold_unchanged=True,
        files_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
    (target/'prefetch_manifest.json').write_text(json.dumps(manifest,indent=2))
    smoke="import sys;sys.path[:0]=sys.argv[1:3];import nodule_prefetch_benchmark,nodule_canvas_input,nodule_canvas_direct;assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('portable prefetch imports PASS')"
    subprocess.run([sys.executable,'-B','-c',smoke,str(target),str(parent)],cwd=root,check=True,capture_output=True)
    archive=root/'nodule_prefetch_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in target.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Invalid ZIP')
        for n,h in manifest['files_sha256'].items():
            if hashlib.sha256(z.read('nodule_prefetch/'+n)).hexdigest()!=h:raise ValueError('Archive mismatch')
    release=dict(status='routed_exact_prefetch_ready_for_board',board_tested=False,archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive),manifest_sha256=sha(target/'prefetch_manifest.json'),utilization=resources,
        route=manifest['route'],simulation=comparison,imported_sha256={str(p):sha(p) for p in imported})
    (root/'release.json').write_text(json.dumps(release,indent=2));print(json.dumps(release,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);package(p.parse_args().root)
