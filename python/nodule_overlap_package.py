"""Timing-, source- and exactness-gated pixel-overlap board package."""
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
from nodule_overlap_build import ROOT,checked
from nodule_plan import load_package,sha
from verify_lung_axil import timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware


def package(root):
    root,prep=checked(root);target=root/'nodule_overlap'
    if target.exists():raise FileExistsError('Preserve sealed package')
    evidence=[];proofs={}
    for name,status in [('sim_128','actual_xsim_pixel_overlap_exact_128'),('sim_448','actual_xsim_pixel_overlap_exact_448'),
        ('protocol','actual_xsim_pixel_overlap_protocol_pass'),('ownership','actual_xsim_overlap_ownership_pass'),
        ('netlist_standalone','actual_post_synthesis_overlap_ownership_pass'),('edges','actual_xsim_window_edge_arithmetic_oracle_pass')]:
        p=root/name/'verification.json';d=json.loads(p.read_text())
        if d['status']!=status:raise ValueError('Missing proof '+name)
        for f,h in d['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed tested source '+f)
        for f,h in d.get('vectors_sha256',{}).items():
            file=Path(f) if Path(f).is_absolute() else root/name/'vectors'/f
            if sha(file)!=h:raise ValueError('Changed tested gold '+str(file))
        if name=='protocol':
            if set(d['results'])!={'tb_lung_cached_axi','tb_lung_memory_axi','tb_lung_axil_accel','mac','reuse'}:raise ValueError('Missing arithmetic/protocol branches')
            logs=[]
            for n,v in d['results'].items():
                file=Path(d['log_paths'][n]);logs.append(file)
                if sha(file)!=v['log_sha256']:raise ValueError('Changed protocol log')
        else:
            logs=[root/name/'rtl_run.log']
            if sha(logs[0])!=d['log_sha256']:raise ValueError('Changed replay log')
            if name=='netlist_standalone':
                if sha(root/name/'vivado.log')!=d['synthesis_log_sha256']:
                    raise ValueError('Wrong synthesized core evidence')
                evidence.append(root/name/'vivado.log')
        for f in logs:
            log=f.read_text()
            if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I) or 'RTL: PASS' not in log:raise ValueError('Failed RTL log '+str(f))
        proofs[name]=d;evidence+=[p]+logs
    for size in ['sim_128','sim_448']:
        d=proofs[size]
        if len(d['commands'])!=15 or d['overlap_events']<=0 or d['total_cycle_ratio']<=1:raise ValueError('No exact full-graph gain')
    parent_rtl=Path(prep['parent'])/'rtl'
    changed=sorted(p.name for p in (root/'rtl').glob('*.v') if sha(p)!=sha(parent_rtl/p.name))
    expected=sorted(['lung_mac_tile8.v','lung_mac_tile8_reuse.v','lung_conv8_window.v','lung_feature_cache.v','lung_feature_window.v'])
    if changed!=expected:raise ValueError('Unexpected functional RTL change')
    # The timing constraints must remain identical to the parent release.
    old_ev=Path(prep['parent'])/'nodule_prefetch/evidence'
    for n in ['lung_axil_config_timing.xdc','validate_lung_window_step.tcl']:
        if sha(ROOT/'fpga'/n)!=sha(old_ev/'baseline_proofs'/n):raise ValueError('Changed constraints/BD checks')
        evidence.append(ROOT/'fpga'/n)
    raw=(root/'timing_routed.rpt').read_text();wns=timing(root/'timing_routed.rpt')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',raw,re.S)
    if not match or wns<0 or float(match[1].split()[4])<0:raise ValueError('Routed setup/hold failed')
    for label in ['no_clock','unconstrained_internal_endpoints','loops','latch_loops']:
        if not re.search(r'checking '+label+r' \(0\)',raw):raise ValueError('Incomplete timing constraints '+label)
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text()):raise ValueError('Routed DRC failed')
    imported=list((root/'ip_repo').rglob('lung_*.v'))
    if not set(expected).issubset({p.name for p in imported}):raise ValueError('Missing implemented source')
    for f in imported:
        if sha(f)!=sha(root/'rtl'/f.name):raise ValueError('Simulated/implemented source mismatch')
    synth=root/'project/lung_network.runs/design_1_lung_conv8_axil_0_0_synth_1/runme.log'
    synth_text=synth.read_text()
    if 'synth_design completed successfully' not in synth_text or re.search(r'CRITICAL WARNING: \[Synth|ERROR:|multi-driven net',synth_text):
        raise ValueError('Convolution synthesis warnings/errors require repair')
    evidence.append(synth)
    build=root/'build_console.log';phys=root/'physopt_console.log'
    if phys.exists():
        if 'post-route physical optimization and bit generation PASS.' not in phys.read_text():raise ValueError('Incomplete current-design bit generation')
        if sha(root/'resume_lung_lut_physopt.tcl')!=sha(ROOT/'fpga/resume_lung_lut_physopt.tcl'):raise ValueError('Changed physical optimization flow')
        evidence+=[phys,root/'resume_lung_lut_physopt.tcl',root/'route_status_physopt.rpt']
    elif 'Resumed route and timing-gated bit generation complete;' not in build.read_text():raise ValueError('Missing timing-gated bit generation')
    resources=utilization(root/'utilization_routed.rpt');hw=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    host=ROOT/'data/nodule_native_host_20261024_v1/nodule_host_fast'
    hm=json.loads((host/'host_manifest.json').read_text())
    for n,h in hm['files_sha256'].items():
        if sha(host/n)!=h:raise ValueError('Changed native host baseline '+n)
    if hm['parent_manifest_sha256']!=sha(parent/'nodule_manifest.json'):raise ValueError('Wrong model baseline')
    test=root/'driver_validation.json';d=json.loads(test.read_text())
    if d['status']!='one_driver_integration_test_passed':raise ValueError('Missing driver validation')
    testlog=root/'driver_tests.log'
    if sha(testlog)!=d['test_log_sha256'] or not re.search(r'Ran 1 test.*?\n\s*\nOK\s*$',testlog.read_text(encoding='utf-8-sig'),re.S):
        raise ValueError('Wrong driver test log')
    for n,h in d['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Changed driver after validation')
    evidence.extend([test,testlog])
    comparison={}
    for size in ['sim_128','sim_448']:
        d=proofs[size];before=sum(x['before'] for x in d['gains']);after=sum(x['after'] for x in d['gains'])
        comparison[size]=dict(before_cycles=before,after_cycles=after,ratio=before/after,reduction_percent=100*(before-after)/before,
            per_command=d['gains'])
    # Preserve the pure pixel-overlap ablation where available; its arithmetic
    # proof is independent of the added cache/window feed changes.
    ablation=ROOT/'data/nodule_overlap_20261024_v3/sim_448/verification.json'
    if ablation.exists():
        a=json.loads(ablation.read_text())
        for f,h in a['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed ablation proof source')
        if a['status']!='actual_xsim_pixel_overlap_exact_448':raise ValueError('Wrong ablation proof')
        before=sum(x['before'] for x in a['gains']);after=sum(x['after'] for x in a['gains'])
        comparison['pixel_only_sim_448']=dict(before_cycles=before,after_cycles=after,ratio=before/after,reduction_percent=100*(before-after)/before)
        evidence.append(ablation)
    (root/'matched_comparison.json').write_text(json.dumps(comparison,indent=2));evidence.append(root/'matched_comparison.json')
    target.mkdir()
    names=['nodule_canvas.py','nodule_canvas_pl.py','nodule_canvas_cpu.py','nodule_fast_input.py',
        'nodule_canvas_input.py','nodule_canvas_input.cpp','nodule_canvas_direct.py',
        'nodule_sampling_native.cpp','nodule_native_sampling.py','nodule_fast_decode.py','nodule_host_pipeline.py',
        'slice_case.json','slice_case.npz','lung_fast_io.cpp','lung_fast_io.h']
    for n in names:shutil.copy2(host/n,target/n)
    shutil.copy2(ROOT/'pynq/nodule_overlap_benchmark.py',target/'nodule_overlap_benchmark.py')
    for n in ['lung_network.bit','lung_network.hwh']:shutil.copy2(root/'deploy'/n,target/n)
    baseline=target/'baseline';baseline.mkdir()
    for n in ['lung_network.bit','lung_network.hwh','lung_fast_io.cpp','lung_fast_io.h']:shutil.copy2(host/n,baseline/n)
    shutil.copytree(root/'rtl',target/'rtl')
    shutil.copy2(ROOT/'docs/肺结节像素重叠与连续供数加速_20261024.md',target/'README.md')
    ev=target/'evidence';ev.mkdir()
    evidence += [root/n for n in ['preparation.json','timing_routed.rpt','utilization_routed.rpt','drc.rpt','configuration_exceptions.rpt','build_console.log','build_flow.tcl','ps_settings.tcl']]
    evidence += [host/'host_manifest.json',host/'evidence/verification.json']
    for f in evidence:
        rel=f.relative_to(root) if root in f.parents else Path('baseline_proofs')/f.name
        dest=ev/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    if (root/'route_initial_failed').exists():shutil.copytree(root/'route_initial_failed',ev/'route_initial_failed')
    dev=target/'development';dev.mkdir()
    for f in [Path(__file__),ROOT/'python/nodule_overlap_build.py',ROOT/'python/nodule_overlap_feed_build.py',ROOT/'python/nodule_overlap_synthsafe_build.py',
        ROOT/'python/nodule_overlap_boundary_build.py',ROOT/'python/nodule_overlap_regression.py',ROOT/'python/nodule_overlap_ownership.py',ROOT/'python/nodule_overlap_netlist.py',ROOT/'python/nodule_overlap_edges.py',ROOT/'fpga/tb_nodule_overlap_ownership.sv',ROOT/'fpga/tb_nodule_window_edges.sv',
        ROOT/'python/verify_nodule_overlap.py',ROOT/'python/tests/test_nodule_overlap_driver.py']:
        shutil.copy2(f,dev/f.name)
    replay=target/'simulation';replay.mkdir();shutil.copytree(root/'sim_448/vectors',replay/'vectors')
    for f in (root/'sim_448').glob('*.sv'):shutil.copy2(f,replay/f.name)
    shutil.copy2(root/'sim_448/run.tcl',replay/'run.tcl')
    (replay/'run_rtl_sim.py').write_text('''import argparse,re,subprocess\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--vivado-bin',required=True);a=p.parse_args()\nr=Path(__file__).resolve().parent;b=Path(a.vivado_bin);s=list((r.parent/'rtl').glob('*.v'))+list(r.glob('*.sv'))\nfor c in [[str(b/'xvlog.bat'),'--sv',*map(str,s)],[str(b/'xelab.bat'),'tb_nodule_network','-s','overlap_replay','--debug','typical','--mt','2'],[str(b/'xsim.bat'),'overlap_replay','-tclbatch','run.tcl','-log','replay.log']]:subprocess.run(c,cwd=r,check=True)\nt=(r/'replay.log').read_text();assert 'RTL: PASS' in t and not re.search(r'Fatal:|ERROR:',t);print('Actual full overlap replay PASS')\n''')
    manifest=dict(schema='nodule_pixel_overlap_and_continuous_hit_feed_v1',board_tested=False,clock_mhz=100,
        parent_manifest_sha256=sha(parent/'nodule_manifest.json'),host_manifest_sha256=sha(host/'host_manifest.json'),
        baseline_bit_sha256=sha(baseline/'lung_network.bit'),overlap_bit_sha256=sha(target/'lung_network.bit'),
        model_parameters_sampling_quantization_threshold_unchanged=True,mac_lanes=64,result_slots=2,
        route=dict(wns_ns=wns,whs_ns=float(match[1].split()[4])),utilization=resources,hardware=hw,simulation=comparison,
        files_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
    (target/'overlap_manifest.json').write_text(json.dumps(manifest,indent=2))
    smoke="import sys;sys.path[:0]=sys.argv[1:3];import nodule_overlap_benchmark,nodule_host_pipeline,nodule_native_sampling;assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('portable overlap imports PASS')"
    subprocess.run([sys.executable,'-B','-c',smoke,str(target),str(parent)],cwd=root,capture_output=True,check=True)
    archive=root/'nodule_overlap_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in target.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Invalid ZIP')
        for n,h in manifest['files_sha256'].items():
            if hashlib.sha256(z.read('nodule_overlap/'+n)).hexdigest()!=h:raise ValueError('Archive hash mismatch')
    release=dict(status='routed_exact_pixel_overlap_ready_for_board',board_tested=False,route=manifest['route'],utilization=resources,
        simulation=comparison,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        manifest_sha256=sha(target/'overlap_manifest.json'),bit_sha256=sha(target/'lung_network.bit'),
        imported_sha256={str(p):sha(p) for p in imported})
    (root/'release.json').write_text(json.dumps(release,indent=2));print(json.dumps(release,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data/nodule_overlap_final_20261024_v1'));package(p.parse_args().root)
