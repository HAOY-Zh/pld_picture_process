"""Audit frozen arithmetic + real burst tests + routed design, bind NEW board package."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from lung_tensor_burst_prepare import WORKSPACE,engine_source
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes,timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import validate_addon
from lung_selected_benchmark import validate_selection,selected_entrypoints


def rtl_proof(root):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    old=WORKSPACE/'data/lung_word_20261015'
    check_hashes({str(root/'vectors'/n):h for n,h in prep['old_word_vectors_sha256'].items()})
    generated=(root/'rtl/lung_tensor_burst_engine.v').read_text(encoding='utf-8')
    if generated!=engine_source((WORKSPACE/'fpga/lung_tensor_word_engine.v').read_text(encoding='utf-8')):
        raise ValueError('Generated burst engine no longer matches arithmetic-preserving transform')
    sim=root/'sim_burst';v=json.loads((sim/'verification.json').read_text(encoding='utf-8'))
    log=(sim/'rtl_run.log').read_text(encoding='utf-8')
    if (v['status']!='passed_actual_xsim_tensor_burst' or v['frames']!=35 or v['output_words']!=379310 or v['error_cases']!=26
            or file_sha256(sim/'rtl_run.log')!=v['log_sha256'] or v['independent_traffic']!=prep['independent_traffic']):
        raise ValueError('Incomplete tensor RTL proof')
    rows=[dict(zip(['frame','op','ca','cycles','words'],map(int,m))) for m in re.findall(
        r'BURST_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)',log)]
    if rows!=v['results'] or len(rows)!=35 or '26 error cases;' not in log: raise ValueError('Cycle/coverage proof changed')
    writer=root/'sim_writer/writer_run.log';wlog=writer.read_text(encoding='utf-8')
    if 'Lung AXI burst write64 RTL: PASS (13 cases;' not in wlog: raise ValueError('Missing independent burst writer protocol proof')
    mid=root/'sim_mid_faults';m=json.loads((mid/'verification.json').read_text(encoding='utf-8'))
    check_hashes(m['source_sha256'])
    if m['status']!='passed_actual_xsim_burst_mid_command_faults' or file_sha256(mid/'rtl_run.log')!=m['log_sha256']:
        raise ValueError('Missing mid-command failure/drain proof')
    for text in [log,wlog,(mid/'rtl_run.log').read_text(encoding='utf-8')]:
        if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I): raise ValueError('Actual RTL error/warning')
    baseline=root/'sim_baseline';original=json.loads((baseline/'verification.json').read_text(encoding='utf-8'))
    check_hashes(original['source_sha256'])
    if original['status']!='passed_actual_xsim_word_same_burst_bfm' or file_sha256(baseline/'rtl_run.log')!=original['log_sha256']:
        raise ValueError('Missing matched same-BFM old-engine proof')
    full=[]
    for case,a,b in zip(prep['cases'],original['results'],rows):
        if b['words']!=case['words'] or (b['op'],b['ca'])!=(case['op'],case['ca']): raise ValueError('Fixture/cycle identity mismatch')
        if case['name'].startswith('FULL_'):
            if b['cycles']>=a['cycles']: raise ValueError('Burst did not reduce full-layer cycles')
            full.append(dict(name=case['name'],old_cycles=a['cycles'],new_cycles=b['cycles'],simulated_ratio=a['cycles']/b['cycles']))
    if len(full)!=3: raise ValueError('Missing full frozen-model concat layers')
    ooc=utilization(root/'ooc/utilization.rpt');wns=timing(root/'ooc/timing.rpt')
    if ooc['DSPs']!=40 or ooc['Block RAM Tile']!=0: raise ValueError('Unexpected tensor resource inference')
    return prep,dict(actual_xsim=True,frames=35,output_words=379310,error_cases=26,burst_writer_cases=13,
        mid_command_read_and_b_errors=True,same_burst_bfm_baseline=True,full_size_concat=full,logical_reads=sum(t[0] for t in prep['independent_traffic']),
        physical_read_bursts=sum(t[1] for t in prep['independent_traffic']),physical_read_beats=sum(t[2] for t in prep['independent_traffic']),
        physical_write_bursts=sum(t[3] for t in prep['independent_traffic']),ooc_utilization=ooc,ooc_wns_ns=wns,
        scope='Physical burst counts independently checked under stalled BFM, clipped spans and 4K offsets; NOT real-board timing')


def audit(root):
    root=Path(root).resolve();out=root/'verification.json';target=root/'lung_tensor_burst'
    if out.exists() or target.exists(): raise FileExistsError('Preserve old proof/package')
    prep,simulation=rtl_proof(root)
    resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not match or float(match[1].split()[4])<0: raise ValueError('Hold timing failed')
    whs=float(match[1].split()[4])
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text(encoding='utf-8')):
        raise ValueError('Final DRC failed')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    oldroot=WORKSPACE/'data/lung_cache_fast_20261016';previous=json.loads((oldroot/'verification.json').read_text(encoding='utf-8'))
    if file_sha256(oldroot/'verification.json')!=prep['previous_cache_proof_sha256']: raise ValueError('Prior cache proof changed')
    for key in ['source_sha256','packaged_source_sha256','evidence_sha256']: check_hashes(previous[key])
    old=oldroot/'lung_cache_fast';check_hashes({str(old/n):h for n,h in previous['package_files_sha256'].items()})
    packaged={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        archived={Path(p).name:h for p,h in previous['packaged_source_sha256'].items() if top in Path(p).parts}
        expected=dict(archived)
        if top=='lung_tensor_axil':
            del expected['lung_tensor_word_engine.v'];del expected['lung_axi_write64.v']
            expected.update({n:file_sha256(root/'rtl'/n) for n in ['lung_tensor_burst_engine.v','lung_tensor_axil.v']})
            expected['lung_axi_burst_write64.v']=file_sha256(WORKSPACE/'fpga/lung_axi_burst_write64.v')
        files=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if len(files)!=len(expected) or {p.name for p in files}!=set(expected): raise ValueError('Packaged dependency set changed')
        for p in files:
            if file_sha256(p)!=expected[p.name]: raise ValueError('Packaged RTL differs: '+str(p))
            packaged[str(p)]=file_sha256(p)
    raw=(root/'python_tests.log').read_bytes();tests=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    count=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',tests)
    if not count or int(count[1])<270 or 'skipped=' in tests: raise ValueError('Need full 270-test regression')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    target.mkdir()
    for p in old.iterdir():
        if not p.is_file(): raise ValueError('Unexpected nonfile in old board package')
        shutil.copy2(p,target/p.name)
    for n in ['lung_network.bit','lung_network.hwh']: shutil.copy2(root/'deploy'/n,target/n)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    for n in ['continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json']:
        data=json.loads((target/n).read_text(encoding='utf-8'));data['files_sha256']={s:file_sha256(target/s) for s in data['files_sha256']}
        if n=='continuous_manifest.json': data.update(original_continuous_manifest_sha256=file_sha256(old/n),bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'])
        elif n=='cpu_compare_manifest.json': data['original_continuous_package_sha256']=file_sha256(target/'continuous_manifest.json')
        else:
            data.update(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),hardware_revision='tensor_burst_v1',bit_changed=True,board_tested=False,
                warning='Same frozen model/arithmetic/conv cache; tensor bounded burst read caches/write combining. READS remains logical delivered words, not physical beats. New board correctness/latency pending.')
        write_json(target/n,data)
    clock=json.loads((target/'clock_manifest.json').read_text(encoding='utf-8'))
    clock.update(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route)
    write_json(target/'clock_manifest.json',clock)
    selection=json.loads((target/'selection_manifest.json').read_text(encoding='utf-8'))
    selection.update(target_folder='/home/xilinx/ct_test/lung_tensor_burst',files_sha256={n:file_sha256(target/n) for n in selection['files_sha256']})
    write_json(target/'selection_manifest.json',selection)
    validate_package(target);validate_addon(target);_,fast=validate_selection(target)
    prior=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        with selected_entrypoints(target,fast) as (bench,clock,origins): clock.validate_addon(target)
    finally: sys.dont_write_bytecode=prior
    if any(not p.is_file() for p in target.iterdir()): raise ValueError('No desktop caches in upload package')
    evidence=[root/n for n in ['preparation.json','build_burst.tcl','synth_burst.tcl','ps_settings.tcl','python_tests.log',
        'utilization_routed.rpt','timing_routed.rpt','configuration_exceptions.rpt','drc.rpt','sim_burst/verification.json',
        'sim_burst/rtl_run.log','sim_writer/writer_run.log','sim_mid_faults/verification.json','sim_mid_faults/rtl_run.log',
        'sim_mid_faults/tb_mid.sv','sim_baseline/verification.json','sim_baseline/rtl_run.log','sim_baseline/tb_baseline.sv','ooc/utilization.rpt','ooc/timing.rpt']]
    sources=[Path(__file__).resolve(),WORKSPACE/'python/lung_tensor_burst_faults.py',WORKSPACE/'python/tests/test_lung_tensor_burst.py',
        WORKSPACE/'fpga/tb_lung_axi_burst_write64.sv',WORKSPACE/'python/verify_lung_prepared_board.py',WORKSPACE/'python/lung_tensor_burst_baseline.py']
    write_json(out,dict(status='built_routed_tensor_burst_not_board_tested',board_tested=False,board_speedup_known=False,
        simulation=simulation,hardware=hardware,routed_design=route,utilization=resources,python_tests=int(count[1]),
        protected_medical_sources_unchanged=len(protected),model_changed=False,arithmetic_changed=False,convolution_rtl_changed=False,
        source_sha256=prep['source_sha256']|{str(p):file_sha256(p) for p in sources},packaged_source_sha256=packaged,
        evidence_sha256={str(p):file_sha256(p) for p in evidence},package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()}))
    print('New tensor burst package audited:',target,flush=True);print(route,resources,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_tensor_burst_20261016_v3')
    parser.add_argument('--rtl-only',action='store_true');args=parser.parse_args()
    if args.rtl_only: print(json.dumps(rtl_proof(args.root)[1],indent=2))
    else: audit(args.root)
