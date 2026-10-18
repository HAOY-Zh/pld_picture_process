"""Audit hit-return RTL, routed clocks/AXI and immutable sources before deployment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from lung_cache_return_prepare import WORKSPACE,OLD,ROOT,TOPS,MARKERS,EXTRA,checked_previous
from lung_cache_fast_prepare import cycle_rows
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
    root=Path(root).resolve();prior=checked_previous();p=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    for k in ['source_sha256','frozen_vectors_sha256']:check_hashes(p[k])
    check_hashes({str(root/n):h for n,h in p['generated_sha256'].items()})
    if p['previous_proof_sha256']!=file_sha256(OLD/'verification.json') or p['accepted_board_proof_sha256']!=file_sha256(OLD/'lung_window_step_result_verification.json'):
        raise ValueError('Accepted baseline changed')
    folder=root/'sim_return';proof=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    names=TOPS+['tb_lung_window_step','tb_lung_cache_return'];markers=MARKERS+[EXTRA,'Lung direct cache return RTL: PASS']
    if proof['status']!='passed_actual_xsim_cache_return' or set(proof['results'])!=set(names) or proof['source_sha256']!=p['source_sha256']:
        raise ValueError('Incomplete actual RTL regression')
    ratios={}
    for top,marker in zip(names,markers):
        log=folder/(top+'_run.log');text=log.read_text(encoding='utf-8')
        if file_sha256(log)!=proof['results'][top]['log_sha256'] or marker not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
            raise ValueError('Invalid RTL log: '+str(log))
        if top in ['tb_lung_cached_axi','tb_lung_axil_accel']:
            before=sum(r['cycles'] for r in cycle_rows((OLD/'sim_step'/(top+'_run.log')).read_text(encoding='utf-8')))
            after=sum(r['cycles'] for r in cycle_rows(text))
            if after>=before:raise ValueError('No matched frozen-BFM cycle gain')
            ratios[top]=dict(old_cycles=before,new_cycles=after,ratio=before/after)
    return p,prior,dict(actual_xsim=True,tops=names,convolution_frames=147,exact_codes=68849,
        read_requests=312018,physical_bursts=2612,read_beats=41606,register_words=8866,independent_hit_checks=16,
        cycle_comparison=ratios,scope='Frozen same-BFM RTL comparison; not board latency')


def audit(root):
    root=Path(root).resolve();out=root/'verification.json';target=root/'lung_cache_return'
    if out.exists() or target.exists():raise FileExistsError('Preserve accepted proof/package')
    prep,prior,sim=rtl_proof(root);resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not match or float(match[1].split()[4])<0:raise ValueError('Hold timing failed')
    whs=float(match[1].split()[4])
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text(encoding='utf-8')):raise ValueError('Final DRC failed')
    if resources['Block RAM Tile']!=31.5 or resources['DSPs']>120:raise ValueError('Unexpected DSP/BRAM inference')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot());imported={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        expected={Path(n).name:h for n,h in prior['packaged_source_sha256'].items() if top in Path(n).parts}
        if 'lung_feature_cache.v' in expected:expected['lung_feature_cache.v']=file_sha256(root/'rtl/lung_feature_cache.v')
        files=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if len(files)!=len(expected) or {p.name for p in files}!=set(expected):raise ValueError('Dependency set changed')
        for p in files:
            if file_sha256(p)!=expected[p.name]:raise ValueError('Unexpected changed RTL: '+str(p))
            imported[str(p)]=file_sha256(p)
    raw=(root/'python_tests.log').read_bytes();tests=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    count=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',tests)
    if not count or int(count[1])<285 or 'skipped=' in tests:raise ValueError('Need full regression >=285 tests, no skips')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256'];check_hashes(protected)
    old=OLD/'lung_window_step';target.mkdir()
    for p in old.iterdir():
        if not p.is_file():raise ValueError('Unexpected previous package folder')
        shutil.copy2(p,target/p.name)
    for n in ['lung_network.bit','lung_network.hwh']:shutil.copy2(root/'deploy'/n,target/n)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    for n in ['continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json']:
        m=json.loads((target/n).read_text(encoding='utf-8'));m['files_sha256']={s:file_sha256(target/s) for s in m['files_sha256']}
        if n=='continuous_manifest.json':m.update(original_continuous_manifest_sha256=file_sha256(old/n),bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'])
        elif n=='cpu_compare_manifest.json':m['original_continuous_package_sha256']=file_sha256(target/'continuous_manifest.json')
        else:m.update(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),hardware_revision='cache_return_v1',bit_changed=True,board_tested=False,
            warning='Same frozen model/math, window and tensor. Cache hit response one cycle earlier. New board output/latency required.')
        write_json(target/n,m)
    clock=json.loads((target/'clock_manifest.json').read_text(encoding='utf-8'));clock.update(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route);write_json(target/'clock_manifest.json',clock)
    selection=json.loads((target/'selection_manifest.json').read_text(encoding='utf-8'));selection.update(target_folder='/home/xilinx/ct_test/lung_cache_return',files_sha256={n:file_sha256(target/n) for n in selection['files_sha256']});write_json(target/'selection_manifest.json',selection)
    validate_package(target);validate_addon(target);_,fast=validate_selection(target)
    prior_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        with selected_entrypoints(target,fast) as (_,clock,_):clock.validate_addon(target)
    finally:sys.dont_write_bytecode=prior_bytecode
    if any(not p.is_file() for p in target.iterdir()):raise ValueError('No host caches in board upload')
    sources=[Path(__file__).resolve(),WORKSPACE/'python/tests/test_lung_cache_return.py']
    evidence=[root/n for n in ['preparation.json','build_return.tcl','ps_settings.tcl','python_tests.log','timing_routed.rpt','utilization_routed.rpt','drc.rpt','configuration_exceptions.rpt','sim_return/verification.json']]
    evidence += [root/'sim_return'/(n+'_run.log') for n in sim['tops']]
    write_json(out,dict(status='built_routed_cache_return_not_board_tested',board_tested=False,model_changed=False,arithmetic_changed=False,
        hardware=hardware,routed_design=route,utilization=resources,simulation=sim,python_tests=int(count[1]),protected_sources_unchanged=len(protected),
        source_sha256=prep['source_sha256']|{str(p):file_sha256(p) for p in sources},packaged_source_sha256=imported,
        evidence_sha256={str(p):file_sha256(p) for p in evidence},package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()}))
    print('Audited new board package:',target,route,resources,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default=str(ROOT));ap.add_argument('--rtl-only',action='store_true')
    a=ap.parse_args();print(json.dumps(rtl_proof(a.root)[2],indent=2)) if a.rtl_only else audit(a.root)
