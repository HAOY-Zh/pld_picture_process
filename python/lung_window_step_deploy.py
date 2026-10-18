"""Require frozen same-BFM RTL, timing, source and input proofs before new deployment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from lung_window_step_prepare import WORKSPACE, OLD, TOPS, MARKERS, EXTRA, checked_old
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
    root=Path(root).resolve();p=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    checked_old();check_hashes(p['source_sha256']);check_hashes(p['frozen_vectors_sha256'])
    check_hashes({str(root/n):h for n,h in p['generated_sha256'].items()})
    if file_sha256(OLD/'verification.json')!=p['previous_proof_sha256'] or file_sha256(OLD/'lung_tensor_burst_confirm_result_verification.json')!=p['accepted_board_proof_sha256']:
        raise ValueError('Prior build/physical acceptance changed')
    variants=[]
    for variant in ['baseline','step']:
        folder=root/('sim_'+variant);proof=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        if proof['status']!='passed_actual_xsim_step_window_'+variant or set(proof['results'])!=set(TOPS+['tb_lung_window_step']):
            raise ValueError('Missing actual RTL checks')
        if proof['source_sha256']!=p['source_sha256']: raise ValueError('Different simulated sources')
        for top,marker in zip(TOPS+['tb_lung_window_step'],MARKERS+[EXTRA]):
            path=folder/(top+'_run.log');text=path.read_text(encoding='utf-8')
            if file_sha256(path)!=proof['results'][top]['log_sha256'] or marker not in text:
                raise ValueError('Missing/changed RTL proof: '+str(path))
            if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I): raise ValueError('RTL failure')
            if top in ['tb_lung_cached_axi','tb_lung_axil_accel'] and cycle_rows(text)!=proof['results'][top]['cycles']:
                raise ValueError('Changed cycle proof')
        variants.append(proof)
    ratios={}
    for top in ['tb_lung_cached_axi','tb_lung_axil_accel']:
        old,new=[sum(r['cycles'] for r in v['results'][top]['cycles']) for v in variants]
        if new>=old: raise ValueError('No matched-BFM cycle benefit')
        ratios[top]=dict(old_cycles=old,new_cycles=new,simulated_ratio=old/new)
    ooc=utilization(root/'ooc/utilization.rpt');wns=timing(root/'ooc/timing_synth.rpt')
    if ooc['Block RAM Tile']!=31.5 or ooc['DSPs']>70: raise ValueError('Unexpected memory/MAC resource increase')
    return p,dict(actual_xsim=True,top_checks_per_variant=7,frames_per_convolution_check=147,
        exact_codes_per_convolution_check=68849,windows_per_check=371974,logical_requests=312018,
        bursts_per_check=2612,read_beats=41606,register_driven_words=8866,
        independent_window_cases=10,cycle_comparison=ratios,ooc_utilization=ooc,ooc_wns_ns=wns,
        scope='Same frozen gold, cache policy and stalled memory BFM; not real-board performance')


def audit(root):
    root=Path(root).resolve();out=root/'verification.json';target=root/'lung_window_step'
    if out.exists() or target.exists(): raise FileExistsError('Preserve prior proof/package')
    prep,simulation=rtl_proof(root);previous=checked_old()
    resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not match or float(match[1].split()[4])<0: raise ValueError('Hold timing failed')
    whs=float(match[1].split()[4])
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text(encoding='utf-8')): raise ValueError('Final DRC failed')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    imported={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        expected={Path(n).name:h for n,h in previous['packaged_source_sha256'].items() if top in Path(n).parts}
        if 'lung_feature_window.v' in expected: expected['lung_feature_window.v']=file_sha256(root/'rtl/lung_feature_window.v')
        files=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if len(files)!=len(expected) or {p.name for p in files}!=set(expected): raise ValueError('RTL dependency set changed')
        for p in files:
            if file_sha256(p)!=expected[p.name]: raise ValueError('Unexpected RTL change: '+str(p))
            imported[str(p)]=file_sha256(p)
    test_path=root/'python_tests_resume.log'
    raw=test_path.read_bytes();tests=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    count=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',tests)
    if not count or int(count[1])<279 or 'skipped=' in tests: raise ValueError('Need full 279-test resumed regression')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    old=OLD/'lung_tensor_burst';target.mkdir()
    for p in old.iterdir():
        if not p.is_file(): raise ValueError('Unexpected old package subdirectory')
        shutil.copy2(p,target/p.name)
    for n in ['lung_network.bit','lung_network.hwh']: shutil.copy2(root/'deploy'/n,target/n)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    for n in ['continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json']:
        data=json.loads((target/n).read_text(encoding='utf-8'));data['files_sha256']={s:file_sha256(target/s) for s in data['files_sha256']}
        if n=='continuous_manifest.json': data.update(original_continuous_manifest_sha256=file_sha256(old/n),bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'])
        elif n=='cpu_compare_manifest.json': data['original_continuous_package_sha256']=file_sha256(target/'continuous_manifest.json')
        else:
            data.update(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),hardware_revision='window_step_v1',bit_changed=True,board_tested=False,
                warning='Same frozen model/math/cache/tensor/CPU; registered window address stepping only. New physical outputs and latency required.')
        write_json(target/n,data)
    clock=json.loads((target/'clock_manifest.json').read_text(encoding='utf-8'))
    clock.update(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route);write_json(target/'clock_manifest.json',clock)
    selection=json.loads((target/'selection_manifest.json').read_text(encoding='utf-8'))
    selection.update(target_folder='/home/xilinx/ct_test/lung_window_step',files_sha256={n:file_sha256(target/n) for n in selection['files_sha256']})
    write_json(target/'selection_manifest.json',selection)
    validate_package(target);validate_addon(target);_,fast=validate_selection(target)
    prior=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        with selected_entrypoints(target,fast) as (bench,clock,origins): clock.validate_addon(target)
    finally: sys.dont_write_bytecode=prior
    if any(not p.is_file() for p in target.iterdir()): raise ValueError('No host caches in upload package')
    evidence=[root/n for n in ['preparation.json','build_step.tcl','synth_step.tcl','ps_settings.tcl','python_tests.log','python_tests_resume.log',
        'utilization_routed.rpt','timing_routed.rpt','configuration_exceptions.rpt','drc.rpt','ooc/utilization.rpt','ooc/timing_synth.rpt']]
    evidence += [root/('sim_'+v)/n for v in ['baseline','step'] for n in ['verification.json']+[t+'_run.log' for t in TOPS+['tb_lung_window_step']]]
    sources=[Path(__file__).resolve(),WORKSPACE/'python/tests/test_lung_window_step.py',WORKSPACE/'python/verify_lung_prepared_board.py',
        WORKSPACE/'fpga/resume_lung_window_step.tcl',WORKSPACE/'fpga/validate_lung_window_step.tcl',
        WORKSPACE/'python/tests/test_lung_window_step_project.py']
    write_json(out,dict(status='built_routed_window_step_not_board_tested',board_tested=False,board_speedup_known=False,
        simulation=simulation,hardware=hardware,routed_design=route,utilization=resources,python_tests=int(count[1]),
        protected_medical_sources_unchanged=len(protected),model_changed=False,arithmetic_changed=False,tensor_rtl_changed=False,
        source_sha256=prep['source_sha256']|{str(p):file_sha256(p) for p in sources},packaged_source_sha256=imported,
        evidence_sha256={str(p):file_sha256(p) for p in evidence},package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()}))
    print('Routed window-step package audited:',target,flush=True);print(route,resources,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_window_step_20261016_v2')
    parser.add_argument('--rtl-only',action='store_true');args=parser.parse_args()
    print(json.dumps(rtl_proof(args.root)[1],indent=2)) if args.rtl_only else audit(args.root)
