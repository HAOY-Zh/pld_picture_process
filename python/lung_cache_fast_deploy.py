"""Audit actual cache/window RTL and routed Z2 design, then bind a NEW package."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from lung_cache_fast_prepare import WORKSPACE,TOPS,MARKERS,cycle_rows
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
    check_hashes(prep['source_sha256']);check_hashes(prep['frozen_vectors_sha256'])
    check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    previous=WORKSPACE/'data/lung_word_20261015/verification.json'
    if file_sha256(previous)!=prep['previous_word_proof_sha256']: raise ValueError('Previous word proof changed')
    variants=[]
    for variant in ['baseline','fast']:
        folder=root/('sim_'+variant)
        proof=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        if proof['status']!='passed_actual_xsim_cache_'+variant or set(proof['results'])!=set(TOPS):
            raise ValueError('Missing actual RTL checks')
        for top,marker in zip(TOPS,MARKERS):
            path=folder/(top+'_run.log');text=path.read_text(encoding='utf-8')
            if file_sha256(path)!=proof['results'][top]['log_sha256'] or marker not in text:
                raise ValueError('RTL evidence changed/incomplete: '+str(path))
            if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I): raise ValueError('RTL fault')
            if top in ['tb_lung_cached_axi','tb_lung_axil_accel']:
                if cycle_rows(text)!=proof['results'][top]['cycles']: raise ValueError('Cycle proof differs')
        variants.append(proof)
    ratios={}
    for top in ['tb_lung_cached_axi','tb_lung_axil_accel']:
        old,new=[sum(r['cycles'] for r in p['results'][top]['cycles']) for p in variants]
        if new>=old: raise ValueError('No cycle reduction under the same BFM policy')
        ratios[top]=dict(old_cycles=old,new_cycles=new,simulated_ratio=old/new)
    ooc=root/'ooc'
    resources=utilization(ooc/'utilization.rpt');wns=timing(ooc/'timing_synth.rpt')
    if resources['Block RAM Tile']!=31.5 or resources['DSPs']!=70: raise ValueError('Unexpected cache/MAC memory or DSP inference')
    return prep,dict(actual_xsim=True,top_checks_per_variant=6,frames_per_convolution_check=147,
        exact_codes_per_convolution_check=68849,register_driven_words=8866,
        cache_capacity_reads=18437,cache_protocol_reads=28,memory_rejection_error_cases=14,
        axil_error_cases=5,cycle_comparison=ratios,ooc_utilization=resources,ooc_wns_ns=wns,
        scope='Same frozen independent gold and stalled BFM policy; NOT board end-to-end speedup')


def audit(root):
    root=Path(root).resolve();output=root/'verification.json';target=root/'lung_cache_fast'
    if output.exists() or target.exists(): raise FileExistsError('Preserve prior proof/package')
    prep,simulation=rtl_proof(root)
    resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not match or float(match[1].split()[4])<0: raise ValueError('Routed hold timing failed')
    whs=float(match[1].split()[4])
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text(encoding='utf-8')):
        raise ValueError('Final DRC Error/Critical Warning')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    oldroot=WORKSPACE/'data/lung_word_20261015'
    previous=json.loads((oldroot/'verification.json').read_text(encoding='utf-8'))
    check_hashes(previous['source_sha256']);check_hashes(previous['packaged_source_sha256']);check_hashes(previous['evidence_sha256'])
    timing_source=WORKSPACE/'fpga/lung_axil_config_timing.xdc'
    original_control=json.loads((WORKSPACE/'data/lung_axil_20261015/verification.json').read_text(encoding='utf-8'))
    if file_sha256(timing_source)!=original_control['source_sha256'][str(timing_source)]:
        raise ValueError('Original scoped configuration timing exceptions changed')
    old=oldroot/'lung_word'
    check_hashes({str(old/n):h for n,h in previous['package_files_sha256'].items()})
    imported={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        copies=list((root/'ip_repo'/top).rglob('lung_*.v'))
        archived={Path(p).name:h for p,h in previous['packaged_source_sha256'].items() if top in Path(p).parts}
        if len(copies)!=len(archived) or {p.name for p in copies}!=set(archived): raise ValueError('Packaged dependency set changed')
        for p in copies:
            expected=file_sha256(root/'rtl'/p.name) if p.name in ['lung_feature_cache.v','lung_feature_window.v'] else archived[p.name]
            if file_sha256(p)!=expected: raise ValueError('Packaged RTL mismatch: '+str(p))
            imported[str(p)]=file_sha256(p)
    lograw=(root/'python_tests.log').read_bytes()
    tests=lograw.decode('utf-16' if lograw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    count=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',tests)
    if not count or int(count[1])<266 or 'skipped=' in tests: raise ValueError('Need complete 266-test regression')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    # Copy the known package only AFTER all hardware/test gates succeed.
    target.mkdir()
    for p in old.iterdir(): shutil.copy2(p,target/p.name)
    for name in ['lung_network.bit','lung_network.hwh']: shutil.copy2(root/'deploy'/name,target/name)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,
        bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    for name in ['continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json']:
        data=json.loads((target/name).read_text(encoding='utf-8'))
        data['files_sha256']={n:file_sha256(target/n) for n in data['files_sha256']}
        if name=='continuous_manifest.json':
            data.update(original_continuous_manifest_sha256=file_sha256(old/name),bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'])
        elif name=='cpu_compare_manifest.json': data['original_continuous_package_sha256']=file_sha256(target/'continuous_manifest.json')
        else:
            data.update(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),bit_changed=True,
                hardware_revision='cache_window_fast_v1',board_tested=False,
                warning='Fast synchronous cache lookup/channel address stepping, same word tensor and frozen math/model; new board correctness/timing pending.')
        write_json(target/name,data)
    clocks=json.loads((target/'clock_manifest.json').read_text(encoding='utf-8'))
    clocks.update(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route,
                  warning='New routed cache/window bit; calibrated actual FCLK0. Use pinned entry; board correctness/speed pending.')
    write_json(target/'clock_manifest.json',clocks)
    selected=WORKSPACE/'data/lung_selected_20261015/deploy'
    selection_proof=json.loads((selected.parent/'verification.json').read_text(encoding='utf-8'))
    check_hashes(selection_proof['source_sha256'])
    check_hashes({str(selected/n):h for n,h in selection_proof['deploy_sha256'].items()})
    shutil.copy2(selected/'lung_selected_benchmark.py',target/'lung_selected_benchmark.py')
    write_json(target/'selection_manifest.json',dict(schema='lung_pinned_entrypoint_v1',
        target_folder='/home/xilinx/ct_test/lung_cache_fast',files_sha256={n:file_sha256(target/n) for n in
        ['lung_selected_benchmark.py','fast_manifest.json','clock_manifest.json','lung_network.bit','lung_network.hwh']},
        warning='Explicit file-rooted benchmark and pre-download bit identity guard. New board result required.'))
    validate_package(target);validate_addon(target);_,fast=validate_selection(target)
    prior_bytecode=sys.dont_write_bytecode
    try:
        # Desktop import checks must not put host-version __pycache__ into the
        # board upload directory or its signed file inventory.
        sys.dont_write_bytecode=True
        with selected_entrypoints(target,fast) as (bench,clock,origins): clock.validate_addon(target)
    finally: sys.dont_write_bytecode=prior_bytecode
    if any(not p.is_file() for p in target.iterdir()): raise ValueError('Only declared files may enter board package')
    evidence=[p for folder in [root/'sim_baseline',root/'sim_fast'] for p in folder.glob('*_run.log')]
    evidence += [root/n for n in ['preparation.json','build_cache.tcl','synth_cache.tcl','ps_settings.tcl',
        'python_tests.log','utilization_routed.rpt','timing_routed.rpt','drc.rpt','configuration_exceptions.rpt',
        'ooc/utilization.rpt','ooc/utilization_hier.rpt','ooc/timing_synth.rpt','sim_fast/verification.json','sim_baseline/verification.json']]
    record=dict(status='built_routed_cache_window_fast_not_board_tested',board_tested=False,
        board_speedup_known=False,hardware=hardware,simulation=simulation,routed_design=route,
        utilization=resources,python_tests=int(count[1]),protected_medical_sources_unchanged=len(protected),
        medical_model_changed=False,arithmetic_changed=False,tensor_word_rtl_changed=False,
        source_sha256=prep['source_sha256']|{str(Path(__file__).resolve()):file_sha256(__file__),str(timing_source):file_sha256(timing_source),
            str(WORKSPACE/'python/tests/test_lung_cache_fast.py'):file_sha256(WORKSPACE/'python/tests/test_lung_cache_fast.py')},
        packaged_source_sha256=imported,evidence_sha256={str(p):file_sha256(p) for p in evidence},
        package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()})
    write_json(output,record)
    print('New routed cache/window package audited:',target,flush=True)
    print('Route:',route,'resources:',resources,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_cache_fast_20261016')
    parser.add_argument('--rtl-only',action='store_true')
    args=parser.parse_args()
    if args.rtl_only: print(json.dumps(rtl_proof(args.root)[1],indent=2))
    else: audit(args.root)
