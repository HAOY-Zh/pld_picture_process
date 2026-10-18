"""Audit word-parallel RTL/route and create a NEW bound deployment, never overwrite old proofs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes,timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import validate_addon


def parse_cycles(text):
    if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I): raise ValueError('RTL tool error')
    marker=re.search(r'Lung tensor AXI-Lite RTL: PASS \((\d+) frames, (\d+) words, (\d+) reads, (\d+) error cases;',text)
    rows=[dict(zip(['frame','op','ca','cycles','words'],map(int,m))) for m in re.findall(
        r'WORD_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)',text)]
    if not marker or len(rows)!=int(marker[1]) or int(marker[4])!=24:
        raise ValueError('Missing complete normal/fault coverage')
    if [r['frame'] for r in rows]!=list(range(len(rows))) or any(r['cycles']<=0 for r in rows):
        raise ValueError('Bad frame cycle records')
    if sum(r['words'] for r in rows)!=int(marker[2]): raise ValueError('Output coverage mismatch')
    return rows


def rtl_proof(root):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    results=[]
    for variant in ['baseline','word']:
        folder=root/('sim_'+variant);v=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        if v['status']!='passed_actual_xsim_tensor_'+variant: raise ValueError('Missing actual XSim success')
        check_hashes(v['source_sha256'])
        if file_sha256(folder/'rtl_run.log')!=v['log_sha256']: raise ValueError('Changed RTL log')
        rows=parse_cycles((folder/'rtl_run.log').read_text(encoding='utf-8'))
        if rows!=v['results'] or len(rows)!=len(prep['cases']): raise ValueError('Simulation counters changed')
        for r,case in zip(rows,prep['cases']):
            if (r['op'],r['ca'],r['words'])!=(case['op'],case['ca'],case['words']): raise ValueError('Fixture identity mismatch')
        results.append(rows)
    old,new=results;full=[]
    for case,a,b in zip(prep['cases'],old,new):
        if case['name'].startswith('FULL_'):
            if not b['cycles']<a['cycles']: raise ValueError('Full-size word path did not reduce cycles')
            full.append(dict(name=case['name'],words=b['words'],old_cycles=a['cycles'],new_cycles=b['cycles'],
                             simulated_cycle_ratio=a['cycles']/b['cycles']))
    if len(full)!=3: raise ValueError('Missing full frozen concat frames')
    return prep,dict(frames=len(new),output_words=sum(r['words'] for r in new),fault_cases_per_variant=24,
                     full_size_concat=full,scope='Actual XSim, deterministic stalled AXI; not board end-to-end speedup')


def audit(root):
    root=Path(root).resolve(); output=root/'verification.json';target=root/'lung_word'
    if output.exists() or target.exists(): raise FileExistsError('Preserve prior package/proof')
    prep,sim=rtl_proof(root)
    ooc=utilization(root/'ooc/utilization.rpt');ooc_wns=timing(root/'ooc/timing.rpt')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not match or float(match[1].split()[4])<0: raise ValueError('Routed hold timing failed')
    whs=float(match[1].split()[4])
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text(encoding='utf-8')):
        raise ValueError('DRC error/critical warning')
    required={
        'lung_conv8_axil':{'lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v','lung_feature_window.v',
            'lung_conv8_window.v','lung_feature_cache.v','lung_axi_read64.v','lung_conv8_cached_axi.v','lung_axi_write64.v',
            'lung_output_writer.v','lung_conv8_memory_axi.v','lung_axil_control.v','lung_conv8_axil.v'},
        'lung_tensor_axil':{'lung_requant.v','lung_axi_read64.v','lung_axi_write64.v','lung_tensor_word_engine.v',
            'lung_tensor_control.v','lung_tensor_axil.v'}}
    original_board=json.loads((WORKSPACE/'data/lung_network_board_20261015/verification.json').read_text(encoding='utf-8'))
    check_hashes(original_board['packaged_source_sha256'])
    imported={}
    for top,names in required.items():
        copies=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if {p.name for p in copies}!=names or len(copies)!=len(names): raise ValueError('Incorrect packaged dependencies')
        for p in copies:
            source=root/'rtl/word/lung_tensor_axil.v' if p.name=='lung_tensor_axil.v' else WORKSPACE/'fpga'/p.name
            if file_sha256(p)!=file_sha256(source): raise ValueError('Packaged RTL changed: '+str(p))
            if p.name not in ['lung_tensor_axil.v','lung_tensor_word_engine.v']:
                archived=[digest for path,digest in original_board['packaged_source_sha256'].items()
                          if top in Path(path).parts and Path(path).name==p.name]
                if len(archived)!=1 or file_sha256(p)!=archived[0]:
                    raise ValueError('Previously frozen convolution/control/AXI/RNE dependency changed: '+p.name)
            imported[str(p)]=file_sha256(p)
    lograw=(root/'python_tests.log').read_bytes()
    tests=lograw.decode('utf-16' if lograw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    count=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',tests)
    if not count or int(count[1])<254 or 'skipped=' in tests: raise ValueError('Need full 254-test regression')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    original=WORKSPACE/'data/lung_fast_20261015/lung_fast';_,_,_,fast=validate_package(original)
    check_hashes(fast['source_sha256'])
    target.mkdir()
    for p in original.iterdir():
        if p.name not in ['lung_network.bit','lung_network.hwh']: shutil.copy2(p,target/p.name)
    for name in ['lung_network.bit','lung_network.hwh']: shutil.copy2(root/'deploy'/name,target/name)
    bitsha=file_sha256(target/'lung_network.bit');hwhsha=file_sha256(target/'lung_network.hwh')
    # Rebind only NEW copies, in dependency order. Original files remain intact.
    for name in ['continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json']:
        data=json.loads((target/name).read_text(encoding='utf-8'))
        data['files_sha256']={n:file_sha256(target/n) for n in data['files_sha256']}
        if name=='continuous_manifest.json':
            data.update(original_continuous_manifest_sha256=file_sha256(original/name),bit_sha256=bitsha,hwh_sha256=hwhsha)
        elif name=='cpu_compare_manifest.json': data['original_continuous_package_sha256']=file_sha256(target/'continuous_manifest.json')
        else:
            data.update(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),bit_changed=True,
                hardware_revision='word_concat_v1',board_tested=False,
                warning='New word-parallel tensor bit with unchanged convolution, arithmetic, model, codes and runtime; real-board correctness/latency pending.')
        write_json(target/name,data)
    shutil.copy2(WORKSPACE/'pynq/lung_clock_benchmark.py',target/'lung_clock_benchmark.py')
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=bitsha,hwh_sha256=hwhsha)
    write_json(target/'clock_manifest.json',dict(schema='lung_routed_clock_benchmark_v1',target_clock_mhz=100,
        original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),
        files_sha256={'lung_clock_benchmark.py':file_sha256(target/'lung_clock_benchmark.py')},routed_design=route,
        warning='New tensor RTL bit, runtime calibrates actual FCLK0 after load; physical board output and timing still required.'))
    validate_package(target);validate_addon(target)
    evidence=[root/n for n in ['preparation.json','build_word.tcl','synth_word.tcl','ps_settings.tcl','python_tests.log',
        'utilization_routed.rpt','timing_routed.rpt','drc.rpt','configuration_exceptions.rpt',
        'sim_word/rtl_run.log','sim_baseline/rtl_run.log','sim_word/verification.json','sim_baseline/verification.json',
        'ooc/utilization.rpt','ooc/timing.rpt']]
    record=dict(status='built_routed_word_parallel_tensor_not_board_tested',hardware=hardware,simulation=sim,
        utilization=resources,routed_wns_ns=wns,routed_whs_ns=whs,ooc_utilization=ooc,ooc_wns_ns=ooc_wns,
        python_tests=int(count[1]),protected_medical_sources_unchanged=len(protected),board_tested=False,
        medical_model_changed=False,convolution_rtl_changed=False,board_speedup_known=False,
        source_sha256=prep['source_sha256'] | {str(Path(__file__).resolve()):file_sha256(__file__),
            str(WORKSPACE/'python/tests/test_lung_word.py'):file_sha256(WORKSPACE/'python/tests/test_lung_word.py')},
        packaged_source_sha256=imported,evidence_sha256={str(p):file_sha256(p) for p in evidence},
        package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()},
        old_fast_manifest_sha256=file_sha256(original/'fast_manifest.json'),routed_design=route)
    write_json(output,record)
    print('New word-tensor package audited:',target,flush=True)
    print('Routed resources:',resources,'WNS',wns,'WHS',whs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_word_20261015')
    audit(parser.parse_args().root)
