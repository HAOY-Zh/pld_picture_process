"""Verify returned prepared-runtime files and actual head; summarize real node cycles."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

from lung_segmentation import write_json
from lung_unet import file_sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_continuous import unpack_words,word_count
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import validate_addon
from lung_selected_benchmark import validate_selection


def commands(row,model):
    expected=[(n['name'],'conv' if n['kind']=='conv' else 'tensor',b if n['kind']=='conv' else None,
               n['shape'][1]*n['shape'][2] if n['kind']=='conv' else word_count(n['shape']))
        for n in model['nodes'] for b in range((n['shape'][0]+7)//8 if n['kind']=='conv' else 1)]
    actual=row['commands'];last={}
    if row['command_count']!=len(expected) or len(actual)!=len(expected): raise ValueError('Missing commands')
    for c,(name,engine,block,words) in zip(actual,expected):
        if (c['node'],c['engine'],c['block'],c['written_words'])!=(name,engine,block,words): raise ValueError('Wrong command/order/count')
        if type(c['cycles']) is not int or not 0<c['cycles']<2**32-1: raise ValueError('Invalid/saturated cycle counter')
        if type(c['completed']) is not int or not 0<=c['completed']<2**32: raise ValueError('Invalid completion counter')
        if engine in last and c['completed']!=(last[engine]+1)&0xffffffff: raise ValueError('Completion count skipped')
        last[engine]=c['completed']
    if row['pl_command_cycles']!=sum(c['cycles'] for c in actual): raise ValueError('Cycle sum differs')
    return actual


def verify(root,package,stem):
    root=Path(root).resolve();package=Path(package).resolve();path=root/(stem+'.json')
    destination=root/(stem+'_verification.json')
    if destination.exists(): raise FileExistsError('Preserve earlier returned-result acceptance')
    manifest,model,params,fast=validate_package(package);validate_addon(package);validate_selection(package)
    def read(suffix): return json.loads((root/(stem+suffix+'.json')).read_text(encoding='utf-8'))
    report=read('');clock=read('_clock');selection=read('_selection');diag=read('_diagnostic');cpu=read('_cpu')
    if report['status']!='passed_real_board_prepared_runtime_comparison': raise ValueError('Not prepared PL PASS')
    if (report['case'],report['slice_index'],report['input_shape'])!=(manifest['case'],manifest['slice_index'],model['input_shape']):
        raise ValueError('Different CT input')
    if report['package_sha256']!=file_sha256(package/'fast_manifest.json'): raise ValueError('Different runtime package')
    for ext in ['bit','hwh']:
        if report[ext+'_sha256']!=file_sha256(package/('lung_network.'+ext)): raise ValueError('Different '+ext)
    if clock['status']!='passed_real_board_100mhz_clock_verified_benchmark' or selection['status']!='passed_pinned_real_board_benchmark':
        raise ValueError('Missing clock/selected-path acceptance')
    if clock['benchmark_sha256']!=file_sha256(path) or selection['benchmark_sha256']!=file_sha256(path): raise ValueError('Changed main result')
    if selection['clock_evidence_sha256']!=file_sha256(root/(stem+'_clock.json')): raise ValueError('Changed clock result')
    if clock['addon_sha256']!=file_sha256(package/'clock_manifest.json') or selection['manifest_sha256']!=file_sha256(package/'selection_manifest.json'):
        raise ValueError('Different addon/selection package')
    if selection['expected_bit_sha256']!=report['bit_sha256'] or selection['expected_fast_manifest_sha256']!=report['package_sha256']:
        raise ValueError('Selected bit/package identity differs')
    if len(selection['loads'])!=1 or selection['loads'][0]['bit_sha256']!=report['bit_sha256']: raise ValueError('Wrong selected download')
    if any(abs(t['cpu_mhz']-650)>.05 or (t['stage']!='overlay_loaded_before_clock' and abs(t['fclk0_mhz']-100)>.001)
           for t in clock['clock_trace']): raise ValueError('Different actual clocks')
    for section in ['cpu_build','io_build']:
        if any(file_sha256(package/n)!=h for n,h in report[section]['sources_sha256'].items()): raise ValueError('Different native source')
    if diag['status']!='passed_real_board_prepared_runtime_all_nodes' or diag['node_checks']!=[n['name'] for n in model['nodes']]:
        raise ValueError('Missing all-node diagnostic')
    if diag['package_sha256']!=report['package_sha256'] or report['diagnostic_sha256']!=file_sha256(root/(stem+'_diagnostic.json')):
        raise ValueError('Changed diagnostic')
    commands(diag['stats'],model)
    if cpu['status']!='passed_optimized_comparison_cpu_phase' or cpu['package_sha256']!=report['package_sha256'] or cpu['cpu_two_threads']!=report['cpu_two_threads']:
        raise ValueError('CPU phase identity differs')
    head=root/(stem+'_head.npz')
    if file_sha256(head)!=report['head_codes_sha256']: raise ValueError('Changed actual FPGA head')
    with np.load(head,allow_pickle=False) as a: actual=a['head_codes']
    with np.load(package/'reference.npz',allow_pickle=False) as a: expected=unpack_words(a['head'],model['nodes'][-1]['shape'])
    if actual.dtype!=np.uint8 or actual.shape!=expected.shape or not np.array_equal(actual,expected): raise ValueError('Actual head codes differ')
    details=report['optimized_pl']['details'];samples=report['optimized_pl']['samples_ms']
    if len(details)!=report['runs'] or len(samples)!=len(details) or report['optimized_pl']['all_21_nodes_verified'] is not True:
        raise ValueError('Incomplete measured frames')
    if report['optimized_pl']['all_heads_match'] is not True: raise ValueError('Head mismatch reported')
    nodes=defaultdict(float);engines=defaultdict(float)
    for row in details:
        for c in commands(row,model):
            ms=c['cycles']/100000/len(details);nodes[c['node']]+=ms;engines[c['engine']]+=ms
        for key in ['clock_before','clock_after']:
            if abs(row[key]['fclk0_mhz']-100)>.001 or abs(row[key]['cpu_mhz']-650)>.05: raise ValueError('Frame clock changed')
    for section in ['cpu_two_threads','optimized_pl']:
        values=report[section]['samples_ms']
        if len(values)!=report['runs'] or any(not math.isfinite(v) or v<=0 for v in values) or abs(sum(values)/len(values)-report[section]['mean_ms'])>1e-6:
            raise ValueError('Invalid mean/time samples')
    if abs(report['speedup_cpu2_over_fast_pl']-report['cpu_two_threads']['mean_ms']/report['optimized_pl']['mean_ms'])>1e-9:
        raise ValueError('Speedup arithmetic differs')
    proof=dict(status='verified_returned_prepared_board_actual_head_and_profile',actual_head_words=int(actual.size),
        actual_head_codes_bit_exact=True,scope='One validation CT slice, not dataset-wide accuracy or sustained clinical throughput',
        cpu_ms=report['cpu_two_threads']['mean_ms'],pl_ms=report['optimized_pl']['mean_ms'],engine_ms=dict(engines),
        node_ms=dict(sorted(nodes.items(),key=lambda x:-x[1])),
        input_package_sha256=report['package_sha256'],bit_sha256=report['bit_sha256'],
        returned_files_sha256={p.name:file_sha256(p) for p in [path,head]+[root/(stem+s+'.json') for s in ['_clock','_selection','_diagnostic','_cpu']]},
        verifier_sha256=file_sha256(__file__))
    write_json(destination,proof);print(json.dumps({k:v for k,v in proof.items() if k not in ['returned_files_sha256']},indent=2))
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_cache_fast_20261016')
    parser.add_argument('--package',default='data/lung_cache_fast_20261016/lung_cache_fast')
    parser.add_argument('--stem',default='lung_cache_fast_result')
    args=parser.parse_args();verify(args.root,args.package,args.stem)
