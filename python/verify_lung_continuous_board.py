"""Audit user-returned continuous PL report and, if present, actual head codes.

Report-only validation is explicitly NOT independent output-array acceptance.
Keep the original build/PC-emulation proof unchanged. No inference or training.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from lung_unet import file_sha256
from lung_segmentation import write_json
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_continuous import GUARD_WORDS, unpack_words, word_count
from lung_continuous_test import load_package


def positive_time(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0


def exact_int(row,key,expected):
    if type(row.get(key)) is not int or row[key] != expected:
        raise ValueError('Wrong integer statistic: '+key)


def planned_peak(model):
    uses=Counter(s for n in model['nodes'] for s in n['sources'])
    live={'input':(word_count(model['input_shape'])+GUARD_WORDS)*8}; peak=sum(live.values())
    for node in model['nodes']:
        live[node['name']]=(word_count(node['shape'])+GUARD_WORDS)*8
        peak=max(peak,sum(live.values()))
        for source in node['sources']:
            uses[source]-=1
            if uses[source]==0: del live[source]
    return peak


def validate_report(report,manifest,model,package_sha):
    if (report.get('status')!='passed_real_board_continuous_unet_codes'
            or report.get('continuous_network') is not True or report.get('core_forward_on_pl') is not True):
        raise ValueError('Not a continuous real-board PASS report')
    if report.get('package_sha256') != package_sha:
        raise ValueError('Wrong continuous package identity')
    for suffix in ['bit','hwh']:
        if report.get(suffix+'_sha256') != manifest['files_sha256']['lung_network.'+suffix]:
            raise ValueError('Wrong board artifact: '+suffix)
    for key in ['case','slice_index']:
        if report.get(key) != manifest[key]: raise ValueError('Wrong CT case/slice')
    for key,value in [('command_count',51),('convolution_commands',45),('tensor_commands',6),
                      ('clock_mhz',100),('cma_peak_bytes',planned_peak(model))]:
        exact_int(report,key,value)
    if report.get('head_shape')!=model['nodes'][-1]['shape']:
        raise ValueError('Wrong head geometry')
    if not re.fullmatch('[0-9a-f]{64}',str(report.get('head_codes_sha256',''))):
        raise ValueError('Invalid returned head digest')
    nodes=report.get('nodes',[]); checks=report.get('node_checks',[]); commands=report.get('commands',[])
    if len(nodes)!=21 or len(checks)!=21 or len(commands)!=51:
        raise ValueError('Incomplete command/node coverage')
    offset=0; counters=Counter(); shapes={'input':model['input_shape']}; pl_cycles=0
    engines=defaultdict(lambda:dict(commands=0,cycles=0,driver_call_ms=0.0))
    layer_times=[]
    for node,timing,check in zip(model['nodes'],nodes,checks):
        name=node['name']; words=word_count(node['shape']); shape=shapes[node['sources'][0]]
        for row in [timing,check]:
            if row.get('node')!=name: raise ValueError('Wrong node order')
            exact_int(row,'output_words',words)
        if check.get('same') is not True: raise ValueError('Node did not match reference')
        exact_int(check,'different_words',0)
        if not positive_time(timing.get('host_node_ms')) or not positive_time(timing.get('observer_ms')):
            raise ValueError('Invalid node timing')
        if timing['observer_ms']>timing['host_node_ms']: raise ValueError('Impossible observer timing')
        conv=node['kind']=='conv'; engine='conv' if conv else 'tensor'
        count=(node['shape'][0]+7)//8 if conv else 1
        batch=commands[offset:offset+count]; offset+=count; node_cycles=0; calls=0.0
        for block,command in enumerate(batch):
            if (command.get('node')!=name or command.get('engine')!=engine
                    or command.get('output_block')!=(block if conv else None)):
                raise ValueError('Wrong hardware command order/block')
            counters[engine]+=1; exact_int(command,'completed',counters[engine])
            exact_int(command,'written_words',node['shape'][1]*node['shape'][2] if conv else words)
            cycles=command.get('cycles')
            if type(cycles) is not int or not 0 < cycles < 0xffffffff:
                raise ValueError('Invalid or saturated hardware cycle counter')
            if not positive_time(command.get('driver_call_ms')): raise ValueError('Invalid command duration')
            # Busy engine cycles include DDR/cache/backpressure stalls, not just MACs.
            if cycles/100000 > command['driver_call_ms']+1.0:
                raise ValueError('Hardware duration exceeds enclosing driver call')
            if conv:
                for key in ['cache_hits','cache_misses']:
                    if type(command.get(key)) is not int or command[key]<0: raise ValueError('Invalid cache counter')
                cin,h,w=shape; kernel=node['layer']['kernel']; padding=node['layer']['padding']
                rows=sum(max(0,min(h,y-padding+kernel)-max(0,y-padding)) for y in range(h))
                cols=sum(max(0,min(w,x-padding+kernel)-max(0,x-padding)) for x in range(w))
                if command['cache_hits']+command['cache_misses']!=rows*cols*((cin+7)//8):
                    raise ValueError('Convolution feature request count differs')
            else:
                # Frozen network's branches are C8-aligned. Concat cache reuses
                # one raw word across its lanes; pool reads four words per output.
                if shape[0]%8 or node['shape'][0]%8: raise ValueError('Expected frozen C8-aligned tensor graph')
                exact_int(command,'read_words',words*(4 if node['kind']=='pool' else 1))
            node_cycles+=cycles; calls+=command['driver_call_ms']
            engines[engine]['commands']+=1; engines[engine]['cycles']+=cycles
            engines[engine]['driver_call_ms']+=command['driver_call_ms']
        if calls+timing['observer_ms']>timing['host_node_ms']+1.0:
            raise ValueError('Subtimers exceed enclosing node')
        pl_cycles+=node_cycles; shapes[name]=node['shape']
        layer_times.append(dict(node=name,hardware_active_ms=node_cycles/100000,
                                driver_call_ms=calls,host_node_ms=timing['host_node_ms'],observer_ms=timing['observer_ms']))
    exact_int(report,'pl_command_cycles',pl_cycles)
    total=report.get('host_total_ms')
    if not positive_time(total) or sum(n['host_node_ms'] for n in nodes)>total+1.0:
        raise ValueError('Invalid total timing')
    hardware_ms=pl_cycles/100000
    call_ms=sum(e['driver_call_ms'] for e in engines.values())
    observer_ms=sum(n['observer_ms'] for n in nodes)
    return dict(command_count=51,node_count=21,all_reported_node_checks_match=True,
                host_total_ms=total,hardware_active_ms=hardware_ms,
                convolution_active_ms=engines['conv']['cycles']/100000,
                tensor_active_ms=engines['tensor']['cycles']/100000,
                observer_ms=observer_ms,driver_call_ms=call_ms,
                driver_time_outside_engine_ms=call_ms-hardware_ms,
                other_host_unattributed_ms=total-call_ms-observer_ms,
                host_time_outside_engines_ms=total-hardware_ms,
                hardware_active_fraction=hardware_ms/total,
                layers=layer_times,
                timing_warning='Busy-cycle times include DDR/cache/backpressure stalls. Remaining host time is not separately instrumented into weights/config/allocation/guards; do not call it all weight loading. One diagnostic run, not steady-state FPS.')


def validate_head(path,report,model,reference):
    path=Path(path)
    if file_sha256(path)!=report['head_codes_sha256']: raise ValueError('Returned hardware head hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        if archive.files!=['head_codes']: raise ValueError('Unexpected hardware output arrays')
        actual=archive['head_codes']
    shape=model['nodes'][-1]['shape']
    if actual.dtype!=np.uint8 or list(actual.shape)!=shape: raise ValueError('Invalid hardware output dtype/shape')
    expected=unpack_words(reference,shape)
    if not np.array_equal(actual,expected): raise ValueError('Returned actual head differs from frozen reference')
    return dict(actual_head_independently_verified=True,different_codes=0,shape=shape,
                output_codes=int(actual.size),head_codes_sha256=file_sha256(path))


def audit(args):
    root=Path(args.root).resolve(); folder=root/'lung_continuous'
    output=Path(args.out).resolve() if args.out else root/'board_report_verification.json'
    if output.exists(): raise FileExistsError('Preserve previous verification; choose a fresh --out')
    manifest,model,_=load_package(folder)
    proof_path=root/'verification.json'; proof=json.loads(proof_path.read_text(encoding='utf-8'))
    if proof['status']!='prepared_audited_continuous_network_not_board_tested': raise ValueError('Wrong host preparation proof')
    check_hashes(proof['source_sha256']); check_hashes(proof['audit_sources_sha256'])
    report_path=root/'lung_continuous_result.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    stats=validate_report(report,manifest,model,file_sha256(folder/'continuous_manifest.json'))
    head_path=Path(args.head).resolve() if args.head else root/'lung_continuous_result_head.npz'
    head=dict(actual_head_independently_verified=False,reason='Actual head NPZ has not been found; do not render reference as FPGA output')
    if head_path.exists():
        with np.load(folder/'reference.npz',allow_pickle=False) as reference:
            head=validate_head(head_path,report,model,reference['head'])
    result=dict(status=('verified_returned_real_board_continuous_output' if head['actual_head_independently_verified']
                        else 'verified_returned_continuous_report_actual_head_pending'),
                report_sha256=file_sha256(report_path),package_sha256=file_sha256(folder/'continuous_manifest.json'),
                preparation_proof_sha256=file_sha256(proof_path),**head,timing=stats,
                source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)},
                provenance='User-transferred report checked against immutable local package; not a live remote board attestation',
                medical_accuracy_evaluated=False,clinical_use=False)
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','timing']},indent=2),flush=True)
    print('Timing ms:',json.dumps({k:v for k,v in stats.items() if k.endswith('_ms')},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_continuous_20261015')
    parser.add_argument('--head',help='Actual user-copied head NPZ, if stored elsewhere')
    parser.add_argument('--out',help='New audit output; never overwrite previous evidence')
    audit(parser.parse_args())
