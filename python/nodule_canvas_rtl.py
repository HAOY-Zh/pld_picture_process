"""Actual full 448x448 real-plane graph through unchanged RTL/shared DDR.

Intermediate RTL outputs feed later operators. Integer references are write
oracles only. No tile or head output is injected as an intermediate input.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
from nodule_plan import load_package,sha
from nodule_canvas import expanded_plan,lower_canvas
from nodule_fast_input import SharedPlaneSampler
from nodule_int8_cuda_reference import IntegerNodule
from lung_nodule_cnn import config_from_dict
from nodule_rtl_sim import TOP,wrapper,mem
from lung_axil_smoke import feature_words
from lung_fast import prepare_block


def run(out,bin):
    out=Path(out).resolve();out.mkdir(exist_ok=True);sim=out/'rtl_sim';sim.mkdir();vectors=sim/'vectors';vectors.mkdir()
    parent,_,model,params=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
    src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
    meta=json.loads((src/'slice_case.json').read_text());cfg=config_from_dict(model['detector_config'])
    with np.load(src/'slice_case.npz') as z:hu=z['hu'];jobs=[tuple(v) for v in z['jobs']]
    sample=SharedPlaneSampler(hu,meta['local_geometry']['spacing_zyx_mm'],jobs,cfg)
    ref=IntegerNodule(model,params);q=ref.lut[sample.context.astype(np.int32)+32768]
    _,trace=ref.forward_codes(torch.from_numpy(q[None]),True)
    trace={k:v[0] for k,v in trace.items()};trace['input']=lower_canvas(q)
    plan,shapes=expanded_plan(model,params,*map(int,sample.context.shape[1:]))
    addresses={};offset=512;capacity=1000000
    gold=np.full(capacity,0xdeadbeefcafef00d,np.uint64);initial=gold.copy()
    written=np.zeros(capacity,int);expected=written.copy()
    for name,(shape,zero) in shapes.items():
        words=feature_words(trace[name],zero);addresses[name]=(offset,len(words))
        if offset+len(words)+16>capacity:raise ValueError('Simulation capacity insufficient')
        gold[offset:offset+len(words)]=words;expected[offset:offset+len(words)]=1
        if name=='input':initial[offset:offset+len(words)]=words;written[offset:offset+len(words)]=1
        offset+=len(words)+16
    commands=[];weights=[];bias=[];mult=[];shift=[]
    for node in plan['nodes']:
        a=node['sources'][0];ca,h,w=trace[a].shape
        dst,dwords=addresses[node['name']];src,awords=addresses[a]
        if node['kind']=='conv':
            row=node['layer'];arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']};cout=node['shape'][0]
            for block in range((cout+7)//8):
                p=prepare_block(arrays,block,row);wb=len(weights);pb=len(bias)
                weights.extend(p['packets']);bias.extend(p['bias']);mult.extend(p['mult']);shift.extend(p['shift'])
                commands.extend([0,src,dst,0,ca,h,w,cout,block,row['kernel'],row['padding'],row['input_zero'],row['output_zero'],int(row['relu']),
                    wb,len(p['packets']),pb,0,0,0,1,0,1,0,dwords,awords,0,p['keep']])
        else:
            qn=node['quant'];b=node['sources'][1] if node['kind']=='fused_concat' else None
            baddr,bwords=addresses[b] if b else (0,0);cb=trace[b].shape[0] if b else 0
            commands.extend([1,src,dst,baddr,ca,h,w,cb,3 if b else 0,0,0,0,0,0,0,0,0,
                qn['za'],qn.get('zb',0),qn['zo'],qn.get('ma',1),qn.get('sa',0),qn.get('mb',1),qn.get('sb',0),dwords,awords,bwords,0])
    mem(vectors/'header.mem',[15,offset,len(weights),len(bias),1],32)
    for name,a,bits in [('commands',commands,32),('initial',initial[:offset],64),('gold',gold[:offset],64),
        ('initial_written',written[:offset],1),('expected_written',expected[:offset],1),('weights',weights,512),
        ('bias',bias,32),('mult',mult,32),('shift',shift,32)]:mem(vectors/(name+'.mem'),a,bits)
    top=TOP.replace('[0:99999]','[0:999999]').replace('header[1]>100000','header[1]>1000000')
    top=top.replace('Nodule continuous network RTL: PASS','Nodule full canvas RTL: PASS')
    top=top.replace('#500000000;','#2000000000;')
    (sim/'nodule_sim_engine.sv').write_text(wrapper(),encoding='utf-8')
    (sim/'tb_nodule_network.sv').write_text(top,encoding='utf-8');(sim/'run.tcl').write_text('run all\nexit\n')
    sources=list((parent/'rtl').glob('*.v'))+[ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv',sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv']
    bin=Path(bin);tick=time.perf_counter()
    for cmd,name in [([str(bin/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(bin/'xelab.bat'),'tb_nodule_network','-s','nodule_canvas_snapshot','--debug','typical','--mt','2'],'elaborate'),
        ([str(bin/'xsim.bat'),'nodule_canvas_snapshot','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        print('Canvas actual XSim:',name,flush=True)
        with (sim/(name+'.log')).open('x',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=sim,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Canvas XSim failed: '+name)
    verify_existing(out,time.perf_counter()-tick)


def verify_existing(out,elapsed=None):
    out=Path(out).resolve();sim=out/'rtl_sim';vectors=sim/'vectors'
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board'
    sources=list((parent/'rtl').glob('*.v'))+[ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv',sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv']
    log=(sim/'rtl_run.log').read_text(encoding='utf-8')
    if 'Nodule full canvas RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise RuntimeError('Canvas RTL mismatch')
    rows=re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)
    if len(rows)!=15 or [int(r[0]) for r in rows]!=list(range(15)):raise ValueError('Incomplete graph simulation')
    proof=dict(status='actual_xsim_full_448_canvas_pass',canvas_hw=[448,448],
        commands=rows,elapsed_sec=elapsed,source_sha256={str(p):sha(p) for p in sources},
        vectors_sha256={p.name:sha(p) for p in vectors.iterdir()},log_sha256=sha(sim/'rtl_run.log'),
        parent_sha256=sha(parent/'nodule_manifest.json'),board_tested=False)
    (sim/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('Full 448 canvas actual RTL PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify_existing(a.out)
    else:run(a.out,a.vivado_bin)
