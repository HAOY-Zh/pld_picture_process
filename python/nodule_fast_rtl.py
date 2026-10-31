"""Real XSim: two different tiles, layer-major, weights loaded once per block.

Actual shared-DDR outputs feed all subsequent nodes; intermediate references
are checked only as write oracles, never injected as next-layer input.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from nodule_rtl_sim import TOP,wrapper,mem
from lung_axil_smoke import feature_words
from lung_fast import prepare_block


def run(out,bin):
    out=Path(out).resolve();out.mkdir(exist_ok=True);sim=out/'rtl_sim';sim.mkdir();vectors=sim/'vectors';vectors.mkdir()
    package,manifest,plan,params=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
    with np.load(package/'nodule_reference.npz') as z:
        traces=[{n:z[f'case{i}_'+n].copy() for n in ['input']+[node['name'] for node in plan['nodes']]} for i in [0,2]]
    addresses={};offset=512;gold=np.full(200000,0xdeadbeefcafef00d,np.uint64);initial=gold.copy()
    written=np.zeros(200000,int);expected=written.copy()
    for slot,trace in enumerate(traces):
        for name in ['input']+[node['name'] for node in plan['nodes']]:
            zero=0 if name=='input' else next(n['zero'] for n in plan['nodes'] if n['name']==name)
            words=feature_words(trace[name],zero);addresses[(slot,name)]=(offset,len(words))
            gold[offset:offset+len(words)]=words;expected[offset:offset+len(words)]=1
            if name=='input':initial[offset:offset+len(words)]=words;written[offset:offset+len(words)]=1
            offset+=len(words)+16
    commands=[];weights=[];bias=[];mult=[];shift=[]
    for node in plan['nodes']:
        a=node['sources'][0];ca,h,w=traces[0][a].shape
        if node['kind']=='conv':
            row=node['layer'];arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']};cout=node['shape'][0]
            for block in range((cout+7)//8):
                p=prepare_block(arrays,block,row);wb=len(weights);pb=len(bias)
                weights.extend(p['packets']);bias.extend(p['bias']);mult.extend(p['mult']);shift.extend(p['shift'])
                for slot in range(2):
                    src,awords=addresses[(slot,a)];dst,dwords=addresses[(slot,node['name'])]
                    commands.extend([0,src,dst,0,ca,h,w,cout,block,row['kernel'],row['padding'],row['input_zero'],row['output_zero'],int(row['relu']),
                        wb,len(p['packets']) if slot==0 else 0,pb,0,0,0,1,0,1,0,dwords,awords,0,p['keep']])
        else:
            q=node['quant'];b=node['sources'][1] if node['kind']=='fused_concat' else None
            for slot in range(2):
                src,awords=addresses[(slot,a)];dst,dwords=addresses[(slot,node['name'])]
                baddr,bwords=addresses[(slot,b)] if b else (0,0);cb=traces[slot][b].shape[0] if b else 0
                commands.extend([1,src,dst,baddr,ca,h,w,cb,3 if b else 0,0,0,0,0,0,0,0,0,
                    q['za'],q.get('zb',0),q['zo'],q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0),dwords,awords,bwords,0])
    assert offset<200000 and len(commands)==30*28
    mem(vectors/'header.mem',[30,offset,len(weights),len(bias),2],32)
    for name,a,bits in [('commands',commands,32),('initial',initial[:offset],64),('gold',gold[:offset],64),
        ('initial_written',written[:offset],1),('expected_written',expected[:offset],1),('weights',weights,512),
        ('bias',bias,32),('mult',mult,32),('shift',shift,32)]:mem(vectors/(name+'.mem'),a,bits)
    top=TOP.replace('[0:99999]','[0:199999]').replace('header[0]!=15||header[1]>100000','header[0]!=30||header[1]>200000')
    start=top.index('                conv.put(32,commands[t+27]');end=top.index('                conv.put(4,1);',start)
    top=top[:start]+'                if(commands[t+15]>0)begin\n'+top[start:end]+'                end\n'+top[end:]
    old='$display("Nodule continuous network RTL: PASS (15 commands, 11 nodes, seven convolutions, shared DDR producer-consumer, five signed head channels, AXI stalls and final-B checks)");'
    new='conv.host.read_reg(10\'h038,0,value);if(value!==header[2])$fatal(1,"Weights reloaded instead of reused");\n        $display("Nodule batch reuse RTL: PASS (30 commands, two different tiles, all nodes shared-DDR, resident weights reused, exact heads and canaries)");'
    if old not in top:raise ValueError('Unexpected original bench')
    top=top.replace(old,new)
    (sim/'nodule_sim_engine.sv').write_text(wrapper(),encoding='utf-8');(sim/'tb_nodule_network.sv').write_text(top,encoding='utf-8')
    (sim/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    sources=list((package/'rtl').glob('*.v'))+[ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv',sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv']
    bin=Path(bin)
    calls=[([str(bin/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(bin/'xelab.bat'),'tb_nodule_network','-s','nodule_batch_snapshot','--debug','typical','--mt','2'],'elaborate'),
        ([str(bin/'xsim.bat'),'nodule_batch_snapshot','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]
    for cmd,name in calls:
        print('Batch actual XSim:',name,flush=True)
        with (sim/(name+'.log')).open('x',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=sim,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('XSim failed: '+name)
    log=(sim/'rtl_run.log').read_text(encoding='utf-8')
    if 'Nodule batch reuse RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise RuntimeError('Batch RTL mismatch')
    rows=re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)
    if len(rows)!=30:raise ValueError('Incomplete batch simulation')
    proof=dict(status='actual_xsim_two_tile_resident_weight_reuse_pass',commands=30,tiles=2,weight_loads=11,
        transferred_weight_packets=len(weights),without_reuse_packets=len(weights)*2,board_tested=False,
        source_sha256={str(p):sha(p) for p in sources},log_sha256=sha(sim/'rtl_run.log'),
        vectors_sha256={p.name:sha(p) for p in vectors.iterdir()},
        parent_manifest_sha256=sha(package/'nodule_manifest.json'))
    (sim/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print('Batch full-graph RTL PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();run(a.out,a.vivado_bin)
