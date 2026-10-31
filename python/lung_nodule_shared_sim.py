"""Real RTL mixed-model stream, no reset between graphs; small full U-Net topology.

U-Net geometries are 32x32 with actual frozen original/pruned weights. Nodule
uses an actual 128x128 patch lowered to 45x64x64. This is not full-256 U-Net RTL
coverage; the board acceptance checks the complete full-256 frozen references.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from lung_nodule_shared_prepare import ROOT,checked
from nodule_plan import sha
from nodule_rtl_sim import mem,wrapper,TOP
from lung_axil_smoke import feature_words
from lung_fast import prepare_block


def prepare(root):
    root,p=checked(root);oracle=json.loads((root/'oracle/verification.json').read_text())
    if oracle['status']!='frozen_full_models_and_small_dense_integer_oracles_pass':raise ValueError('Missing independent integer oracle')
    for f,h in oracle['outputs_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed gold '+f)
    out=root/'mixed_sim';out.mkdir();vectors=out/'vectors';vectors.mkdir()
    target=root/'lung_nodule_shared';sequence=['unet_original','nodule','unet_pruned','nodule','unet_original']
    gold=[];initial=[];written=[];expected=[];commands=[];weights=[];bias=[];mult=[];shift=[];stages=[]
    def reserve(values,is_input):
        start=len(gold);values=list(map(int,values));filler=0xdeadbeefcafef00d
        gold.extend(values+[filler]*16);initial.extend((values if is_input else [filler]*len(values))+[filler]*16)
        written.extend(([1]*len(values) if is_input else [0]*len(values))+[0]*16);expected.extend([1]*len(values)+[0]*16)
        return start,len(values)
    reserve([],True)
    for stage,label in enumerate(sequence):
        folder=target/'models'/label
        model=json.loads(((folder/'model.json') if label=='nodule' else root/'oracle'/(label+'_32_model.json')).read_text())
        params=dict(np.load(folder/'parameters.npz'))
        tracefile=root/'oracle'/('nodule_trace.npz' if label=='nodule' else label+'_32_trace.npz')
        with np.load(tracefile) as z:trace={n:z[n] for n in z.files}
        addresses={}
        for name in ['input']+[n['name'] for n in model['nodes']]:
            zero=model['input_zero'] if name=='input' else next(n['zero'] for n in model['nodes'] if n['name']==name)
            addresses[name]=reserve(feature_words(trace[name],zero),name=='input')
        first=len(commands)//28
        for node in model['nodes']:
            a=node['sources'][0];ca,h,w=trace[a].shape;dst,dwords=addresses[node['name']];src,awords=addresses[a]
            if node['kind']=='conv':
                row=node['layer'];arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
                cout=node['shape'][0]
                for block in range((cout+7)//8):
                    v=prepare_block(arrays,block,row);wb=len(weights);pb=len(bias)
                    weights.extend(v['packets']);bias.extend(v['bias']);mult.extend(v['mult']);shift.extend(v['shift'])
                    commands.extend([0,src,dst,0,ca,h,w,cout,block,row['kernel'],row['padding'],row['input_zero'],row['output_zero'],int(row['relu']),
                        wb,len(v['packets']),pb,0,0,0,1,0,1,0,dwords,awords,0,v['keep']])
            else:
                q=node['quant'];b=node['sources'][1] if node['kind']=='fused_concat' else None
                baddr,bwords=addresses[b] if b else (0,0);cb=trace[b].shape[0] if b else 0
                commands.extend([1,src,dst,baddr,ca,h,w,cb,3 if b else 0,0,0,0,0,0,0,0,0,
                    q['za'],q.get('zb',0),q['zo'],q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0),dwords,awords,bwords,0])
        stages.append(dict(stage=stage,label=label,input_shape=model['input_shape'],nodes=len(model['nodes']),
            first_command=first,after_command=len(commands)//28,head_zero=model['nodes'][-1]['zero']))
    count=len(commands)//28
    mem(vectors/'header.mem',[count,len(gold),len(weights),len(bias),1],32)
    for name,a,bits in [('commands',commands,32),('initial',initial,64),('gold',gold,64),('initial_written',written,1),
        ('expected_written',expected,1),('weights',weights,512),('bias',bias,32),('mult',mult,32),('shift',shift,32)]:mem(vectors/(name+'.mem'),a,bits)
    t=TOP.replace('99999','999999').replace('commands[0:999]','commands[0:9999]').replace('bias[0:999],mult[0:999],shift[0:999]',
        'bias[0:9999],mult[0:9999],shift[0:9999]').replace('weights[0:9999]','weights[0:49999]')
    t=t.replace('header[0]!=15||header[1]>100000||header[2]>10000||header[3]>1000',
        'header[0]!='+str(count)+'||header[1]>1000000||header[2]>50000||header[3]>10000')
    t=t.replace('#500000000;','#2000000000;')
    marker='            t=f*28;'
    t=t.replace(marker,marker+'''\n            if(conv.con.dut.engine.conv.conv.core.pending_groups!=0 ||
                conv.con.dut.engine.conv.conv.core.tile.inflight!=0 || conv.con.dut.engine.conv.conv.meta_count!=0)
                $fatal(1,"Cross-command model state not drained");''')
    for s in stages:
        marker='        for(f=0;f<header[0];f=f+1)begin'
    t=t.replace('$display("Nodule continuous network RTL: PASS (15 commands, 11 nodes, seven convolutions, shared DDR producer-consumer, five signed head channels, AXI stalls and final-B checks)");',
        '$display("Shared U-Net/nodule mixed RTL: PASS ('+str(count)+' commands, five graphs, original32/nodule128/pruned32/nodule128/original32, no inter-model reset, all node outputs, final B, DDR isolation)");')
    (out/'tb_nodule_network.sv').write_text(t);(out/'nodule_sim_engine.sv').write_text(wrapper());(out/'run.tcl').write_text('run all\nexit\n')
    for n in ['tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']:shutil.copy2(ROOT/'fpga'/n,out/n)
    sources=list((target/'rtl').glob('*.v'))+list(out.glob('*.sv'))
    identity=dict(stages=stages,commands=count,memory_words=len(gold),weight_beats=len(weights),parameters=len(bias),
        source_sha256={str(p):sha(p) for p in sources},vectors_sha256={p.name:sha(p) for p in vectors.iterdir()})
    (out/'inputs.json').write_text(json.dumps(identity,indent=2));return out,sources,identity


def run(root,binary):
    out,sources,identity=prepare(root);binary=Path(binary);tick=time.perf_counter()
    for c,n in [([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(binary/'xelab.bat'),'tb_nodule_network','-s','shared_mixed','--debug','typical','--mt','2'],'elaborate'),
        ([str(binary/'xsim.bat'),'shared_mixed','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        print('Shared actual XSim',n,flush=True)
        with (out/(n+'.log')).open('x') as s:r=subprocess.run(c,cwd=out,stdout=s,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Actual RTL tool failed: '+str(out/(n+'.log')))
    log=(out/'rtl_run.log').read_text()
    if 'Shared U-Net/nodule mixed RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Mixed RTL not exact: '+str(out/'rtl_run.log'))
    rows=[dict(command=int(a),kind=int(b),cycles=int(c)) for a,b,c in re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)]
    if len(rows)!=identity['commands'] or [r['command'] for r in rows]!=list(range(identity['commands'])):raise ValueError('Incomplete mixed graph')
    for f,h in identity['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed simulated source')
    proof=dict(status='actual_xsim_shared_models_no_reset_exact',board_tested=False,full256_unet_rtl=False,
        scope='U-Net real weights full topology at32x32; nodule real128 patch. Full256 references checked separately on CPU, must still validate full256 on physical shared bit.',
        elapsed_sec=time.perf_counter()-tick,command_results=rows,log_sha256=sha(out/'rtl_run.log'),**identity)
    (out/'verification.json').write_text(json.dumps(proof,indent=2));checked(root);print(proof['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();run(a.root,a.vivado_bin)
