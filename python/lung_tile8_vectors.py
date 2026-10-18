"""Frozen scalar vectors regrouped into 8x8 MAC packets; no medical labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_fpga_contract import conv_accumulator,requantize,multiplier
from lung_fixed_network import checked_contract
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from lung_rtl_vectors import hex_bank


def packed_bytes(values):
    return sum((int(v)&255)<<(8*i) for i,v in enumerate(values))


def pack_frame(q,w,output_start,y,x,padding,input_zero):
    """Kernel order KY/KX, then blocks of 8 input channels; exact tile sum."""
    cin=w.shape[1];kernel=w.shape[-1];cout=w.shape[0]
    output_mask=(1<<min(8,cout-output_start))-1
    packets=[]
    for ky in range(kernel):
        for kx in range(kernel):
            for start in range(0,cin,8):
                sy=y+ky-padding;sx=x+kx-padding
                keep=(1<<min(8,cin-start))-1
                # Inactive slots intentionally contain garbage: mask must work.
                acts=[(91+i*17)&255 for i in range(8)]
                weights=[((i*13+71)%256)-128 for i in range(64)]
                for i in range(min(8,cin-start)):
                    acts[i]=int(q[start+i,sy,sx]) if 0<=sy<q.shape[1] and 0<=sx<q.shape[2] else input_zero
                    for o in range(min(8,cout-output_start)):
                        weights[o*8+i]=int(w[output_start+o,start+i,ky,kx])
                packets.append((packed_bytes(acts),packed_bytes(weights),keep))
    return packets,output_mask


def emit_frame_banks(cases,output):
    packets=[];descriptors=[];biases=[];multipliers=[];shifts=[];accs=[];codes=[];lanes=[];lasts=[];frames=[];active_macs=0
    for case_index,case in enumerate(cases):
        q,w,b,m,s,iz,oz,relu,pad=[case[k] for k in ['q','w','bias','m','shift','iz','oz','relu','pad']]
        acc=conv_accumulator(q,w,b,iz,pad);code=requantize(acc,m,s,oz,relu)
        active_macs+=int(np.prod(w.shape))*acc.shape[1]*acc.shape[2]
        for y in range(acc.shape[1]):
            for x in range(acc.shape[2]):
                for o in range(0,w.shape[0],8):
                    words,mask=pack_frame(q,w,o,y,x,pad,iz)
                    start=len(packets);param_start=len(biases);result_start=len(accs)
                    descriptors.extend([start,len(words),mask,iz,oz,int(relu),param_start,result_start,case_index,y,x,o])
                    for lane in range(8):
                        active=o+lane<len(b)
                        biases.append(int(b[o+lane]) if active else 0)
                        multipliers.append(int(m[o+lane]) if active else 1)
                        shifts.append(int(s[o+lane]) if active else 0)
                        if active:
                            accs.append(int(acc[o+lane,y,x]));codes.append(int(code[o+lane,y,x]));lanes.append(lane)
                            lasts.append(lane==min(8,len(b)-o)-1)
                    for index,(a,wt,keep) in enumerate(words):
                        # [585:last][584:first][583:keep8][575:weights512][63:acts64]
                        packets.append((int(index==len(words)-1)<<585)|(int(index==0)<<584)|(keep<<576)|(wt<<64)|a)
                    frames.append(dict(case=case['name'],packets=len(words),output_mask=mask))
    header=[len(frames),len(packets),len(biases),len(accs),len(cases)]
    banks=dict(header=(header,32),packets=(packets,586),descriptors=(descriptors,32),bias=(biases,32),
        multipliers=(multipliers,32),shifts=(shifts,6),accumulators=(accs,32),codes=(codes,8),lanes=(lanes,3),lasts=(lasts,1))
    for name,(values,bits) in banks.items():hex_bank(output/(name+'.mem'),values,bits)
    return dict(header=header,frames=len(frames),packet_count=len(packets),outputs=len(accs),
        max_frame_packets=max(f['packets'] for f in frames),scalar_active_macs=active_macs,
        cases=[dict(name=c['name'],input_channels=c['w'].shape[1],output_channels=c['w'].shape[0]) for c in cases])


def generate(args):
    path,contract,tracked=checked_contract(args.contract)
    output=fresh_output(args.outdir);cases=[]
    with np.load(path.parent/'parameters.npz') as p,np.load(path.parent/'operator_vectors.npz') as v:
        for row in contract['layers']:
            prefix=row['name'].replace('.','_')
            for tag in ['real_zero','ramp127','ramp255']:
                cases.append(dict(name=prefix+'_'+tag,q=v[prefix+'_'+tag+'_input_uint8'],w=p[prefix+'_weight_int8'],
                    bias=p[prefix+'_bias_int32'],m=p[prefix+'_multiplier_int32'],shift=p[prefix+'_right_shift_int32'],
                    iz=row['input_zero_point'],oz=row['output_zero_point'],relu=row['fused_relu'],pad=row['padding'][0]))
    rng=np.random.default_rng(881)
    for cin,cout,relu,iz in [(3,4,False,128),(11,13,True,255),(1,1,False,4)]:
        q=rng.integers(0,256,(cin,3,5),dtype=np.uint8);w=rng.integers(-128,128,(cout,cin,3,3),dtype=np.int8)
        m,s=multiplier(np.linspace(.002,.03,cout))
        cases.append(dict(name=f'tail_{cin}_{cout}',q=q,w=w,bias=rng.integers(-20,21,cout,dtype=np.int32),m=m,shift=s,iz=iz,oz=128,relu=relu,pad=1))
    report=emit_frame_banks(cases,output)
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Frozen source changed: '+p)
    report.update(status='complete_tile8_vector_banks',source_sha256=tracked,generated_by_sha256=file_sha256(__file__),
        outputs_sha256={p.name:file_sha256(p) for p in output.glob('*.mem')},
        layout='12-word frame descriptors; 586-bit packets: last/first/input_keep/weights(OxI)/activations',
        warning='Full-range and masked-tail golden vectors; no CT/GT or precision retuning')
    write_json(output/'vectors.json',report)
    print('Tile8 vectors:',report['frames'],'groups,',report['packet_count'],'packets,',report['outputs'],'outputs',flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':generate(parser().parse_args())
