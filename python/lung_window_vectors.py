"""CHW frozen inputs -> external HWC8 words and independent window/golden banks."""
import argparse
from pathlib import Path

import numpy as np

from lung_fixed_network import checked_contract
from lung_fpga_contract import conv_accumulator,requantize,multiplier
from lung_rtl_vectors import hex_bank
from lung_segmentation import fresh_output,write_json
from lung_tile8_vectors import packed_bytes,pack_frame
from lung_unet import file_sha256


def feature_words(q):
    q=np.asarray(q)
    if q.dtype!=np.uint8 or q.ndim!=3 or not 1<=q.shape[0]<=96 or not all(1<=v<=512 for v in q.shape[1:]):
        raise ValueError('Need CHW UINT8, C=1..96, H/W=1..512')
    words=[];cin,h,w=q.shape
    for y in range(h):
        for x in range(w):
            for start in range(0,cin,8):
                slots=[(91+i*17)&255 for i in range(8)]
                slots[:min(8,cin-start)]=q[start:start+8,y,x].tolist()
                words.append(packed_bytes(slots))
    return words


def window_plan(q,base,kernel,pad,zero):
    """Independent coordinate plan. Addresses are 64-bit WORD indices, not bytes."""
    q=np.asarray(q);cin,h,w=q.shape;blocks=(cin+7)//8
    if kernel not in (1,3) or pad not in (0,1) or kernel==1 and pad or not 0<=zero<=255:
        raise ValueError('Unsupported kernel/padding/zero')
    oh=h+2*pad-kernel+1;ow=w+2*pad-kernel+1
    if min(oh,ow)<1 or base<0 or base+h*w*blocks>2**32:raise ValueError('Bad shape/address range')
    words=feature_words(q)
    for y in range(oh):
        for x in range(ow):
            for ky in range(kernel):
                for kx in range(kernel):
                    for cb in range(blocks):
                        sy=y+ky-pad;sx=x+kx-pad;keep=(1<<min(8,cin-cb*8))-1
                        inside=0<=sy<h and 0<=sx<w
                        offset=(sy*w+sx)*blocks+cb
                        data=words[offset] if inside else packed_bytes([zero]*8)
                        first=ky==0 and kx==0 and cb==0
                        last=ky==kernel-1 and kx==kernel-1 and cb==blocks-1
                        yield (base+offset if inside else None,data,keep,first,last,x,y,last and y==oh-1 and x==ow-1)


def emit(cases,out):
    banks={n:[] for n in ['features','weights','frames','bias','multipliers','shifts','windows',
        'window_coords','requests','accumulators','codes','lanes','lasts','coords']}
    frame_count=0;active_macs=0
    for case in cases:
        q,w,b,m,s,iz,oz,relu,pad=[case[k] for k in ['q','w','bias','m','shift','iz','oz','relu','pad']]
        cin,h,width=q.shape;kernel=w.shape[-1];base=len(banks['features']);banks['features'].extend(feature_words(q))
        expected_acc=conv_accumulator(q,w,b,iz,pad);expected_code=requantize(expected_acc,m,s,oz,relu)
        oh,ow=expected_acc.shape[1:];active_macs+=int(np.prod(w.shape))*oh*ow
        plan=list(window_plan(q,base,kernel,pad,iz))
        for oc in range(0,w.shape[0],8):
            packets,mask=pack_frame(q,w,oc,0,0,pad,iz)
            pbase=len(banks['bias']);wbase=len(banks['weights']);rbase=len(banks['codes']);vbase=len(banks['windows'])
            banks['weights'].extend(p[1] for p in packets)
            banks['frames'].extend([base,h,width,cin,kernel,pad,iz,oz,mask,int(relu),wbase,len(packets),pbase,rbase,vbase,oh*ow])
            for name,values in [('bias',b),('multipliers',m),('shifts',s)]:
                banks[name].extend(int(values[oc+i]) if oc+i<len(b) else (1 if name=='multipliers' else 0) for i in range(8))
            for address,data,keep,first,last,x,y,final in plan:
                banks['windows'].append((int(final)<<74)|(int(last)<<73)|(int(first)<<72)|(keep<<64)|data)
                banks['window_coords'].append((y<<10)|x)
                if address is not None:banks['requests'].append(address)
            for y in range(oh):
                for x in range(ow):
                    for lane in range(min(8,w.shape[0]-oc)):
                        last=lane==min(8,w.shape[0]-oc)-1
                        banks['accumulators'].append(int(expected_acc[oc+lane,y,x]))
                        banks['codes'].append(int(expected_code[oc+lane,y,x]));banks['lanes'].append(lane);banks['lasts'].append(last)
                        banks['coords'].append((int(last and y==oh-1 and x==ow-1)<<20)|(y<<10)|x)
            frame_count+=1
    header=[frame_count,len(banks['features']),len(banks['weights']),len(banks['bias']),len(banks['codes']),len(banks['windows']),len(banks['requests'])]
    widths=dict(features=64,weights=512,frames=32,bias=32,multipliers=32,shifts=6,windows=75,
        window_coords=20,requests=32,accumulators=32,codes=8,lanes=3,lasts=1,coords=21)
    hex_bank(out/'header.mem',header,32)
    for name,values in banks.items():hex_bank(out/(name+'.mem'),values,widths[name])
    return dict(header=header,frames=frame_count,features=header[1],weight_beats=header[2],outputs=header[4],
        window_beats=header[5],memory_requests=header[6],padding_beats=header[5]-header[6],active_scalar_macs=active_macs,
        cases=[dict(name=c['name'],shape=list(c['q'].shape),outputs=c['w'].shape[0],kernel=c['w'].shape[-1],padding=c['pad']) for c in cases])


def generate(args):
    path,contract,tracked=checked_contract(args.contract);out=fresh_output(args.outdir);cases=[]
    with np.load(path.parent/'parameters.npz') as p,np.load(path.parent/'operator_vectors.npz') as v:
        for row in contract['layers']:
            prefix=row['name'].replace('.','_')
            for tag in ['real_zero','ramp127','ramp255']:
                cases.append(dict(name=prefix+'_'+tag,q=v[prefix+'_'+tag+'_input_uint8'],w=p[prefix+'_weight_int8'],
                    bias=p[prefix+'_bias_int32'],m=p[prefix+'_multiplier_int32'],shift=p[prefix+'_right_shift_int32'],
                    iz=row['input_zero_point'],oz=row['output_zero_point'],relu=row['fused_relu'],pad=row['padding'][0]))
    rng=np.random.default_rng(319)
    for cin,cout,h,w,k,pad,zero in [(3,4,3,5,3,1,128),(11,13,3,5,3,1,255),
        (1,1,1,1,3,1,4),(96,9,3,5,3,1,128),(11,13,5,7,3,0,128),
        (9,9,1,7,1,0,255),(3,4,7,1,3,1,128),(3,4,9,11,3,1,0)]:
        m,s=multiplier(np.linspace(.002,.03,cout))
        cases.append(dict(name=f'synthetic_{cin}_{cout}_{h}_{w}_k{k}_p{pad}',
            q=rng.integers(0,256,(cin,h,w),dtype=np.uint8),w=rng.integers(-128,128,(cout,cin,k,k),dtype=np.int8),
            bias=rng.integers(-20,21,cout,dtype=np.int32),m=m,shift=s,iz=zero,oz=128,relu=True,pad=pad))
    report=emit(cases,out)
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Frozen source changed: '+p)
    report.update(status='complete_feature_window_vectors',source_sha256=tracked,generated_by_sha256=file_sha256(__file__),
        outputs_sha256={p.name:file_sha256(p) for p in out.glob('*.mem')},
        layout='HWC8 64-bit word addresses; one 16-word frame descriptor per output-channel tile',
        warning='External memory latency is simulated, not real DDR/AXI; no medical precision tuning')
    write_json(out/'vectors.json',report)
    print({k:report[k] for k in ['frames','features','weight_beats','outputs','window_beats','memory_requests','padding_beats']},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    parser.add_argument('--outdir',required=True)
    generate(parser.parse_args())
