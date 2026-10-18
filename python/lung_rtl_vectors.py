"""Generate compact readmemh banks from frozen integer contract, no CT/labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_fixed_network import checked_contract
from lung_fpga_contract import conv_accumulator,multiplier,requantize
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256


def hex_bank(path,values,bits):
    mask=(1<<bits)-1;width=(bits+3)//4
    path.write_text(''.join(f'{int(v)&mask:0{width}x}\n' for v in values),encoding='ascii')


def quant_word(acc,m,shift,z,relu,code,error=0):
    return ((int(acc)&0xffffffff)<<56)|((int(m)&0xffffffff)<<24)|(int(shift)<<18)|(int(z)<<10)|(int(relu)<<9)|(int(code)<<1)|int(error)


def generate(args):
    source,contract,tracked=checked_contract(args.contract)
    output=fresh_output(args.outdir);rng=np.random.default_rng(731)
    banks={key:[] for key in ['inputs','weights','bias','multipliers','shifts','codes','accumulators','descriptors','requant']}
    cases=[]
    def append(name,q,w,b,m,s,iz,oz,relu,pad):
        acc=conv_accumulator(q,w,b,iz,pad);expected=requantize(acc,m,s,oz,relu)
        offsets={key:len(banks[key]) for key in ['inputs','weights','bias','codes']}
        desc=[w.shape[1],w.shape[0],q.shape[1],q.shape[2],w.shape[2],pad,iz,oz,int(relu),
              offsets['inputs'],offsets['weights'],offsets['bias'],offsets['codes'],0,0,0]
        banks['descriptors'].extend(desc)
        for key,a in [('inputs',q),('weights',w),('bias',b),('multipliers',m),('shifts',s),('codes',expected),('accumulators',acc)]:banks[key].extend(a.ravel().tolist())
        for c in range(len(b)):
            for a,code in zip(acc[c].ravel(),expected[c].ravel()):banks['requant'].append(quant_word(a,m[c],s[c],oz,relu,code))
        cases.append(dict(name=name,outputs=int(acc.size),beats_per_output=int(np.prod(w.shape[1:])),offsets=offsets))
    with np.load(source.parent/'parameters.npz') as p,np.load(source.parent/'operator_vectors.npz') as v:
        for row in contract['layers']:
            prefix=row['name'].replace('.','_')
            for tag in ['real_zero','ramp127','ramp255']:
                key=prefix+'_'+tag
                append(key,v[key+'_input_uint8'],p[prefix+'_weight_int8'],p[prefix+'_bias_int32'],
                    p[prefix+'_multiplier_int32'],p[prefix+'_right_shift_int32'],row['input_zero_point'],row['output_zero_point'],row['fused_relu'],row['padding'][0])
                if not np.array_equal(v[key+'_output_uint8'],np.asarray(banks['codes'][-v[key+'_output_uint8'].size:],np.uint8).reshape(v[key+'_output_uint8'].shape)):
                    raise ValueError('Generated golden output differs from frozen contract')
    for relu in [False,True]:
        q=rng.integers(0,256,(3,4,5),dtype=np.uint8);w=rng.integers(-128,128,(4,3,3,3),dtype=np.int8)
        b=np.array([-7,3,-20,0],np.int32);m,s=multiplier([.01,.02,.04,.005])
        append('synthetic_nonzero_zp_relu_'+str(relu),q,w,b,m,s,128,128,relu,1)
    m,s=multiplier([.5,.5]);append('synthetic_signed_ties',np.array([[[0,1,3,4,5,7,9,11]]],np.uint8),
        np.array([[[[1]]],[[[-1]]]],np.int8),np.array([0,0],np.int32),m,s,4,128,False,0)
    # Signed extremes, shift boundaries, negative ties, random valid/invalid metadata.
    for shift in range(63):
        for a in [-(2**31),-7,-3,-1,0,1,3,7,2**31-1]:
            m=1073741824;code=int(requantize(np.array([[[a]]],np.int32),[m],[shift],128)[0,0,0])
            banks['requant'].append(quant_word(a,m,shift,128,False,code))
    for _ in range(1000):
        a=int(rng.integers(-(2**31),2**31));m=int(rng.integers(1,2**31));s=int(rng.integers(0,63));z=int(rng.integers(0,256));relu=bool(rng.integers(0,2))
        code=int(requantize(np.array([[[a]]],np.int32),[m],[s],z,relu)[0,0,0]);banks['requant'].append(quant_word(a,m,s,z,relu,code))
    for a,m,s in [(1,0,0),(3,-1,31),(9,1073741824,63)]:banks['requant'].append(quant_word(a,m,s,0,False,0,1))
    widths=dict(inputs=8,weights=8,bias=32,multipliers=32,shifts=6,codes=8,accumulators=32,descriptors=32,requant=88)
    header=[len(cases),len(banks['inputs']),len(banks['weights']),len(banks['bias']),len(banks['codes']),len(banks['requant'])]
    hex_bank(output/'header.mem',header,32)
    for key,values in banks.items():hex_bank(output/(key+'.mem'),values,widths[key])
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Frozen input changed: '+p)
    report=dict(status='complete_rtl_vector_banks',cases=cases,header=header,
        dot_outputs=header[4],dot_input_beats=sum(c['outputs']*c['beats_per_output'] for c in cases),
        requant_outputs=header[5],source_sha256=tracked,
        outputs={p.name:file_sha256(p) for p in output.glob('*.mem')},
        metadata='CHW/OIHW, first/last per output dot; descriptors 16 uint32 words/case',
        generated_by_sha256=file_sha256(__file__))
    write_json(output/'vectors.json',report)
    print('RTL vectors:',len(cases),'cases,',header[4],'dot outputs,',report['dot_input_beats'],'MAC beats',flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':generate(parser().parse_args())
