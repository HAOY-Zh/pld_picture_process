"""Independent HWC8 tensor-operator fixtures, synthetic bounds and real activations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from lung_fixed_network import checked_contract
from lung_fpga_contract import maxpool2, nearest2, requantize
from lung_rtl_vectors import hex_bank
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_axil_smoke import feature_words


def reference(op,a,b,za,zb,zo,ma=1,sa=0,mb=1,sb=0):
    if op==0: return maxpool2(a)
    if op==1: return nearest2(a)
    if op not in (2,3) or b is None: raise ValueError('Unknown tensor operation')
    if op==3: a=nearest2(a)
    if a.shape[1:]!=b.shape[1:]: raise ValueError('Concat spatial mismatch')
    scaled=[]
    for values,zero,m,s in [(a,za,ma,sa),(b,zb,mb,sb)]:
        scaled.append(requantize(values.astype(np.int64)-zero,np.full(len(values),m,np.int64),
                                np.full(len(values),s,np.int64),zo))
    return np.concatenate(scaled,axis=0)


def source_words(a,zero):
    result=feature_words(a,zero)
    # Intentionally dirty inactive input lanes: output tails must use output ZP.
    channels,height,width=a.shape
    if channels%8:
        view=result.view(np.uint8).reshape(height,width,(channels+7)//8,8)
        view[:,:,-1,channels%8:]=np.arange(channels%8,8)*17+19
    return result


def generate(args):
    path,contract,tracked=checked_contract(args.contract)
    package=Path(args.layers).resolve(); manifest=json.loads(package.read_text(encoding='utf-8'))
    check_hashes(manifest['source_sha256'])
    check_hashes({str(package.parent/n):h for n,h in manifest['files_sha256'].items()})
    tracked.update({str(package):file_sha256(package),str(package.parent/'vectors.npz'):file_sha256(package.parent/'vectors.npz')})
    rng=np.random.default_rng(581);cases=[]
    def add(name,op,a,b=None,za=128,zb=64,zo=None,ma=1073741824,sa=31,mb=1610612736,sb=30):
        if zo is None: zo=za if op<2 else 99
        expected=reference(op,a,b,za,zb,zo,ma,sa,mb,sb)
        cases.append(dict(name=name,op=op,a=a,b=b,za=za,zb=zb,zo=zo,ma=ma,sa=sa,mb=mb,sb=sb,expected=expected))
    def random(c,h,w): return rng.integers(0,256,(c,h,w),dtype=np.uint8)
    for c,h,w in [(8,6,10),(3,4,6),(96,2,2),(1,512,2),(11,2,512)]:
        add('pool_{}_{}_{}'.format(c,h,w),0,random(c,h,w))
    for c,h,w in [(1,3,5),(11,256,1),(96,1,256)]:
        add('nearest_{}_{}_{}'.format(c,h,w),1,random(c,h,w))
    for ca,cb,h,w in [(5,11,3,5),(1,95,1,1),(13,3,2,7)]:
        add('concat_{}_{}'.format(ca,cb),2,random(ca,h,w),random(cb,h,w))
    add('fused_unaligned',3,random(13,3,5),random(3,6,10))
    add('signed_ties_clip',2,np.array([[[127,125,129,131,0,255]]],np.uint8),
        np.array([[[63,61,65,67,0,255]]],np.uint8))
    add('shift_zero_62',2,random(3,1,7),random(5,1,7),ma=1,sa=0,mb=2147483647,sb=62)
    with np.load(package.parent/'vectors.npz',allow_pickle=False) as archive:
        for number,(pooled,decoder,skip) in enumerate([('enc1.3','neck.3','enc3.3'),
                ('enc2.3','dec3.3','enc2.3'),('enc3.3','dec2.3','enc1.3')]):
            a=np.array(archive[pooled.replace('.','_')+'_expected'][:,:8,:8],copy=True)
            zero=next(r['output_zero_point'] for r in contract['layers'] if r['name']==pooled)
            add('real_pool_'+str(number),0,a,za=zero)
            a=np.array(archive[decoder.replace('.','_')+'_expected'][:,:8,:8],copy=True)
            b=np.array(archive[skip.replace('.','_')+'_expected'][:,:16,:16],copy=True)
            cat=contract['concat'][number];branch_a,branch_b=cat['branches']
            add('real_nearest_'+str(number),1,a,za=branch_a['input_zero_point'])
            quant=dict(za=branch_a['input_zero_point'],zb=branch_b['input_zero_point'],zo=cat['output_zero_point'],
                       ma=branch_a['multiplier'],sa=branch_a['right_shift'],mb=branch_b['multiplier'],sb=branch_b['right_shift'])
            add('real_concat_'+str(number),2,nearest2(a),b,**quant)
            add('real_fused_'+str(number),3,a,b,**quant)
    output=fresh_output(args.outdir);a_bank=[];b_bank=[];gold=[];descriptors=[];archive={};rows=[]
    for index,case in enumerate(cases):
        a,b,expected=case['a'],case['b'],case['expected'];ca,h,w=a.shape;cb=0 if b is None else len(b)
        aa=source_words(a,case['za']);bb=[] if b is None else source_words(b,case['zb']);gg=feature_words(expected,case['zo'])
        descriptors.extend([case['op'],h,w,ca,cb,case['za'],case['zb'],case['zo'],case['ma'],case['sa'],case['mb'],case['sb'],
                            len(a_bank),len(aa),len(b_bank),len(bb),len(gold),len(gg)])
        a_bank.extend(aa.tolist());b_bank.extend(bb if isinstance(bb,list) else bb.tolist());gold.extend(gg.tolist())
        prefix='c'+str(index)
        for name,value in [('a',a),('b',np.empty((0,0,0),np.uint8) if b is None else b),('expected',expected)]:archive[prefix+'_'+name]=value
        rows.append({k:v for k,v in case.items() if k not in ['a','b','expected']}|dict(prefix=prefix,input_shape=list(a.shape),
                    b_shape=None if b is None else list(b.shape),output_shape=list(expected.shape)))
    header=[len(cases),len(a_bank),len(b_bank),len(gold),18]
    for name,values,bits in [('header',header,32),('commands',descriptors,32),('a',a_bank,64),('b',b_bank,64),('gold',gold,64)]:
        hex_bank(output/(name+'.mem'),values,bits)
    np.savez_compressed(output/'vectors.npz',**archive)
    tracked[str(Path(__file__).resolve())]=file_sha256(__file__);check_hashes(tracked)
    record=dict(schema='tensor_ops_hwc8_v1',status='complete_tensor_operator_vectors',cases=rows,header=header,
                source_sha256=tracked,files_sha256={p.name:file_sha256(p) for p in output.iterdir()},
                warning='Software-generated independent reference fixtures, not RTL or board execution')
    write_json(output/'vectors.json',record)
    print('Tensor vectors:',len(cases),'cases,',len(gold),'HWC8 output words',flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    parser.add_argument('--layers',default='data/lung_model_layers_20261015/lung_model_layers/layers.json')
    parser.add_argument('--outdir',required=True)
    generate(parser.parse_args())
