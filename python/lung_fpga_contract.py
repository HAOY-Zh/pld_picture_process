"""Proposed W8/A8/ACC32 hardware contract, exact golden operators and probes.

Not RTL, not a bit-exact copy of x86 quantized kernels. CPU differences are
reported, not silently ignored. No medical images or labels used here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_quantize import restore_quantized
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256, torch_module

SCHEMA='w8_a8_acc32_q31_rne_v1'


def integer_array(value):
    a=np.asarray(value)
    if not np.issubdtype(a.dtype,np.integer):raise ValueError('Need integer codes, not floating values')
    if a.size and (int(a.min()) < -(2**31) or int(a.max()) >= 2**31):
        raise ValueError('Input exceeds supported signed INT32 domain')
    return a.astype(np.int64)


def multiplier(ratio):
    ratio=np.asarray(ratio,dtype=np.float64)
    if not np.isfinite(ratio).all() or (ratio<=0).any():raise ValueError('Need positive finite scale ratio')
    fraction,exponent=np.frexp(ratio)
    m=np.rint(fraction*2**31).astype(np.int64)
    overflow=m==2**31
    m=np.where(overflow,m//2,m);exponent=exponent+overflow
    shift=31-exponent
    if (shift<0).any() or (shift>62).any():raise ValueError('Scale ratio outside contract right-shift domain')
    return m.astype(np.int32),shift.astype(np.int32)


def round_shift_even(product, shift):
    """Signed floor quotient + nonnegative remainder; ties to even, one round."""
    a=np.asarray(product)
    if not np.issubdtype(a.dtype,np.integer) or not isinstance(shift,(int,np.integer)) or not 0<=shift<=62:
        raise ValueError('Need integer product and shift 0..62')
    if a.size and (int(a.min()) < -(2**63) or int(a.max()) >= 2**63):
        raise ValueError('Product outside signed INT64 domain')
    a=a.astype(np.int64)
    if shift==0:return a
    divisor=np.int64(2**int(shift));q=np.floor_divide(a,divisor);r=np.remainder(a,divisor)
    return q+((r>divisor//2)|((r==divisor//2)&((q&1)!=0)))


def requantize(accumulator, multipliers, shifts, zero_point, relu=False):
    a=integer_array(accumulator);m=integer_array(multipliers);s=integer_array(shifts)
    if a.ndim<1 or m.shape!=(a.shape[0],) or s.shape!=m.shape or (m<=0).any() or (s<0).any() or (s>62).any() or not 0<=zero_point<=255:
        raise ValueError('Invalid per-output-channel requantization parameters')
    result=np.empty(a.shape,np.uint8)
    for c in range(a.shape[0]):
        q=round_shift_even(a[c]*m[c],int(s[c]))+int(zero_point)
        result[c]=np.clip(q,zero_point if relu else 0,255).astype(np.uint8)
    return result


def conv_accumulator(image, weights, bias, input_zero_point, padding=1):
    """Exact CHW/OIHW cross-correlation; UINT8 padding code is input zero point."""
    image,weights,bias=[np.asarray(a) for a in [image,weights,bias]]
    if (image.dtype!=np.uint8 or weights.dtype!=np.int8 or bias.dtype!=np.int32 or
        image.ndim!=3 or weights.ndim!=4 or image.shape[0]!=weights.shape[1] or
        bias.shape!=(weights.shape[0],) or not 0<=input_zero_point<=255 or padding not in [0,1] or
        weights.shape[2:] not in [(3,3),(1,1)]):raise ValueError('Invalid supported convolution')
    kernel=weights.shape[-1];h=image.shape[1]+2*padding-kernel+1;w=image.shape[2]+2*padding-kernel+1
    if min(h,w)<=0:raise ValueError('Kernel larger than padded input')
    centered=np.pad(image.astype(np.int64)-int(input_zero_point),((0,0),(padding,padding),(padding,padding)))
    acc=np.broadcast_to(bias[:,None,None].astype(np.int64),(len(bias),h,w)).copy()
    for y in range(kernel):
        for x in range(kernel):
            acc+=np.einsum('ihw,oi->ohw',centered[:,y:y+h,x:x+w],weights[:,:,y,x].astype(np.int64),optimize=False)
    if (acc<-(2**31)).any() or (acc>=2**31).any():raise ValueError('ACC32 overflow; no wrap allowed')
    return acc.astype(np.int32)


def maxpool2(image):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[1]%2 or image.shape[2]%2:
        raise ValueError('Need even-sized CHW UINT8 input')
    return image.reshape(image.shape[0],image.shape[1]//2,2,image.shape[2]//2,2).max((2,4))


def nearest2(image):
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3:raise ValueError('Need CHW UINT8 input')
    return np.repeat(np.repeat(a,2,axis=1),2,axis=2)


def rescale_codes(image,input_scale,input_zero_point,output_scale,output_zero_point):
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or not 0<=input_zero_point<=255:
        raise ValueError('Need CHW UINT8 codes and valid input zero point')
    m,s=multiplier(np.full(a.shape[0],input_scale/output_scale))
    return requantize(a.astype(np.int32)-input_zero_point,m,s,output_zero_point)


def concatenate(branches,output_scale,output_zero_point):
    if not branches:raise ValueError('Need at least one branch')
    arrays=[rescale_codes(a,s,z,output_scale,output_zero_point) for a,s,z in branches]
    if len({a.shape[1:] for a in arrays})!=1:raise ValueError('Concat spatial shapes differ')
    return np.concatenate(arrays,axis=0)


def run(args):
    source=Path(args.quantization).resolve(strict=True)
    manifest=json.loads((source/'core_manifest.json').read_text(encoding='utf-8'))
    if len(manifest['layers'])!=15 or manifest['integer_convolution_count']!=15:
        raise ValueError('Expected frozen 15-layer INT8 core')
    inputs={str((source/name).resolve()):file_sha256(source/name) for name in
        ['core_manifest.json','int8_core_parameters.npz','int8_model.pt','protocol.json']}
    inputs[str(Path(__file__).resolve())]=file_sha256(__file__)
    model,saved=restore_quantized(source/'int8_model.pt');torch=torch_module();torch.set_num_threads(2)
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if saved['source_checkpoint_sha256']!=protocol['checkpoint_sha256']:raise ValueError('Model identity differs')
    modules=dict(model.named_modules());output=fresh_output(args.outdir);params={};vectors={};rows=[]
    with np.load(source/'int8_core_parameters.npz') as arrays:
        for layer in manifest['layers']:
            name=layer['name'];prefix=name.replace('.','_');module=modules[name]
            weights=arrays[prefix+'_weight_int8'];zeros=arrays[prefix+'_weight_zero_point']
            bias=arrays[prefix+'_proposed_bias_int32'];scales=arrays[prefix+'_weight_scale']
            if weights.dtype!=np.int8 or np.any(zeros!=0):raise ValueError('Contract requires symmetric zero-point-zero weights')
            actual_w=module.weight()
            if (not np.array_equal(actual_w.int_repr().numpy(),weights) or
                not np.array_equal(actual_w.q_per_channel_scales().numpy(),scales) or
                module.scale!=layer['output_scale'] or module.zero_point!=layer['output_zero_point']):
                raise ValueError('Core parameters do not match frozen model')
            expected_bias=np.rint(module.bias().detach().numpy()/(layer['input_scale']*scales)).astype(np.int32)
            if not np.array_equal(expected_bias,bias):raise ValueError('Bias convention mismatch')
            ratios=layer['input_scale']*scales/layer['output_scale'];m,s=multiplier(ratios)
            bound=int((np.abs(weights.astype(np.int64)).sum((1,2,3))*max(layer['input_zero_point'],255-layer['input_zero_point'])+np.abs(bias.astype(np.int64))).max())
            if bound>=2**31:raise ValueError('Conservative accumulator bound overflow')
            for key,value in [('weight_int8',weights),('bias_int32',bias),('multiplier_int32',m),('right_shift_int32',s)]:params[prefix+'_'+key]=value
            comparisons=[]
            for tag in ['real_zero','ramp127','ramp255']:
                dims=(weights.shape[1],8,8)
                q=np.full(dims,layer['input_zero_point'],np.uint8) if tag=='real_zero' else (np.arange(np.prod(dims)).reshape(dims)*37%(128 if tag=='ramp127' else 256)).astype(np.uint8)
                acc=conv_accumulator(q,weights,bias,layer['input_zero_point'],layer['padding'][0])
                golden=requantize(acc,m,s,layer['output_zero_point'],layer['fused_relu'])
                real=(q.astype(np.float32)-layer['input_zero_point'])*layer['input_scale']
                tensor=torch.quantize_per_tensor(torch.from_numpy(real)[None],layer['input_scale'],layer['input_zero_point'],torch.quint8)
                if not np.array_equal(tensor.int_repr()[0].numpy(),q):raise ValueError('Probe quantization changed integer codes')
                with torch.inference_mode():cpu=module(tensor).int_repr()[0].numpy()
                error=np.abs(cpu.astype(np.int16)-golden.astype(np.int16))
                comparisons.append(dict(probe=tag,elements=int(error.size),different_codes=int(np.count_nonzero(error)),max_abs_code_error=int(error.max())))
                for key,value in [('input_uint8',q),('accumulator_int32',acc),('output_uint8',golden),('x86_output_uint8',cpu)]:vectors[prefix+'_'+tag+'_'+key]=value
            params[prefix+'_weight_scale_fp64']=scales
            rows.append(dict(name=name,input_shape=layer['input_shape'],output_shape=layer['output_shape'],
                weight_shape=layer['weight_shape'],padding=layer['padding'],stride=layer['stride'],
                input_scale=layer['input_scale'],input_zero_point=layer['input_zero_point'],output_scale=layer['output_scale'],
                output_zero_point=layer['output_zero_point'],fused_relu=layer['fused_relu'],
                ratio_max_abs_error=float(np.max(np.abs(m.astype(float)*np.exp2(-s.astype(float))-ratios))),
                conservative_accumulator_bound=bound,x86_probes=comparisons))
    np.savez(output/'parameters.npz',**params);np.savez_compressed(output/'operator_vectors.npz',**vectors)
    # Boundary grids and concat branch ORDER are explicit, not inherited implicitly.
    cats=[]
    for n,(up,skip) in enumerate([('neck.3','enc3.3'),('dec3.3','enc2.3'),('dec2.3','enc1.3')]):
        lookup={row['name']:row for row in rows};dest=manifest['boundaries']['cats.'+str(n)]
        branches=[]
        for name in [up,skip]:
            row=lookup[name];m,s=multiplier(row['output_scale']/dest['scale'])
            branches.append(dict(source=name,input_scale=row['output_scale'],input_zero_point=row['output_zero_point'],multiplier=int(m),right_shift=int(s)))
        cats.append(dict(name='cats.'+str(n),order='upsampled decoder feature THEN encoder skip',
            output_scale=dest['scale'],output_zero_point=dest['zero_point'],branches=branches))
    for p,h in inputs.items():
        if file_sha256(p)!=h:raise ValueError('Frozen input changed: '+p)
    mismatch=sum(p['different_codes'] for r in rows for p in r['x86_probes'])
    contract=dict(schema=SCHEMA,status='complete_proposed_integer_contract',input_sha256=inputs,
        checkpoint_sha256=saved['source_checkpoint_sha256'],protected_inputs_unchanged=True,
        layers=rows,concat=cats,input_quantization=manifest['boundaries']['input_quant'],
        arithmetic=dict(activation='UINT8 affine [0,255], centered signed 9-bit [-255,255]',
            weight='symmetric signed INT8, zero point 0',accumulator='signed INT32; overflow forbidden',
            bias='RNE(fp32_bias/(input_scale*weight_scale_per_channel)) in INT32',
            requantization='clip(RNE((acc_int32 * multiplier_int32) / 2**right_shift)+output_zero_point, 0,255)',
            product='signed INT64 reference; signed 64-bit register in initial RTL design',
            rounding='ONE signed round-to-nearest ties-to-even; no staged/double rounding',
            relu='clamp minimum to output zero point after requantization',
            padding='real zero represented by input zero point, NOT always code 0'),
        operators=dict(maxpool='2x2 stride2, max codes, keep grid',nearest='2x2 repeat codes, keep grid',
            concat='Requantize EACH branch to destination grid, then CHW concat; decoder branch first'),
        storage=dict(feature_layout='CHW uint8, X fastest; little-endian integers',weight_layout='OIHW int8, KX fastest',
            skip_bytes=manifest['naive_full_skip_buffer_bytes'],weights_bytes=manifest['int8_weight_bytes'],
            planning='DDR layer ping-pong + 3 retained skip buffers; BRAM tile/line caches; no full network unrolling'),
        probes=dict(total_convolution_vectors=45,x86_different_codes=mismatch,x86_bit_exact_on_probes=mismatch==0,
            warning='Tiny synthetic full-range probes, NOT whole-network or medical accuracy proof; no clinical data used'),
        scope='PROPOSED hardware numerical reference; x86 differences reported. RTL and whole-network fixed-point evaluation pending.',
        hardware_ready=False,rtl_present=False,whole_network_fixed_point_validated=False,
        outputs={name:file_sha256(output/name) for name in ['parameters.npz','operator_vectors.npz']})
    write_json(output/'contract.json',contract)
    print('Integer contract exported; x86 different codes on probes:',mismatch,flush=True)
    return contract


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quantization',default='data/lung_training_20261011/int8_ptq_validation_v2')
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
