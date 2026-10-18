"""Candidate PL arithmetic: int8 weights, signed int16 features, power-of-two
scales, int32 accumulators, round nearest ties away from zero, saturation.
NumPy uses int64 to audit int32 bounds. This is NOT an FPGA implementation.
"""
import json
from pathlib import Path
import numpy as np
from .filters import quantize_hu


def fraction_bits(maximum, bits):
    if not np.isfinite(maximum) or maximum<0: raise ValueError('Invalid calibration range')
    if maximum==0: return 10
    return int(np.clip(np.floor(np.log2(((1<<(bits-1))-1)/maximum)), -16, 24))


def round_shift(values, shift):
    values=np.asarray(values,dtype=np.int64)
    if shift<=0:
        if -shift>60 or np.max(np.abs(values))>(np.iinfo(np.int64).max >> (-shift)):
            raise OverflowError('Requantization shift exceeds reference int64 range')
        return values*(2**(-shift))
    return np.sign(values)*((np.abs(values)+(1<<(shift-1)))>>shift)


def build_quantized(model, activation_maxima):
    if len(activation_maxima)!=2: raise ValueError('Two ReLU activation ranges required')
    # Input normalized Q10 equals integer HU. Last noise output also integer HU.
    fractions=[10]+[fraction_bits(1.1*v,16) for v in activation_maxima]+[10]
    arrays={};layers=[]
    for i,layer in enumerate(model.layers):
        w=layer.weight.detach().cpu().numpy().astype(float)
        b=layer.bias.detach().cpu().numpy().astype(float)
        fw=fraction_bits(float(np.max(np.abs(w))),8)
        qw=np.clip(np.rint(w*2.0**fw),-127,127).astype(np.int8)
        qb=np.rint(b*2.0**(fractions[i]+fw)).astype(np.int64)
        # Worst-case signed int16 features; sum abs separately for every output.
        bounds=np.abs(qw.astype(np.int64)).sum(axis=(1,2,3))*32768+np.abs(qb)
        if np.max(bounds)>np.iinfo(np.int32).max: raise ValueError('int32 accumulator bound exceeded')
        arrays[f'w{i}']=qw;arrays[f'b{i}']=qb.astype(np.int32)
        layers.append(dict(input_fraction=fractions[i],weight_fraction=fw,output_fraction=fractions[i+1],
            right_shift=fractions[i]+fw-fractions[i+1],relu=i<2,
            worst_case_accumulator=int(np.max(bounds))))
    return arrays,dict(layers=layers,channels=model.channels,input='signed int16 HU = normalized Q10',
        output='signed int16 HU = input HU - rounded predicted noise HU, saturated',
        padding='replicate separately at each layer',rounding='nearest; half away from zero',
        weights='int8 power-of-two scales; bias int32 accumulator units',
        activations='int16 with train-only range calibration and 10% headroom',
        status='Arithmetic software reference only; no RTL/HLS/board result')


def infer_fixed(image, arrays, spec):
    original=quantize_hu(image).astype(np.int64)
    a=original[None];saturated=[]
    for i,layer in enumerate(spec['layers']):
        w=arrays[f'w{i}'].astype(np.int64);b=arrays[f'b{i}'].astype(np.int64)
        padded=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
        h,width=a.shape[1:]
        acc=np.broadcast_to(b[:,None,None],(len(b),h,width)).copy()
        for y in range(3):
            for x in range(3):
                acc+=np.einsum('oi,ihw->ohw',w[:,:,y,x],padded[:,y:y+h,x:x+width],optimize=False)
        if acc.min()<np.iinfo(np.int32).min or acc.max()>np.iinfo(np.int32).max:
            raise OverflowError('Actual int32 accumulator overflow')
        a=round_shift(acc,layer['right_shift'])
        if layer['relu']: a=np.maximum(a,0)
        saturated.append(int(np.count_nonzero((a < -32768)|(a > 32767))))
        a=np.clip(a,-32768,32767)
    result=original-a[0]
    saturated.append(int(np.count_nonzero((result < -32768)|(result > 32767))))
    return np.clip(result,-32768,32767).astype(np.int16),saturated


def load_quantized(folder):
    import hashlib
    folder=Path(folder)
    spec=json.loads((folder/'quantization.json').read_text(encoding='utf-8'))
    # Standalone board deployment must not import training/DICOM dependencies.
    weights_hash=hashlib.sha256((folder/'weights_integer.npz').read_bytes()).hexdigest()
    if weights_hash!=spec['weights_sha256']:
        raise ValueError('Quantized weights hash differs')
    with np.load(folder/'weights_integer.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    return arrays,spec
