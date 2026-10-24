"""Portable INT8 nodule ABI. NumPy only; no Torch, annotations or web server.

The stride-two 5-channel stem is exactly lowered to a 45-channel 1x1
convolution. Channel order is (input channel, kernel y, kernel x).
"""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'nodule_hwc8_plan_v1'
TOPOLOGY = [
    ('stem0', 'conv', ['input'], 8, 1),
    ('stem1', 'conv', ['stem0'], 8, 3),
    ('pool1', 'pool', ['stem1'], 8, 0),
    ('enc1', 'conv', ['pool1'], 16, 3),
    ('pool2', 'pool', ['enc1'], 16, 0),
    ('enc2', 'conv', ['pool2'], 24, 3),
    ('cat1', 'fused_concat', ['enc2', 'enc1'], 40, 0),
    ('dec1', 'conv', ['cat1'], 16, 3),
    ('cat0', 'fused_concat', ['dec1', 'stem1'], 24, 0),
    ('dec0', 'conv', ['cat0'], 8, 3),
    ('head', 'conv', ['dec0'], 5, 1),
]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def quantize_hu(image):
    image = np.asarray(image)
    if image.shape != (5, 128, 128) or not np.isfinite(image).all():
        raise ValueError('Expected finite five-plane 128x128 HU patch')
    return np.rint((image.astype(np.float64).clip(-1000, 400)+1000)*(255./1400.)).astype(np.uint8)


def lower_stem(codes):
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or codes.shape != (5, 128, 128):
        raise ValueError('Expected five-channel UINT8 128x128 patch')
    padded = np.pad(codes, ((0,0),(1,1),(1,1)), constant_values=0)
    return np.ascontiguousarray(np.stack([
        padded[c, ky:ky+128:2, kx:kx+128:2]
        for c in range(5) for ky in range(3) for kx in range(3)]))


def requant(acc, mult, shift, zero, relu=False):
    """Signed INT64 product, round to nearest ties to even, then U8 saturate."""
    acc = np.asarray(acc, np.int64)
    if np.any(acc < -2**31) or np.any(acc >= 2**31):
        raise OverflowError('INT32 accumulator overflow')
    m, s = np.asarray(mult, np.int64), np.asarray(shift, np.int64)
    product = acc*m
    quotient = product >> s
    remainder = product & ((np.ones_like(s) << s)-1)
    half = np.ones_like(s) << (np.maximum(s,1)-1)
    increment = (s>0) & ((remainder>half) | ((remainder==half) & ((quotient&1)!=0)))
    return np.clip(quotient+increment+zero, zero if relu else 0, 255).astype(np.uint8)


def multiplier(ratio):
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Invalid positive quantization ratio')
    shift = min(62, int(np.floor(np.log2((2**30)/ratio))))
    value = int(round(ratio*2.**shift))
    if not 0 < value < 2**31 or not 0 <= shift <= 62:
        raise ValueError('Ratio outside supported Q31 range')
    return value, shift


def validate(model, parameters):
    if (model.get('schema') != SCHEMA or model.get('input_shape') != [45,64,64]
            or model.get('input_zero') != 0 or len(model.get('nodes',[])) != len(TOPOLOGY)):
        raise ValueError('Wrong nodule graph/input contract')
    available = {'input': ([45,64,64],0)}; commands = 0
    for node, (name, kind, sources, cout, kernel) in zip(model['nodes'], TOPOLOGY):
        if (node['name'],node['kind'],node['sources']) != (name,kind,sources):
            raise ValueError('Unexpected nodule topology: '+name)
        ishape, iz = available[sources[0]]; cin,h,w = ishape
        if kind == 'conv':
            row=node['layer']; p=row['prefix']; weight=parameters[p+'_weight']
            if (weight.dtype != np.int8 or weight.shape != (cout,cin,kernel,kernel)
                    or row['input_shape'] != ishape or row['input_zero'] != iz
                    or row['name'] != name or row['prefix'] != name or row['kernel'] != kernel
                    or row['padding'] != kernel//2 or row['relu'] != (name!='head')):
                raise ValueError('Bad convolution: '+name)
            for key in ['bias','multiplier','shift']:
                a=parameters[p+'_'+key]
                if a.dtype != np.int32 or a.shape != (cout,):
                    raise ValueError('Invalid channel parameters')
            m,s=parameters[p+'_multiplier'],parameters[p+'_shift']
            if np.any(m<=0) or np.any(s<0) or np.any(s>62): raise ValueError('Bad requantization')
            bound=np.abs(weight.astype(np.int64)).sum((1,2,3))*max(iz,255-iz)+np.abs(parameters[p+'_bias'].astype(np.int64))
            if bound.max()>=2**31: raise OverflowError('Unsafe accumulator')
            shape=[cout,h,w]; oz=row['output_zero'];commands+=(cout+7)//8
            if oz!=(128 if name=='head' else 0):raise ValueError('Unexpected head/hidden zero point')
            if row['output_shape']!=shape: raise ValueError('Bad convolution output')
        else:
            q=node['quant']; oz=q['zo']
            if q['za']!=iz: raise ValueError('Tensor grid mismatch')
            if kind=='pool':
                shape=[cout,h//2,w//2]
                if h%2 or w%2 or oz!=iz: raise ValueError('Bad pool')
            else:
                sb,zb=available[sources[1]];shape=[cout,h*2,w*2]
                if sb[1:]!=shape[1:] or q['zb']!=zb or cin+sb[0]!=cout: raise ValueError('Bad skip')
                for branch in ['a','b']:
                    if not 0<q['m'+branch]<2**31 or not 0<=q['s'+branch]<=62: raise ValueError('Bad concat quantization')
            commands+=1
        if type(oz) is not int or not 0<=oz<=255 or node['zero']!=oz or node['shape']!=shape:
            raise ValueError('Bad output grid')
        available[name]=(shape,oz)
    if model['commands']!=commands or np.asarray(model['head_scales']).shape!=(5,):
        raise ValueError('Incomplete graph')
    if not np.isfinite(model['head_scales']).all() or min(model['head_scales'])<=0:
        raise ValueError('Bad head scales')
    return available


def dequantize_head(codes, model):
    if codes.dtype!=np.uint8 or codes.shape!=(5,64,64): raise ValueError('Bad head codes')
    return ((codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None])


def load_package(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'nodule_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='nodule_board_package_v1': raise ValueError('Wrong package')
    for name,digest in manifest['files_sha256'].items():
        p=(folder/name).resolve()
        if folder not in p.parents or not p.is_file() or sha(p)!=digest:
            raise ValueError('Changed/missing package file: '+name)
    model=json.loads((folder/'nodule_model.json').read_text(encoding='utf-8'))
    with np.load(folder/'nodule_parameters.npz',allow_pickle=False) as arrays:
        params={n:arrays[n].copy() for n in arrays.files}
    validate(model,params)
    return folder,manifest,model,params
