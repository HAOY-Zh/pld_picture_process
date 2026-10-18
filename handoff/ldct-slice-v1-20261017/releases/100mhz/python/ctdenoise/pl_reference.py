"""Integer contracts for the tiled PL denoiser. Not board performance evidence."""
from pathlib import Path
import numpy as np
from .filters import quantize_hu
from .cnn_fixed import infer_fixed

MODES={0:'bypass',1:'gaussian5',2:'bilateral5',3:'joint7_guide5',
       4:'joint7_guide3',5:'guided3',6:'cnn',7:'bilateral7'}
RANGE=np.rint(4096*np.exp(-.5*(np.arange(1024)/64.)**2)).astype(np.int64)


def gaussian_integer(image,radius):
    from math import comb
    a=quantize_hu(image).astype(np.int64)+32768
    v=np.array([comb(2*radius,i) for i in range(2*radius+1)],dtype=np.int64)
    p=np.pad(a,radius,mode='edge');s=np.zeros_like(a)
    for y,x in np.ndindex((2*radius+1,2*radius+1)):
        s+=int(v[y]*v[x])*p[y:y+a.shape[0],x:x+a.shape[1]]
    d=int(v.sum()**2)
    return ((s+d//2)//d-32768).astype(np.int16)


def spatial_integer(mode):
    radius=2 if mode==2 else 3
    sigma=1.0 if mode==7 else 1.8
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    return np.rint(4096*np.exp(-.5*(x*x+y*y)/sigma**2)).astype(np.int64)


def window_sum(a,radius=1):
    a=np.asarray(a,dtype=np.int64);p=np.pad(a,radius,mode='edge');s=np.zeros_like(a)
    for y,x in np.ndindex((2*radius+1,2*radius+1)):
        s+=p[y:y+a.shape[0],x:x+a.shape[1]]
    return s


def fixed_filter(image,mode):
    a=quantize_hu(image).astype(np.int64)
    if mode==0:return a.astype(np.int16)
    if mode==1:return gaussian_integer(a,2)
    if mode==5:
        s=window_sum(a);s2=window_sum(a*a)
        var=np.maximum(9*s2-s*s,0);den=var+81*1048576
        coeff=((var<<16)+den//2)//den
        b9=s*(65536-coeff)
        numerator=9*a*window_sum(coeff)+window_sum(b9)
        denominator=81*65536
        result=np.sign(numerator)*((np.abs(numerator)+denominator//2)//denominator)
        return np.clip(result,-32768,32767).astype(np.int16)
    if mode not in (2,3,4,7):raise ValueError('Nonfilter mode')
    guide=gaussian_integer(a,2 if mode==3 else 1).astype(np.int64) if mode in (3,4) else a
    radius=2 if mode==2 else 3;shift=2 if mode in (3,4) else 4
    spatial=spatial_integer(mode);p=np.pad(a+32768,radius,mode='edge');g=np.pad(guide,radius,mode='edge')
    num=np.zeros_like(a);den=np.zeros_like(a)
    for y,x in np.ndindex(spatial.shape):
        n=p[y:y+a.shape[0],x:x+a.shape[1]]
        index=np.abs(g[y:y+a.shape[0],x:x+a.shape[1]]-guide)>>shift
        rw=np.where(index<1024,RANGE[np.minimum(index,1023)],0)
        w=(spatial[y,x]*rw+2048)>>12
        num+=n*w;den+=w
    assert np.all(den>0)
    return ((num+den//2)//den-32768).astype(np.int16)


def tiled_reference(image,mode,arrays=None,spec=None,tile=48):
    a=quantize_hu(image);out=np.empty_like(a)
    halo={0:0,1:2,2:2,3:5,4:4,5:2,6:3,7:3}[mode]
    for y in range(0,a.shape[0],tile):
        for x in range(0,a.shape[1],tile):
            y2,x2=min(y+tile,len(a)),min(x+tile,a.shape[1])
            ya,xa=max(0,y-halo),max(0,x-halo)
            ye,xe=min(len(a),y2+halo),min(a.shape[1],x2+halo)
            patch=a[ya:ye,xa:xe]
            q=infer_fixed(patch,arrays,spec)[0] if mode==6 else fixed_filter(patch,mode)
            out[y:y2,x:x2]=q[y-ya:y2-ya,x-xa:x2-xa]
    return out


def export_roms(quantized,outdir):
    from .cnn_fixed import load_quantized
    # ROM export also works from a lightweight handoff without DICOM/training.
    import hashlib,json
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    arrays,spec=load_quantized(quantized);out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    lines=['// Generated from the frozen, audited integer model. Do not edit.']
    for i,layer in enumerate(spec['layers']):
        shift=int(layer['right_shift'])
        if not 1<=shift<=30:raise ValueError('RTL requires requantization right shifts in 1..30')
        lines.append(f'localparam integer CNN_SHIFT_{i}={shift};')
    def fn(name,values,bits=8):
        flat=np.asarray(values).ravel()
        text=[f'function signed [{bits-1}:0] {name};', 'input integer index;', 'begin case(index)']
        for i,v in enumerate(flat):
            literal=('-' if int(v)<0 else '')+f"{bits}'sd{abs(int(v))}"
            text.append(f'{i}: {name}={literal};')
        text.extend([f'default: {name}=0;','endcase end endfunction'])
        return text
    for i in range(3):
        lines+=fn('cnn_w'+str(i),arrays['w'+str(i)])
        lines+=fn('cnn_b'+str(i),arrays['b'+str(i)],32)
    for mode in (2,3,7):lines+=fn('spatial_'+str(mode),spatial_integer(mode),14)
    lines+=fn('range_weight',RANGE,14)
    (out/'ct_denoise_roms.vh').write_text('\n'.join(lines)+'\n',encoding='ascii')
    write_json(out/'model_manifest.json',dict(quantization=spec,rom_sha256=digest(out/'ct_denoise_roms.vh'),
        modes=MODES,max_tile=64,tile_default=48,halo={0:0,1:2,2:2,3:5,4:4,5:2,6:3,7:3}))
    return arrays,spec
