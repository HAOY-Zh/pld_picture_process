"""Independent numpy reference vectors: all int16 HU, odd/extreme windows,
signed thresholds, padded morphology, backpressure and mode changes.
"""
from pathlib import Path
import sys
import numpy as np
from ct_basic_processing import process

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
frames=[]
native='--native' in sys.argv[2:]
for wl,ww in ([] if native else [(-600,1500),(40,400),(400,1800),(0,5),(-32768,65535),(32767,1),(32767,65535)]):
    for hu in np.arange(-32768,32768,dtype=np.int32).reshape(-1,16,16):
        frames.append((0,hu,dict(operation='window',wl=wl,ww=ww)))
rng=np.random.default_rng(1907)
for cycle in range(0 if native else 6):
    for op in [1,2,3,4,5,0,5,2,4,1,3]:
        hu=rng.integers(-1200,1200,size=(16,16),dtype=np.int32)
        if cycle==0:hu[:]=-800
        elif cycle==1:hu[:]=0
        elif cycle==2:hu[:]=0;hu[3:13,2:14]=-800
        p=dict(operation=['window','threshold','erosion','dilation','opening','closing'][op],
               wl=-600,ww=1500,lower=-1000,upper=-400)
        frames.append((op,hu,p))
if native:
    from lung_live_data import read_ct
    hu,geom,details=read_ct(r'D:\CT2\lctsc\LCTSC-Test-S1-101',slices='44')
    hu=hu[0]
    np.savez_compressed(out/'actual_ct_input.npz',hu=hu)
    import json
    (out/'actual_ct_input.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    y,x=np.mgrid[:512,:512]
    for sample in [hu,((131*x+73*y)%4000-2000).astype(np.int32),np.full((512,512),-800,np.int32)]:
        for op,name in enumerate(['window','threshold','erosion','dilation','opening','closing']):
            frames.append((op,sample,dict(operation=name,wl=0,ww=5,lower=-1000,upper=-400)))
with (out/'vectors.txt').open('x',encoding='ascii') as f:
    f.write(str(len(frames))+'\n')
    for mode,hu,p in frames:
        ww=p.get('ww',1500);wl=p.get('wl',-600)
        low=p.get('lower',-1000);high=p.get('upper',-400)
        actual_input=np.where((hu>=low)&(hu<=high),255,0) if mode>=2 else hu
        expected=process(hu,p)
        f.write(f'{mode} {((ww<<16)|(wl&65535)):08x} {(((high&65535)<<16)|(low&65535)):08x}\n')
        for v,e in zip(actual_input.ravel(),expected.ravel()):f.write(f'{int(v)&0xffffffff:08x} {int(e):08x}\n')
print('vectors:',len(frames),'frames',sum(v[1].size for v in frames),'pixels; numpy reference, not real board execution')
