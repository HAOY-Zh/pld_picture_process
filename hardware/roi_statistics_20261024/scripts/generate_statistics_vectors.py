from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / 'vectors'
folder.mkdir(exist_ok=True)
rng = np.random.default_rng(20261024)
cases = []

def add(name, hu, roi=None, hu_range=(-32768,32767), lung=None, bad_keep=False):
    h,w = hu.shape
    roi = roi or (0,h,0,w)
    y0,y1,x0,x1 = roi
    keep = np.zeros((h,w),bool)
    keep[y0:y1,x0:x1] = True
    keep &= (hu >= hu_range[0]) & (hu <= hu_range[1])
    if lung is not None:
        keep &= lung != 0
    values = hu[keep].astype(np.int64)
    words = hu.view(np.uint16).astype(np.uint32)
    if lung is not None:
        words |= lung.astype(np.uint32)<<16
    cases.append(dict(name=name,hu=hu,words=words.ravel(),mask=keep.astype(np.uint8).ravel(),
        geometry=w|(h<<16),bx=x0|(x1<<16),by=y0|(y1<<16),
        thresholds=(hu_range[0]&65535)|((hu_range[1]&65535)<<16),modes=1|(2 if lung is not None else 0),
        count=int(values.size),sum=int(values.sum()),sumsq=int((values*values).sum()),
        min=int(values.min()) if values.size else None,max=int(values.max()) if values.size else None,
        bad_keep=bad_keep))

add('full_512_minimum',np.full((512,512),-32768,np.int16))
add('full_512_maximum',np.full((512,512),32767,np.int16))
add('full_512_alternating',np.tile(np.array([-32768,32767],np.int16),(512,256)))
add('random_roi_hu_lung',rng.integers(-2000,2001,(128,128),dtype=np.int16),(5,120,3,118),(-700,300),rng.integers(0,2,(128,128),dtype=np.uint8))
add('empty_roi',rng.integers(-1000,1000,(17,19),dtype=np.int16),(0,0,0,19))
add('one_negative',np.array([[-37]],np.int16))
add('one_positive',np.array([[777]],np.int16))
add('zero_values',np.zeros((5,7),np.int16))
add('invalid_keep',np.array([[-2,3,4,-5]],np.int16),bad_keep=True)
offset=0
with (folder/'statistics_input.mem').open('w') as wi,(folder/'statistics_mask.mem').open('w') as wm,(folder/'statistics_last.mem').open('w') as wl,(folder/'statistics_cases.vh').open('w') as vh:
    for index,c in enumerate(cases):
        for word,mask in zip(c['words'],c['mask']): wi.write(f'{int(word):08x}\n');wm.write(f'{int(mask):x}\n')
        wl.write('0\n'*(len(c['words'])-1)+'1\n')
        minmax=0 if c['count']==0 else ((c['min']&65535)|((c['max']&65535)<<16))
        vh.write(f"run_case({index},{offset},{len(c['words'])},32'h{c['geometry']:08x},32'h{c['bx']:08x},32'h{c['by']:08x},32'h{c['thresholds']:08x},32'h{c['modes']:08x},{c['count']},64'h{c['sum']&((1<<64)-1):016x},64'h{c['sumsq']:016x},32'h{minmax:08x},{int(c['bad_keep'])});\n")
        c['offset']=offset;offset+=len(c['words'])
metadata=[{k:v for k,v in c.items() if k not in ('hu','words','mask')} for c in cases]
(folder/'statistics_manifest.json').write_text(json.dumps({'provenance':'SYNTHETIC_TEST_ONLY','cases':metadata,'total_pixels':offset},indent=2),encoding='utf-8')
print(f'{len(cases)} frames, {offset} pixels; independent NumPy oracle')
