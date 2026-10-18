"""Generate frozen ROMs and independent multi-mode protocol simulation cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from ctdenoise.pl_reference import export_roms,fixed_filter,MODES
from ctdenoise.cnn_fixed import infer_fixed


def main(args):
    out=Path(args.outdir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use empty/new vector directory')
    out.mkdir(parents=True,exist_ok=True)
    arrays,spec=export_roms(args.quantized_dir,args.romdir)
    shifts=[int(s['right_shift']) for s in spec['layers']]
    rng=np.random.default_rng(20261017);cases=[]
    for mode in MODES:
        a=rng.integers(-1400,1800,(5,7),dtype=np.int16)
        cases.append(dict(mode=mode,image=a,status=0))
    for shape in ((1,1),(1,7),(7,1)):
        for mode in (1,3,5,6,7):
            a=rng.integers(-32768,32768,shape,dtype=np.int16)
            cases.append(dict(mode=mode,image=a,status=0))
    # A real native-HU crop exercises the trained weights on actual CT values.
    if args.cache or args.sample_input:
        a=(np.load(Path(args.cache)/'C067/low.npy',mmap_mode='r')[180,208:217,112:121].copy()
           if args.cache else np.load(args.sample_input,allow_pickle=False)[208:217,112:121].copy())
        if a.shape!=(9,9) or a.dtype!=np.int16:raise ValueError('Real sample must supply a signed HU 9x9 crop at the frozen location')
        for mode in MODES:cases.append(dict(mode=mode,image=a,status=0))
    if args.stress:
        a=rng.integers(-32768,32768,(64,64),dtype=np.int16)
        for mode in MODES:cases.append(dict(mode=mode,image=a,status=0))
        # Compare legacy/compact outputs on identical input tiles, including
        # every combination of true global edges and internal halo edges.
        for flags in range(16):
            trims=tuple(3 if flags & (1<<bit) else 0 for bit in range(4))
            shape=(64,64) if flags in (0,15) else (9,11)
            a=rng.integers(-32768,32768,shape,dtype=np.int16)
            for compact in (False,True):
                cases.append(dict(mode=6,image=a,status=0,trims=trims if compact else (0,0,0,0),compact=compact))
        for shape in ((1,1),(1,64),(64,1)):
            cases.append(dict(mode=6,image=rng.integers(-32768,32768,shape,dtype=np.int16),status=0,compact=True))
        cases.append(dict(mode=6,image=np.ones((3,3),dtype=np.int16),status=8,invalid_crop=True))
    cases.append(dict(mode=0,image=np.array([[-32768,32767],[-1,0]],dtype=np.int16),status=3,bad_sidebands=True))
    cases.append(dict(mode=9,image=np.array([[123]],dtype=np.int16),status=4))
    cases.append(dict(mode=0,image=np.array([[-999]],dtype=np.int16),status=8,bad_width=True))
    inputs=[];expected=[];settings=[];offsets=[];sizes=[];statuses=[];bad=[];total=0
    output_offsets=[];output_sizes=[]
    for c in cases:
        a=c['image'];mode=c['mode'];h,w=a.shape
        q=infer_fixed(a,arrays,spec)[0] if mode==6 else fixed_filter(a,mode if mode<=7 else 0)
        cfg=mode|((0 if c.get('bad_width') else w)<<4)|(h<<11)
        if c.get('compact'):
            left,right,top,bottom=c.get('trims',(0,0,0,0))
            cfg|=(1<<26)|(left<<18)|(right<<20)|(top<<22)|(bottom<<24)
            q=q[top:h-bottom,left:w-right]
        if c.get('invalid_crop'):cfg|=(1<<26)|(1<<18)
        inputs.extend(a.view(np.uint16).ravel().astype(np.uint32))
        output_offsets.append(len(expected));output_sizes.append(q.size)
        expected.extend(q.view(np.uint16).ravel().astype(np.uint32))
        settings.append(cfg);offsets.append(total);sizes.append(a.size);statuses.append(c['status']);bad.append(int(c.get('bad_sidebands',False)));total+=a.size
    for name,values in [('input',inputs),('expected',expected),('settings',settings),('offsets',offsets),('sizes',sizes),('statuses',statuses),('bad',bad),('output_offsets',output_offsets),('output_sizes',output_sizes)]:
        (out/(name+'.hex')).write_text(''.join(f'{int(v):08x}\n' for v in values),encoding='ascii')
    (out/'vectors.vh').write_text(f'localparam FRAMES={len(cases)},PIXELS={total},OUTPUT_PIXELS={len(expected)};\n'+
        ''.join(f'localparam MODEL_SHIFT_{i}={shift};\n' for i,shift in enumerate(shifts)),encoding='ascii')
    (out/'cases.json').write_text(json.dumps([dict(mode=c['mode'],shape=list(c['image'].shape),status=c['status'],
        compact=c.get('compact',False),trims=c.get('trims',[0,0,0,0])) for c in cases],indent=2),encoding='utf-8')
    print(json.dumps(dict(frames=len(cases),pixels=total,out=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quantized-dir',required=True);p.add_argument('--romdir',required=True)
    p.add_argument('--outdir',required=True);p.add_argument('--cache')
    p.add_argument('--sample-input',help='Frozen 512x512 HU NPY for self-contained handoff simulation without the patient cache')
    p.add_argument('--stress',action='store_true')
    main(p.parse_args())
