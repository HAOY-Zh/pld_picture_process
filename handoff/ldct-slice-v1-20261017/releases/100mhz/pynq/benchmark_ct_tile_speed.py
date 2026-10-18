"""Board-only speed comparison: same CNN weights, four transfer strategies."""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import sleep
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ct_denoise_tile_pl_test import (check_bundle,sha,run_frame,run_frame_fast,reset_dma)
from ctdenoise.cnn_fixed import load_quantized,infer_fixed


def main(a):
    root=Path(a.bundle).resolve();manifest=check_bundle(root)
    if not manifest.get('cnn_compact_protocol'):raise ValueError('Matching compact CNN bundle required')
    if a.repeats<3:raise ValueError('At least 3 measured repeats required')
    out=Path(a.outdir)
    if out.exists():raise ValueError('Choose a new result directory')
    stages=[('legacy48',48,False,False),('fast48',48,True,False),
            ('fast58',58,True,False),('compact58',58,True,True)]
    arrays,spec=load_quantized(root/'model')
    from pynq import Overlay,allocate,Clocks
    import pynq
    overlay=Overlay(str(root/'overlay/ct_denoise_tile.bit'))
    if abs(float(Clocks.fclk0_mhz)-manifest['clock_mhz'])>.5:raise ValueError('Clock differs from timed implementation')
    dma,gpio=overlay.axi_dma_0,overlay.axi_gpio_0
    source=allocate(shape=(4096,),dtype=np.uint32);destination=None;active=False
    out.mkdir(parents=True)
    rows=[];case_metrics=[]
    try:
        destination=allocate(shape=(4096,),dtype=np.uint32)
        gpio.write(0,0x80000000);sleep(.001);gpio.write(0,0)
        for patient in ['C067','C077']:
            image=np.load(root/f'samples/{patient}_input_hu.npy',allow_pickle=False)
            # This slow numerical oracle is computed once, outside every timer.
            print('Checking integer reference',patient,flush=True)
            reference=infer_fixed(image,arrays,spec)[0]
            timings={name:[] for name,_,_,_ in stages};outputs={};counts={}
            for repeat in range(a.repeats+1):
                # Rotate order to limit thermal/order bias; pass 0 is warmup.
                ordered=stages[repeat%4:]+stages[:repeat%4]
                for name,tile,fast,compact in ordered:
                    active=True
                    if fast:q,ms,tiles=run_frame_fast(image,6,tile,dma,gpio,source,destination,a.timeout,compact)
                    else:q,ms,tiles=run_frame(image,6,tile,dma,gpio,source,destination,a.timeout)
                    active=False
                    if not np.array_equal(q,reference):
                        raise AssertionError(f'{patient}/{name}: output differs from audited integer CNN')
                    outputs[name]=q.copy();counts[name]=tiles
                    if repeat:timings[name].append(ms)
                    print(json.dumps(dict(patient=patient,stage=name,repeat=repeat,warmup=repeat==0,frame_ms=ms,bit_exact=True)),flush=True)
            normal=np.load(root/f'samples/{patient}_reference_hu.npy',allow_pickle=False)
            case_metrics.append(dict(patient=patient,rmse_hu=float(np.sqrt(np.mean((reference.astype(float)-normal)**2))),
                                     input_sha256=sha(root/f'samples/{patient}_input_hu.npy')))
            baseline=float(np.median(timings['legacy48']))
            for name,tile,_,_ in stages:
                median=float(np.median(timings[name]))
                rows.append(dict(patient=patient,stage=name,tile=tile,tiles=counts[name],repeats=a.repeats,
                    median_frame_ms=median,p95_frame_ms=float(np.percentile(timings[name],95)),
                    fps=1000/median,speedup_vs_legacy48=baseline/median,bit_exact=True,trials_ms=timings[name]))
                np.save(out/f'{patient}_{name}.npy',outputs[name],allow_pickle=False)
    finally:
        if active:reset_dma(dma)
        if destination is not None:destination.freebuffer()
        source.freebuffer()
    summary=[]
    for name,_,_,_ in stages:
        values=[r for r in rows if r['stage']==name]
        summary.append(dict(stage=name,mean_frame_ms=float(np.mean([r['median_frame_ms'] for r in values])),
                            mean_speedup_vs_legacy48=float(np.mean([r['speedup_vs_legacy48'] for r in values]))))
    result=dict(actual_PL_execution=True,bit_sha256=sha(root/'overlay/ct_denoise_tile.bit'),
        model_checkpoint_sha256=spec['checkpoint_sha256'],pl_clock_mhz=float(Clocks.fclk0_mhz),
        cpu_clock_mhz=float(Clocks.cpu_mhz),pynq=str(pynq.__version__),python=platform.python_version(),
        all_outputs_bit_exact=True,summary=summary,rows=rows,case_metrics=case_metrics,
        scope='Two demonstration slices; same frozen model in all stages, identified by checkpoint hash. Not a full-dataset quality benchmark.',
        timing='Frame packing, DMA/cache, PL compute, polling and assembly; excludes files, reference computation and display.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='trials_ms'])
        w.writeheader();w.writerows([{k:v for k,v in r.items() if k!='trials_ms'} for r in rows])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--outdir',required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--timeout',type=float,default=10)
    main(p.parse_args())
