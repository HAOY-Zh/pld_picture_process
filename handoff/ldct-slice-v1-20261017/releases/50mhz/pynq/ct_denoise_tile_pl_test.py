"""PYNQ CPU controls DMA; all selected denoising arithmetic executes in PL.

Use the generated deployment bundle. No CPU substitute for a failed PL run.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter,sleep
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ctdenoise.filters import pack_hu,unpack_hu,quantize_hu
from ctdenoise.pl_reference import fixed_filter,MODES
from ctdenoise.cnn_fixed import infer_fixed

HALO={0:0,1:2,2:2,3:5,4:4,5:2,6:3,7:3}


def cpu_gaussian_baseline(image,repeats):
    """Optimized separable C++ baseline with the same integer Gaussian output."""
    try:import cv2
    except ImportError:return None
    kernel=np.array([1,4,6,4,1],dtype=np.float64)
    timings=[]
    for repeat in range(repeats+1):
        start=perf_counter()
        a=quantize_hu(image).astype(np.float64)+32768
        numerator=cv2.sepFilter2D(a,cv2.CV_64F,kernel,kernel,borderType=cv2.BORDER_REPLICATE)
        q=((numerator+128)//256-32768).astype(np.int16)
        if repeat:timings.append((perf_counter()-start)*1000)
    if not np.array_equal(q,fixed_filter(image,1)):raise AssertionError('Optimized CPU Gaussian differs from integer contract')
    return dict(method='gaussian5',backend='OpenCV sepFilter2D float64 with exact integer Gaussian rounding',
        opencv=cv2.__version__,threads=cv2.getNumThreads(),bit_exact=True,
        median_frame_ms=float(np.median(timings)),repeats=repeats,
        timing='CPU conversion, separable filtering and integer output; excludes I/O/checking')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def check_bundle(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for relative,expected in manifest['files_sha256'].items():
        path=(root/relative).resolve()
        if root not in path.parents:raise ValueError('Unsafe bundle path')
        if sha(path)!=expected:raise ValueError('Bundle file changed: '+relative)
    return manifest


def wait_transfer(dma,timeout):
    deadline=perf_counter()+timeout
    while True:
        tx,rx=dma.mmio.read(4),dma.mmio.read(0x34)
        if (tx|rx)&0x70:raise RuntimeError('DMA error tx=0x%x rx=0x%x'%(tx,rx))
        if tx&2 and rx&2:break
        if perf_counter()>deadline:raise TimeoutError('DMA timeout; reset before freeing buffers')
        sleep(.0001)
    # PYNQ invalidates receive cache and marks the channel transfer complete.
    dma.sendchannel.wait();dma.recvchannel.wait()


def reset_dma(dma,timeout=2):
    dma.mmio.write(0,4);deadline=perf_counter()+timeout
    while dma.mmio.read(0)&4:
        if perf_counter()>deadline:raise TimeoutError('DMA reset failed; reboot board')
        sleep(.001)


def run_frame(image,mode,tile,dma,gpio,source,destination,timeout):
    halo=HALO[mode];a=quantize_hu(image);out=np.empty_like(a);tiles=0
    # This timer includes PS tile extraction/packing, cache sync, DMA, PL,
    # polling and reassembly. File I/O, plots and reference checking excluded.
    start=perf_counter()
    for y in range(0,a.shape[0],tile):
        for x in range(0,a.shape[1],tile):
            y2,x2=min(y+tile,a.shape[0]),min(x+tile,a.shape[1])
            ya,xa=max(0,y-halo),max(0,x-halo)
            ye,xe=min(a.shape[0],y2+halo),min(a.shape[1],x2+halo)
            patch=a[ya:ye,xa:xe];h,w=patch.shape;n=h*w
            if h>64 or w>64:raise ValueError('Tile plus overlap exceeds PL memory')
            source[:n]=pack_hu(patch).ravel()
            gpio.write(0,mode|(w<<4)|(h<<11))
            dma.recvchannel.transfer(destination[:n])
            dma.sendchannel.transfer(source[:n])
            wait_transfer(dma,timeout)
            if gpio.read(8):raise RuntimeError('PL protocol status 0x%x'%gpio.read(8))
            q=unpack_hu(np.asarray(destination[:n])).reshape(h,w)
            out[y:y2,x:x2]=q[y-ya:y2-ya,x-xa:x2-xa];tiles+=1
    return out,(perf_counter()-start)*1000,tiles


def run_frame_fast(image,mode,tile,dma,gpio,source,destination,timeout,compact=False,lifecycle=None):
    """Pack once per frame; optional CNN core-only protocol needs new RTL.

    Every arithmetic operation remains in PL. PS only packs/transfers/assembles.
    The timer includes frame packing, cache/DMA, PL, polling and assembly.
    """
    if compact and mode!=6:raise ValueError('Compact protocol is CNN only')
    halo=HALO[mode]
    # Native int16 HU is already exactly quantized; avoid a full float64/rint
    # round trip. Packing below still creates independent uint32 frame words.
    a=image if isinstance(image,np.ndarray) and image.dtype==np.int16 and image.ndim==2 and image.size else quantize_hu(image)
    out=np.empty_like(a);tiles=0
    start=perf_counter()
    words=a.view(np.uint16).astype(np.uint32)
    for y in range(0,a.shape[0],tile):
        for x in range(0,a.shape[1],tile):
            y2,x2=min(y+tile,a.shape[0]),min(x+tile,a.shape[1])
            ya,xa=max(0,y-halo),max(0,x-halo)
            ye,xe=min(a.shape[0],y2+halo),min(a.shape[1],x2+halo)
            h,w=ye-ya,xe-xa;n=h*w
            if h>64 or w>64:raise ValueError('Tile plus overlap exceeds PL memory')
            np.copyto(source[:n].reshape(h,w),words[ya:ye,xa:xe])
            cfg=mode|(w<<4)|(h<<11)
            if compact:
                left,right,top,bottom=x-xa,xe-x2,y-ya,ye-y2
                if any(v not in (0,3) for v in (left,right,top,bottom)):
                    raise ValueError('CNN core trim must be 0 or 3')
                cfg|=(1<<26)|(left<<18)|(right<<20)|(top<<22)|(bottom<<24)
                output_h,output_w=y2-y,x2-x
            else:output_h,output_w=h,w
            count=output_h*output_w
            gpio.write(0,cfg)
            if lifecycle is not None:lifecycle.before_dma()
            dma.recvchannel.transfer(destination[:count])
            dma.sendchannel.transfer(source[:n])
            wait_transfer(dma,timeout)
            received=np.asarray(destination[:count])
            if np.any(received>>16):raise RuntimeError('Nonzero upper output bits')
            # Native little-endian ARM/Windows: low halfword is signed HU.
            # A strided view avoids the old uint64/uint32/uint16 copies.
            if sys.byteorder!='little':raise RuntimeError('Fast packing requires little-endian host')
            q=received.view(np.uint16)[::2].view(np.int16).reshape(output_h,output_w)
            out[y:y2,x:x2]=q if compact else q[y-ya:y2-ya,x-xa:x2-xa]
            tiles+=1
    # RTL protocol errors are sticky until reset, so one frame-end read suffices.
    status=gpio.read(8)
    if status:raise RuntimeError('PL protocol status 0x%x'%status)
    return out,(perf_counter()-start)*1000,tiles


def main(args):
    root=Path(args.bundle).resolve();manifest=check_bundle(root)
    out=Path(args.outdir)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose new/empty output')
    if any(mode not in MODES for mode in args.modes):raise ValueError('Unknown PL mode')
    if not args.modes:raise ValueError('Choose at least one PL mode')
    max_tile=min(64-2*HALO[mode] for mode in args.modes)
    if not 1<=args.tile<=max_tile or args.repeats<1 or args.timeout<=0:
        raise ValueError('Invalid tile/repeats/timeout; selected modes allow output tile <=%d'%max_tile)
    if args.cnn_compact:
        if args.modes!=[6] or args.ps_path!='fast':raise ValueError('Compact requires --modes 6 --ps-path fast')
        if not manifest.get('cnn_compact_protocol'):
            raise ValueError('This bundle does not declare compact CNN RTL; never send compact commands to an old bitstream')
    a=np.load(args.input if args.input else root/'samples/C067_input_hu.npy',allow_pickle=False)
    a=quantize_hu(a)
    reference_path=Path(args.reference) if args.reference else (root/'samples/C067_reference_hu.npy' if not args.input else None)
    normal=np.load(reference_path,allow_pickle=False) if reference_path else None
    if normal is not None and (normal.shape!=a.shape or not np.isfinite(normal).all()):raise ValueError('Reference shape/data mismatch')
    with np.load(root/'model/weights_integer.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    spec=json.loads((root/'model/quantization.json').read_text())
    from pynq import Overlay,allocate,Clocks
    overlay=Overlay(str(root/'overlay/ct_denoise_tile.bit'))
    if not {'axi_dma_0','axi_gpio_0'}.issubset(overlay.ip_dict):raise RuntimeError('Wrong overlay IPs')
    pl_clock=float(Clocks.fclk0_mhz)
    if abs(pl_clock-float(manifest['clock_mhz']))>.5:raise RuntimeError('PL clock differs from timed implementation')
    dma,gpio=overlay.axi_dma_0,overlay.axi_gpio_0
    source=allocate(shape=(4096,),dtype=np.uint32);destination=None;dma_may_access=False
    images={'Input':a};records=[]
    try:
        destination=allocate(shape=(4096,),dtype=np.uint32)
        gpio.write(0,0x80000000);sleep(.001);gpio.write(0,0)
        for mode in args.modes:
            ref_start=perf_counter()
            reference=infer_fixed(a,arrays,spec)[0] if mode==6 else fixed_filter(a,mode)
            ref_ms=(perf_counter()-ref_start)*1000
            timings=[]
            for repeat in range(args.repeats+1):
                dma_may_access=True
                if args.ps_path=='fast':
                    q,ms,tiles=run_frame_fast(a,mode,args.tile,dma,gpio,source,destination,args.timeout,args.cnn_compact)
                else:q,ms,tiles=run_frame(a,mode,args.tile,dma,gpio,source,destination,args.timeout)
                dma_may_access=False
                if not np.array_equal(q,reference):
                    raise AssertionError('%s: %d mismatches, max error %d HU'%(MODES[mode],
                        np.count_nonzero(q!=reference),np.max(np.abs(q.astype(np.int64)-reference))))
                if repeat:timings.append(ms)
            if mode!=0:images[MODES[mode]]=q
            row=dict(mode=mode,method=MODES[mode],bit_exact=True,tile_output=args.tile,tiles=tiles,
                ps_path=args.ps_path,cnn_compact=args.cnn_compact,
                median_frame_ms=float(np.median(timings)),p95_frame_ms=float(np.percentile(timings,95)),
                fps=1000/float(np.median(timings)),cpu_integer_reference_ms=ref_ms,repeats=args.repeats)
            if normal is not None:
                row['rmse_hu_vs_normal']=float(np.sqrt(np.mean((q.astype(float)-normal.astype(float))**2)))
            records.append(row);print(json.dumps(row),flush=True)
        if normal is not None:images['NormalDose']=normal
        cpu_baseline=cpu_gaussian_baseline(a,args.repeats) if 1 in args.modes else None
        if cpu_baseline:print(json.dumps(dict(cpu_optimized_baseline=cpu_baseline)),flush=True)
        out.mkdir(parents=True,exist_ok=True)
        for name,image in images.items():np.save(out/(name+'.npy'),image,allow_pickle=False)
        with (out/'board_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        import pynq
        evidence=dict(status='PASS',actual_PL_execution=True,python=platform.python_version(),
            pl_clock_mhz=pl_clock,cpu_clock_mhz=float(Clocks.cpu_mhz),
            pynq=str(pynq.__version__),platform=platform.platform(),input_sha256=sha(args.input if args.input else root/'samples/C067_input_hu.npy'),
            reference_sha256=sha(reference_path) if reference_path else None,
            bit_sha256=sha(root/'overlay/ct_denoise_tile.bit'),rows=records,
            cpu_optimized_gaussian=cpu_baseline,
            timing='PS packing, DMA/cache, PL compute, polling, reassembly; excludes file I/O, display and integer reference.',
            cpu_reference_note='Correctness reference, not a tuned ARM implementation; not a fair optimized CPU speedup claim.')
        (out/'board_result.json').write_text(json.dumps(evidence,indent=2))
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            n=len(images);cols=3;rows=(n+cols-1)//cols
            fig,axes=plt.subplots(rows,cols,figsize=(12,4*rows),squeeze=False)
            for ax in axes.flat:ax.axis('off')
            for ax,(name,image) in zip(axes.flat,images.items()):
                ax.imshow(image,cmap='gray',vmin=-1000,vmax=400);ax.set_title(name)
            fig.tight_layout();fig.savefig(out/'actual_board_outputs.png',dpi=130);plt.close(fig)
        except ImportError:pass
    finally:
        if dma_may_access:
            # If reset fails, do not free memory that the DMA could still access.
            try:reset_dma(dma)
            except Exception:
                print('DMA reset failed. Keep buffers allocated and reboot the board.',file=sys.stderr)
                raise
        if destination is not None:destination.freebuffer()
        source.freebuffer()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--input');parser.add_argument('--reference');parser.add_argument('--outdir',required=True)
    parser.add_argument('--modes',type=int,nargs='+',default=list(MODES))
    parser.add_argument('--tile',type=int,default=48)
    parser.add_argument('--ps-path',choices=['legacy','fast'],default='legacy')
    parser.add_argument('--cnn-compact',action='store_true',help='Requires matching new RTL/bundle; return only CNN tile cores')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=10.)
    main(parser.parse_args())
