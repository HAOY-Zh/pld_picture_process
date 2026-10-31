"""Native XY interpolation; original NumPy Z mixing and HU rounding retained."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
_DLLS=[]


def compile_sampling(folder,compiler=None):
    folder=Path(folder).resolve();compiler=compiler or shutil.which('g++')
    if not compiler:raise RuntimeError('Need local g++')
    src=folder/'nodule_sampling_native.cpp'
    binary=folder/('nodule_xy_sampling_native.dll' if os.name=='nt' else 'nodule_xy_sampling_native.so')
    meta=folder/'nodule_xy_sampling_build.json'
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    command=[str(compiler),'-O3','-std=c++11','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math']
    command+=['-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC']
    if platform.machine().lower() in ('armv7l','armv7'):command+=['-mcpu=cortex-a9','-mfpu=neon']
    command += [str(src),'-o',str(binary)]
    if os.name=='nt' and hasattr(os,'add_dll_directory'):_DLLS.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    if binary.exists() or meta.exists():
        receipt=json.loads(meta.read_text())
        if receipt['source_sha256']!=digest(src) or receipt['binary_sha256']!=digest(binary) or receipt['command']!=command:
            raise ValueError('Changed sampling build; use fresh folder')
        return binary,receipt
    env=dict(os.environ);env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
    tick=time.perf_counter();p=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Native sampling compile failed:\n'+p.stdout)
    receipt=dict(source_sha256=digest(src),binary_sha256=digest(binary),command=command,build_ms=(time.perf_counter()-tick)*1000,output=p.stdout)
    with meta.open('x') as f:json.dump(receipt,f,indent=2)
    return binary,receipt


class NativeXY:
    def __init__(self,binary,threads=2):
        if threads not in (1,2):raise ValueError('Use one or two sampling threads')
        self.threads=threads;self.lib=ctypes.CDLL(str(Path(binary).resolve()))
        a16=np.ctypeslib.ndpointer(dtype=np.int16)
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        self.lib.nodule_resample_xy.argtypes=[a16,*([ctypes.c_size_t]*3),*([ctypes.c_uint]*3),i32,ctypes.c_uint,f64,ctypes.c_uint,f64,ctypes.c_uint,f32,ctypes.c_int]
        self.lib.nodule_resample_xy.restype=ctypes.c_int

    def resample(self,hu,planes,y,x):
        hu=np.asarray(hu);y=np.ascontiguousarray(y,np.float64);x=np.ascontiguousarray(x,np.float64)
        raw=np.asarray(planes)
        if (hu.dtype!=np.int16 or hu.ndim!=3 or not all(hu.shape) or any(s<=0 or s%2 for s in hu.strides)
            or raw.ndim!=1 or not raw.size or not np.isfinite(raw).all() or not np.equal(raw,np.rint(raw)).all()
            or min(raw)<0 or max(raw)>=len(hu) or y.ndim!=1 or x.ndim!=1 or not len(y) or not len(x)
            or not np.isfinite(y).all() or not np.isfinite(x).all()):raise ValueError('Invalid sampling input')
        ids=np.ascontiguousarray(raw,np.int32);out=np.empty((len(ids),len(y),len(x)),np.float32)
        status=self.lib.nodule_resample_xy(hu,*(s//2 for s in hu.strides),*hu.shape[1:],hu.shape[0],ids,len(ids),y,len(y),x,len(x),out,self.threads)
        if status:raise RuntimeError('Native resample rejected: '+str(status))
        return out


class NativePlaneSampler:
    def __init__(self,hu,spacing,jobs,config,native):
        spacing=np.asarray(spacing,float)
        if not jobs or spacing.shape!=(3,) or (spacing<=0).any() or not np.isfinite(spacing).all():raise ValueError('Need valid spacing/jobs')
        zm=float(jobs[0][0]);self.config=config
        if any(float(j[0])!=zm or int(j[1])!=j[1] or int(j[2])!=j[2] for j in jobs):raise ValueError('Mixed/fractional plane jobs')
        self.jobs={tuple(j) for j in jobs};self.y0=int(min(j[1] for j in jobs));self.x0=int(min(j[2] for j in jobs))
        y1=int(max(j[1] for j in jobs))+config.patch;x1=int(max(j[2] for j in jobs))+config.patch
        y=(self.y0+np.arange(y1-self.y0))*config.xy_mm/spacing[1]
        x=(self.x0+np.arange(x1-self.x0))*config.xy_mm/spacing[2]
        indices=[];zplan=[]
        for offset in config.context_mm:
            z=np.clip((zm+offset)/spacing[0],0,len(hu)-1);lo,hi=int(np.floor(z)),int(np.ceil(z));alpha=z-lo
            for index in (lo,hi):
                if index not in indices:indices.append(index)
            zplan.append((lo,hi,alpha))
        sampled=native.resample(hu,indices,y,x);planes={index:sampled[i] for i,index in enumerate(indices)};context=[]
        for lo,hi,alpha in zplan:
            a=planes[lo]
            if hi!=lo:a=a*(1-alpha)+planes[hi]*alpha
            context.append(np.rint(a).clip(-32768,32767).astype(np.int16))
        self.context=np.stack(context);self.context.flags.writeable=False
        self.resampled_native_planes=len(planes);self.resampled_pixels=len(planes)*len(y)*len(x)

    def patch(self,job):
        if tuple(job) not in self.jobs:raise ValueError('Tile outside sampler')
        y=int(job[1])-self.y0;x=int(job[2])-self.x0;p=self.config.patch
        return self.context[:,y:y+p,x:x+p]
