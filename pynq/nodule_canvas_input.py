"""Exact native INT16 HU -> lowered CHW/HWC8, with independent build receipt."""
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


def compile_input(folder,compiler=None):
    folder=Path(folder).resolve();compiler=compiler or shutil.which('g++')
    if not compiler:raise RuntimeError('Need local g++')
    src=folder/'nodule_canvas_input.cpp'
    target=folder/('nodule_canvas_pack_native.dll' if os.name=='nt' else 'nodule_canvas_pack_native.so')
    meta=folder/'nodule_canvas_pack_build.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    command=[str(compiler),'-O3','-std=c++11','-shared']
    command+=['-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC']
    if platform.machine().lower() in ('armv7l','armv7'):command+=['-mcpu=cortex-a9','-mfpu=neon']
    command += [str(src),'-o',str(target)]
    if os.name=='nt' and hasattr(os,'add_dll_directory'):_DLLS.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    if target.exists() or meta.exists():
        d=json.loads(meta.read_text())
        if d['source_sha256']!=sha(src) or d['binary_sha256']!=sha(target) or d['command']!=command:
            raise ValueError('Changed native pack build; use fresh folder')
        return target,d
    env=dict(os.environ);env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
    tick=time.perf_counter();p=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(p.stdout)
    d=dict(source_sha256=sha(src),binary_sha256=sha(target),command=command,build_ms=(time.perf_counter()-tick)*1000,output=p.stdout)
    with meta.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
    return target,d


class NativeCanvasInput:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(Path(library).resolve()));self.chw=None
        a=np.arange(-32768,32768,dtype=np.float64)
        self.lut=np.rint((a.clip(-1000,400)+1000)*(255./1400.)).astype(np.uint8)
        a16=np.ctypeslib.ndpointer(dtype=np.int16);u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        self.lib.nodule_canvas_pack.argtypes=[a16,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,
            ctypes.c_uint,ctypes.c_uint,u8,u8,ctypes.c_size_t,ctypes.c_int]
        self.lib.nodule_canvas_pack.restype=ctypes.c_int

    def pack(self,image,destination,hwc8=False):
        a=np.asarray(image);out=np.asarray(destination)
        if (a.dtype!=np.int16 or a.ndim!=3 or a.shape[0]!=5
                or any(v<128 or v>1024 or v%8 for v in a.shape[1:]) or any(s<=0 or s%2 for s in a.strides)):
            raise ValueError('Need positive-stride INT16 five-plane aligned canvas')
        h,w=a.shape[1:];count=h//2*(w//2)*(48 if hwc8 else 45)
        if (out.dtype not in (np.dtype('uint8'),np.dtype('uint64')) or not out.flags.writeable
                or not out.flags.c_contiguous or out.nbytes<count or np.shares_memory(a,out)):
            raise ValueError('Need separate contiguous writable output of sufficient size')
        raw=out.view(np.uint8).reshape(-1)
        result=self.lib.nodule_canvas_pack(a,*(s//2 for s in a.strides),h,w,self.lut,raw,len(raw),int(hwc8))
        if result:raise RuntimeError('Native canvas pack failed '+str(result))
        return destination

    def lower(self,image):
        a=np.asarray(image)
        if a.ndim!=3:raise ValueError('Need five-plane canvas')
        shape=(45,a.shape[1]//2,a.shape[2]//2)
        if self.chw is None or self.chw.shape!=shape:self.chw=np.empty(shape,np.uint8)
        self.pack(a,self.chw,False);return self.chw
