"""Original FP32 weights, fused BN, native NEON/OpenMP, no board Torch needed."""
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

_DLL_DIRS=[]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_fp32(folder,compiler=None):
    folder=Path(folder).resolve();compiler=compiler or shutil.which('g++')
    if not compiler: raise RuntimeError('Need g++, no slow Python fallback')
    output=folder/('lung_fp32_native.dll' if os.name=='nt' else 'lung_fp32_native.so')
    metadata=folder/'lung_fp32_build.json'
    command=[str(compiler),'-O3','-std=c++11','-shared','-fopenmp','-ffp-contract=off']
    command+=['-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC']
    if platform.machine().lower() in ('armv7','armv7l'): command+=['-mcpu=cortex-a9','-mfpu=neon']
    # Explicit NEON kernel; no fast-math needed on the PC or hidden FP16/INT8.
    command += [str(folder/'lung_fp32_native.cpp'),'-o',str(output)]
    sources={n:sha(folder/n) for n in ['lung_fp32_native.cpp','lung_fp32_kernel.h']}
    env=dict(os.environ);env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
    if os.name=='nt' and hasattr(os,'add_dll_directory'): _DLL_DIRS.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    if output.exists() or metadata.exists():
        if not output.is_file() or not metadata.is_file(): raise FileExistsError('Incomplete FP32 build; use fresh folder')
        old=json.loads(metadata.read_text(encoding='utf-8'))
        if old['command']!=command or old['sources_sha256']!=sources or old['binary_sha256']!=sha(output): raise ValueError('FP32 binary identity changed')
        return output,dict(old,reused=True)
    start=time.perf_counter();p=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if p.returncode: raise RuntimeError('FP32 compilation failed:\n'+p.stdout)
    record=dict(command=command,sources_sha256=sources,binary_sha256=sha(output),build_ms=(time.perf_counter()-start)*1000,
        compiler_version=subprocess.check_output([str(compiler),'--version'],env=env,text=True).splitlines()[0],output=p.stdout)
    with metadata.open('x',encoding='utf-8') as f: json.dump(record,f,indent=2)
    return output,record


class NativeFP32:
    def __init__(self,library,descriptors,weights,bias,threads=2):
        self.lib=ctypes.CDLL(str(Path(library).resolve()));self.handle=None
        lib=self.lib;arr=lambda dt:np.ctypeslib.ndpointer(dtype=dt,flags='C_CONTIGUOUS')
        lib.lung_fp32_error.restype=ctypes.c_char_p
        lib.lung_fp32_create.restype=ctypes.c_void_p
        lib.lung_fp32_create.argtypes=[arr(np.int32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,arr(np.float32),ctypes.c_int,arr(np.float32),ctypes.c_int,ctypes.c_int]
        lib.lung_fp32_forward.argtypes=[ctypes.c_void_p,arr(np.float32),ctypes.c_size_t,arr(np.float32),ctypes.c_size_t]
        lib.lung_fp32_trace.argtypes=[ctypes.c_void_p,ctypes.c_int,arr(np.float32),ctypes.c_size_t]
        lib.lung_fp32_destroy.argtypes=[ctypes.c_void_p]
        d=np.ascontiguousarray(descriptors,dtype=np.int32)
        if d.ndim!=2 or d.shape[1]!=21 or not len(d): raise ValueError('Need N x 21 descriptors')
        weights=np.ascontiguousarray(weights,dtype=np.float32);bias=np.ascontiguousarray(bias,dtype=np.float32)
        self.shapes=[tuple(map(int,r[6:9])) for r in d];self.shape=tuple(map(int,d[0,3:6]))
        self.handle=lib.lung_fp32_create(d,len(d),*self.shape,weights,weights.size,bias,bias.size,threads)
        if not self.handle: raise RuntimeError(lib.lung_fp32_error().decode())
        self.output=np.empty(self.shapes[-1],np.float32);self.features=lib.lung_fp32_features()

    def forward(self,image):
        if not self.handle: raise RuntimeError('FP32 network closed')
        if image.dtype!=np.float32 or tuple(image.shape)!=self.shape or not image.flags.c_contiguous: raise ValueError('Need contiguous CHW FP32 input')
        if self.lib.lung_fp32_forward(self.handle,image,image.size,self.output,self.output.size): raise RuntimeError(self.lib.lung_fp32_error().decode())
        return self.output

    def trace(self,index):
        if not self.handle or not 0<=index<len(self.shapes): raise ValueError('Invalid FP32 trace')
        out=np.empty(self.shapes[index],np.float32)
        if self.lib.lung_fp32_trace(self.handle,index,out,out.size): raise RuntimeError(self.lib.lung_fp32_error().decode())
        return out

    def close(self):
        if self.handle: self.lib.lung_fp32_destroy(self.handle);self.handle=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
