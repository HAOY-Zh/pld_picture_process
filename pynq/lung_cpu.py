"""Native exact-INT8 CPU U-Net wrapper; no Torch/ONNX runtime dependency."""
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

from lung_continuous import validate_model

FIELDS=21
_DLL_DIRECTORIES=[]


def compile_native(folder,compiler=None):
    folder=Path(folder).resolve(); compiler=compiler or shutil.which('g++')
    if not compiler: raise RuntimeError('g++ is required; do not silently replace with Python loops')
    target=folder/('lung_cpu_native.dll' if os.name=='nt' else 'lung_cpu_native.so')
    command=[str(compiler),'-O3','-std=c++11','-shared','-fopenmp']
    if os.name=='nt': command+=['-static-libgcc','-static-libstdc++']
    else: command+=['-fPIC']
    machine=platform.machine().lower()
    if machine in ('armv7l','armv7'): command+=['-mcpu=cortex-a9','-mfpu=neon']
    command+=[str(folder/'lung_cpu_native.cpp'),'-o',str(target)]
    environment=dict(os.environ); environment['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+environment.get('PATH','')
    sources={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['lung_cpu_native.cpp','lung_cpu_kernel.h']}
    metadata=folder/'lung_cpu_native_build.json'
    if os.name=='nt' and hasattr(os,'add_dll_directory'):
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    if target.exists() or metadata.exists():
        if not target.is_file() or not metadata.is_file(): raise FileExistsError('Incomplete previous native build; use a fresh folder')
        prior=json.loads(metadata.read_text(encoding='utf-8'))
        if (prior['sources_sha256']!=sources or prior['command']!=command or
                prior['binary_sha256']!=hashlib.sha256(target.read_bytes()).hexdigest()):
            raise ValueError('Existing compiled binary/build identity changed; use a fresh folder')
        return target,dict(prior,reused_verified_binary=True)
    start=time.perf_counter()
    process=subprocess.run(command,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError('Native compiler failed:\n'+process.stdout)
    version=subprocess.run([str(compiler),'--version'],env=environment,text=True,stdout=subprocess.PIPE,check=True).stdout.splitlines()[0]
    record=dict(command=command,compiler_version=version,build_ms=(time.perf_counter()-start)*1000,output=process.stdout,
                sources_sha256=sources,binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with metadata.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
    return target,record


def encode_model(model,parameters):
    validate_model(model,parameters)
    descriptors=[]; weights=[]; biases=[]; mults=[]; shifts=[]; ids={'input':-1}
    shapes={'input':model['input_shape']}; weight_offset=param_offset=0
    for index,node in enumerate(model['nodes']):
        ca,h,w=shapes[node['sources'][0]]; c,oh,ow=node['shape']; conv=node['kind']=='conv'
        q=node.get('quant',{}); row=node.get('layer',{})
        descriptors.append([1 if conv else (2 if node['kind']=='pool' else 3),ids[node['sources'][0]],
                            ids[node['sources'][1]] if len(node['sources'])==2 else -1,
                            ca,h,w,c,oh,ow,row.get('kernel',0),row.get('padding',0),
                            row['input_zero'] if conv else q['za'],node['zero'],int(row.get('relu',False)),
                            q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0),q.get('zb',0),
                            weight_offset,param_offset])
        if conv:
            prefix=row['prefix']; value=parameters[prefix+'_weight'].reshape(-1)
            weights.append(value); weight_offset+=len(value)
            for bank,key in [(biases,'bias'),(mults,'multiplier'),(shifts,'shift')]: bank.append(parameters[prefix+'_'+key])
            param_offset+=c
        ids[node['name']]=index; shapes[node['name']]=node['shape']
    return (np.ascontiguousarray(descriptors,dtype=np.int32),np.concatenate(weights).astype(np.int8),
            *[np.concatenate(bank).astype(np.int32) for bank in [biases,mults,shifts]])


class NativeUNet:
    def __init__(self,library,model,parameters,threads=1):
        self.model=model; self.library=ctypes.CDLL(str(Path(library).resolve())); self.handle=None
        lib=self.library; array=lambda dt:np.ctypeslib.ndpointer(dtype=dt,flags='C_CONTIGUOUS')
        lib.lung_cpu_error.restype=ctypes.c_char_p
        lib.lung_cpu_create.restype=ctypes.c_void_p
        lib.lung_cpu_create.argtypes=[array(np.int32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                                     array(np.int8),ctypes.c_int,array(np.int32),array(np.int32),array(np.int32),ctypes.c_int,ctypes.c_int]
        lib.lung_cpu_forward.argtypes=[ctypes.c_void_p,array(np.uint8),ctypes.c_size_t,array(np.uint8),ctypes.c_size_t]
        lib.lung_cpu_copy_node.argtypes=[ctypes.c_void_p,ctypes.c_int,array(np.uint8),ctypes.c_size_t]
        lib.lung_cpu_destroy.argtypes=[ctypes.c_void_p]
        lib.lung_cpu_requant.argtypes=[ctypes.c_int32,ctypes.c_int32,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        lib.lung_cpu_requant.restype=ctypes.c_uint8
        d,weights,bias,mult,shift=encode_model(model,parameters)
        self.handle=lib.lung_cpu_create(d,len(d),*model['input_shape'],weights,len(weights),bias,mult,shift,len(bias),threads)
        if not self.handle: raise RuntimeError(lib.lung_cpu_error().decode())
        self.output=np.empty(model['nodes'][-1]['shape'],np.uint8)
        self.features=lib.lung_cpu_features(); self.threads=threads

    def forward(self,image):
        if self.handle is None: raise RuntimeError('CPU network already closed')
        if image.dtype!=np.uint8 or list(image.shape)!=self.model['input_shape'] or not image.flags.c_contiguous:
            raise ValueError('Need contiguous CHW UINT8 input codes')
        if self.library.lung_cpu_forward(self.handle,image,image.size,self.output,self.output.size):
            raise RuntimeError(self.library.lung_cpu_error().decode())
        return self.output  # Reused output, overwritten by the next call.

    def node_codes(self,index):
        if self.handle is None: raise RuntimeError('CPU network already closed')
        if not 0<=index<len(self.model['nodes']): raise ValueError('Invalid trace node')
        output=np.empty(self.model['nodes'][index]['shape'],np.uint8)
        if self.library.lung_cpu_copy_node(self.handle,index,output,output.size):
            raise RuntimeError(self.library.lung_cpu_error().decode())
        return output

    def close(self):
        if self.handle is not None: self.library.lung_cpu_destroy(self.handle); self.handle=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
