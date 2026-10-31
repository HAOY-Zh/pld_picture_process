"""Hash-bound local native packing. No CPU network operator is implemented."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np

_DLLS=[]


def compile_pack(folder, compiler=None):
    folder=Path(folder).resolve()
    compiler=compiler or shutil.which('g++')
    if not compiler:raise RuntimeError('Native g++ required; no slow-path fallback')
    source=folder/'nodule_quantized_pack.cpp'
    output=folder/('nodule_quantized_pack_native.dll'if os.name=='nt'else'nodule_quantized_pack_native.so')
    receipt=folder/'nodule_quantized_pack_build.json'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cmd=[str(compiler),'-O3','-std=c++11','-shared']
    cmd+=['-static-libgcc','-static-libstdc++']if os.name=='nt'else['-fPIC']
    if platform.machine().lower()in('armv7','armv7l'):cmd+=['-mcpu=cortex-a9','-mfpu=neon']
    cmd+=[str(source),'-o',str(output)]
    if output.exists()or receipt.exists():
        value=json.loads(receipt.read_text())
        if value['source_sha256']!=sha(source)or value['binary_sha256']!=sha(output)or value['command']!=cmd:
            raise ValueError('Native packing build identity differs; use a fresh directory')
    else:
        task_env=dict(os.environ);task_env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+task_env.get('PATH','')
        result=subprocess.run(cmd,env=task_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode:raise RuntimeError(result.stdout)
        receipt.write_text(json.dumps(dict(source_sha256=sha(source),binary_sha256=sha(output),command=cmd),indent=2))
    if os.name=='nt'and hasattr(os,'add_dll_directory'):
        _DLLS.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    return output


class QuantizedPacker:
    def __init__(self, library):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        array=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        self.lib.nodule_pack_quantized.argtypes=[array,ctypes.c_uint,ctypes.c_uint,array,ctypes.c_size_t]
        self.lib.nodule_pack_quantized.restype=ctypes.c_int

    def pack(self,image,destination):
        if (not isinstance(image,np.ndarray)or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[0]!=5
                or not image.flags.c_contiguous):raise ValueError('Need contiguous five-channel uint8 input')
        h,w=image.shape[1:]
        out=np.asarray(destination)
        if out.dtype not in(np.dtype('uint8'),np.dtype('uint64'))or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('Need writable contiguous separate output buffer')
        if np.shares_memory(image,out):raise ValueError('Input/output overlap')
        raw=out.view(np.uint8).reshape(-1)
        status=self.lib.nodule_pack_quantized(image,h,w,raw,raw.nbytes)
        if status:raise ValueError('Native quantized packing rejected '+str(status))
