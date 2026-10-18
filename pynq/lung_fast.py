"""Prepared, persistent-buffer runtime for the UNCHANGED dual-engine bit/ABI.

All 21 operators still run in PL. No CPU inference fallback. Intermediate
buffers are device-owned: never flush stale ARM payloads between PL producers
and consumers. Only CPU-written input is flushed each frame. CPU reads first
invalidate. Every allocation is checked at frame end, not after every command.
"""
import copy
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

from lung_axil import LungAxil
from lung_axil_smoke import feature_words
from lung_continuous import validate_model, word_count, unpack_words, GUARD_WORDS
from lung_model_layers_test import lane_parameters, SENTINEL

_UNSAFE = []
_DLL_DIRECTORIES = []


def prepare_block(arrays, block, row):
    """OIHW -> packet[ky,kx,cin-block][output-lane,input-lane], once at setup."""
    weight = arrays['weight']; cout,cin,k,_ = weight.shape
    count = min(8,cout-block*8)
    if block < 0 or count <= 0: raise ValueError('Invalid output block')
    padded = np.zeros((8,(cin+7)//8*8,k,k),np.int8)
    padded[:count,:cin] = weight[block*8:block*8+count]
    data = padded.reshape(8,(cin+7)//8,8,k,k).transpose(3,4,1,0,2).copy()
    words = data.reshape(-1,64).view('<u4').reshape(-1,16).copy()
    (bias,mult,shift),keep = lane_parameters(arrays,block)
    params = np.asarray([v & 0xffffffff for v in bias]+mult+shift,dtype=np.uint32)
    # Cached integers are only for a diagnostic Python loader, never repacked per frame.
    packets = tuple(int.from_bytes(v.tobytes(),'little') for v in words)
    words.flags.writeable = params.flags.writeable = False
    return dict(words=words,parameters=params,packets=packets,bias=bias,mult=mult,shift=shift,
                iz=row['input_zero'],oz=row['output_zero'],keep=keep,relu=row['relu'])


def compile_io(folder, compiler=None):
    folder=Path(folder).resolve(); compiler=compiler or shutil.which('g++')
    if not compiler: raise RuntimeError('Need local g++; no network downloads')
    target=folder/('lung_fast_io.dll' if os.name=='nt' else 'lung_fast_io.so')
    command=[str(compiler),'-O3','-std=c++11','-shared']
    command += ['-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC']
    if platform.machine().lower() in ('armv7l','armv7'): command+=['-mcpu=cortex-a9','-mfpu=neon']
    command += [str(folder/'lung_fast_io.cpp'),'-o',str(target)]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources={n:digest(folder/n) for n in ['lung_fast_io.cpp','lung_fast_io.h']}
    metadata=folder/'lung_fast_io_build.json'
    if os.name=='nt' and hasattr(os,'add_dll_directory'):
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))
    if target.exists() or metadata.exists():
        if not target.is_file() or not metadata.is_file(): raise FileExistsError('Incomplete native build; use a fresh folder')
        prior=json.loads(metadata.read_text(encoding='utf-8'))
        if prior['sources_sha256']!=sources or prior['command']!=command or prior['binary_sha256']!=digest(target):
            raise ValueError('Native build identity changed; use a fresh folder')
        return target,dict(prior,reused_verified_binary=True)
    env=dict(os.environ); env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
    start=time.perf_counter()
    proc=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if proc.returncode: raise RuntimeError('MMIO loader compilation failed:\n'+proc.stdout)
    record=dict(command=command,sources_sha256=sources,binary_sha256=digest(target),
                build_ms=(time.perf_counter()-start)*1000,output=proc.stdout)
    with metadata.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
    return target,record


class NativeLoader:
    def __init__(self, library, mmio):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        registers=getattr(mmio,'array',None)
        if (not isinstance(registers,np.ndarray) or registers.dtype!=np.uint32
                or registers.ndim!=1 or not registers.flags.c_contiguous
                or not registers.flags.writeable or len(registers)<0x1a0//4):
            raise ValueError('Native loader requires a writable local PYNQ uint32 MMIO mapping')
        self.registers=registers; self.mmio=mmio
        arr=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
        self.lib.lung_fast_load.argtypes=[arr,ctypes.c_size_t,arr,ctypes.c_uint,arr,ctypes.c_uint32,ctypes.c_double]
        self.lib.lung_fast_load.restype=ctypes.c_int

    def load(self, prepared, timeout):
        p=prepared; quant=p['keep'] | p['iz']<<8 | p['oz']<<16 | int(p['relu'])<<24
        result=self.lib.lung_fast_load(self.registers,len(self.registers),p['words'],len(p['words']),
                                      p['parameters'],quant,timeout)
        if result==-4: raise TimeoutError('Native weight load timeout; shared board recovery required')
        if result: raise RuntimeError('Native weight load rejected/incomplete (code {})'.format(result))


class PreparedConv(LungAxil):
    def __init__(self, mmio, native_loader=None, **kwargs):
        super().__init__(mmio,**kwargs); self.native_loader=native_loader

    def load_prepared(self,p):
        if self.native_loader is None:
            return self.load_weights(p['packets'],p['bias'],p['mult'],p['shift'],p['iz'],p['oz'],p['keep'],p['relu'])
        self._idle(); self._resident=None
        try:
            self.native_loader.load(p,self.timeout)
            self._resident=dict(packet_count=len(p['words']),keep=p['keep'])
        except BaseException:
            self._poisoned=True
            raise


class FastUNet:
    """Single-owner runtime. Explicit close only when both engines are idle.

    Buffers persist across frames (about 7.5 MiB for this model). Unlike the old
    liveness allocator, this trades modest DDR space for zero steady allocation.
    """
    def __init__(self,conv,tensor,allocate,model,parameters):
        self.model=copy.deepcopy(model); self.shapes=validate_model(self.model,parameters)
        self.conv,self.tensor=conv,tensor; self.buffers={}; self.prepared={}
        self.closed=False; self.unsafe=False; self.running=False; self._retained=False
        start=time.perf_counter()
        try:
            for node in self.model['nodes']:
                if node['kind']=='conv':
                    row=node['layer']; arrays={k:parameters[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
                    self.prepared[node['name']]=tuple(prepare_block(arrays,b,row) for b in range((node['shape'][0]+7)//8))
            self.preparation_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            for name,(shape,_) in self.shapes.items():
                words=word_count(shape); count=(words+7)//8*8+GUARD_WORDS
                buf=allocate(shape=(count,),dtype=np.uint64)
                self.buffers[name]=buf  # Own before filling/sync, including on exception.
                address=LungAxil._buffer_address(buf)
                if (buf.dtype!=np.uint64 or buf.ndim!=1 or not buf.flags.c_contiguous or buf.nbytes<count*8
                        or address%64 or address+buf.nbytes>2**32
                        or not all(callable(getattr(buf,n,None)) for n in ['flush','invalidate','freebuffer'])):
                    raise ValueError('Need 64-byte-aligned contiguous PYNQ allocations with cache sync')
                for other in list(self.buffers.values())[:-1]:
                    a=LungAxil._buffer_address(other)
                    if address<a+other.nbytes and a<address+buf.nbytes: raise ValueError('Device allocations overlap')
                buf[:]=SENTINEL; buf.flush()
            self.allocation_ms=(time.perf_counter()-start)*1000
            self.cma_bytes=sum(b.nbytes for b in self.buffers.values())
        except BaseException:
            self.close()
            raise

    def _outstanding(self):
        return any(d._inflight or d._poisoned for d in [self.conv,self.tensor])

    def _retain(self):
        self.unsafe=True
        if not self._retained: _UNSAFE.append(self); self._retained=True

    def close(self):
        if self.running or self._outstanding():
            self._retain()
            raise RuntimeError('Cannot free CMA with uncertain/outstanding hardware; recover shared board')
        for name in list(self.buffers):
            self.buffers[name].freebuffer(); del self.buffers[name]
        self.closed=True

    def predict_codes(self,image,observer=None):
        if self.closed or self.unsafe or self.running or self._outstanding():
            raise RuntimeError('Closed/failed/outstanding runtime; do not reuse')
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or list(image.shape)!=self.model['input_shape']:
            raise ValueError('Expected quantized UINT8 CHW input')
        self.running=True; start=time.perf_counter(); commands=[]; counts={}
        timing=dict(input_sync_ms=0.,configure_ms=0.,weight_load_ms=0.,execute_wait_ms=0.,
                    observer_ms=0.,final_sync_guard_ms=0.,output_unpack_ms=0.)
        def timed(key,call):
            begin=time.perf_counter(); result=call(); timing[key]+=(time.perf_counter()-begin)*1000; return result
        def record(engine,node,block,stats):
            expected=word_count(node['shape']) if engine=='tensor' else node['shape'][1]*node['shape'][2]
            if stats['written_words']!=expected: raise AssertionError('Unexpected device word count')
            completed=stats['completed']
            if engine in counts and completed!=(counts[engine]+1)&0xffffffff:
                raise AssertionError('Unexpected completion counter')
            counts[engine]=completed; commands.append(dict(node=node['name'],engine=engine,block=block,**stats))
        try:
            src=self.buffers['input']
            def upload():
                src[:word_count(self.model['input_shape'])]=feature_words(image,self.model['input_zero'])
                src.flush()
            timed('input_sync_ms',upload)
            for node in self.model['nodes']:
                name=node['name']; sources=node['sources']; dst=self.buffers[name]; a=self.buffers[sources[0]]
                ca,h,w=self.shapes[sources[0]][0]
                if node['kind']=='conv':
                    row=node['layer']
                    for block,p in enumerate(self.prepared[name]):
                        timed('configure_ms',lambda:self.conv.configure(LungAxil._buffer_address(a),LungAxil._buffer_address(dst),
                            h,w,ca,node['shape'][0],block,row['kernel'],row['padding']))
                        timed('weight_load_ms',lambda:self.conv.load_prepared(p))
                        def run_conv():
                            self.conv.start(); return self.conv.wait()
                        stats=timed('execute_wait_ms',run_conv)
                        record('conv',node,block,stats)
                else:
                    b=None if node['kind']=='pool' else self.buffers[sources[1]]
                    cb=0 if b is None else self.shapes[sources[1]][0][0]
                    timed('configure_ms',lambda:self.tensor.configure(LungAxil._buffer_address(a),LungAxil._buffer_address(dst),
                        0 if b is None else 3,h,w,ca,b=0 if b is None else LungAxil._buffer_address(b),channels_b=cb,**node['quant']))
                    def run_tensor():
                        self.tensor.start(); return self.tensor.wait()
                    stats=timed('execute_wait_ms',run_tensor)
                    record('tensor',node,None,stats)
                if observer is not None:
                    def observe():
                        dst.invalidate()
                        snapshot=np.array(dst[:word_count(node['shape'])],dtype='<u8',copy=True)
                        snapshot.flags.writeable=False; observer(name,snapshot)
                    timed('observer_ms',observe)
            def guards():
                # No CPU writes to payload/guard after initial setup except input.
                # Thus cached intermediate lines are clean, never flushed back.
                for name,buf in self.buffers.items():
                    buf.invalidate()
                    if np.any(np.asarray(buf)[word_count(self.shapes[name][0]):]!=SENTINEL):
                        raise AssertionError('Device allocation canary overwritten: '+name)
            timed('final_sync_guard_ms',guards)
            codes=timed('output_unpack_ms',lambda:unpack_words(self.buffers['head'],self.shapes['head'][0]))
            if len(commands)!=self.model['commands']: raise AssertionError('Incomplete command count')
            total=(time.perf_counter()-start)*1000
            return codes,dict(command_count=len(commands),commands=commands,timing_ms=timing,
                host_total_ms=total,unclassified_host_ms=max(0.,total-sum(timing.values())),
                pl_command_cycles=sum(c['cycles'] for c in commands),cma_resident_bytes=self.cma_bytes,
                buffer_allocations_per_frame=0,weight_packets_repacked_per_frame=0,clock_mhz=100,
                scope='Resident input -> CHW head; includes all per-frame MMIO weights, input sync, engine waits, frame-end guards and output copy; no model setup/bit download')
        except BaseException:
            self.unsafe=True
            if self._outstanding(): self._retain()
            raise
        finally:
            self.running=False
