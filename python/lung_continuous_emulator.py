"""PC-only scheduler test doubles. NEVER board results or performance evidence.

Arithmetic reads the ACTUAL upstream emulated memory; it does not copy golden
activations. This module is deliberately excluded from the board package.
"""
from pathlib import Path
import sys

import numpy as np

from lung_fpga_contract import conv_accumulator, requantize
from lung_tensor_vectors import reference

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_axil_smoke import feature_words
from lung_continuous import unpack_words, word_count


class Buffer(np.ndarray):
    def __new__(cls, count, address):
        value = np.empty(count,np.uint64).view(cls)
        value.physical_address = address; value.freed = False
        return value
    def flush(self):
        if self.freed: raise AssertionError('Using freed emulated buffer')
    def invalidate(self): self.flush()
    def freebuffer(self):
        if self.freed: raise AssertionError('Double free')
        self.freed = True


class Memory:
    def __init__(self):
        self.address = 0x100000; self.buffers = []; self.events = []
    def allocate(self,shape,dtype):
        if dtype != np.uint64: raise ValueError('Expected HWC8 allocation')
        value = Buffer(shape[0],self.address)
        self.address += ((value.nbytes+63)//64)*64+64
        self.buffers.append(value)
        return value


class Conv:
    def __init__(self,memory):
        self.memory = memory; self._poisoned = self._inflight = False; self.completed = 0
    def configure(self,a,dst,h,w,cin,cout,block,kernel,pad):
        self.configuration = (a,dst,h,w,cin,cout,block,kernel,pad)
    def load_weights(self,packets,bias,mult,shift,iz,oz,keep,relu):
        _,_,_,_,cin,_,_,kernel,_ = self.configuration
        self.weights = np.zeros((8,cin,kernel,kernel),np.int8)
        index = 0
        for ky in range(kernel):
            for kx in range(kernel):
                for block in range((cin+7)//8):
                    values = np.frombuffer(int(packets[index]).to_bytes(64,'little'),np.int8).reshape(8,8)
                    count = min(8,cin-block*8)
                    self.weights[:,block*8:block*8+count,ky,kx] = values[:,:count]; index += 1
        if index != len(packets): raise AssertionError('Unexpected weight packets')
        self.quant = bias,mult,shift,iz,oz,keep,relu
    def run(self,src,dst):
        src.flush(); dst.flush()
        a,out,h,w,cin,cout,block,kernel,pad = self.configuration
        if (src.physical_address,dst.physical_address) != (a,out): raise AssertionError('Address mismatch')
        bias,mult,shift,iz,oz,keep,relu = self.quant
        image = unpack_words(src,(cin,h,w))
        acc = conv_accumulator(image,self.weights,np.asarray(bias,np.int32),iz,pad)
        codes = requantize(acc,np.asarray(mult,np.int32),np.asarray(shift,np.int32),oz,relu)
        for lane in range(8):
            if not keep & (1 << lane): codes[lane] = oz
        pixels = codes.shape[1]*codes.shape[2]; blocks = (cout+7)//8
        dst[:pixels*blocks].reshape(pixels,blocks)[:,block] = feature_words(codes,oz)
        self.completed += 1
        self.memory.events.append(dict(engine='conv',sources=[a],dst=out,block=block,
                                       source_codes=image.copy(),output_shape=(cout,codes.shape[1],codes.shape[2])))
        return dict(written_words=pixels,cycles=0,completed=self.completed,cache_hits=0,cache_misses=0)


class Tensor:
    def __init__(self,memory):
        self.memory = memory; self._poisoned = self._inflight = False; self.completed = 0
    def run(self,src,dst,op,height,width,channels_a,src_b=None,channels_b=0,**q):
        src.flush(); dst.flush()
        a = unpack_words(src,(channels_a,height,width)); b = None
        if src_b is not None:
            src_b.flush(); b = unpack_words(src_b,(channels_b,height*2,width*2))
        codes = reference(op,a,b,q['za'],q.get('zb',0),q['zo'],q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0))
        dst[:word_count(codes.shape)] = feature_words(codes,q['zo'])
        self.completed += 1
        self.memory.events.append(dict(engine='tensor',sources=[src.physical_address]+([] if b is None else [src_b.physical_address]),
                                       dst=dst.physical_address,op=op,source_codes=a.copy(),b_codes=b))
        return dict(written_words=word_count(codes.shape),read_words=0,cycles=0,completed=self.completed)
