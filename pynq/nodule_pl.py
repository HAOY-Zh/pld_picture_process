"""Nodule scheduler on the existing 100MHz convolution/tensor engines.

Persistent CMA + native MMIO weight loader + device-owned intermediates.
No CPU convolution fallback, Torch dependency or per-patch overlay download.
"""
import copy
from time import perf_counter
import numpy as np
from lung_axil import LungAxil
from lung_fast import FastUNet, prepare_block
from lung_continuous import word_count, GUARD_WORDS
from lung_model_layers_test import SENTINEL
from nodule_plan import validate, quantize_hu, lower_stem, dequantize_head


class NodulePL(FastUNet):
    """Inherited execution/ownership rules; independent frozen nodule validation."""
    def __init__(self,conv,tensor,allocate,model,parameters):
        self.model=copy.deepcopy(model);self.shapes=validate(self.model,parameters)
        self.conv,self.tensor=conv,tensor;self.buffers={};self.prepared={}
        self.closed=False;self.unsafe=False;self.running=False;self._retained=False
        tick=perf_counter()
        for node in self.model['nodes']:
            if node['kind']=='conv':
                row=node['layer']; arrays={k:parameters[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
                self.prepared[node['name']]=tuple(prepare_block(arrays,b,row) for b in range((node['shape'][0]+7)//8))
        self.preparation_ms=(perf_counter()-tick)*1000;tick=perf_counter()
        try:
            for name,(shape,_) in self.shapes.items():
                words=word_count(shape);count=(words+7)//8*8+GUARD_WORDS
                buf=allocate(shape=(count,),dtype=np.uint64);self.buffers[name]=buf
                addr=LungAxil._buffer_address(buf)
                if (buf.dtype!=np.uint64 or buf.ndim!=1 or not buf.flags.c_contiguous or buf.nbytes<count*8
                        or addr%64 or addr+buf.nbytes>2**32
                        or not all(callable(getattr(buf,n,None)) for n in ['flush','invalidate','freebuffer'])):
                    raise ValueError('Need aligned contiguous cache-coherent PYNQ buffers')
                for other in list(self.buffers.values())[:-1]:
                    a=LungAxil._buffer_address(other)
                    if addr<a+other.nbytes and a<addr+buf.nbytes: raise ValueError('Overlapping CMA')
                buf[:]=SENTINEL;buf.flush()
            self.allocation_ms=(perf_counter()-tick)*1000
            self.cma_bytes=sum(b.nbytes for b in self.buffers.values())
        except BaseException:
            self.close();raise

    def predict_hu(self,patch,observer=None):
        tick=perf_counter();q=lower_stem(quantize_hu(patch));lower_ms=(perf_counter()-tick)*1000
        codes,stats=self.predict_codes(q,observer)
        pred=dequantize_head(codes,self.model)
        stats['arm_quantize_stride2_pack_ms']=lower_ms
        stats['patch_total_ms']=(perf_counter()-tick)*1000
        stats['scope']='Five 128x128 HU planes -> five 64x64 signed head maps; ARM quantize/pack, all seven convolutions + four tensor operators in PL. Excludes CT sampling, lung segmentation and candidate postprocessing.'
        return pred,stats
