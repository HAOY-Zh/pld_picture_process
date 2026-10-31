"""One enlarged rectangle through the unchanged 100MHz INT8 PL graph.

Reuse the proven single-owner scheduler, allocation checks, counters, canaries
and timeout retention. Only validated spatial sizes/allocation lengths change.
"""
import numpy as np
from nodule_pl import NodulePL
from nodule_canvas import expanded_plan,lower_canvas
from lung_continuous import word_count,GUARD_WORDS


class CanvasPL(NodulePL):
    def __init__(self,conv,tensor,allocate,model,parameters,height,width):
        expanded,shapes=expanded_plan(model,parameters,height,width)
        order=iter(shapes)
        def reserve(shape,dtype):
            name=next(order);words=word_count(shapes[name][0])
            count=(words+7)//8*8+GUARD_WORDS
            buf=allocate(shape=(count,),dtype=dtype)
            if getattr(buf,'nbytes',0)<count*8:
                if callable(getattr(buf,'freebuffer',None)):buf.freebuffer()
                raise ValueError('Allocator returned an undersized canvas arena')
            return buf
        # The canonical constructor checks original model/weights and every
        # actual returned allocation (address range, coherence, overlap, guards).
        # All prepared weight packets depend on channels/kernel, never H/W.
        super().__init__(conv,tensor,reserve,model,parameters)
        self.model=expanded;self.shapes=shapes
        a=np.arange(-32768,32768,dtype=np.float64)
        self.lut=np.rint((a.clip(-1000,400)+1000)*(255./1400.)).astype(np.uint8)
        self.height=height;self.width=width

    def predict_hu(self,canvas,observer=None):
        from time import perf_counter
        start=perf_counter();a=np.asarray(canvas)
        if a.dtype!=np.int16 or a.shape!=(5,self.height,self.width):raise ValueError('Wrong INT16 canvas shape')
        codes=lower_canvas(self.lut[a.astype(np.int32)+32768]);packed=perf_counter()
        head,stats=self.predict_codes(codes,observer)
        pred=(head.astype(np.float32)-128)*np.asarray(self.model['head_scales'],np.float32)[:,None,None]
        stats['arm_quantize_stride2_pack_ms']=(packed-start)*1000
        stats['patch_total_ms']=(perf_counter()-start)*1000
        stats['canvas_hw']=[self.height,self.width]
        return pred,stats
