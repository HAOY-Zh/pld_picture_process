"""Layer-major nodule execution with resident per-block weights across tiles.

Same bit/INT8 graph. Persistent per-slot guarded CMA, device-owned intermediate
features, no CPU convolution fallback. A timeout retains all CMA owners.
"""
from time import perf_counter
import numpy as np
from nodule_pl import NodulePL
from lung_axil import LungAxil
from lung_continuous import word_count,unpack_words
from lung_model_layers_test import SENTINEL
from nodule_plan import dequantize_head


class BatchNodulePL(NodulePL):
    def __init__(self,conv,tensor,allocate,model,parameters,native_input,batch_size=8):
        if type(batch_size) is not int or not 1<=batch_size<=32:raise ValueError('Batch size must be 1..32')
        self.capacity=batch_size;self.native_input=native_input;self.slot_words={}
        # The original constructor validates shape, address, alignment, ownership
        # and preparation. Each requested single-frame arena becomes N arenas.
        def reserve(shape,dtype):return allocate(shape=(shape[0]*batch_size,),dtype=dtype)
        super().__init__(conv,tensor,reserve,model,parameters)
        for name,buf in self.buffers.items():self.slot_words[name]=len(buf)//batch_size

    def address(self,name,slot):
        return LungAxil._buffer_address(self.buffers[name])+slot*self.slot_words[name]*8

    def predict_batch(self,patches,observer=None):
        if self.closed or self.unsafe or self.running or self._outstanding():raise RuntimeError('Closed/unsafe/outstanding batch runtime')
        if not 1<=len(patches)<=self.capacity:raise ValueError('Invalid batch length')
        start=perf_counter();clear_start=start
        if len(patches)<self.capacity:self.clear_unused_slots(len(patches))
        clear_ms=(perf_counter()-clear_start)*1000
        self.running=True;commands=[];counts={};weight_loads=0
        timing=dict(input_pack_ms=0.,input_sync_ms=0.,configure_ms=0.,weight_load_ms=0.,
            execute_wait_ms=0.,final_sync_guard_ms=0.,output_ms=0.,observer_ms=0.,inactive_slot_clear_ms=clear_ms)
        def timed(key,call):
            t=perf_counter();result=call();timing[key]+=(perf_counter()-t)*1000;return result
        def record(engine,node,slot,block,stats):
            expected=word_count(node['shape']) if engine=='tensor' else node['shape'][1]*node['shape'][2]
            if stats['written_words']!=expected:raise AssertionError('Incomplete batch write')
            if engine in counts and stats['completed']!=(counts[engine]+1)&0xffffffff:raise AssertionError('Bad completion sequence')
            counts[engine]=stats['completed'];commands.append(dict(node=node['name'],slot=slot,engine=engine,block=block,**stats))
        try:
            src=self.buffers['input'];stride=self.slot_words['input'];words=word_count(self.model['input_shape'])
            for slot,patch in enumerate(patches):
                timed('input_pack_ms',lambda slot=slot,patch=patch:self.native_input.pack(patch,src[slot*stride:slot*stride+words],True))
            timed('input_sync_ms',src.flush)
            for node in self.model['nodes']:
                name=node['name'];sources=node['sources'];ca,h,w=self.shapes[sources[0]][0]
                if node['kind']=='conv':
                    row=node['layer']
                    for block,p in enumerate(self.prepared[name]):
                        for slot in range(len(patches)):
                            timed('configure_ms',lambda slot=slot:self.conv.configure(self.address(sources[0],slot),self.address(name,slot),
                                h,w,ca,node['shape'][0],block,row['kernel'],row['padding']))
                            if slot==0:
                                timed('weight_load_ms',lambda:self.conv.load_prepared(p));weight_loads+=1
                            def execute():self.conv.start();return self.conv.wait()
                            record('conv',node,slot,block,timed('execute_wait_ms',execute))
                else:
                    b=None if node['kind']=='pool' else sources[1]
                    cb=0 if b is None else self.shapes[b][0][0]
                    for slot in range(len(patches)):
                        timed('configure_ms',lambda slot=slot:self.tensor.configure(self.address(sources[0],slot),self.address(name,slot),
                            0 if b is None else 3,h,w,ca,b=0 if b is None else self.address(b,slot),channels_b=cb,**node['quant']))
                        def execute_tensor():self.tensor.start();return self.tensor.wait()
                        record('tensor',node,slot,None,timed('execute_wait_ms',execute_tensor))
                if observer is not None:
                    def observe():
                        self.buffers[name].invalidate()
                        for slot in range(len(patches)):
                            at=slot*self.slot_words[name]
                            a=np.array(self.buffers[name][at:at+word_count(node['shape'])],copy=True)
                            a.flags.writeable=False;observer(name,slot,a)
                    timed('observer_ms',observe)
            def guards():
                for name,buf in self.buffers.items():
                    buf.invalidate();n=word_count(self.shapes[name][0]);stride=self.slot_words[name]
                    matrix=np.asarray(buf).reshape(self.capacity,stride)
                    if np.any(matrix[:len(patches),n:]!=SENTINEL) or np.any(matrix[len(patches):]!=SENTINEL):
                        raise AssertionError('Batch canary/unscheduled slot changed: '+name)
            timed('final_sync_guard_ms',guards)
            def outputs():
                result=[];buf=self.buffers['head'];stride=self.slot_words['head'];n=word_count(self.shapes['head'][0])
                for slot in range(len(patches)):
                    codes=unpack_words(buf[slot*stride:slot*stride+n],self.shapes['head'][0])
                    result.append(dequantize_head(codes,self.model))
                return np.stack(result)
            pred=timed('output_ms',outputs)
            if len(commands)!=self.model['commands']*len(patches):raise AssertionError('Incomplete batch graph')
            total=(perf_counter()-start)*1000
            return pred,dict(patch_total_ms=total,host_total_ms=total-timing['input_pack_ms'],
                arm_quantize_stride2_pack_ms=timing['input_pack_ms'],timing_ms=timing,
                command_count=len(commands),weight_loads=weight_loads,tiles=len(patches),commands=commands,
                pl_command_cycles=sum(c['cycles'] for c in commands),cma_resident_bytes=self.cma_bytes,
                buffer_allocations_per_batch=0,weight_packets_repacked_per_batch=0,
                scope='Complete same-graph tile batch with native fused HWC8 input and per-block weights reused across slots')
        except BaseException:
            self.unsafe=True
            if self._outstanding():self._retain()
            raise
        finally:self.running=False

    def clear_unused_slots(self,active):
        """Prepare unused slots only while hardware is provably idle.

        Used when a short final batch follows a full batch. Full arena flush is
        safe only after invalidate, preserving all prior device-produced bytes.
        """
        if self.closed or self.unsafe or self.running or self._outstanding():raise RuntimeError('Cannot clear active/unsafe batch arena')
        if not 1<=active<=self.capacity:raise ValueError('Invalid active slots')
        if active==self.capacity:return
        for name,buf in self.buffers.items():
            buf.invalidate();buf[active*self.slot_words[name]:]=SENTINEL;buf.flush()
