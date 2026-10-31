"""Native direct-CMA input, retaining the original scheduler safety checks."""
from time import perf_counter
import numpy as np
from nodule_canvas_pl import CanvasPL
from lung_axil import LungAxil
from lung_continuous import word_count,unpack_words
from lung_model_layers_test import SENTINEL


class DirectCanvasPL(CanvasPL):
    def __init__(self,conv,tensor,allocate,model,parameters,height,width,native_input):
        super().__init__(conv,tensor,allocate,model,parameters,height,width)
        self.native_input=native_input

    def predict_hu(self,canvas,observer=None):
        if self.closed or self.unsafe or self.running or self._outstanding():
            raise RuntimeError('Closed/failed/outstanding runtime; do not reuse')
        a=np.asarray(canvas)
        if a.dtype!=np.int16 or a.shape!=(5,self.height,self.width):raise ValueError('Wrong canvas shape')
        self.running=True;start=perf_counter();commands=[];counts={}
        timing=dict(input_pack_ms=0.,input_sync_ms=0.,configure_ms=0.,weight_load_ms=0.,execute_wait_ms=0.,
            observer_ms=0.,final_sync_guard_ms=0.,output_unpack_ms=0.)
        def timed(key,call):
            tick=perf_counter();result=call();timing[key]+=(perf_counter()-tick)*1000;return result
        def record(engine,node,block,stats):
            expected=word_count(node['shape']) if engine=='tensor' else node['shape'][1]*node['shape'][2]
            if stats['written_words']!=expected:raise AssertionError('Unexpected device word count')
            completed=stats['completed']
            if engine in counts and completed!=(counts[engine]+1)&0xffffffff:raise AssertionError('Unexpected completion counter')
            counts[engine]=completed;commands.append(dict(node=node['name'],engine=engine,block=block,**stats))
        try:
            src=self.buffers['input'];words=word_count(self.model['input_shape'])
            timed('input_pack_ms',lambda:self.native_input.pack(a,src[:words],True))
            timed('input_sync_ms',src.flush)
            for node in self.model['nodes']:
                name=node['name'];sources=node['sources'];dst=self.buffers[name];source=self.buffers[sources[0]]
                ca,h,w=self.shapes[sources[0]][0]
                if node['kind']=='conv':
                    row=node['layer']
                    for block,p in enumerate(self.prepared[name]):
                        timed('configure_ms',lambda:self.conv.configure(LungAxil._buffer_address(source),LungAxil._buffer_address(dst),
                            h,w,ca,node['shape'][0],block,row['kernel'],row['padding']))
                        timed('weight_load_ms',lambda:self.conv.load_prepared(p))
                        def run():self.conv.start();return self.conv.wait()
                        record('conv',node,block,timed('execute_wait_ms',run))
                else:
                    b=None if node['kind']=='pool' else self.buffers[sources[1]]
                    cb=0 if b is None else self.shapes[sources[1]][0][0]
                    timed('configure_ms',lambda:self.tensor.configure(LungAxil._buffer_address(source),LungAxil._buffer_address(dst),
                        0 if b is None else 3,h,w,ca,b=0 if b is None else LungAxil._buffer_address(b),channels_b=cb,**node['quant']))
                    def run_tensor():self.tensor.start();return self.tensor.wait()
                    record('tensor',node,None,timed('execute_wait_ms',run_tensor))
                if observer is not None:
                    def observe():
                        dst.invalidate();snapshot=np.array(dst[:word_count(node['shape'])],dtype='<u8',copy=True)
                        snapshot.flags.writeable=False;observer(name,snapshot)
                    timed('observer_ms',observe)
            def guards():
                for name,buf in self.buffers.items():
                    buf.invalidate()
                    if np.any(np.asarray(buf)[word_count(self.shapes[name][0]):]!=SENTINEL):raise AssertionError('Device allocation canary overwritten: '+name)
            timed('final_sync_guard_ms',guards)
            head=timed('output_unpack_ms',lambda:unpack_words(self.buffers['head'],self.shapes['head'][0]))
            if len(commands)!=self.model['commands']:raise AssertionError('Incomplete command count')
            pred=(head.astype(np.float32)-128)*np.asarray(self.model['head_scales'],np.float32)[:,None,None]
            total=(perf_counter()-start)*1000
            return pred,dict(command_count=len(commands),commands=commands,timing_ms=timing,patch_total_ms=total,
                host_total_ms=total-timing['input_pack_ms'],arm_quantize_stride2_pack_ms=timing['input_pack_ms'],
                pl_command_cycles=sum(c['cycles'] for c in commands),cma_resident_bytes=self.cma_bytes,
                buffer_allocations_per_frame=0,weight_packets_repacked_per_frame=0,clock_mhz=100,
                scope='Direct native HU-to-HWC8 CMA, input flush, original PL graph, counters, all guards and head readback')
        except BaseException:
            self.unsafe=True
            if self._outstanding():self._retain()
            raise
        finally:self.running=False
