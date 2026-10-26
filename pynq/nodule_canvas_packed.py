"""Same guarded PL scheduler; native pack directly to the existing input CMA.

Only upload differs from the frozen FastUNet scheduler. All network operators,
configuration, counters, waits, synchronization and memory guards are retained.
"""
import time
import numpy as np
from .nodule_canvas_pl import CanvasPL
from .lung_axil import LungAxil
from .lung_continuous import word_count,unpack_words
from .lung_model_layers_test import SENTINEL


class PackedCanvasPL(CanvasPL):
    def predict_quantized(self,image,observer=None):
        if self.closed or self.unsafe or self.running or self._outstanding():
            raise RuntimeError('Closed/failed/outstanding runtime; do not reuse')
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(5,self.height,self.width):
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
                self.quantized_packer.pack(image,src[:word_count(self.model['input_shape'])])
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
