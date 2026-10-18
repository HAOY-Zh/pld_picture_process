"""Continuous HWC8 U-Net scheduling on the two existing PL engines.

Only the initial UINT8 image is copied from the CPU. All intermediate inputs
are device outputs, including retained encoder skips. There is no CPU operator
fallback, reference activation input, PyTorch dependency, or overlay reload.
"""
from collections import Counter
import time

import numpy as np

from lung_axil import LungAxil
from lung_axil_smoke import feature_words, weight_packets
from lung_model_layers_test import lane_parameters, SENTINEL
from lung_tensor import layout

GUARD_WORDS = 16
_UNSAFE_RUNS = []  # Keep CMA owners AND drivers alive until shared board recovery.


def word_count(shape):
    c, h, w = shape
    return h*w*((c+7)//8)


def unpack_words(words, shape):
    c, h, w = shape
    raw = np.asarray(words[:word_count(shape)], dtype='<u8')
    return raw.view(np.uint8).reshape(h, w, (c+7)//8*8)[:, :, :c].transpose(2, 0, 1).copy()


def validate_model(model, parameters):
    """Validate the entire executable plan before any allocation or MMIO write."""
    if model.get('schema') != 'lung_continuous_plan_v1':
        raise ValueError('Wrong continuous network plan')
    shape = model.get('input_shape', [])
    if (len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape)
            or shape[0] != 1 or max(shape[1:]) > 512 or any(v % 8 for v in shape[1:])):
        raise ValueError('Expected one-channel spatial input divisible by eight')
    available = {'input': (shape, model['input_zero'])}; names = []; commands = 0
    if type(model['input_zero']) is not int or not 0 <= model['input_zero'] <= 255:
        raise ValueError('Invalid input zero point')
    for node in model['nodes']:
        name = node['name']; sources = node['sources']; kind = node['kind']
        if name in available or not sources or any(n not in available for n in sources):
            raise ValueError('Duplicate node or non-topological dependency: '+name)
        if kind == 'conv':
            if len(sources) != 1: raise ValueError('Convolution needs one source')
            row = node['layer']; prefix = row['prefix']
            arrays = {k: parameters[prefix+'_'+k] for k in ['weight', 'bias', 'multiplier', 'shift']}
            weight = arrays['weight']; ishape, iz = available[sources[0]]
            if weight.dtype != np.int8 or weight.ndim != 4:
                raise ValueError('Invalid convolution weights')
            cout, cin, kh, kw = weight.shape; pad = row['padding']
            if (not 1 <= cout <= 96 or not 1 <= cin <= 96 or (kh, kw) not in [(1,1),(3,3)]
                    or row['kernel'] != kh or pad != (1 if kh == 3 else 0)
                    or ishape != row['input_shape'] or cin != ishape[0] or iz != row['input_zero']
                    or type(row['relu']) is not bool or row['name'] != name):
                raise ValueError('Convolution source geometry/grid mismatch: '+name)
            if any(arrays[k].dtype != np.int32 or arrays[k].shape != (cout,)
                   for k in ['bias', 'multiplier', 'shift']):
                raise ValueError('Invalid channel parameters')
            if np.any(arrays['multiplier'] <= 0) or np.any(arrays['shift'] < 0) or np.any(arrays['shift'] > 62):
                raise ValueError('Invalid channel requantization')
            out_shape = [cout, ishape[1], ishape[2]]; oz = row['output_zero']
            if row['output_shape'] != out_shape: raise ValueError('Convolution output shape mismatch')
            commands += (cout+7)//8
        elif kind in ('pool', 'fused_concat'):
            op = 0 if kind == 'pool' else 3
            if len(sources) != (1 if op == 0 else 2): raise ValueError('Wrong tensor dependencies')
            (ca,h,w), za = available[sources[0]]; q = node['quant']
            cb = 0 if op == 0 else available[sources[1]][0][0]
            sizes = layout(op,h,w,ca,cb)
            if q['za'] != za: raise ValueError('Tensor A quantization mismatch')
            oz = q['zo']
            if op == 0 and oz != za: raise ValueError('Pool changes quantization grid')
            if op == 3:
                bshape, zb = available[sources[1]]
                if bshape[1:] != [sizes['oh'],sizes['ow']] or q['zb'] != zb:
                    raise ValueError('Skip spatial/grid mismatch')
                for k in ['ma','mb']:
                    if type(q[k]) is not int or not 1 <= q[k] < 2**31: raise ValueError('Invalid concat multiplier')
                for k in ['sa','sb']:
                    if type(q[k]) is not int or not 0 <= q[k] <= 62: raise ValueError('Invalid concat shift')
            out_shape = [sizes['cout'],sizes['oh'],sizes['ow']]; commands += 1
        else:
            raise ValueError('Unknown node kind: '+str(kind))
        if type(oz) is not int or not 0 <= oz <= 255 or node['shape'] != out_shape or node['zero'] != oz:
            raise ValueError('Invalid output grid/geometry: '+name)
        available[name] = (out_shape, oz); names.append(name)
    if (len(names) != 21 or len(set(names)) != 21 or names[-1] != 'head'
            or sum(n['kind']=='conv' for n in model['nodes']) != 15
            or sum(n['kind']=='pool' for n in model['nodes']) != 3
            or sum(n['kind']=='fused_concat' for n in model['nodes']) != 3
            or commands != model['commands']):
        raise ValueError('Incomplete frozen U-Net topology')
    return available


class ContinuousUNet:
    def __init__(self, conv, tensor, allocate, model, parameters, log=print):
        self.shapes = validate_model(model, parameters)
        self.conv, self.tensor, self.allocate = conv, tensor, allocate
        self.model, self.parameters, self.log = model, parameters, log
        self._unsafe = False

    def _busy_or_unsafe(self):
        return any(d._inflight or d._poisoned for d in (self.conv,self.tensor))

    def predict_codes(self, input_codes, observer=None):
        """Return head codes and timing/stats. Observer gets detached READ-ONLY snapshots.

        Observer is optional diagnostic output, never a source of device data.
        On an uncertain completion, retain every live CMA allocation. Do not
        interrupt/reload/free the buffers as a substitute for shared recovery.
        """
        if self._unsafe or self._busy_or_unsafe():
            raise RuntimeError('Outstanding/poisoned run: shared board recovery required')
        if input_codes.dtype != np.uint8 or list(input_codes.shape) != self.model['input_shape']:
            raise ValueError('Expected quantized CHW UINT8 input, not HU/float image')
        active = {}; allocated_bytes = peak_bytes = 0; commands = []; node_results = []
        uses = Counter(s for n in self.model['nodes'] for s in n['sources'])
        counts = {}; started = time.perf_counter()

        def reserve(name):
            nonlocal allocated_bytes, peak_bytes
            shape = self.shapes[name][0]; words = word_count(shape)
            buf = self.allocate(shape=(words+GUARD_WORDS,),dtype=np.uint64)
            # Register ownership BEFORE initialization, so exceptions cannot leak CMA.
            active[name] = buf; allocated_bytes += buf.nbytes; peak_bytes = max(peak_bytes,allocated_bytes)
            buf[:] = SENTINEL
            return buf

        def release(name):
            nonlocal allocated_bytes
            buf = active[name]; buf.freebuffer(); allocated_bytes -= buf.nbytes; del active[name]

        def guard_check():
            for name, buf in active.items():
                buf.invalidate()
                if np.any(np.asarray(buf)[word_count(self.shapes[name][0]):] != SENTINEL):
                    raise AssertionError('Device allocation canary overwritten: '+name)

        def record(kind, name, block, stats, elapsed):
            if stats['written_words'] != (self.shapes[name][0][1]*self.shapes[name][0][2]
                                            if kind == 'conv' else word_count(self.shapes[name][0])):
                raise AssertionError('Incomplete device write: '+name)
            completed = stats['completed']
            if kind in counts and completed != (counts[kind]+1)&0xffffffff:
                raise AssertionError('Unexpected device completion counter: '+kind)
            counts[kind] = completed
            commands.append(dict(node=name,engine=kind,output_block=block,driver_call_ms=elapsed,**stats))
            guard_check()

        try:
            src = reserve('input')
            src[:word_count(self.model['input_shape'])] = feature_words(input_codes,self.model['input_zero'])
            src.flush()
            for index,node in enumerate(self.model['nodes']):
                name = node['name']; sources = node['sources']; shape = node['shape']
                self.log('{}/21 {}: FPGA {} -> {}'.format(index+1,name,' + '.join(sources),name),flush=True)
                begin = time.perf_counter(); dst = reserve(name); dst.flush()
                a = active[sources[0]]; ca,h,w = self.shapes[sources[0]][0]
                if node['kind'] == 'conv':
                    row = node['layer']; prefix = row['prefix']
                    arrays = {k:self.parameters[prefix+'_'+k] for k in ['weight','bias','multiplier','shift']}
                    blocks = (shape[0]+7)//8; pixels = shape[1]*shape[2]
                    for block in range(blocks):
                        self.conv.configure(LungAxil._buffer_address(a),LungAxil._buffer_address(dst),
                                            h,w,ca,shape[0],block,row['kernel'],row['padding'])
                        (bias,mult,shift),keep = lane_parameters(arrays,block)
                        self.conv.load_weights(weight_packets(arrays['weight'],block),bias,mult,shift,
                                               row['input_zero'],row['output_zero'],keep,row['relu'])
                        call_start = time.perf_counter(); stats = self.conv.run(a,dst)
                        record('conv',name,block,stats,(time.perf_counter()-call_start)*1000)
                        # Later blocks MUST remain untouched until they are scheduled.
                        remaining = np.asarray(dst)[:pixels*blocks].reshape(pixels,blocks)[:,block+1:]
                        if np.any(remaining != SENTINEL): raise AssertionError('Unscheduled convolution block overwritten')
                else:
                    b = None if node['kind']=='pool' else active[sources[1]]
                    cb = 0 if b is None else self.shapes[sources[1]][0][0]
                    call_start = time.perf_counter()
                    stats = self.tensor.run(a,dst,0 if b is None else 3,h,w,ca,src_b=b,channels_b=cb,**node['quant'])
                    record('tensor',name,None,stats,(time.perf_counter()-call_start)*1000)
                check_start = time.perf_counter()
                if observer is not None:
                    # A copy, not a view of CMA: diagnostics cannot inject next-layer inputs.
                    snapshot = np.array(dst[:word_count(shape)],dtype='<u8',copy=True)
                    snapshot.flags.writeable = False
                    observer(name,snapshot)
                check_ms = (time.perf_counter()-check_start)*1000
                node_results.append(dict(node=name,host_node_ms=(time.perf_counter()-begin)*1000,
                                         observer_ms=check_ms,output_words=word_count(shape)))
                for source in sources:
                    uses[source] -= 1
                    if uses[source] == 0: release(source)
            result = unpack_words(active['head'],self.shapes['head'][0])
            if len(commands) != self.model['commands']: raise AssertionError('Incomplete network command count')
            return result,dict(commands=commands,nodes=node_results,command_count=len(commands),
                               convolution_commands=sum(c['engine']=='conv' for c in commands),
                               tensor_commands=sum(c['engine']=='tensor' for c in commands),
                               cma_peak_bytes=peak_bytes,host_total_ms=(time.perf_counter()-started)*1000,
                               clock_mhz=100,pl_command_cycles=sum(c['cycles'] for c in commands),
                               timing_scope='Diagnostic single slice; includes allocation, weight loading, polling, cache sync and reference checks; not steady-state FPS')
        finally:
            if self._busy_or_unsafe():
                self._unsafe = True
                _UNSAFE_RUNS.append((self,active))
            else:
                for name in list(active): release(name)
