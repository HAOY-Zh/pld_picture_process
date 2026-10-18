import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
from lung_axil_smoke import feature_words
import lung_continuous as runner
from lung_continuous_package import make_plan
from lung_continuous_emulator import Memory,Conv,Tensor


def fixture():
    # Same topology/channel counts, 8x8 image for fast independent CPU unit tests.
    rng = np.random.default_rng(829)
    pairs = [('enc1.0',1,8,8),('enc1.3',8,8,8),('enc2.0',8,16,4),('enc2.3',16,16,4),
             ('enc3.0',16,32,2),('enc3.3',32,32,2),('neck.0',32,64,1),('neck.3',64,64,1),
             ('dec3.0',96,32,2),('dec3.3',32,32,2),('dec2.0',48,16,4),('dec2.3',16,16,4),
             ('dec1.0',24,8,8),('dec1.3',8,8,8),('head',8,1,8)]
    rows = []; parameters = {}
    for name,cin,cout,size in pairs:
        prefix = name.replace('.','_'); kernel = 1 if name=='head' else 3
        rows.append(dict(name=name,prefix=prefix,input_shape=[cin,size,size],output_shape=[cout,size,size],
                         kernel=kernel,padding=0 if kernel==1 else 1,input_zero=128,output_zero=128,relu=False))
        parameters[prefix+'_weight'] = rng.integers(-2,3,(cout,cin,kernel,kernel),dtype=np.int8)
        parameters[prefix+'_bias'] = rng.integers(-10,11,cout,dtype=np.int32)
        parameters[prefix+'_multiplier'] = np.full(cout,1073741824,np.int32)
        parameters[prefix+'_shift'] = np.full(cout,32,np.int32)
    cats = []
    for a,b in [('neck.3','enc3.3'),('dec3.3','enc2.3'),('dec2.3','enc1.3')]:
        cats.append(dict(output_zero_point=128,branches=[dict(source=s,input_zero_point=128,multiplier=1073741824,right_shift=30) for s in [a,b]]))
    contract = dict(concat=cats,input_quantization=dict(zero_point=128,scale=1),layers=[dict(output_scale=1)])
    return make_plan(rows,contract),parameters,rng.integers(0,256,(1,8,8),dtype=np.uint8)


class ContinuousTests(unittest.TestCase):
    def make(self):
        model,params,image = fixture(); memory = Memory(); conv = Conv(memory); tensor = Tensor(memory)
        net = runner.ContinuousUNet(conv,tensor,memory.allocate,model,params,log=lambda *a,**k:None)
        return model,params,image,memory,conv,tensor,net

    def test_complete_graph_real_upstream_addresses_and_skips(self):
        model,params,image,memory,conv,tensor,net = self.make(); snapshots = {}
        def observe(name,words): snapshots[name] = words.copy()
        codes,stats = net.predict_codes(image,observe)
        self.assertEqual(codes.shape,(1,8,8))
        self.assertEqual((stats['command_count'],stats['convolution_commands'],stats['tensor_commands']),(51,45,6))
        self.assertEqual(len(snapshots),21)
        self.assertTrue(all(b.freed for b in memory.buffers))
        addresses = {'input':memory.buffers[0].physical_address}; event_index = 0
        for node in model['nodes']:
            count = (node['shape'][0]+7)//8 if node['kind']=='conv' else 1
            events = memory.events[event_index:event_index+count]; event_index += count
            for event in events:
                self.assertEqual(event['sources'],[addresses[s] for s in node['sources']])
                upstream = image if node['sources'][0]=='input' else runner.unpack_words(snapshots[node['sources'][0]],net.shapes[node['sources'][0]][0])
                np.testing.assert_array_equal(event['source_codes'],upstream)
            addresses[node['name']] = events[0]['dst']
        self.assertEqual(len(set(addresses.values())),22)

    def test_observer_cannot_inject_reference_into_device_memory(self):
        model,params,image,memory,conv,tensor,net = self.make()
        baseline,_ = net.predict_codes(image)
        def hostile_observer(name,words):
            self.assertFalse(words.flags.writeable)
            words.flags.writeable=True; words[:]=0  # Detached copy, never actual CMA.
        actual,_ = net.predict_codes(image,hostile_observer)
        np.testing.assert_array_equal(actual,baseline)

    def test_allocation_peak_is_bounded_and_success_allows_second_slice(self):
        model,params,image,memory,conv,tensor,net = self.make()
        first,stats = net.predict_codes(image); second,_ = net.predict_codes(image)
        np.testing.assert_array_equal(first,second)
        self.assertLess(stats['cma_peak_bytes'],sum(b.nbytes for b in memory.buffers[:22]))
        self.assertEqual((conv.completed,tensor.completed),(90,12))
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_invalid_plan_rejected_before_any_device_allocation(self):
        model,params,_,memory,conv,tensor,_ = self.make()
        for mutation in ['skip','zero','shift','shape','missing']:
            bad = copy.deepcopy(model)
            node = next(n for n in bad['nodes'] if n['kind']=='fused_concat')
            if mutation=='skip': node['sources'][1]='enc1.3'
            elif mutation=='zero': node['quant']['za']=0
            elif mutation=='shift': node['quant']['sa']=63
            elif mutation=='shape': node['shape'][0]=1
            else: bad['nodes'].pop()
            with self.assertRaises(ValueError): runner.ContinuousUNet(conv,tensor,memory.allocate,bad,params)
        self.assertEqual(len(memory.buffers),0)

    def test_reject_hu_or_wrong_size_input_before_allocation(self):
        *_,net = self.make()
        for image in [np.zeros((1,8,8),np.int16),np.zeros((1,9,8),np.uint8)]:
            with self.assertRaises(ValueError): net.predict_codes(image)

    def test_timeout_retains_all_live_buffers_and_prevents_reuse(self):
        model,params,image,memory,conv,tensor,net = self.make(); start=len(runner._UNSAFE_RUNS)
        def timeout(*args,**kwargs):
            tensor._inflight=tensor._poisoned=True
            raise TimeoutError('PC-only fake bus timeout')
        tensor.run=timeout
        try:
            with self.assertRaises(TimeoutError): net.predict_codes(image)
            live=[b for b in memory.buffers if not b.freed]
            self.assertEqual(len(live),2)  # Retained enc1 skip + pending pool output.
            self.assertEqual(len(runner._UNSAFE_RUNS)-start,1)
            with self.assertRaises(RuntimeError): net.predict_codes(image)
        finally: del runner._UNSAFE_RUNS[start:]

    def test_guard_corruption_detected_and_idle_buffers_freed(self):
        model,params,image,memory,conv,tensor,net = self.make(); old=conv.run
        def corrupt(src,dst):
            result=old(src,dst); dst[-1]=0; return result
        conv.run=corrupt
        with self.assertRaisesRegex(AssertionError,'canary'): net.predict_codes(image)
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_allocation_failure_or_observer_failure_frees_idle_buffers(self):
        for failure in ['allocation','observer']:
            model,params,image,memory,conv,tensor,net = self.make()
            allocate=net.allocate
            def broken_allocate(*args,**kwargs):
                if len(memory.buffers)==2: raise MemoryError('PC-only allocation failure')
                return allocate(*args,**kwargs)
            def broken_observer(*args): raise RuntimeError('PC-only observer failure')
            if failure=='allocation': net.allocate=broken_allocate
            with self.assertRaises((MemoryError,RuntimeError)):
                net.predict_codes(image,broken_observer if failure=='observer' else None)
            self.assertTrue(all(b.freed for b in memory.buffers))

    def test_completion_counter_jump_detected(self):
        model,params,image,memory,conv,tensor,net = self.make(); old=conv.run
        def wrong_counter(src,dst):
            result=old(src,dst)
            if conv.completed==2: result['completed']+=1
            return result
        conv.run=wrong_counter
        with self.assertRaisesRegex(AssertionError,'counter'): net.predict_codes(image)
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_wrong_block_or_incomplete_count_rejected(self):
        for failure in ['future','count']:
            model,params,image,memory,conv,tensor,net = self.make(); old=conv.run
            def corrupt(src,dst):
                result=old(src,dst)
                if failure=='count': result['written_words']-=1
                else:
                    blocks=(conv.configuration[5]+7)//8
                    if blocks>1: dst[1]=0
                return result
            conv.run=corrupt
            with self.assertRaises(AssertionError): net.predict_codes(image)
            self.assertTrue(all(b.freed for b in memory.buffers))

    def test_invalid_parameters_never_reach_hardware(self):
        model,params,_,memory,conv,tensor,_ = self.make()
        for bad in [np.ones(8,np.int64),np.zeros(8,np.int32)]:
            changed=dict(params,enc1_0_multiplier=bad)
            with self.assertRaises(ValueError): runner.ContinuousUNet(conv,tensor,memory.allocate,model,changed)
        self.assertEqual(len(memory.buffers),0)

    def test_pack_unpack_preserves_tail_padding(self):
        image=np.arange(3*2*5,dtype=np.uint8).reshape(3,2,5)
        words=feature_words(image,137)
        np.testing.assert_array_equal(runner.unpack_words(words,image.shape),image)
        self.assertTrue(np.all(words.view(np.uint8).reshape(2,5,8)[:,:,3:]==137))


if __name__ == '__main__': unittest.main()
