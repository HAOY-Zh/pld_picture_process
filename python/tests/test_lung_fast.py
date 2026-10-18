import copy
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_fast as fast
from lung_axil_smoke import weight_packets
from lung_continuous import ContinuousUNet
from lung_continuous_emulator import Buffer,Memory,Conv,Tensor
from test_lung_continuous import fixture
from test_lung_axil import MMIO
from test_lung_cpu import compiler


class CachedBuffer(np.ndarray):
    """CPU cache and DDR deliberately differ; missing invalidate/extra flush fails."""
    def __new__(cls,count,address):
        value=np.empty(count,np.uint64).view(cls)
        value.physical_address=address; value.device=Buffer(count,address)
        value.freed=False; value.flushes=value.invalidates=0
        return value
    def flush(self):
        if self.freed: raise AssertionError('Freed buffer')
        self.device[:]=np.asarray(self);self.flushes+=1
    def invalidate(self):
        if self.freed: raise AssertionError('Freed buffer')
        self[:]=self.device;self.invalidates+=1
    def freebuffer(self):
        if self.freed: raise AssertionError('Double free')
        self.freed=True


class CachedMemory(Memory):
    def allocate(self,shape,dtype):
        value=CachedBuffer(shape[0],self.address);self.address+=((value.nbytes+63)//64)*64+64
        self.buffers.append(value);return value
    def device(self,address):
        return next(b.device for b in self.buffers if b.physical_address==address)


class DeviceConv(Conv):
    def load_prepared(self,p):
        self.load_weights(p['packets'],p['bias'],p['mult'],p['shift'],p['iz'],p['oz'],p['keep'],p['relu'])
    def start(self): self._inflight=True
    def wait(self):
        result=self.run(self.memory.device(self.configuration[0]),self.memory.device(self.configuration[1]))
        self._inflight=False;return result


class DeviceTensor(Tensor):
    def configure(self,a,dst,op,height,width,channels_a,b=0,channels_b=0,**q):
        self.configuration=(a,dst,op,height,width,channels_a,b,channels_b,q)
    def start(self): self._inflight=True
    def wait(self):
        a,dst,op,h,w,ca,b,cb,q=self.configuration
        result=self.run(self.memory.device(a),self.memory.device(dst),op,h,w,ca,
                        self.memory.device(b) if b else None,cb,**q)
        self._inflight=False;return result


class FastTests(unittest.TestCase):
    def make(self):
        model,params,image=fixture(); memory=CachedMemory();conv=DeviceConv(memory);tensor=DeviceTensor(memory)
        net=fast.FastUNet(conv,tensor,memory.allocate,model,params)
        return model,params,image,memory,conv,tensor,net

    def test_packet_words_exact_old_pack_all_layers_and_tail_channels(self):
        model,params,_=fixture()
        for node in model['nodes']:
            if node['kind']!='conv': continue
            row=node['layer'];arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
            for block in range((node['shape'][0]+7)//8):
                p=fast.prepare_block(arrays,block,row)
                self.assertEqual(list(p['packets']),weight_packets(arrays['weight'],block))
                self.assertFalse(p['words'].flags.writeable)
        rng=np.random.default_rng(10)
        arrays=dict(weight=rng.integers(-128,128,(13,11,3,3),dtype=np.int8),bias=np.full(13,-1,np.int32),
                    multiplier=np.ones(13,np.int32),shift=np.zeros(13,np.int32))
        row=dict(input_zero=127,output_zero=129,relu=False)
        p=fast.prepare_block(arrays,1,row)
        self.assertEqual(list(p['packets']),weight_packets(arrays['weight'],1))
        self.assertEqual(p['keep'],31); self.assertEqual(int(p['parameters'][0]),0xffffffff)

    def test_full_graph_and_repeated_changed_inputs_with_real_cache_ownership(self):
        model,params,image,memory,conv,tensor,net=self.make()
        try:
            for q in [image,255-image,image]:
                plain=Memory(); refs={}
                baseline=ContinuousUNet(Conv(plain),Tensor(plain),plain.allocate,model,params,log=lambda *a,**k:None)
                expected,_=baseline.predict_codes(q,lambda n,w:refs.update({n:w.copy()}))
                actual,stats=net.predict_codes(q)
                np.testing.assert_array_equal(actual,expected)
                for node in model['nodes']:
                    buf=net.buffers[node['name']]
                    np.testing.assert_array_equal(buf.device[:len(refs[node['name']])],refs[node['name']])
                self.assertEqual(stats['command_count'],51)
                self.assertEqual(stats['buffer_allocations_per_frame'],0)
                self.assertEqual(stats['weight_packets_repacked_per_frame'],0)
            self.assertEqual(len(memory.buffers),22)
            self.assertEqual(net.buffers['input'].flushes,4)  # setup + three inputs
            self.assertTrue(all(b.flushes==1 for n,b in net.buffers.items() if n!='input'))
            self.assertTrue(all(b.invalidates==3 for b in memory.buffers))
            self.assertFalse(any(b.freed for b in memory.buffers))
        finally: net.close()
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_observer_is_detached_and_never_becomes_upstream(self):
        _,_,image,_,_,_,net=self.make()
        try:
            expected,_=net.predict_codes(image)
            def observer(name,words):
                self.assertFalse(words.flags.writeable);words.flags.writeable=True;words[:]=0
            actual,_=net.predict_codes(image,observer)
            np.testing.assert_array_equal(actual,expected)
        finally: net.close()

    def test_timeout_retains_all_buffers_and_rejects_free_or_reuse(self):
        _,_,image,memory,conv,_,net=self.make(); start=len(fast._UNSAFE)
        def timeout(): conv._poisoned=True;raise TimeoutError('Fake timeout')
        conv.wait=timeout
        try:
            with self.assertRaises(TimeoutError): net.predict_codes(image)
            with self.assertRaises(RuntimeError): net.close()
            with self.assertRaises(RuntimeError): net.predict_codes(image)
            self.assertFalse(any(b.freed for b in memory.buffers))
            self.assertEqual(len(fast._UNSAFE),start+1)
        finally: del fast._UNSAFE[start:]

    def test_frame_end_canary_and_bad_completion_detected(self):
        for failure in ['guard','counter','count']:
            _,_,image,memory,conv,_,net=self.make(); old=conv.wait
            def corrupt():
                result=old()
                if failure=='guard': memory.device(conv.configuration[1])[-1]=0
                elif failure=='counter' and conv.completed==2: result['completed']+=1
                elif failure=='count': result['written_words']-=1
                return result
            conv.wait=corrupt
            try:
                with self.assertRaises(AssertionError): net.predict_codes(image)
                with self.assertRaises(RuntimeError): net.predict_codes(image)
            finally: net.close()

    def test_invalid_plan_and_allocation_failure_no_cma_leak(self):
        model,params,_=fixture();mem=CachedMemory();bad=copy.deepcopy(model);bad['nodes'][0]['shape'][0]=0
        with self.assertRaises(ValueError): fast.FastUNet(DeviceConv(mem),DeviceTensor(mem),mem.allocate,bad,params)
        self.assertEqual(mem.buffers,[])
        def allocate(**kwargs):
            if len(mem.buffers)==3: raise MemoryError('Fake allocation failure')
            return mem.allocate(**kwargs)
        with self.assertRaises(MemoryError): fast.FastUNet(DeviceConv(mem),DeviceTensor(mem),allocate,model,params)
        self.assertTrue(all(b.freed for b in mem.buffers))

    def test_close_and_invalid_input(self):
        _,_,image,_,_,_,net=self.make()
        with self.assertRaises(ValueError): net.predict_codes(image.astype(np.float32))
        net.close();net.close()
        with self.assertRaises(RuntimeError): net.predict_codes(image)

    def test_prepared_driver_keeps_old_start_validation_and_poisons_native_failure(self):
        _,params,_=fixture(); arrays={k:params['enc1_0_'+k] for k in ['weight','bias','multiplier','shift']}
        p=fast.prepare_block(arrays,0,dict(input_zero=128,output_zero=128,relu=False))
        mmio=MMIO(); driver=fast.PreparedConv(mmio)
        driver.configure(0x1000,0x2000,8,8,1,8);driver.load_prepared(p);driver.start()
        self.assertEqual(driver.wait()['written_words'],64)
        class Broken:
            def load(self,*a): raise KeyboardInterrupt('Simulated uncertain native call')
        driver=fast.PreparedConv(MMIO(),native_loader=Broken())
        with self.assertRaises(KeyboardInterrupt): driver.load_prepared(p)
        self.assertTrue(driver._poisoned)


class NativeIOTests(unittest.TestCase):
    def test_actual_compiled_protocol_and_volatile_entry_validation(self):
        cxx=compiler()
        if cxx is None: self.skipTest('Native compiler absent; native MMIO protocol NOT tested')
        tmp=tempfile.TemporaryDirectory(); folder=Path(tmp.name)
        try:
            root=Path(__file__).resolve().parents[2]
            for name in ['lung_fast_io.cpp','lung_fast_io.h']: shutil.copy2(root/'pynq'/name,folder/name)
            exe=folder/('protocol.exe' if os.name=='nt' else 'protocol')
            env=dict(os.environ);env['PATH']=str(Path(cxx).parent)+os.pathsep+env.get('PATH','')
            command=[cxx,'-O2','-std=c++11','-static-libgcc','-static-libstdc++','-I',str(folder),
                     str(root/'python/tests/lung_fast_io_test.cpp'),'-o',str(exe)]
            subprocess.run(command,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            result=subprocess.run([str(exe)],env=env,check=True,stdout=subprocess.PIPE,text=True)
            self.assertIn('protocol: PASS',result.stdout)
            library,_=fast.compile_io(folder,cxx)
            _,record=fast.compile_io(folder,cxx);self.assertTrue(record['reused_verified_binary'])
            class Mapping: array=np.zeros(256,np.uint32)
            mmio=Mapping(); loader=fast.NativeLoader(library,mmio)
            # Invalid timeout is rejected before touching the mapped registers.
            packet=np.zeros((1,16),np.uint32); params=np.zeros(24,np.uint32)
            self.assertEqual(loader.lib.lung_fast_load(mmio.array,256,packet,1,params,1,-1.),-1)
            self.assertTrue(np.all(mmio.array==0))
            class BadMapping: array=np.zeros(10,np.uint64)
            with self.assertRaises(ValueError): fast.NativeLoader(library,BadMapping())
        finally:
            if os.name=='nt': tmp._finalizer.detach()  # Loaded Windows DLL lifetime is process-wide.
            else: tmp.cleanup()


if __name__=='__main__': unittest.main()
