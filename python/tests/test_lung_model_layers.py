from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_model_layers_test as runner


def fixture():
    row=dict(name='fixture',prefix='fixture',input_shape=[11,5,7],output_shape=[9,5,7],
             kernel=3,padding=1,input_zero=128,output_zero=17,relu=True)
    values=dict(input=np.arange(11*5*7,dtype=np.uint8).reshape(11,5,7),
                weight=np.zeros((9,11,3,3),np.int8),bias=np.arange(9,dtype=np.int32)-4,
                multiplier=np.arange(1,10,dtype=np.int32),shift=np.arange(9,dtype=np.int32),
                expected=np.arange(9*5*7,dtype=np.uint8).reshape(9,5,7))
    archive={'fixture_'+k:v for k,v in values.items()}
    return archive,row


class Buffer(np.ndarray):
    def __new__(cls,count,address):
        value=np.empty(count,dtype=np.uint64).view(cls)
        value.physical_address=address; value.freed=False
        return value
    def freebuffer(self): self.freed=True


class FakeDriver:
    """Host orchestration test double only; never evidence of PL execution."""
    def __init__(self,arrays,mode='ok'):
        self.arrays=arrays; self.mode=mode; self._inflight=self._poisoned=False
        self.buffers=[]; self.block=0
    def allocate(self,shape,dtype):
        result=Buffer(shape[0],0x1000+len(self.buffers)*0x10000)
        self.buffers.append(result); return result
    def configure(self,*args): self.block=args[6]
    def load_weights(self,*args): pass
    def run(self,src,dst):
        if self.mode=='timeout':
            self._inflight=self._poisoned=True
            raise TimeoutError('unit-test fake timeout')
        expected=runner.feature_words(self.arrays['expected'],17).reshape(35,2)
        dst[:70].reshape(35,2)[:,self.block]=expected[:,self.block]
        if self.mode=='overrun': dst[-1]=0
        return dict(written_words=35,cycles=100,completed=self.block+1,cache_hits=0,cache_misses=1)


class LayerTests(unittest.TestCase):
    def test_valid_layer_and_channel_padding(self):
        archive,row=fixture(); values=runner.read_layer(archive,row)
        packed=runner.feature_words(values['expected'],row['output_zero']).view(np.uint8).reshape(5,7,2,8)
        self.assertTrue(np.all(packed[:,:,1,1:]==17))

    def test_invalid_input_and_output_shapes_rejected(self):
        for key in ['input','expected','weight']:
            archive,row=fixture(); archive['fixture_'+key]=archive['fixture_'+key][:-1]
            with self.assertRaises(ValueError): runner.read_layer(archive,row)

    def test_invalid_quantization_rejected(self):
        for key,value in [('multiplier',0),('shift',63)]:
            archive,row=fixture(); archive['fixture_'+key][0]=value
            with self.assertRaises(ValueError): runner.read_layer(archive,row)

    def test_lane_parameter_order_signed_bias_and_tail(self):
        archive,row=fixture(); values=runner.read_layer(archive,row)
        (b,m,s),keep=runner.lane_parameters(values,0)
        self.assertEqual(b,list(range(-4,4))); self.assertEqual(m,list(range(1,9))); self.assertEqual(keep,255)
        (b,m,s),keep=runner.lane_parameters(values,1)
        self.assertEqual(b,[4]+[0]*7); self.assertEqual(m,[9]+[1]*7)
        self.assertEqual(s,[8]+[0]*7); self.assertEqual(keep,1)

    def test_host_orchestration_multiple_blocks_and_release(self):
        archive,row=fixture(); values=runner.read_layer(archive,row); driver=FakeDriver(values)
        results=runner.run_layer(driver,driver.allocate,values,row)
        self.assertEqual(len(results),2); self.assertTrue(all(b.freed for b in driver.buffers))

    def test_suffix_canary_overwrite_detected(self):
        archive,row=fixture(); values=runner.read_layer(archive,row); driver=FakeDriver(values,'overrun')
        with self.assertRaisesRegex(AssertionError,'canary'): runner.run_layer(driver,driver.allocate,values,row)
        self.assertTrue(all(b.freed for b in driver.buffers))

    def test_timeout_retains_buffers_without_free(self):
        archive,row=fixture(); values=runner.read_layer(archive,row); driver=FakeDriver(values,'timeout')
        start=len(runner._TIMED_OUT_BUFFERS)
        try:
            with self.assertRaises(TimeoutError): runner.run_layer(driver,driver.allocate,values,row)
            self.assertEqual(len(runner._TIMED_OUT_BUFFERS)-start,2)
            self.assertFalse(any(b.freed for b in driver.buffers))
        finally: del runner._TIMED_OUT_BUFFERS[start:]


if __name__ == '__main__': unittest.main()
