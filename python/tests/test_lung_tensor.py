from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_tensor as t
from lung_tensor_smoke import expected_reads
from lung_tensor_vectors import reference,source_words


class MMIO:
    def __init__(self):
        self.reg={t.ID:0x54454e53,t.VERSION:0x10000,t.STATUS:0,t.COMPLETED:0}
        self.writes=[];self.ticks=0;self.error=False;self.hang=False
    def write(self,address,value):
        self.writes.append((address,value));self.reg[address]=value
        if address==t.CTRL and value==1:
            shape=self.reg[t.SHAPE];geom=self.reg[t.GEOMETRY]
            spec=t.layout(geom&3,shape&1023,shape>>16,(geom>>8)&127,(geom>>16)&127)
            self.reg[t.WRITTEN]=spec['words'];self.reg[t.STATUS]=1;self.ticks=2
    def read(self,address):
        if address==t.STATUS and self.ticks and not self.hang:
            self.ticks-=1
            if self.ticks==0:
                self.reg[t.STATUS]=t.DONE|(t.ERROR if self.error else 0)
                self.reg[t.COMPLETED]+=1;self.reg[t.READS]=9;self.reg[t.CYCLES]=31
        return self.reg.get(address,0)


class Buffer:
    def __init__(self,address,size,name,events):
        self.physical_address=address;self.nbytes=size;self.name=name;self.events=events
    def flush(self): self.events.append(self.name+'.flush')
    def invalidate(self): self.events.append(self.name+'.invalidate')


class TensorTests(unittest.TestCase):
    def test_layout_sizes_and_boundary_rejection(self):
        result=t.layout(3,3,5,13,3)
        self.assertEqual([result[k] for k in ['a_bytes','b_bytes','dst_bytes','words']],[240,480,960,120])
        for args in [(0,3,4,8,0),(1,257,1,8,0),(2,1,1,96,1),(3,1,1,8,0),(0,2,2,0,0)]:
            with self.assertRaises(ValueError): t.layout(*args)

    def test_register_mapping_and_abi(self):
        mmio=MMIO();driver=t.TensorAxil(mmio)
        driver.configure(0x1000,0x4000,3,3,5,13,b=0x2000,channels_b=3,
                         za=128,zb=64,zo=99,ma=123,sa=31,mb=456,sb=30)
        self.assertEqual(mmio.reg[t.SHAPE],3|(5<<16))
        self.assertEqual(mmio.reg[t.GEOMETRY],3|(13<<8)|(3<<16))
        self.assertEqual(mmio.reg[t.ZEROS],128|(64<<8)|(99<<16))
        self.assertEqual([mmio.reg[x] for x in [t.MA,t.SA,t.MB,t.SB]],[123,31,456,30])
        mmio.reg[t.ID]=0
        with self.assertRaises(RuntimeError): t.TensorAxil(mmio)

    def test_invalid_ranges_quantization_and_overlap_do_not_write(self):
        mmio=MMIO();driver=t.TensorAxil(mmio)
        for extra in [dict(a=0x1001),dict(dst=0x1000),dict(a=0xfffffff8),
                      dict(za=1),dict(ma=2**31),dict(sa=63),dict(b=0x4000,op=2,channels_b=8)]:
            values=dict(a=0x1000,dst=0x4000,op=0,height=4,width=4,channels_a=8);values.update(extra)
            with self.assertRaises(ValueError): driver.configure(**values)
        self.assertEqual(mmio.writes,[])

    def test_run_synchronizes_both_inputs_and_output(self):
        mmio=MMIO();driver=t.TensorAxil(mmio);events=[]
        a=Buffer(0x1000,8,'a',events);b=Buffer(0x2000,8,'b',events);dst=Buffer(0x3000,8,'dst',events)
        result=driver.run(a,dst,2,1,1,3,src_b=b,channels_b=5)
        self.assertEqual(result['written_words'],1)
        self.assertEqual(events,['a.flush','dst.flush','b.flush','dst.invalidate'])
        dst.nbytes=1;before=len(mmio.writes)
        with self.assertRaises(ValueError): driver.run(a,dst,2,1,1,3,src_b=b,channels_b=5)
        self.assertEqual(len(mmio.writes),before)

    def test_timeout_busy_and_compute_error(self):
        mmio=MMIO();driver=t.TensorAxil(mmio)
        driver.configure(0x1000,0x2000,0,2,2,1);mmio.error=True;driver.start()
        with self.assertRaises(RuntimeError): driver.wait()
        mmio.error=False;mmio.hang=True;driver.start()
        with self.assertRaises(RuntimeError): driver.start()
        with patch.object(t.time,'monotonic',side_effect=[0,40]):
            with self.assertRaises(TimeoutError): driver.wait()
        self.assertTrue(driver._poisoned and driver._inflight)
        with self.assertRaises(RuntimeError): driver.start()

    def test_uncertain_start_keeps_inflight(self):
        driver=t.TensorAxil(MMIO());driver.configure(0x1000,0x2000,0,2,2,1)
        with patch.object(driver.mmio,'write',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): driver.start()
        self.assertTrue(driver._inflight and driver._poisoned)

    def test_independent_read_address_model(self):
        for op,a,b,count in [(2,[5,1,1],[11,1,1],3),(3,[3,1,1],[5,2,2],8),
                              (1,[9,1,2],None,16),(0,[9,2,4],None,16)]:
            self.assertEqual(expected_reads(dict(op=op,input_shape=a,b_shape=b)),count)

    def test_signed_rounding_concat_order_and_pool_nearest(self):
        a=np.array([[[127,125,129,131]]],np.uint8);b=np.array([[[63,61,65,67]]],np.uint8)
        result=reference(2,a,b,128,64,99,1073741824,31,1610612736,30)
        self.assertEqual(result.tolist(),[[[99,97,99,101]],[[97,95,101,103]]])
        image=np.arange(8,dtype=np.uint8).reshape(1,2,4)
        self.assertEqual(reference(0,image,None,0,0,0).tolist(),[[[5,7]]])
        self.assertEqual(reference(1,image,None,0,0,0).shape,(1,4,8))

    def test_dirty_input_padding_is_not_output_padding(self):
        image=np.zeros((3,2,2),np.uint8)
        words=source_words(image,128).view(np.uint8).reshape(2,2,1,8)
        self.assertTrue(np.all(words[:,:,:,3:]!=128))
        self.assertTrue(np.all(words[:,:,:,:3]==0))


if __name__=='__main__': unittest.main()
