"""Exercise PS transfer framing with a fake DMA and independent filter oracle."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
spec=importlib.util.spec_from_file_location('tile_driver',ROOT/'pynq/ct_denoise_tile_pl_test.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
from ctdenoise.pl_reference import fixed_filter


class GPIO:
    def __init__(self):self.cfg=0
    def write(self,offset,value):self.cfg=value
    def read(self,offset):return 0


class Channel:
    def __init__(self,dma,receive):self.dma=dma;self.receive=receive
    def transfer(self,buffer):
        if self.receive:self.dma.destination=buffer;return
        cfg=self.dma.gpio.cfg;mode=cfg&15;w=(cfg>>4)&127;h=(cfg>>11)&127
        a=driver.unpack_hu(buffer).reshape(h,w)
        # Identity is a legal all-zero CNN. Other modes use the numerical oracle.
        q=a if mode==6 else fixed_filter(a,mode)
        if cfg&(1<<26):
            l,r,t,b=[(cfg>>shift)&3 for shift in (18,20,22,24)]
            q=q[t:h-b,l:w-r]
        if q.size!=self.dma.destination.size:raise AssertionError('Wrong receive byte count')
        self.dma.destination[:]=q.view(np.uint16).astype(np.uint32).ravel()
        self.dma.calls+=1
    def wait(self):pass


class DMA:
    def __init__(self,gpio):
        self.gpio=gpio;self.mmio=self;self.calls=0
        self.sendchannel=Channel(self,False);self.recvchannel=Channel(self,True)
    def read(self,offset):return 2


class Driver(unittest.TestCase):
    def test_fast_assembly_matches_legacy_and_whole_image(self):
        rng=np.random.default_rng(41)
        for mode in range(8):
            image=rng.integers(-32768,32768,(19,23),dtype=np.int16)
            expected=image if mode==6 else fixed_filter(image,mode)
            for fast,compact in [(False,False),(True,False)]+([(True,True)] if mode==6 else []):
                gpio=GPIO();dma=DMA(gpio);src=np.empty(4096,np.uint32);dst=np.empty_like(src)
                if fast:q,_,tiles=driver.run_frame_fast(image,mode,8,dma,gpio,src,dst,1,compact)
                else:q,_,tiles=driver.run_frame(image,mode,8,dma,gpio,src,dst,1)
                np.testing.assert_array_equal(q,expected)
                self.assertEqual(tiles,9);self.assertEqual(dma.calls,9)

    def test_58_pixel_cores_cover_edges_and_reduce_transfers(self):
        image=np.arange(512*512,dtype=np.int32).astype(np.int16).reshape(512,512)
        gpio=GPIO();dma=DMA(gpio);src=np.empty(4096,np.uint32);dst=np.empty_like(src)
        q,_,tiles=driver.run_frame_fast(image,6,58,dma,gpio,src,dst,1,True)
        np.testing.assert_array_equal(q,image);self.assertEqual(tiles,81)
        for shape in [(1,1),(1,69),(67,1),(67,69)]:
            a=image[:shape[0],:shape[1]]
            q,_,_=driver.run_frame_fast(a,6,58,dma,gpio,src,dst,1,True)
            np.testing.assert_array_equal(q,a)


if __name__=='__main__':unittest.main()
