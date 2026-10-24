"""Contract, borrowed-DMA and HTTP tests. Physical acceptance is recorded separately."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'pynq/roi_vendor')]
from roi_statistics_contract import statistics,validate_statistics,read_statistics,STATISTICS_ABI,IDENTITY
import test_ct_studio_roi as old
from roi_statistics_processor import RoiStatisticsProcessor
from ct_roi_reference import CpuRoiProcessor


def scalar(values):
    values=np.asarray(values,dtype=np.int64)
    return statistics(int(values.size),int(values.sum()),int((values*values).sum()),
        int(values.min()) if values.size else None,int(values.max()) if values.size else None)


def registers(stats):
    return dict(identity=IDENTITY,valid=1,sum_lo=stats['sum']&0xffffffff,
        sum_hi=(stats['sum']>>32)&0xffffffff,sumsq_lo=stats['sumsq']&0xffffffff,
        sumsq_hi=stats['sumsq']>>32,minmax=((stats['min'] or 0)&65535)|(((stats['max'] or 0)&65535)<<16))


class ContractTests(unittest.TestCase):
    def test_sign_extension_extremes_and_empty(self):
        for values in ([],[-32768]*262144,[32767]*262144,[-32768,-1,0,32767]):
            expected=scalar(values);words=registers(expected)
            actual=read_statistics(words.__getitem__,len(values),0x01000002)
            self.assertEqual(actual,expected)
            if values:self.assertAlmostEqual(actual['std_population'],np.std(values),places=9)
    def test_invalid_scalar_identity_valid_and_status_are_rejected(self):
        expected=scalar([-4,8,10]);words=registers(expected)
        for key,value in [('identity',0),('valid',0),('sum_hi',0xffffffff)]:
            changed=dict(words);changed[key]=value
            with self.assertRaises(ValueError):read_statistics(changed.__getitem__,3,0x01000002)
        with self.assertRaises(ValueError):read_statistics(words.__getitem__,3,0x01000006)
        changed=dict(expected);changed['sumsq']=0
        with self.assertRaises(ValueError):validate_statistics(changed,3)
        with self.assertRaises(ValueError):validate_statistics(expected,4)
    def test_valid_must_remain_set_during_read(self):
        words=registers(scalar([2]));reads=iter([1,0])
        def read(name):return next(reads) if name=='valid' else words[name]
        with self.assertRaisesRegex(ValueError,'changed'):read_statistics(read,1,0x01000002)


class BorrowedDriverTests(unittest.TestCase):
    def make_driver(self):
        def gpio(first=0,second=0):
            return SimpleNamespace(channel1=Mock(read=Mock(return_value=first)),channel2=Mock(read=Mock(return_value=second)))
        gpios={name:gpio() for name in ('roi_stats_sum','roi_stats_sumsq','roi_stats_meta','roi_stats_identity')}
        gpios['roi_stats_identity'].channel1.read.return_value=IDENTITY
        resources=dict(dma=SimpleNamespace(recvchannel=Mock(),sendchannel=Mock()),
            geometry_gpio=gpio(),bounds_gpio=gpio(),hu_status_gpio=gpio(),count_identity_gpio=gpio(0,0x524f4931),
            input_buffer=np.zeros(1024,np.uint32),output_buffer=np.zeros(1024,np.uint32),clock_hz=100e6)
        def wait(dma,timeout):
            hu=self.image;reference=CpuRoiProcessor().process_slice(hu,self.roi,self.limits,self.use_hu,self.lung)
            mask=reference['filtered_mask'];resources['output_buffer'][:hu.size]=mask.ravel()
            resources['hu_status_gpio'].channel2.read.return_value=0x01000002
            resources['count_identity_gpio'].channel1.read.return_value=reference['valid_count']
            words=registers(scalar(hu[mask!=0]))
            for name,channel,key in [('roi_stats_sum',1,'sum_lo'),('roi_stats_sum',2,'sum_hi'),
                ('roi_stats_sumsq',1,'sumsq_lo'),('roi_stats_sumsq',2,'sumsq_hi'),
                ('roi_stats_meta',1,'minmax'),('roi_stats_meta',2,'valid')]:
                getattr(gpios[name],'channel'+str(channel)).read.return_value=words[key]
        resources['wait_dma']=wait
        return RoiStatisticsProcessor(statistics_gpio=gpios,**resources),resources,gpios
    def test_mask_and_stats_parity_without_cpu_result_builder(self):
        driver,resources,_=self.make_driver()
        self.image=np.array([[-32768,5,22,32767],[-4,2,99,7]],np.int16)
        for roi,limits,use_hu,lung in [(None,None,False,None),((0,2,1,4),(0,100),True,np.ones((2,4),np.uint8)),
            ((2,2,0,4),None,False,None)]:
            self.roi,self.limits,self.use_hu,self.lung=roi,limits,use_hu,lung
            with patch('ct_roi_processor.make_result',side_effect=AssertionError('CPU statistics forbidden')):
                actual=driver.process_slice(self.image,roi,limits,use_hu,lung)
            expected=CpuRoiProcessor().process_slice(self.image,roi,limits,use_hu,lung)
            np.testing.assert_array_equal(actual['filtered_mask'],expected['filtered_mask'])
            self.assertEqual(actual['hu_statistics'],scalar(self.image[expected['filtered_mask']!=0]))
        self.assertFalse(driver.faulted)
        self.assertIs(driver.input_buffer,resources['input_buffer'])
    def test_uncertain_transfer_is_latched_and_not_retried(self):
        driver,resources,_=self.make_driver();resources['dma'].recvchannel.transfer.side_effect=TimeoutError('uncertain')
        with self.assertRaises(TimeoutError):driver.process_slice(np.ones((1,1),np.int16),use_hu=False)
        self.assertTrue(driver.faulted)
        with self.assertRaisesRegex(Exception,'Faulted'):driver.process_slice(np.ones((1,1),np.int16),use_hu=False)
        self.assertEqual(resources['dma'].recvchannel.transfer.call_count,1)


class StatisticsProtocolTests(old.ProtocolTests):
    def setUp(self):
        super().setUp();self.hw.manifest['roi']=dict(old.ROW,statistics_abi=STATISTICS_ABI)
        self.client.manifest=copy.deepcopy(dict(self.hw.manifest,platform_manifest_sha256='f'*64))
        original=self.hw.roi
        def hardware(hu,params,lung):
            result=original(hu,params,lung);result['statistics_abi']=STATISTICS_ABI
            result['hu_statistics']=scalar(hu[result['filtered_mask']!=0]);return result
        self.hw.roi=hardware
    def test_new_path_never_recomputes_masked_hu_on_host(self):
        with patch('ct_studio_roi_client.hu_statistics',side_effect=AssertionError('CPU recompute')):
            result=self.client.roi(self.hu,None,None,False)
        self.assertEqual(result['hu_statistics'],scalar(self.hu))
        self.assertEqual(result['statistics_abi'],STATISTICS_ABI)
    def test_wrong_statistics_receipt_is_rejected(self):
        original=self.client.call
        def altered(*args,**kwargs):
            result=original(*args,**kwargs);result['hu_statistics']['sum']+=1;return result
        self.client.call=altered
        with self.assertRaises(ValueError):self.client.roi(self.hu,None,None,False)


class StatisticsWorkflowTests(old.WorkflowTests):
    def setUp(self):
        super().setUp();original=self.c.client.roi
        self.c.client.manifest={'roi':dict(old.ROW,statistics_abi=STATISTICS_ABI)}
        def hardware(hu,*args,**kwargs):
            result=original(hu,*args,**kwargs);result['statistics_abi']=STATISTICS_ABI
            result['hu_statistics']=scalar(hu[result['filtered_mask']!=0]);return result
        self.c.client.roi=hardware
    def test_whole_slab_variance_uses_exact_hardware_moments(self):
        self.c.submit('roi',self.values());state=old.wait(self.c)
        self.assertEqual(state['phase'],'done',state['error'])
        actual=self.c.roi_result['hu_statistics']
        selected=self.c.hu[:,1:4,2:7].astype(np.int64).ravel()
        self.assertEqual(actual['sum'],int(selected.sum()))
        self.assertEqual(actual['sumsq'],int((selected*selected).sum()))
        self.assertEqual(actual['computed_on'],'FPGA_STATISTICS_RST1')
        self.assertAlmostEqual(actual['std_population'],selected.std(),places=10)


if __name__=='__main__':unittest.main()
