"""Bounded transport and scheduling equivalence; these tests use no physical FPGA."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_ct_studio_shared import ProtocolTests,INPUT_SHAPE,OUTPUT_SHAPE
from shared_patch_contract import MAX_BATCH,BATCH_INTERFACE,digest
from lung_live_web import Controller
from nodule_detector import NoduleCaseProvider,Cancelled
from nodule_vendor.lung_nodule_cnn import config_from_dict


class BatchTests(ProtocolTests):
    def test_distinct_patches_tail_and_max_batch_preserve_codes_and_order(self):
        for count in (1,3,MAX_BATCH):
            images=[(self.image+i).astype(np.uint8)for i in range(count)]
            rows=self.client.nodule_batch(images)
            self.assertEqual(len(rows),count)
            for i,((head,r),image) in enumerate(zip(rows,images)):
                np.testing.assert_array_equal(head,np.broadcast_to(image[0],OUTPUT_SHAPE))
                self.assertEqual(r['input_sha256'],digest(image.tobytes()))
                self.assertEqual(r['batch_index'],i)
                self.assertEqual(r['transport_scope'],'amortized_batch_roundtrip')
    def test_batch_invalid_busy_and_fault_never_retry_or_publish_partial(self):
        for count in (0,9,True):
            with self.assertRaises(ValueError):self.s.execute_batch(b'','0'*32,digest(b''),count)
        self.s.gate.acquire()
        try:
            with self.assertRaises(Exception):self.client.nodule_batch([self.image]*2)
        finally:self.s.gate.release()
        self.assertEqual(self.hw.patch_calls,0)
        original=self.hw.nodule_patch
        def fails_second(image):
            if self.hw.patch_calls==1:self.hw.fail=True
            return original(image)
        self.hw.nodule_patch=fails_second
        with self.assertRaises(Exception):self.client.nodule_batch([self.image]*3)
        self.assertEqual(self.hw.patch_calls,2);self.assertTrue(self.s.failed)
        with self.assertRaises(Exception):self.client.nodule_batch([self.image])
        self.assertEqual(self.hw.patch_calls,2)
    def test_tampered_batch_head_order_clock_and_nested_identity_are_rejected(self):
        original=self.client.call
        mutations=(lambda r:r.update(output_sha256='0'*64),lambda r:r['records'].reverse(),
            lambda r:r['records'][0]['stats'].update(actual_pl_mhz_after=50),
            lambda r:r['records'][0].update(service_bundle_sha256='0'*64))
        for mutate in mutations:
            def changed(*args,**kwargs):
                r=original(*args,**kwargs);mutate(r);return r
            self.client.call=changed
            with self.assertRaises(ValueError):self.client.nodule_batch([self.image,self.image+1])


class CacheTests(unittest.TestCase):
    def test_completed_mask_count_is_cached_and_invalidates_on_mask_or_spacing(self):
        class Mask(np.ndarray):
            calls=0
            def sum(self,*a,**kw):
                Mask.calls+=1;return super().sum(*a,**kw)
        with tempfile.TemporaryDirectory() as tmp:
            c=Controller(None,{},tmp);c.hu=np.zeros((3,8,9));c.known=np.ones(3,bool)
            c.geom=dict(spacing_zyx_mm=[2.,.5,.8]);c.cleaned=np.ones((3,8,9),bool).view(Mask)
            c.status['volume_ready']=True
            before=c.cleaned.copy();first=c.snapshot()['volume_ml']
            for _ in range(10):self.assertEqual(c.snapshot()['volume_ml'],first)
            self.assertEqual(Mask.calls,1);np.testing.assert_array_equal(c.cleaned,before)
            c.geom['spacing_zyx_mm']=[1.,.5,.8];self.assertEqual(c.snapshot()['volume_ml'],first/2)
            c.cleaned=np.zeros((3,8,9),bool).view(Mask);self.assertEqual(c.snapshot()['volume_ml'],0)
            self.assertEqual(Mask.calls,3)
            c.cleaned=None;self.assertIsNone(c.snapshot()['volume_ml'])


class SchedulingTests(unittest.TestCase):
    def test_prefetch_and_batch_preserve_all_sampled_inputs_and_candidates(self):
        root=Path(__file__).resolve().parents[2]
        package=root/'data/ct_studio_shared_20261024_01'
        if not package.is_dir():package=root/'data/ct_studio_integration_20261025'
        if not package.is_dir():self.skipTest('Install the reviewed integration runtime before scheduling equivalence tests')
        bundle=json.loads((package/'shared_host_platform.json').read_text())
        model=json.loads((package/'ct_studio/shared_cnn/models/nodule/model.json').read_text())
        class Client:
            def __init__(self):self.inputs=[]
            def health(self):return dict(shared_batch=dict(interface=BATCH_INTERFACE,max_patches=MAX_BATCH))
            def nodule_patch(self,image):
                self.inputs.append(image.copy())
                head=np.zeros(OUTPUT_SHAPE,np.uint8)
                return head,dict(stats=dict(host_total_ms=0,pl_command_cycles=0),roundtrip_ms=0)
            def nodule_batch(self,images):return [self.nodule_patch(image)for image in images]
        hu=np.random.default_rng(6).normal(-500,250,(9,48,48)).astype(np.float32)
        geom=dict(spacing_zyx_mm=[1.2,.7,.8],origin_lps_mm=[0,0,0],basis_lps_zyx=np.diag([1.2,.7,.8]).tolist())
        context=dict(analysis_source_id='original',source_hu_sha256=digest(hu.tobytes()),geometry_sha256='test',segmentation_signature='test')
        jobs=[(float(z),-32,-32)for z in range(10)]
        results=[];clients=[]
        with tempfile.TemporaryDirectory() as tmp,patch('nodule_detector.tiles',return_value=(jobs,dict(test_only=True))):
            for size in (1,8):
                p=NoduleCaseProvider.__new__(NoduleCaseProvider)
                p.client=Client();p.batch_size=size;p.model=model;p.bundle=bundle;p.config=config_from_dict(model['detector_config']);p.code_sha256={}
                folder=Path(tmp)/str(size);folder.mkdir()
                results.append(p.run(hu,np.ones(hu.shape,bool),geom,context,folder,lambda *a:None,lambda:False)[0])
                clients.append(p.client)
            self.assertEqual(results[0],results[1]);self.assertEqual(len(clients[0].inputs),10)
            for a,b in zip(clients[0].inputs,clients[1].inputs):np.testing.assert_array_equal(a,b)
            cancelled=Path(tmp)/'cancel';cancelled.mkdir()
            calls=len(p.client.inputs)
            with self.assertRaises(Cancelled):p.run(hu,np.ones(hu.shape,bool),geom,context,cancelled,lambda *a:None,lambda:True)
            self.assertEqual(len(p.client.inputs),calls)
            # Cancellation after submission finishes only the current bounded batch.
            flag=[False];original=p.client.nodule_batch
            def stop_after_batch(images):
                result=original(images);flag[0]=True;return result
            p.client.nodule_batch=stop_after_batch
            mid=Path(tmp)/'mid_cancel';mid.mkdir()
            with self.assertRaises(Cancelled):p.run(hu,np.ones(hu.shape,bool),geom,context,mid,lambda *a:None,lambda:flag[0])
            self.assertEqual(len(p.client.inputs)-calls,8)


if __name__=='__main__':unittest.main()
