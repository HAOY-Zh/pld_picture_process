"""Synthetic job-lifecycle tests; actual FPGA acceptance uses separate evidence."""
import copy,json,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_ct_nodule_workspace import NoduleTests
from test_ct_roi_workspace import wait
from nodule_workspace import test_packet as fixture
from nodule_detector import Cancelled
from nodule_detector import FloatPlaneCache
from nodule_vendor.lung_nodule_cnn import DetectorConfig,sample_patch
from nodule_vendor.lung_nodule_sampling_cache import sample_patch_reuse
from ct_workspace_resume import restore_nodule_result


class OnlineTests(NoduleTests):
    def test_cached_sampling_is_bit_exact_with_frozen_reference(self):
        hu=np.random.default_rng(10).normal(-600,300,(9,32,40)).astype(np.float32)
        spacing=[1.2,.7,.9];cfg=DetectorConfig();cache=FloatPlaneCache(hu)
        for job in ((0.,-32,-32),(3.75,-10,-8),(5.,0,0),(3.75,-10,-8)):
            np.testing.assert_array_equal(sample_patch(hu,spacing,job,cfg),sample_patch_reuse(cache,spacing,job,cfg))
    def ready(self):
        self.c.known[:]=True;self.c.body[:]=True;self.c.raw[:]=True
        self.c.receipts={z:'SIMULATED_TEST_'+str(z) for z in range(len(self.c.hu))}
        self.c.status.update(full_volume_ready=True,board_connected=True)
        class Provider:
            def run(self,hu,lung,geometry,context,folder,progress,cancelled):
                progress('inference',1,1)
                return fixture(context)['candidates'],dict(backend='SIMULATED_UNIT_TEST',pipeline='raw_cnn_experimental_v1')
        self.c.nodule_provider=Provider()
    def start(self):return self.c.start_nodule(dict(self.identity(),scoring_mode='raw_cnn'))
    def test_live_current_enhanced_source_and_roi_reference(self):
        self.c.ldct_identity={'model_id':'SIMULATED_UNIT_TEST_MODEL'}
        self.c.ldct_groups['online']={z:dict(hu=np.rint(self.c.hu[z]).astype(np.int16),metadata={}) for z in range(len(self.c.hu))}
        self.c._register_enhanced_source('online','CPU_REFERENCE','bypass')
        self.c.select_analysis_source('enhanced:online');self.ready()
        before=self.c.hu.copy();source=self.c.analysis_source_id;self.start();s=wait(self.c)
        self.assertEqual(s['phase'],'done',s['error']);self.assertEqual(self.c.analysis_source_id,source)
        p=self.c.nodule_result;self.assertEqual(p['ingestion_kind'],'computed_live_fpga');self.assertFalse(p['is_test_data'])
        self.assertEqual(p['source_hu_sha256'],self.c.details['hu_sha256']);np.testing.assert_array_equal(before,self.c.hu)
        candidate=p['candidates'][0]
        ref=self.c.nodule_reference(dict(context_id=p['context_id'],result_id=p['result_id'],candidate_id=candidate['candidate_id'],padding_mm=3),[0,16,0,20],[0,5])
        self.assertEqual(ref['candidate_id'],candidate['candidate_id']);self.assertFalse(ref['is_test_data'])
        self.assertTrue((self.c.case_folder/'nodule'/p['result_id']/'prediction.json').is_file())
    def test_cancel_keeps_old_result_and_does_not_publish_partial(self):
        self.ready();self.install();old=copy.deepcopy(self.c.nodule_result);entered=threading.Event()
        class Provider:
            def run(self,hu,lung,geometry,context,folder,progress,cancelled):
                progress('inference',1,5);entered.set()
                while not cancelled():time.sleep(.005)
                raise Cancelled('TEST stop at safe boundary')
        self.c.nodule_provider=Provider();self.start();self.assertTrue(entered.wait(3))
        self.c.nodule_action('cancel',dict(job_id=self.c.nodule_job['job_id']))
        s=wait(self.c);self.assertEqual(s['phase'],'cancelled');self.assertEqual(self.c.nodule_result,old)
        self.assertEqual(self.c.nodule_job['status'],'CANCELLED')
    def test_hardware_failure_is_not_empty_success_or_cpu_fallback(self):
        self.ready()
        class Provider:
            def run(self,*args):raise TimeoutError('TEST hardware uncertainty')
        self.c.nodule_provider=Provider();self.start();s=wait(self.c)
        self.assertEqual(s['phase'],'error');self.assertFalse(s['board_connected']);self.assertIsNone(self.c.nodule_result)
        self.assertEqual(self.c.nodule_job['status'],'FAILED')
    def test_empty_completed_case_is_valid(self):
        self.ready()
        class Provider:
            def run(self,*args):return [],dict(backend='SIMULATED_UNIT_TEST')
        self.c.nodule_provider=Provider();self.start();s=wait(self.c)
        self.assertEqual(s['phase'],'done');self.assertEqual(self.c.nodule_result['candidate_count'],0)
        self.assertEqual(self.c.nodule_job['status'],'COMPLETE')
    def test_missing_full_mask_and_implicit_scoring_are_rejected(self):
        self.ready();self.c.known[0]=False;self.c.status['full_volume_ready']=False
        with self.assertRaises(ValueError):self.start()
        self.c.known[:]=True;self.c.status['full_volume_ready']=True
        with self.assertRaises(ValueError):self.c.start_nodule(self.identity())

    def test_saved_live_result_restores_without_inference_and_rejects_stale_or_partial(self):
        self.ready()
        packet=fixture(self.c.nodule_context());packet['is_test_data']=False
        packet['ingestion_kind']='computed_live_fpga'
        folder=self.c.case_folder/'nodule'/packet['result_id'];folder.mkdir(parents=True)
        (folder/'prediction.json').write_text(json.dumps(packet),encoding='utf8')
        job=dict(job_id=packet['result_id'],status='COMPLETE',candidate_count=packet['candidate_count'],
            case_id=packet['case_id'],analysis_source_id=packet['analysis_source_id'])
        (folder/'job.json').write_text(json.dumps(job),encoding='utf8')
        selected={k:packet[k]for k in ('case_id','analysis_source_id','context_id','result_id','candidate_count','is_test_data','ingestion_kind')}
        state=dict(nodule=dict(result=selected))
        with patch.object(self.c.nodule_provider,'run',side_effect=AssertionError('must not rerun')):
            restore_nodule_result(self.c,state)
        self.assertEqual(self.c.nodule_result,packet);self.assertEqual(self.c.nodule_job,job)
        self.c.nodule_generation+=1
        with self.assertRaises(ValueError):restore_nodule_result(self.c,state)
        self.c.nodule_generation-=1;job['status']='FAILED'
        (folder/'job.json').write_text(json.dumps(job),encoding='utf8')
        with self.assertRaises(ValueError):restore_nodule_result(self.c,state)


if __name__=='__main__':unittest.main()
