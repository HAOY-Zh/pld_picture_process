"""Preset selection and context-safe re-filtering; no model inference."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib import request,error

APP=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(APP/'python'))
from nodule_threshold_policy import BASE_THRESHOLD,DEFAULT_THRESHOLD,LOW_FP_THRESHOLD,checked_threshold,select_candidates
from nodule_workspace import NoduleWorkspaceMixin,test_packet


class Harness(NoduleWorkspaceMixin):
    def __init__(self,folder):
        self.lock=threading.RLock();self.case_folder=Path(folder);self.status=dict(phase='done')
        self.context=dict(case_id='case',analysis_source_id='original',context_id='ctx',source_hu_sha256='a'*64,
            geometry_sha256='b'*64,segmentation_signature='c'*64,whole_case_ready=True,shape=[30,30,30],
            geometry=dict(origin_lps_mm=[0,0,0],basis_lps_zyx=[[1,0,0],[0,1,0],[0,0,1]],spacing_zyx_mm=[1,1,1]))
        self.nodule_result=test_packet(self.context);self.nodule_result.update(result_id='nodule_initial',is_test_data=False,ingestion_kind='computed_live_fpga')
        rows=self.nodule_result['candidates']
        for row,score in zip(rows,[BASE_THRESHOLD,DEFAULT_THRESHOLD,LOW_FP_THRESHOLD]):
            row.update(score=score,score_kind='frozen_cross_layer_ranked_score_not_malignancy_probability')
        self.nodule_result['detector_provenance']=dict(ranker_applied=True,additional_score_filter=BASE_THRESHOLD)
        folder=self.case_folder/'nodule/nodule_initial';folder.mkdir(parents=True)
        (folder/'prediction.json').write_text(json.dumps(self.nodule_result),encoding='utf8')
        with gzip.open(folder/'all_ranked_candidates.json.gz','wt',encoding='utf8')as f:json.dump(rows,f)
        self.nodule_packet_hashes={};self.nodule_job={}
    def nodule_context(self):return self.context
    def _update(self):pass
    def snapshot(self):return copy.deepcopy(self.nodule_result)
    def request(self,threshold):return dict(**{k:self.context[k]for k in ('case_id','analysis_source_id','context_id')},
        result_id=self.nodule_result['result_id'],selection_threshold=threshold)


class ThresholdTests(unittest.TestCase):
    def test_presets_are_finite_explicit_and_nested(self):
        for invalid in [None,True,.7,float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):checked_threshold(invalid)
        rows=[dict(score=s,center_index_zyx=[0,0,0],candidate_id=str(i))for i,s in enumerate([BASE_THRESHOLD,DEFAULT_THRESHOLD,LOW_FP_THRESHOLD])]
        before=copy.deepcopy(rows)
        self.assertEqual(len(select_candidates(rows,DEFAULT_THRESHOLD)),2)
        self.assertEqual(rows,before)
    def test_refilter_and_restore_original_without_inference_or_geometry_change(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);original=copy.deepcopy(h.nodule_result)
            strict=h.filter_nodule(h.request(DEFAULT_THRESHOLD))
            self.assertEqual(strict['candidate_count'],2)
            self.assertNotEqual(strict['result_id'],original['result_id'])
            self.assertEqual(h.nodule_job['operation'],'threshold_only_no_inference')
            restored=h.filter_nodule(h.request(BASE_THRESHOLD))
            self.assertEqual(restored['candidates'],select_candidates(original['candidates'],BASE_THRESHOLD))
            self.assertEqual(restored['candidate_count'],3)
            self.assertEqual(restored['detector_provenance']['selection_source_result_id'],'nodule_initial')
            self.assertEqual(original['candidates'][1],next(c for c in strict['candidates']if c['candidate_id']==original['candidates'][1]['candidate_id']))
    def test_stale_context_and_active_task_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);r=h.request(DEFAULT_THRESHOLD);r['result_id']='stale'
            with self.assertRaises(ValueError):h.filter_nodule(r)
            h.status['phase']='processing'
            with self.assertRaises(BlockingIOError):h.filter_nodule(h.request(DEFAULT_THRESHOLD))
    def test_test_fixture_is_not_represented_as_detection(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);h.nodule_result['is_test_data']=True
            with self.assertRaises(ValueError):h.filter_nodule(h.request(DEFAULT_THRESHOLD))

    def test_real_http_filter_route_and_stale_result_rejection(self):
        from ct_workspace_web import workspace_handler
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);server=ThreadingHTTPServer(('127.0.0.1',0),workspace_handler(h,'test-ui-token',b''))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            opener=request.build_opener(request.ProxyHandler({}))
            url='http://127.0.0.1:'+str(server.server_port)+'/api/nodule/filter'
            def send(values):
                req=request.Request(url,data=json.dumps(values).encode(),headers={'X-UI-Token':'test-ui-token','Content-Type':'application/json'})
                with opener.open(req,timeout=5)as r:return json.load(r)
            try:
                old=h.request(DEFAULT_THRESHOLD);strict=send(old);self.assertEqual(strict['candidate_count'],2)
                with self.assertRaises(error.HTTPError)as caught:send(old)
                self.assertEqual(caught.exception.code,400)
                restored=send(h.request(BASE_THRESHOLD));self.assertEqual(restored['candidate_count'],3)
            finally:server.shutdown();server.server_close();worker.join(3)


if __name__=='__main__':unittest.main()
