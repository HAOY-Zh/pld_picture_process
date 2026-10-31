"""Fixed policy across HTTP, imports and restored live results; no model calls."""
import copy,gzip,json,sys,tempfile,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib import request,error

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
if not (APP/'python/nodule_workspace.py').is_file():APP=ROOT
sys.path.insert(0,str(APP/'python'))
from nodule_threshold_policy import BASE_THRESHOLD,DEFAULT_THRESHOLD,LOW_FP_THRESHOLD,PRESETS,checked_threshold,select_candidates
from nodule_workspace import NoduleWorkspaceMixin,test_packet,fixed_selection_packet
from ct_workspace_resume import restore_nodule_result
LEGACY_MIDDLE=0.7655561641768475

class Harness(NoduleWorkspaceMixin):
    def __init__(self,folder):
        self.lock=threading.RLock();self.case_folder=Path(folder);self.status=dict(phase='done')
        self.context=dict(case_id='case',analysis_source_id='original',context_id='ctx',source_hu_sha256='a'*64,
            geometry_sha256='b'*64,segmentation_signature='c'*64,whole_case_ready=True,shape=[30,30,30],
            geometry=dict(origin_lps_mm=[0,0,0],basis_lps_zyx=[[1,0,0],[0,1,0],[0,0,1]],spacing_zyx_mm=[1,1,1]))
        self.nodule_result=test_packet(self.context)
        self.nodule_result.update(result_id='nodule_initial',is_test_data=False,ingestion_kind='computed_live_fpga')
        rows=self.nodule_result['candidates']
        for row,score in zip(rows,[BASE_THRESHOLD,LEGACY_MIDDLE,LOW_FP_THRESHOLD]):
            row.update(score=score,score_kind='frozen_cross_layer_ranked_score_not_malignancy_probability')
        self.nodule_result['detector_provenance']=dict(ranker_applied=True,additional_score_filter=BASE_THRESHOLD)
        folder=self.case_folder/'nodule/nodule_initial';folder.mkdir(parents=True)
        (folder/'prediction.json').write_text(json.dumps(self.nodule_result),encoding='utf8')
        with gzip.open(folder/'all_ranked_candidates.json.gz','wt',encoding='utf8')as f:json.dump(rows,f)
        self.nodule_packet_hashes={}
        self.nodule_job=dict(job_id='nodule_initial',status='COMPLETE',candidate_count=len(rows),case_id='case',analysis_source_id='original')
        (folder/'job.json').write_text(json.dumps(self.nodule_job),encoding='utf8')
    def nodule_context(self):return self.context
    def _update(self):pass
    def snapshot(self):return copy.deepcopy(self.nodule_result)
    def request(self,threshold=DEFAULT_THRESHOLD):return dict(**{k:self.context[k]for k in ('case_id','analysis_source_id','context_id')},
        result_id=self.nodule_result['result_id'],selection_threshold=threshold)

class ThresholdTests(unittest.TestCase):
    def test_only_fixed_low_fp_is_accepted_and_boundary_is_inclusive(self):
        self.assertEqual(DEFAULT_THRESHOLD,0.8738163050576156);self.assertEqual(PRESETS,(LOW_FP_THRESHOLD,))
        for invalid in [None,True,BASE_THRESHOLD,LEGACY_MIDDLE,.8738,float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):checked_threshold(invalid)
        rows=[dict(score=s,center_index_zyx=[0,0,0],candidate_id=str(i))for i,s in enumerate([BASE_THRESHOLD,LEGACY_MIDDLE,LOW_FP_THRESHOLD])]
        before=copy.deepcopy(rows);self.assertEqual(select_candidates(rows),[rows[-1]]);self.assertEqual(rows,before)
    def test_refilter_cannot_restore_lower_threshold(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);original=copy.deepcopy(h.nodule_result);strict=h.filter_nodule(h.request())
            self.assertEqual(strict['candidate_count'],1);self.assertEqual(strict['candidates'][0],original['candidates'][-1])
            self.assertNotEqual(strict['result_id'],original['result_id'])
            for lower in (BASE_THRESHOLD,LEGACY_MIDDLE):
                with self.assertRaises(ValueError):h.filter_nodule(h.request(lower))
            self.assertEqual(h.nodule_result,strict)
    def test_stale_context_and_active_task_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);r=h.request();r['result_id']='stale'
            with self.assertRaises(ValueError):h.filter_nodule(r)
            h.status['phase']='processing'
            with self.assertRaises(BlockingIOError):h.filter_nodule(h.request())
    def test_test_fixture_is_not_represented_as_detection(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);h.nodule_result['is_test_data']=True
            self.assertIs(fixed_selection_packet(h.nodule_result),h.nodule_result)
            with self.assertRaises(ValueError):h.filter_nodule(h.request())
    def test_imported_result_filtered_without_mutating_input(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);old=copy.deepcopy(h.nodule_result)
            payload=dict(**{k:h.context[k]for k in ('case_id','analysis_source_id','context_id')},packet=copy.deepcopy(old))
            result=h.nodule_action('import',payload)
            self.assertEqual(result['candidates'],[old['candidates'][-1]])
            self.assertEqual(result['detector_provenance']['additional_score_filter'],LOW_FP_THRESHOLD)
            self.assertNotEqual(result['result_id'],old['result_id']);self.assertEqual(payload['packet'],old)
    def test_restored_session_gets_distinct_filtered_result_without_inference(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);old=copy.deepcopy(h.nodule_result)
            summary={k:old[k]for k in ('case_id','analysis_source_id','context_id','result_id','candidate_count','is_test_data','ingestion_kind')}
            original=(h.case_folder/'nodule/nodule_initial/prediction.json').read_bytes()
            restore_nodule_result(h,dict(nodule=dict(result=summary)))
            self.assertEqual(h.nodule_result['candidates'],[old['candidates'][-1]])
            self.assertNotEqual(h.nodule_result['result_id'],old['result_id'])
            self.assertEqual(h.nodule_job['operation'],'fixed_threshold_restore_no_inference')
            self.assertEqual((h.case_folder/'nodule/nodule_initial/prediction.json').read_bytes(),original)
            self.assertEqual(json.loads((h.case_folder/'nodule'/h.nodule_result['result_id']/'prediction.json').read_text(encoding='utf8')),h.nodule_result)
    def test_already_fixed_result_keeps_identity(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);p=copy.deepcopy(h.nodule_result);p.update(candidates=p['candidates'][-1:],candidate_count=1)
            p['detector_provenance']['additional_score_filter']=LOW_FP_THRESHOLD
            self.assertIs(fixed_selection_packet(p),p)
    def test_real_http_cannot_request_a_lower_threshold(self):
        from ct_workspace_web import workspace_handler
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);server=ThreadingHTTPServer(('127.0.0.1',0),workspace_handler(h,'test-ui-token',b''))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            opener=request.build_opener(request.ProxyHandler({}));url='http://127.0.0.1:'+str(server.server_port)+'/api/nodule/filter'
            def send(values):
                req=request.Request(url,data=json.dumps(values).encode(),headers={'X-UI-Token':'test-ui-token','Content-Type':'application/json'})
                with opener.open(req,timeout=5)as r:return json.load(r)
            try:
                for lower in (BASE_THRESHOLD,LEGACY_MIDDLE):
                    with self.assertRaises(error.HTTPError)as caught:send(h.request(lower))
                    self.assertEqual(caught.exception.code,400)
                self.assertEqual(send(h.request())['candidate_count'],1)
            finally:server.shutdown();server.server_close();worker.join(3)

if __name__=='__main__':unittest.main()
