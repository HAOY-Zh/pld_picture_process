"""Synthetic geometry/protocol/ROI integration tests. No detection or board."""
import copy
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib import request, error

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_workspace_web import WorkspaceController, workspace_handler
from nodule_workspace import test_packet as fixture_packet, display_bounds
from test_ct_roi_workspace import wait
from test_lung_live import RunningServer


class NoduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.path = self.root/'ct.npz'
        np.savez(self.path, hu=np.arange(5*16*20,dtype=np.float32).reshape(5,16,20)-900,
            spacing_zyx_mm=[3.,.5,.8], origin_lps_mm=[39.,-27.,80.], orientation=[-1,0,0,0,1,0])
        self.c = WorkspaceController(None, {}, self.root/'results')
        self.load()

    def load(self):
        self.c.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False))
        self.assertEqual(wait(self.c)['phase'],'idle')

    def tearDown(self):
        wait(self.c); self.tmp.cleanup()

    def identity(self):
        ctx = self.c.nodule_context()
        return {k:ctx[k] for k in ('case_id','analysis_source_id','context_id')}

    def install(self, packet=None):
        packet = packet or fixture_packet(self.c.nodule_context())
        self.c.nodule_action('import',dict(self.identity(),packet=packet))
        return self.c.nodule_result

    def test_lps_direction_origin_anisotropy_and_clipping_preserve_raw_candidate(self):
        p=fixture_packet(self.c.nodule_context());c=p['candidates'][0]
        c['bbox_zyx']=[-2,8,-5,20,-3,24];c['raw_score']=.88
        frozen=copy.deepcopy(c);r=self.install(p)
        self.assertEqual(r['candidates'][0],frozen)
        self.assertEqual(display_bounds(c,[5,16,20],[3,.5,.8],3),[0,5,0,16,0,20])
        point=np.array(c['center_index_zyx']);world=np.array([39.,-27.,80.])+np.array(self.c.geom['basis_lps_zyx'])@point
        np.testing.assert_allclose(c['center_lps_xyz_mm'],world)
        self.assertEqual(r['candidates'],p['candidates'])

    def test_mip_matches_native_hu_crop_without_changing_candidate_or_model_state(self):
        before=self.c.hu.copy();r=self.install();candidate=r['candidates'][0];frozen=copy.deepcopy(r)
        args=[r[k] for k in ('case_id','analysis_source_id','context_id','result_id')]+[candidate['candidate_id']]
        result=self.c.nodule_mip(*args,0)
        z0,z1,y0,y1,x0,x1=result['bbox_zyx']
        actual=np.frombuffer(base64.b64decode(result['hu']),'<f4').reshape(result['shape_yx'])
        np.testing.assert_array_equal(actual,before[z0:z1,y0:y1,x0:x1].max(axis=0))
        self.assertEqual(result['backend'],'CPU_DISPLAY_ONLY');self.assertTrue(result['is_test_data'])
        self.assertEqual(self.c.nodule_mip(*args,50)['bbox_zyx'],[0,5,0,16,0,20])
        self.assertEqual(self.c.nodule_result,frozen);np.testing.assert_array_equal(self.c.hu,before)
        self.assertFalse(self.c.known.any());self.assertEqual(self.c.receipts,{})
        for pad in (True,float('nan'),-1,51):
            with self.assertRaises(ValueError):self.c.nodule_mip(*args,pad)
        self.c.nodule_action('clear',self.identity())
        with self.assertRaises(ValueError):self.c.nodule_mip(*args,0)

    def test_stale_identity_and_inconsistent_geometry_rejected(self):
        for key in ('case_id','analysis_source_id','context_id','source_hu_sha256','geometry_sha256','segmentation_signature'):
            p=fixture_packet(self.c.nodule_context());p[key]='other'
            with self.assertRaises(ValueError):self.install(p)
        p=fixture_packet(self.c.nodule_context());p['candidates'][0]['center_lps_xyz_mm'][0]+=1
        with self.assertRaises(ValueError):self.install(p)
        self.assertIsNone(self.c.nodule_result)

    def test_fractional_boundary_matches_original_decoder_without_moving_center(self):
        p=fixture_packet(self.c.nodule_context());c=p['candidates'][0]
        c['center_index_zyx']=[-.49,15.49,19.49];c['bbox_zyx']=[-2,2,14,18,18,22]
        c['center_lps_xyz_mm']=(np.asarray(self.c.geom['origin_lps_mm'])+
            np.asarray(self.c.geom['basis_lps_zyx'])@c['center_index_zyx']).tolist()
        result=self.install(p)
        self.assertEqual(result['candidates'][0]['center_index_zyx'],[-.49,15.49,19.49])

    def test_invalid_packets_never_turn_into_display_candidates(self):
        for change in ('duplicate','nan','bool','count','bounds','diameter','region','test_flag'):
            p=fixture_packet(self.c.nodule_context());c=p['candidates'][0]
            if change=='duplicate':p['candidates'][1]['candidate_id']=c['candidate_id']
            elif change=='nan':c['score']=float('nan')
            elif change=='bool':c['bbox_zyx'][0]=False
            elif change=='count':p['candidate_count']=99
            elif change=='bounds':c['center_index_zyx'][0]=-1
            elif change=='diameter':c['estimated_scale_diameter_mm']=0
            elif change=='region':c['region_kind']='precise_segmentation'
            else:del p['is_test_data']
            with self.subTest(change=change),self.assertRaises(ValueError):self.install(p)

    def test_empty_result_is_retained_and_previous_selection_data_removed(self):
        old=self.install();empty=fixture_packet(self.c.nodule_context(),empty=True)
        self.install(empty);self.assertEqual(self.c.snapshot()['nodule']['result']['candidate_count'],0)
        with self.assertRaises(ValueError):self.c.nodule_payload(old['case_id'],old['analysis_source_id'],old['context_id'],old['result_id'])

    def test_same_result_id_is_immutable_and_segmentation_change_invalidates(self):
        old=self.install();changed=copy.deepcopy(old);changed['candidates'][0]['score']+=.01
        with self.assertRaisesRegex(ValueError,'result_id'):self.install(changed)
        self.c.receipts[0]='new-segmentation-receipt'
        self.assertIsNone(self.c.snapshot()['nodule']['result'])
        with self.assertRaises(ValueError):self.install(old)

    def test_imported_real_result_requires_whole_original_ct_and_explicit_origin(self):
        p=fixture_packet(self.c.nodule_context());p['is_test_data']=False
        with self.assertRaisesRegex(ValueError,'完整'):self.install(p)
        self.c.known[:]=True;self.c.status['full_volume_ready']=True
        p=fixture_packet(self.c.nodule_context());p['is_test_data']=False
        self.assertEqual(self.install(p)['ingestion_kind'],'imported_result')
        source_hu=self.c.hu.copy();self.c.ldct_identity={'model_id':'TEST ONLY'}
        self.c.ldct_group='test';self.c.ldct_groups['test']={z:{'hu':source_hu[z].astype(np.int16),'metadata':{}} for z in range(5)}
        self.c._register_enhanced_source('test','CPU_REFERENCE','bypass')
        self.c.select_analysis_source('enhanced:test')
        self.assertIsNone(self.c.snapshot()['nodule']['result'])
        with self.assertRaises(ValueError):self.install(p)

    def test_roi_records_candidate_provenance_without_rewriting_hu_or_masks(self):
        before=self.c.hu.copy();r=self.install();c=r['candidates'][0]
        b=display_bounds(c,list(self.c.hu.shape),self.c.geom['spacing_zyx_mm'],3)
        values=dict(case_id=r['case_id'],analysis_source_id=r['analysis_source_id'],index=round(c['center_index_zyx'][0]),
            roi_yx=b[2:],z_bounds=b[:2],mode='range',backend='CPU_REFERENCE',use_hu=False,use_lung_mask=False,
            candidate_reference=dict(context_id=r['context_id'],result_id=r['result_id'],candidate_id=c['candidate_id'],padding_mm=3))
        self.c.submit('roi',values);self.assertEqual(wait(self.c)['phase'],'done')
        ref=self.c.roi_result['candidate_reference']
        self.assertEqual(ref['original_bbox_zyx'],c['bbox_zyx']);self.assertTrue(ref['is_test_data']);self.assertFalse(ref['user_adjusted'])
        self.assertEqual(self.c.roi_result['valid_count'],int(np.prod([b[a+1]-b[a] for a in (0,2,4)])))
        np.testing.assert_array_equal(self.c.hu,before);self.assertFalse(self.c.known.any())
        values['roi_yx']=[0,1,0,1];self.c.submit('roi',values);wait(self.c)
        self.assertTrue(self.c.roi_result['candidate_reference']['user_adjusted'])
        self.c.nodule_action('clear',self.identity())
        with self.assertRaises(ValueError):self.c.submit('roi',values)

    def test_new_case_and_busy_import_do_not_accept_old_result(self):
        r=self.install();self.c.status['phase']='processing'
        with self.assertRaises(BlockingIOError):self.install()
        self.c.status['phase']='idle';self.load()
        self.assertIsNone(self.c.nodule_result)
        with self.assertRaises(ValueError):self.install(r)

    def test_http_auth_context_result_and_no_fake_detection(self):
        token='nodule-tests-only'
        server=RunningServer(workspace_handler(self.c,token,b'/* test stub */'))
        opener=request.build_opener(request.ProxyHandler({}))
        def call(path,data=None,auth=token):
            req=request.Request(server.url+path,data=json.dumps(data).encode() if data is not None else None,
                headers={'X-UI-Token':auth,'Content-Type':'application/json'})
            return json.load(opener.open(req,timeout=5))
        try:
            with self.assertRaises(error.HTTPError) as failure:call('/api/nodule/context',auth='wrong')
            self.assertEqual(failure.exception.code,403)
            self.assertEqual(call('/api/nodule/context')['context'],self.c.nodule_context())
            state=call('/api/nodule/fixture',self.identity());r=state['nodule']['result']
            from urllib.parse import urlencode
            query=urlencode(dict(case=r['case_id'],source=r['analysis_source_id'],context=r['context_id'],result=r['result_id']))
            self.assertEqual(call('/api/nodule/result?'+query)['candidate_count'],3)
            cid=self.c.nodule_result['candidates'][0]['candidate_id']
            mip_query=query+'&'+urlencode(dict(candidate=cid,padding=3))
            with self.assertRaises(error.HTTPError):call('/api/nodule/mip?'+mip_query,auth='wrong')
            self.assertEqual(call('/api/nodule/mip?'+mip_query)['backend'],'CPU_DISPLAY_ONLY')
            with self.assertRaises(error.HTTPError) as failure:call('/api/nodule/mip?'+mip_query.replace('padding=3','padding=nan'))
            self.assertEqual(failure.exception.code,400)
            with self.assertRaises(error.HTTPError) as failure:call('/api/nodule/run',self.identity())
            self.assertEqual(failure.exception.code,501)
            self.assertEqual(call('/api/nodule/clear',self.identity())['nodule']['result'],None)
        finally:server.close()


if __name__=='__main__':unittest.main()
