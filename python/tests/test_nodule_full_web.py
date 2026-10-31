"""Canvas wire checks, unchanged scoring and actual INT8 numerical regression."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
PACKAGE=APP/'data/ct_studio_integration_20261025'
sys.path[:0]=[str(APP/'python'),str(PACKAGE/'ct_studio')]
from nodule_canvas_contract import inflate,shapes,quantize_canvas,digest
from nodule_frozen_scoring import load_scoring,score_candidates
from nodule_detector import plane_groups
from nodule_vendor.nodule_canvas import lower_canvas,extract_tile_heads
from nodule_vendor.nodule_fast_input import SharedPlaneSampler
from nodule_vendor.lung_nodule_cnn import config_from_dict,decode,consolidate
from nodule_vendor.lung_nodule_sampling_cache import sample_patch_reuse
from ct_nodule_canvas_client import CanvasClient


class Tests(unittest.TestCase):
    def setUp(self):
        self.model=json.loads((PACKAGE/'ct_studio/shared_cnn/models/nodule/model.json').read_text())
        self.config=config_from_dict(self.model['detector_config'])

    def test_payload_bounds_and_exact_compression(self):
        raw=bytes(range(256))*720
        self.assertEqual(inflate(zlib.compress(raw),len(raw)),raw)
        for packed,count in ((zlib.compress(raw)+b'x',len(raw)),(zlib.compress(raw),len(raw)-1),
                (zlib.compress(raw)[:-1],len(raw)),(zlib.compress(b'x'*10000),10)):
            with self.assertRaises(ValueError):inflate(packed,count)
        for dims in ((127,128),(128,129),(1032,128),(True,128)):
            with self.assertRaises(ValueError):shapes(*dims)

    def test_lung_schedule_split_preserves_every_job_order(self):
        jobs=[(3.,y,x) for y in range(-32,1600,64) for x in range(-32,1600,64)]
        groups=list(plane_groups(jobs,self.config))
        self.assertEqual([j for g in groups for j in g],jobs)
        self.assertGreater(len(groups),1)
        for g in groups:
            self.assertLessEqual(max(j[1]for j in g)-min(j[1]for j in g)+128,1024)
            self.assertLessEqual(max(j[2]for j in g)-min(j[2]for j in g)+128,1024)

    def test_sampler_and_quantization_match_old_patch(self):
        from nodule_raw_reference import FloatPlaneCache
        hu=np.random.default_rng(4).integers(-1500,600,(9,170,177),dtype=np.int16)
        jobs=[(4.7,-32,-32),(4.7,32,32),(4.7,96,32)]
        sampler=SharedPlaneSampler(hu,[1.5,.7,.8],jobs,self.config)
        cache=FloatPlaneCache(hu)
        q=quantize_canvas(sampler.context)
        for job in jobs:
            expected=sample_patch_reuse(cache,[1.5,.7,.8],job,self.config)
            np.testing.assert_array_equal(sampler.patch(job),expected)
            y=job[1]-sampler.y0;x=job[2]-sampler.x0
            np.testing.assert_array_equal(q[:,y:y+128,x:x+128],
                np.rint((expected.astype(np.float64).clip(-1000,400)+1000)*(255./1400.)).astype(np.uint8))

    def test_scoring_is_frozen_no_topk_or_center_movement(self):
        setup,ranker=load_scoring(PACKAGE)
        self.assertEqual(setup['threshold'],0.6418673229668569)
        hu=np.full((12,35,35),-900,np.int16)
        hu[3:9,12:22,12:22]=-300
        candidates=[dict(candidate_id='keep'+str(i),score=v,center_index_zyx=[6.,17.,17.],
            estimated_scale_diameter_mm=6.,bbox_zyx=[3,10,13,22,13,22])for i,v in enumerate((.02,.2,.6,.99))]
        from nodule_vendor.lung_nodule_continuity_boundary import evidence_for_packet
        from nodule_vendor.lung_nodule_continuity import rerank_scores
        features,_=evidence_for_packet(hu,[1.,1.,1.],candidates)
        expected=rerank_scores([c['score']for c in candidates],features,ranker)
        with tempfile.TemporaryDirectory()as folder:
            actual,info=score_candidates(hu,[1.,1.,1.],candidates,setup,ranker,folder)
            with gzip.open(Path(folder)/'all_ranked_candidates.json.gz','rt')as stream:allrows=json.load(stream)
        np.testing.assert_array_equal([r['score']for r in allrows],expected)
        self.assertEqual({r['candidate_id']for r in actual},
            {c['candidate_id']for c,v in zip(candidates,expected)if v>=setup['threshold']})
        self.assertIsNone(info['top_k_cap'])
        for a,c in zip(allrows,candidates):
            for k in ('candidate_id','center_index_zyx','bbox_zyx','estimated_scale_diameter_mm'):
                self.assertEqual(a[k],c[k])

    def test_actual_integer_canvas_neighbourhood_and_decode(self):
        # Independent existing integer-semantic network; no mocked convolutions.
        sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
        import torch
        torch.set_num_threads(2)
        from nodule_int8_cuda_reference import IntegerNodule
        with np.load(PACKAGE/'ct_studio/shared_cnn/models/nodule/parameters.npz')as f:
            params={k:f[k]for k in f.files}
        net=IntegerNodule(self.model,params,device='cpu')
        image=np.random.default_rng(24).integers(-1100,500,(5,256,256),dtype=np.int16)
        jobs=[(10.,0,0),(10.,64,64),(10.,128,128)]
        old=net(np.stack([image[:,y:y+128,x:x+128]for _,y,x in jobs]))
        full=net(image[None])[0]
        new=extract_tile_heads(full,jobs,0,0,self.config)
        np.testing.assert_array_equal(old[:,:,15:49,15:49],new[:,:,15:49,15:49])
        shape=(25,256,256);spacing=[1.,.8,.8]
        geom=dict(spacing_zyx_mm=spacing,origin_lps_mm=[0.,0.,0.],basis_lps_zyx=[[0.,0.,.8],[0.,.8,0.],[1.,0.,0.]])
        a=decode(old,jobs,geom,self.config,np.ones(shape,bool));b=decode(new,jobs,geom,self.config,np.ones(shape,bool))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(consolidate(*a,geom,self.config),consolidate(*b,geom,self.config))

    def test_client_rejects_wrong_bundle_or_model(self):
        bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
        c=CanvasClient('http://127.0.0.1:8766','a'*32,bundle)
        c.call=lambda *a,**k:dict(schema='ct_shared_nodule_canvas_v1',backend='FPGA_NODULE_SHARED')
        with self.assertRaises(ValueError):c.predict(np.zeros((45,64,64),np.uint8),128,128)

    def test_canvas_service_busy_and_fault_never_retry(self):
        import ct_nodule_canvas_service as server
        class Hardware:
            fault=None
            manifest=json.loads((PACKAGE/'ct_studio/platform_manifest.json').read_text())
            calls=0
            def canvas(self,*args):
                self.calls+=1;self.fault='test timeout';raise TimeoutError('test timeout')
        hw=Hardware();service=server.Service(hw,'a'*32,'bundle')
        raw=bytes(45*64*64)
        service.gate.acquire()
        try:
            with self.assertRaises(BlockingIOError):service.execute_canvas(raw,'a'*32,digest(raw),128,128)
        finally:service.gate.release()
        self.assertEqual(hw.calls,0)
        with self.assertRaises(TimeoutError):service.execute_canvas(raw,'a'*32,digest(raw),128,128)
        with self.assertRaises(RuntimeError):service.execute_canvas(raw,'b'*32,digest(raw),128,128)
        self.assertEqual(hw.calls,1)

    def test_provider_end_to_end_matches_patch_reference(self):
        sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
        import torch
        torch.set_num_threads(2)
        from nodule_int8_cuda_reference import IntegerNodule
        from nodule_raw_reference import FloatPlaneCache
        from nodule_detector import NoduleCaseProvider
        from nodule_vendor.lung_nodule_continuity_boundary import evidence_for_packet
        from nodule_vendor.lung_nodule_continuity import rerank_scores
        with np.load(PACKAGE/'ct_studio/shared_cnn/models/nodule/parameters.npz')as f:
            net=IntegerNodule(self.model,{k:f[k]for k in f.files},device='cpu')
        class OracleClient:
            def health(self):return {'test_only':True}
            def predict(self,codes,h,w):
                # Invert the first stem's stride-two lowering (no model change).
                q=np.empty((5,h,w),np.uint8)
                for c in range(5):
                    for y in range(2):
                        for x in range(2):q[c,y::2,x::2]=codes[c*9+(y+1)*3+x+1]
                head=net.forward_codes(torch.from_numpy(q[None]))[0].cpu().numpy()
                return head,dict(roundtrip_ms=0.,stats=dict(host_total_ms=0.,pl_command_cycles=0.,canvas_setup_ms=0.))
        owner=type('Legacy',(),{'url':'http://127.0.0.1:8766','token':'a'*32})()
        provider=NoduleCaseProvider(PACKAGE,owner);provider.client=OracleClient()
        hu=np.random.default_rng(9).integers(-1000,100,(24,256,256),dtype=np.int16)
        lung=np.ones_like(hu,bool)
        geom=dict(spacing_zyx_mm=[1.,.8,.8],origin_lps_mm=[0.,0.,0.],
            basis_lps_zyx=[[0.,0.,.8],[0.,.8,0.],[1.,0.,0.]])
        jobs=[(10.,0,0),(10.,64,64),(10.,128,128)]
        context=dict(analysis_source_id='original',source_hu_sha256='s',geometry_sha256='g',segmentation_signature='m')
        with tempfile.TemporaryDirectory()as folder,patch('nodule_detector.tiles',return_value=(jobs,{'test_fixture':True})):
            selected,provenance=provider.run(hu,lung,geom,context,folder,lambda *args:None,lambda:False)
            with gzip.open(Path(folder)/'all_ranked_candidates.json.gz','rt')as f:actual_all=json.load(f)
        from nodule_vendor.lung_nodule_cnn import search_mask
        cache=FloatPlaneCache(hu)
        patches=np.stack([sample_patch_reuse(cache,geom['spacing_zyx_mm'],j,self.config)for j in jobs])
        pred=net(patches)
        points=decode(pred,jobs,geom,self.config,search_mask(lung,geom['spacing_zyx_mm'],self.config))
        candidates=consolidate(*points,geom,self.config)
        features,_=evidence_for_packet(hu,geom['spacing_zyx_mm'],candidates)
        expected=rerank_scores([c['score']for c in candidates],features,provider.ranker)
        self.assertEqual(len(actual_all),len(candidates))
        for a,c,value in zip(actual_all,candidates,expected):
            self.assertEqual(a['candidate_id'],c['candidate_id'])
            self.assertEqual(a['center_index_zyx'],c['center_index_zyx'])
            self.assertEqual(a['bbox_zyx'],c['bbox_zyx'])
            self.assertEqual(a['score'],value)
        self.assertEqual(len(selected),int(np.count_nonzero(expected>=provider.scoring['threshold'])))
        self.assertEqual(provenance['transport']['canvases'],1)
        self.assertEqual(provenance['transport']['original_tiles'],3)


if __name__=='__main__':unittest.main()
