"""Exact native-CMA layout, binary wire guards and full detection equivalence."""
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_NoduleCompact_20261026_01'
PACKAGE=APP/'data/ct_studio_integration_20261025'
BOARD=PACKAGE/'ct_studio'
sys.path[:0]=[str(APP/'python'),str(BOARD)]
from nodule_compact_contract import encode_response,decode_response,digest
from nodule_vendor.nodule_canvas import lower_canvas
from shared_cnn.lung_axil_smoke import feature_words
from shared_cnn.nodule_quantized_pack import compile_pack,QuantizedPacker


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=ROOT/'data/nodule_transport_upgrade_20261026/native_test'
        folder.mkdir(exist_ok=True)
        source=ROOT/'pynq/nodule_quantized_pack.cpp'
        destination=folder/source.name
        if not destination.exists():shutil.copy2(source,destination)
        elif destination.read_bytes()!=source.read_bytes():raise ValueError('Native test source changed')
        compiler='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe'
        cls.packer=QuantizedPacker(compile_pack(folder,compiler))

    def test_native_pack_all_bytes_and_tail_canaries(self):
        rng=np.random.default_rng(20261026)
        for h,w in ((128,128),(256,320),(448,512),(128,1024)):
            for mode in ('random','zero','max'):
                q=rng.integers(0,256,(5,h,w),dtype=np.uint8)if mode=='random'else np.full((5,h,w),0 if mode=='zero' else 255,np.uint8)
                original=q.copy();expected=feature_words(lower_canvas(q),0)
                output=np.full(len(expected)+16,np.uint64(0xa5c3917e),np.uint64)
                self.packer.pack(q,output)
                np.testing.assert_array_equal(output[:len(expected)],expected)
                np.testing.assert_array_equal(output[len(expected):],np.full(16,np.uint64(0xa5c3917e)))
                np.testing.assert_array_equal(q,original)

    def test_reject_small_destination_and_shared_memory(self):
        q=np.zeros((5,128,128),np.uint8)
        with self.assertRaises(ValueError):self.packer.pack(q,np.zeros(16,np.uint8))
        with self.assertRaises(ValueError):self.packer.pack(q,q.reshape(-1))

    def test_binary_protocol_rejects_truncation_trailing_and_bad_length(self):
        head=bytes(range(256))*80;encoded=encode_response({'request_id':'a'},head)
        meta,value=decode_response(encoded,len(head));self.assertEqual(value,head)
        for data in (encoded[:-1],encoded+b'extra',b'x',struct.pack('!I',100000)+encoded[4:]):
            with self.assertRaises((ValueError,zlib.error)):decode_response(data,len(head))

    def test_shared_scheduler_only_upload_contract_changed(self):
        import ast
        original=(BOARD/'shared_cnn/lung_fast.py').read_text()
        original=original[original.index('    def predict_codes(self,image,observer=None):'):]
        actual=(BOARD/'shared_cnn/nodule_canvas_packed.py').read_text()
        actual=actual[actual.index('    def predict_quantized(self,image,observer=None):'):]
        actual=actual.replace('def predict_quantized(self,image,observer=None):','def predict_codes(self,image,observer=None):',1)
        actual=actual.replace('image.shape!=(5,self.height,self.width)',"list(image.shape)!=self.model['input_shape']")
        actual=actual.replace("self.quantized_packer.pack(image,src[:word_count(self.model['input_shape'])])",
            "src[:word_count(self.model['input_shape'])]=feature_words(image,self.model['input_zero'])")
        self.assertEqual(ast.dump(ast.parse('class X:\n'+original)),ast.dump(ast.parse('class X:\n'+actual)))

    def test_binary_http_persistent_connection_and_fault_retention(self):
        import threading
        from http.server import ThreadingHTTPServer
        import ct_nodule_compact_service as api
        from ct_nodule_compact_client import CompactClient
        bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
        class Hardware:
            fault=None
            calls=0
            fail=False
            manifest=json.loads((BOARD/'platform_manifest.json').read_text())
            def compact(self,q,h,w):
                self.calls+=1
                if self.fail:self.fault='simulated timeout';raise TimeoutError(self.fault)
                return np.zeros((5,h//2,w//2),np.uint8),dict(command_count=15,commands=[{}for _ in range(15)],
                    selected_model='nodule_canvas',overlay_reloaded=False,buffer_allocations_per_frame=0,
                    host_total_ms=0.,shared_call_ms=0.,overlay_switch_ms=0.,canvas_setup_ms=0.,pl_command_cycles=0,
                    actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,cpu_mhz=650.)
        hw=Hardware();service=api.Service(hw,'a'*32,bundle['service_bundle_sha256'])
        server=ThreadingHTTPServer(('127.0.0.1',0),api.handler_for(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=CompactClient('http://127.0.0.1:'+str(server.server_port),'a'*32,bundle)
        try:
            q=np.zeros((5,128,128),np.uint8)
            client.predict(q,128,128);connection=client.connection.sock
            result,_=client.predict(q,128,128)
            self.assertIs(client.connection.sock,connection)
            np.testing.assert_array_equal(result,np.zeros((5,64,64),np.uint8))
            hw.fail=True
            with self.assertRaises(RuntimeError):client.predict(q,128,128)
            self.assertEqual(hw.calls,3)
            with self.assertRaises(RuntimeError):client.predict(q,128,128)
            self.assertEqual(hw.calls,3)
        finally:
            if client.connection is not None:client.connection.close()
            server.shutdown();server.server_close();thread.join(timeout=2)

    def test_full_provider_is_identical_to_old_canvas_integer_reference(self):
        sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
        import torch
        torch.set_num_threads(2)
        from nodule_int8_cuda_reference import IntegerNodule
        from nodule_detector import NoduleCaseProvider
        from nodule_vendor.lung_nodule_cnn import decode,consolidate,search_mask
        from nodule_vendor.lung_nodule_continuity_boundary import evidence_for_packet
        from nodule_vendor.lung_nodule_continuity import rerank_scores
        import gzip
        model=json.loads((BOARD/'shared_cnn/models/nodule/model.json').read_text())
        with np.load(BOARD/'shared_cnn/models/nodule/parameters.npz')as a:
            net=IntegerNodule(model,{k:a[k]for k in a.files},device='cpu')
        class IntegerClient:
            def health(self):return {'software_test_only':True}
            def predict(self,q,h,w):
                self_shape=q.shape
                if self_shape!=(5,h,w):raise ValueError('Compact input wrong')
                head=net.forward_codes(torch.from_numpy(q[None]))[0].cpu().numpy()
                return head,dict(roundtrip_ms=0.,stats=dict(host_total_ms=0.,pl_command_cycles=0.,canvas_setup_ms=0.))
        owner=type('Legacy',(),{'url':'http://127.0.0.1:8766','token':'a'*32})()
        provider=NoduleCaseProvider(PACKAGE,owner);provider.client=IntegerClient()
        hu=np.random.default_rng(9).integers(-1000,100,(24,256,256),dtype=np.int16);lung=np.ones_like(hu,bool)
        geom=dict(spacing_zyx_mm=[1.,.8,.8],origin_lps_mm=[0.,0.,0.],basis_lps_zyx=[[0.,0.,.8],[0.,.8,0.],[1.,0.,0.]])
        jobs=[(10.,0,0),(10.,64,64),(10.,128,128)]
        context=dict(analysis_source_id='original',source_hu_sha256='s',geometry_sha256='g',segmentation_signature='m')
        with tempfile.TemporaryDirectory()as folder,patch('nodule_detector.tiles',return_value=(jobs,{})):
            selected,record=provider.run(hu,lung,geom,context,folder,lambda *args:None,lambda:False)
            with gzip.open(Path(folder)/'all_ranked_candidates.json.gz','rt')as f:actual=json.load(f)
        from nodule_vendor.nodule_fast_input import SharedPlaneSampler
        sampler=SharedPlaneSampler(hu,geom['spacing_zyx_mm'],jobs,provider.config)
        predictions=net(np.stack([sampler.patch(j)for j in jobs]))
        points=decode(predictions,jobs,geom,provider.config,search_mask(lung,geom['spacing_zyx_mm'],provider.config))
        candidates=consolidate(*points,geom,provider.config)
        features,_=evidence_for_packet(hu,geom['spacing_zyx_mm'],candidates)
        scores=rerank_scores([c['score']for c in candidates],features,provider.ranker)
        for a,b,score in zip(actual,candidates,scores):
            self.assertEqual(a['score'],score);self.assertEqual(a['center_index_zyx'],b['center_index_zyx']);self.assertEqual(a['raw_score'],b['score'])
        self.assertEqual(len(actual),len(candidates))
        self.assertEqual(len(selected),int(np.count_nonzero(scores>=provider.scoring['threshold'])))


if __name__=='__main__':unittest.main()
