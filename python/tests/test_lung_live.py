"""Transport/geometry/orchestration tests. Fake hardware is TEST ONLY, not a fallback."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib import request, error

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'pynq'))
from lung_live_data import read_ct, input_codes, native_masks, checked_shape, parse_slices
from lung_live_web import BoardClient, Controller, handler_for, BIT_SHA, MODEL_SHA, reconstruction_plan
from lung_live_service import InferenceService, handler_for as board_handler
from lung_volume_viewer import geometry, nifti_export
from lung_unet import torch_module
from http.server import ThreadingHTTPServer

MODEL = dict(input_quantization=dict(shape=[1,1,256,256], dtype='torch.quint8',
    scale=.007870171219110489, zero_point=0), output_scale=.3118481934070587, output_zero=72)
TOKEN = 'test-only-not-a-deployment-token-' + 'a'*32


class FakeHardware:
    """Deterministic stub; never used by production entrypoints."""
    def __init__(self):
        self.calls = 0; self.fail = False; self.capture = None
        self.identity = dict(schema='lung_live_identity_v1', backend='FPGA_INT8', bit_sha256=BIT_SHA,
            model_sha256=MODEL_SHA, startup_nodes_checked=[str(n) for n in range(21)],
            startup_command_count=51, **MODEL)

    def infer(self, codes):
        self.calls += 1; self.capture = codes.copy()
        if self.fail: raise TimeoutError('TEST simulated AXI failure')
        out = np.full((1,256,256), 60, np.uint8); out[:,40:210,60:200] = 85
        clocks = dict(fclk0_mhz=100., cpu_mhz=650.)
        return out, dict(command_count=51, commands=[dict(node=str(i)) for i in range(51)],
            clock_before=clocks, clock_after=clocks, host_total_ms=.5)


class RunningServer:
    def __init__(self, handler):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def wait(controller):
    until = time.monotonic() + 20
    while controller.snapshot()['phase'] in ('connecting','loading','computing','reconstructing'):
        if time.monotonic() > until: raise AssertionError('Worker timeout')
        time.sleep(.005)
    return controller.snapshot()


class DataTests(unittest.TestCase):
    def setUp(self):
        torch_module().set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory(); self.folder = Path(self.temp.name)
        self.hu = np.linspace(-1000,300,3*32*40,dtype=np.float32).reshape(3,32,40)
        self.geom = geometry([10,20,30], [1,0,0,0,1,0], [2.5,.8,.7])

    def tearDown(self): self.temp.cleanup()

    def npz(self, **extra):
        path = self.folder / 'ct.npz'
        values = dict(hu=self.hu, spacing_zyx_mm=[2.5,.8,.7],origin_lps_mm=[10,20,30],orientation=[1,0,0,0,1,0])
        values.update(extra); np.savez(path, **values); return path

    def test_npz_requires_hu_confirmation_and_explicit_geometry(self):
        p = self.npz()
        with self.assertRaises(ValueError): read_ct(p)
        hu,g,info = read_ct(p, True)
        np.testing.assert_array_equal(hu,self.hu); self.assertEqual(g,self.geom)
        self.assertFalse(info['annotations_loaded'])

    def test_npz_rejects_stored_prediction_input(self):
        with self.assertRaises(ValueError): read_ct(self.npz(prediction=np.zeros(self.hu.shape)), True)

    def test_slice_selection_inclusive_sorted_unique_and_bounded(self):
        self.assertEqual(parse_slices('5, 2-4,3，0', 8), [0,2,3,4,5])
        self.assertEqual(parse_slices('', 3, empty_all=True), [0,1,2])
        for value in ['', '3-1', '-1', '0-3', '1,,2', '__import__("os")', '1.0', None]:
            with self.assertRaises(ValueError): parse_slices(value, 3)

    def test_partial_npz_keeps_source_indices_and_shifts_physical_origin(self):
        hu,g,info=read_ct(self.npz(), True, slices='1-2')
        np.testing.assert_array_equal(hu,self.hu[1:3])
        self.assertEqual(info['source_slice_indices'],[1,2]);self.assertTrue(info['partial_input'])
        self.assertEqual(info['source_total_slices'],3)
        np.testing.assert_allclose(g['origin_lps_mm'],[10,20,32.5])
        with self.assertRaisesRegex(ValueError,'连续'):read_ct(self.folder/'ct.npz',True,slices='0,2')

    def test_partial_nifti_preserves_subset_world_coordinates(self):
        p=self.folder/'ct.nii.gz';nifti_export(self.hu,self.geom,p)
        hu,g,info=read_ct(p,True,slices='1-2')
        np.testing.assert_array_equal(hu,self.hu[1:3,::-1,::-1])
        np.testing.assert_allclose(g['origin_lps_mm'],[10+39*.7,20+31*.8,32.5],atol=1e-5)
        self.assertEqual(info['source_slice_indices'],[1,2])

    def test_partial_dicom_decodes_only_selected_plane_sorted_by_position(self):
        from test_lung_segmentation import make_case
        import pydicom
        make_case(self.folder)
        read=pydicom.dcmread;decoded=[]
        def track(path,*args,**kwargs):
            ds=read(path,*args,**kwargs)
            if not kwargs.get('stop_before_pixels'):decoded.append(Path(path).name)
            return ds
        with patch.object(pydicom,'dcmread',side_effect=track):
            hu,g,info=read_ct(self.folder,slices='1')
        self.assertEqual(decoded,['slice_0.dcm'])  # filename order is intentionally reversed.
        self.assertEqual(hu.shape,(1,20,30));self.assertEqual(info['source_slice_indices'],[1])
        self.assertTrue(info['partial_input']);self.assertEqual(g['origin_lps_mm'],[10,20,33])

    def test_npz_rejects_nonfinite_hu(self):
        hu=self.hu.copy(); hu[0,0,0]=np.nan
        with self.assertRaises(ValueError): read_ct(self.npz(hu=hu), True)

    def test_nifti_reorientation_preserves_physical_grid_and_values(self):
        p=self.folder/'ct.nii.gz'; nifti_export(self.hu,self.geom,p)
        hu,g,info=read_ct(p,True)
        # Canonical RAS+ changes native X/Y storage order; physical values agree.
        np.testing.assert_array_equal(hu,self.hu[:,::-1,::-1])
        xyz=np.array([2,4,5]); old=np.array([xyz[0],31-xyz[1],39-xyz[2]])
        a=np.array(g['basis_lps_zyx'])@xyz+g['origin_lps_mm']
        b=np.array(self.geom['basis_lps_zyx'])@old+self.geom['origin_lps_mm']
        np.testing.assert_allclose(a,b,atol=1e-5)

    def test_oblique_nifti_is_not_silently_accepted(self):
        affine=np.eye(4); affine[0,2]=.1
        img=nib.Nifti1Image(self.hu.transpose(2,1,0),affine);img.header.set_xyzt_units('mm')
        p=self.folder/'bad.nii';nib.save(img,p)
        with self.assertRaises(ValueError): read_ct(p,True)

    def test_png_is_not_a_hu_input(self):
        p=self.folder/'picture.png';p.touch()
        with self.assertRaises(ValueError):read_ct(p,True)

    def test_oversized_or_nonvolume_shapes_rejected(self):
        for shape in [(4,4),(3,2,2),(2049,32,32),(1024,1024,1024)]:
            with self.assertRaises(ValueError):checked_shape(shape)

    def test_input_quantization_matches_original_cache_order(self):
        from lung_unet import resize_stack
        torch=torch_module();hu=self.hu[0]
        cached=((resize_stack(np.clip(hu,-1000,400)[None],256,1)+1000)/1400).astype(np.float16)
        expected=torch.quantize_per_tensor(torch.from_numpy(cached.astype(np.float32)),MODEL['input_quantization']['scale'],0,torch.quint8).int_repr().numpy()
        np.testing.assert_array_equal(input_codes(hu,MODEL),expected)

    def test_native_probability_resize_and_gate_never_add_foreground(self):
        head=FakeHardware().infer(np.zeros((1,256,256),np.uint8))[0]
        raw,body,plan=native_masks(head,self.hu[0],self.geom,MODEL)
        torch=torch_module()
        logits=(torch.from_numpy(head.astype(np.float32))[None]-72)*MODEL['output_scale']
        expected=(torch.nn.functional.interpolate(torch.sigmoid(logits),self.hu.shape[1:],mode='bilinear',align_corners=False)[0,0]>=.5).numpy()
        np.testing.assert_array_equal(raw,expected);self.assertFalse((body&~raw).any())

    def test_local_plan_offsets_origin_without_changing_input_geometry(self):
        details=dict(source_slice_indices=[10,11,12,13],partial_input=True)
        p=reconstruction_plan([False,True,True,False],details,self.geom)
        self.assertEqual(p['scope'],'partial');self.assertEqual(p['source_first'],11)
        self.assertEqual(p['source_last'],12);self.assertEqual(p['slice_count'],2)
        np.testing.assert_allclose(p['geometry']['origin_lps_mm'],[10,20,32.5])
        self.assertEqual(self.geom['origin_lps_mm'],[10,20,30])

    def test_plan_rejects_singletons_missing_predictions_and_missing_source_planes(self):
        details=dict(source_slice_indices=[0,1,2],partial_input=False)
        for known in [[False]*3,[True,False,False],[True,False,True]]:
            self.assertIsNone(reconstruction_plan(known,details,self.geom))
        self.assertIsNone(reconstruction_plan([True,True],dict(source_slice_indices=[0,2]),self.geom))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.hardware=FakeHardware();self.service=InferenceService(self.hardware,TOKEN)
        self.server=RunningServer(board_handler(self.service));self.client=BoardClient(self.server.url,TOKEN)
    def tearDown(self):self.server.close()

    def test_health_and_actual_input_reach_single_fpga_owner(self):
        self.assertTrue(self.client.health()['ready'])
        codes=np.arange(65536,dtype=np.uint8).reshape(1,256,256)
        out,row=self.client.infer(codes)
        np.testing.assert_array_equal(self.hardware.capture,codes)
        self.assertEqual(self.hardware.calls,1);self.assertEqual(out.shape,codes.shape)
        self.assertEqual(row['backend'],'FPGA_INT8');self.assertNotIn('head_base64',row)

    def test_invalid_auth_and_browser_origins_rejected(self):
        with self.assertRaises(error.HTTPError) as ctx:BoardClient(self.server.url,'wrong').health()
        self.assertEqual(ctx.exception.code,403)
        req=request.Request(self.server.url+'/v1/health',headers={'Authorization':'Bearer '+TOKEN,'Origin':self.server.url})
        with self.assertRaises(error.HTTPError):request.urlopen(req)
        self.assertEqual(self.hardware.calls,0)

    def test_busy_is_rejected_without_a_second_command_or_retry(self):
        self.service.gate.acquire()
        try:
            with self.assertRaises(error.HTTPError) as ctx:self.client.infer(np.zeros((1,256,256),np.uint8))
            self.assertEqual(ctx.exception.code,409);self.assertEqual(self.hardware.calls,0)
        finally:self.service.gate.release()

    def test_fault_poison_prevents_any_more_hardware_calls(self):
        self.hardware.fail=True
        with self.assertRaises(error.HTTPError):self.client.infer(np.zeros((1,256,256),np.uint8))
        self.hardware.fail=False
        with self.assertRaises(error.HTTPError):self.client.infer(np.zeros((1,256,256),np.uint8))
        self.assertEqual(self.hardware.calls,1)
        with self.assertRaises(RuntimeError):self.client.health()

    def test_hash_or_length_rejection_never_starts_hardware(self):
        with self.assertRaises(ValueError):self.service.infer(b'bad','a'*32,'b'*64)
        with self.assertRaises(ValueError):self.service.infer(bytes(65536),'a'*32,'b'*64)
        self.assertEqual(self.hardware.calls,0)

    def test_tampered_reply_binding_and_clock_are_rejected(self):
        raw=bytes(65536)
        for field,value in [('bit_sha256','bad'),('model_sha256','bad'),('backend','CPU'),('shape',[256,256]),('head_sha256','bad')]:
            reply=self.service.infer(raw,'a'*32,hashlib.sha256(raw).hexdigest());reply[field]=value
            def call(path,data=None,headers=None,**kwargs):
                result=dict(reply);result['request_id']=headers['X-Request-ID'];return result
            with patch.object(self.client,'call',side_effect=call):
                with self.assertRaises(ValueError):self.client.infer(np.zeros((1,256,256),np.uint8))
        self.hardware.identity['bit_sha256']='bad'
        with self.assertRaises(ValueError):self.client.health()

    def test_public_url_redirect_and_environment_proxies_not_used(self):
        for url in ['http://8.8.8.8:8765','https://192.168.2.99','http://u:p@192.168.2.99','http://192.168.2.99/x']:
            with self.assertRaises(ValueError):BoardClient(url,TOKEN)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch_module().set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.hardware=FakeHardware();self.service=InferenceService(self.hardware,TOKEN)
        self.server=RunningServer(board_handler(self.service))
        self.control=Controller(BoardClient(self.server.url,TOKEN),MODEL,self.folder/'results')
        self.path=self.folder/'unknown_ct.npz'
        np.savez(self.path,hu=np.full((3,32,40),-700,np.float32),spacing_zyx_mm=[2.5,.8,.7],
            origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
        self.control.submit('load',dict(path=str(self.path),confirm_hu=True));self.assertEqual(wait(self.control)['phase'],'idle')
    def tearDown(self):self.server.close();self.temp.cleanup()
    def connect(self):
        self.control.submit('connect',{});self.assertTrue(wait(self.control)['board_connected'])

    def test_loading_and_preview_do_not_run_any_model(self):
        s=self.control.snapshot();self.control.plane(s['case_id'],0,0,'body')
        self.assertEqual(self.hardware.calls,0);self.assertEqual(s['known_slices'],0)
        with self.assertRaises(ValueError):self.control.surface(s['case_id'],'body')
        with self.assertRaises(ValueError):self.control.submit('run',dict(mode='slice',index=0))

    def test_single_manual_slice_is_not_misrepresented_as_complete_3d(self):
        self.connect();self.control.submit('run',dict(mode='slice',index=1));s=wait(self.control)
        self.assertEqual(s['phase'],'done');self.assertEqual(self.hardware.calls,1)
        self.assertEqual(s['known_slices'],1);self.assertIsNone(s['volume_ml']);self.assertFalse(s['full_volume_ready'])
        plane=self.control.plane(s['case_id'],0,0,'body')
        self.assertFalse(np.unpackbits(np.frombuffer(base64.b64decode(plane['known']),np.uint8)).any())
        self.assertEqual(len(list((self.folder/'results').rglob('actual_head.npz'))),1)

    def test_complete_volume_uses_each_new_input_then_shared_cleaned_mpr(self):
        self.connect();self.control.submit('run',dict(mode='volume',index=0));s=wait(self.control)
        self.assertEqual(s['phase'],'done',s['error']);self.assertEqual(self.hardware.calls,3)
        self.assertTrue(s['full_volume_ready']);self.assertEqual(s['known_slices'],3)
        surface=self.control.surface(s['case_id'],'cleaned');self.assertTrue(surface['faces'])
        for axis in range(3):
            p=self.control.plane(s['case_id'],axis,0,'cleaned')
            mask=np.unpackbits(np.frombuffer(base64.b64decode(p['mask']),np.uint8),bitorder='little')[:np.prod(p['shape'])].reshape(p['shape'])
            np.testing.assert_array_equal(mask,np.take(self.control.cleaned,0,axis))
        report=json.loads(next((self.folder/'results').rglob('*_volume.json')).read_text(encoding='utf-8'))
        self.assertFalse(report['expert_present']);self.assertEqual(len(report['slice_receipts']),3)

    def test_model_fault_stops_job_no_cpu_or_stale_mask_substitution(self):
        self.connect();self.hardware.fail=True
        self.control.submit('run',dict(mode='volume',index=0));s=wait(self.control)
        self.assertEqual(s['phase'],'error');self.assertEqual(s['known_slices'],0)
        self.assertFalse(s['board_connected']);self.assertEqual(self.hardware.calls,1)

    def test_selected_slices_only_run_deduplicated_commands_then_can_be_supplemented(self):
        self.connect()
        self.control.submit('run',dict(mode='selection',index=0,slices='2,0,0'));s=wait(self.control)
        self.assertEqual(s['phase'],'done',s['error']);self.assertEqual(self.hardware.calls,2)
        self.assertEqual(s['known_slices'],2);self.assertFalse(s['full_volume_ready'])
        self.assertFalse(s['volume_ready'])
        self.assertFalse(self.control.known[1])
        self.control.submit('run',dict(mode='selection',index=0,slices='1'));s=wait(self.control)
        self.assertEqual(self.hardware.calls,3);self.assertTrue(s['full_volume_ready'])

    def test_invalid_selected_range_leaves_previous_mask_and_hardware_untouched(self):
        self.connect();self.control.submit('run',dict(mode='slice',index=0));wait(self.control)
        for text in ['3','2-0','','0,,1']:
            with self.assertRaises(ValueError):self.control.submit('run',dict(mode='selection',index=0,slices=text))
        self.assertEqual(self.hardware.calls,1);self.assertTrue(self.control.known[0])

    def test_partial_input_reconstructs_local_mesh_without_full_lung_claim_or_island_cleanup(self):
        self.connect()
        self.control.submit('load',dict(path=str(self.path),confirm_hu=True,load_slices='1-2'));s=wait(self.control)
        self.assertEqual(s['shape'],[2,32,40]);self.assertEqual(self.hardware.calls,0)
        with patch('lung_live_web.clean_prediction',side_effect=AssertionError('Whole-volume cleanup must not run on local anatomy')):
            self.control.submit('run',dict(mode='volume',index=0));s=wait(self.control)
        self.assertEqual(s['phase'],'done',s['error']);self.assertEqual(self.hardware.calls,2)
        self.assertEqual(s['known_slices'],2);self.assertFalse(s['full_volume_ready']);self.assertIsNone(s['volume_ml'])
        self.assertTrue(s['volume_ready']);self.assertEqual(s['volume_scope'],'partial')
        self.assertGreater(s['region_volume_ml'],0)
        surface=self.control.surface(s['case_id'],'body')
        self.assertTrue(surface['faces']);self.assertEqual(surface['reconstruction']['source_first'],1)
        vertices=np.array(surface['vertices'])
        self.assertAlmostEqual(float(vertices[:,2].min()),1.25)
        self.assertAlmostEqual(float(vertices[:,2].max()),6.25)
        with self.assertRaises(ValueError):self.control.surface(s['case_id'],'cleaned')
        self.assertEqual(s['latest']['source_slice_index'],2)
        self.assertEqual(self.control.plane(s['case_id'],0,0,'body')['source_slice_index'],1)
        report=json.loads(next((self.folder/'results').rglob('*_volume.json')).read_text(encoding='utf-8'))
        self.assertEqual(report['volume_scope'],'partial');self.assertIsNone(report['volume_ml'])
        self.assertFalse(report['cleanup']['applied'])
        self.assertEqual(report['geometry']['origin_lps_mm'],[0,0,2.5])
        with np.load(next((self.folder/'results').rglob('*_masks.npz'))) as archive:
            self.assertEqual(set(archive.files),{'raw','body'})
            np.testing.assert_array_equal(archive['body'],self.control.body)

    def test_contiguous_computed_subset_of_full_input_also_builds_correct_local_mesh(self):
        self.connect();self.control.submit('run',dict(mode='selection',index=0,slices='1-2'));s=wait(self.control)
        self.assertEqual(s['phase'],'done',s['error']);self.assertTrue(s['volume_ready'])
        self.assertFalse(s['full_volume_ready']);self.assertEqual(s['volume_scope'],'partial')
        self.assertFalse(self.control.known[0]);self.assertFalse(self.control.body[0].any())
        surface=self.control.surface(s['case_id'],'body')
        self.assertEqual(surface['reconstruction']['local_start'],1)
        self.assertEqual(surface['reconstruction']['local_stop'],3)
        self.assertEqual(surface['geometry']['origin_lps_mm'],[0,0,2.5])
        self.assertGreaterEqual(min(p[2] for p in surface['vertices']),1.25)

    def test_single_loaded_plane_still_cannot_be_extruded_into_fake_3d(self):
        self.connect();self.control.submit('load',dict(path=str(self.path),confirm_hu=True,load_slices='1'));wait(self.control)
        self.control.submit('run',dict(mode='volume',index=0));s=wait(self.control)
        self.assertEqual(s['phase'],'done');self.assertFalse(s['volume_ready'])
        self.assertIsNone(s['region_volume_ml'])
        with self.assertRaises(ValueError):self.control.surface(s['case_id'],'body')

    def test_empty_local_prediction_is_an_empty_mesh_not_an_invented_organ(self):
        self.connect();original=self.hardware.infer
        def empty(codes):
            head,stats=original(codes);head[:]=0;return head,stats
        self.hardware.infer=empty
        self.control.submit('run',dict(mode='selection',index=0,slices='1-2'));s=wait(self.control)
        self.assertEqual(s['phase'],'done',s['error']);self.assertTrue(s['volume_ready'])
        self.assertEqual(s['region_volume_ml'],0)
        self.assertEqual(self.control.surface(s['case_id'],'body')['faces'],[])

    def test_new_individual_clears_old_masks_and_stale_case_request_rejected(self):
        self.connect();old=self.control.snapshot()['case_id']
        self.control.submit('run',dict(mode='slice',index=0));wait(self.control)
        self.control.submit('load',dict(path=str(self.path),confirm_hu=True));s=wait(self.control)
        self.assertNotEqual(old,s['case_id']);self.assertEqual(s['known_slices'],0)
        with self.assertRaises(ValueError):self.control.plane(old,0,0,'body')

    def test_cancel_finishes_current_slice_and_never_claims_full_volume(self):
        self.connect();original=self.hardware.infer
        def infer(codes):
            out=original(codes);self.control.cancel.set();return out
        self.hardware.infer=infer
        self.control.submit('run',dict(mode='volume',index=0));s=wait(self.control)
        self.assertEqual(s['phase'],'cancelled');self.assertEqual(self.hardware.calls,1)
        self.assertEqual(s['known_slices'],1);self.assertFalse(s['full_volume_ready'])

    def test_host_http_token_and_origin_defend_local_path_api(self):
        server=RunningServer(handler_for(self.control,TOKEN,b'HTML',b'JS'))
        try:
            with self.assertRaises(error.HTTPError):request.urlopen(server.url+'/api/state')
            req=request.Request(server.url+'/api/state',headers={'X-UI-Token':TOKEN,'Origin':'https://example.com'})
            with self.assertRaises(error.HTTPError):request.urlopen(req)
            req=request.Request(server.url+'/api/state',headers={'X-UI-Token':TOKEN})
            with request.urlopen(req) as response:self.assertEqual(json.load(response)['known_slices'],0)
        finally:server.close()


if __name__ == '__main__': unittest.main()
