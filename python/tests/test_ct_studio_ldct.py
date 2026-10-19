"""PC dataflow, explicit simulated protocol and ownership; NOT board acceptance."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
from ct_workspace_web import WorkspaceController, workspace_handler
from ct_studio_client import StudioClient
from ct_studio_service import Service, handler_for, FastHardware, ClassicHardware, validate_package
from ldct_contract import quantize_slice, validate_release
from package_ct_studio import package
from test_ct_platform import FakePlatformHardware, HOST, MANIFEST_SHA
from test_lung_live import MODEL, TOKEN, RunningServer
from ct_basic_processing import process, parameters
from lung_live_data import input_codes
from lung_unet import torch_module
from urllib.request import Request, urlopen
from urllib.error import HTTPError

RELEASE=ROOT/'handoff/ldct-slice-v1-20261017/releases/100mhz'
IDENTITY=validate_release(RELEASE)[1]
CLEAN=dict(busy=False,faulted=False,dma_may_access=False,error=None)


def wait(c):
    until=time.monotonic()+25
    while c.snapshot()['phase'] in ('loading','connecting','processing','computing','reconstructing') or c.ldct_running:
        if time.monotonic()>until: raise AssertionError('TEST worker timeout')
        time.sleep(.005)
    return c.snapshot()


class FakeProcessor:
    def assert_idle(self): pass
    def get_state(self): return CLEAN.copy()


class FakeStudioHardware(FakePlatformHardware):
    """Explicit TEST ONLY deterministic transform; production never imports this."""
    def __init__(self):
        super().__init__(); self.manifest['ldct']=IDENTITY
        self.ldct_processor=FakeProcessor(); self.ldct_calls=0; self.captured=[]
    def denoise(self,image,method):
        self.ldct_calls+=1; self.captured.append(image.copy())
        if self.fail: raise TimeoutError('TEST uncertain LDCT DMA')
        self.mode='ldct'; self.active_bit=IDENTITY['bit_sha256']
        return dict(hu=(image.astype(np.int32)+17).clip(-32768,32767).astype(np.int16),
            total_ms=1.,overlay_switch_ms=2.,actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,
            cpu_mhz=650.,tiles=1,method=method,model_id=IDENTITY['model_id'],
            timing_scope='TEST simulated DMA+PL+stitch')


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hw=FakeStudioHardware(); self.service=Service(self.hw,TOKEN,MANIFEST_SHA)
        self.service.startup_verified=True  # TEST-only server; never represents physical validation.
        self.server=RunningServer(handler_for(self.service))
        self.client=StudioClient(self.server.url,TOKEN,dict(HOST,ldct=IDENTITY))
    def tearDown(self): self.server.close()

    def test_raw_hu_rounding_and_existing_basic_segmentation_survive(self):
        self.client.health()
        hu=np.array([[.5,1.5,-.5,-1.5]],np.float32)
        out,receipt=self.client.denoise(hu)
        np.testing.assert_array_equal(self.hw.captured[0],[[0,2,0,-2]])
        np.testing.assert_array_equal(out,[[17,19,17,15]])
        self.assertEqual(receipt['backend'],'FPGA_LDCT')
        self.assertEqual(self.service.health()['integration_board_acceptance'],'startup-selftests-passed')
        basic,_=self.client.basic(np.full((512,512),-800,np.int16),dict(operation='threshold'))
        self.assertTrue((basic==255).all())
        self.client.infer(np.zeros((1,256,256),np.uint8))
        self.assertEqual(self.hw.mode,'segmentation')

    def test_invalid_and_busy_requests_never_start_hardware(self):
        for hu in (np.array([[np.nan]],np.float32),np.zeros((1,1),np.uint8),np.zeros((513,1),np.int16)):
            with self.assertRaises(ValueError): self.client.denoise(hu)
        with self.assertRaises(ValueError): self.client.denoise(np.zeros((1,1),np.int16),'bad')
        self.service.gate.acquire()
        try:
            with self.assertRaises(HTTPError) as e: self.client.denoise(np.zeros((1,1),np.int16))
            self.assertEqual(e.exception.code,409)
        finally: self.service.gate.release()
        self.assertEqual(self.hw.ldct_calls,0)

    def test_fault_and_tampered_clock_never_return_cpu_or_retry(self):
        self.hw.fail=True
        with self.assertRaises(HTTPError): self.client.denoise(np.zeros((1,1),np.int16))
        self.hw.fail=False
        with self.assertRaises(HTTPError): self.client.denoise(np.zeros((1,1),np.int16))
        self.assertEqual(self.hw.ldct_calls,1)
        self.assertFalse(self.service.health()['ready'])
        with self.assertRaises(HTTPError): self.client.basic(np.zeros((512,512),np.int16),dict(operation='window'))
        self.assertEqual(self.hw.basic_calls,0)

    def test_client_rejects_wrong_frequency_or_model_or_hash(self):
        image=np.zeros((1,1),np.int16)
        original=self.client.call
        for key in ('clock','model','hash'):
            def changed(*a,**kw):
                value=original(*a,**kw)
                if key=='clock': value['stats']['actual_pl_mhz_after']=50.
                elif key=='model': value['model_id']='wrong'
                else: value['output_sha256']='0'*64
                return value
            with patch.object(self.client,'call',changed),self.assertRaises(ValueError): self.client.denoise(image)

    def test_unverified_startup_service_is_rejected(self):
        self.service.startup_verified=False
        with self.assertRaises(RuntimeError): self.client.health()


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        torch_module().set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.hw=FakeStudioHardware(); self.service=Service(self.hw,TOKEN,MANIFEST_SHA)
        self.service.startup_verified=True  # Explicit protocol simulation only.
        self.server=RunningServer(handler_for(self.service))
        self.client=StudioClient(self.server.url,TOKEN,dict(HOST,ldct=IDENTITY))
        self.c=WorkspaceController(self.client,MODEL,self.root/'results'); self.c.configure_ldct(RELEASE)
        self.hu=np.linspace(-1000,300,3*32*40,dtype=np.float32).reshape(3,32,40)
        self.path=self.root/'ct.npz'
        np.savez(self.path,hu=self.hu,spacing_zyx_mm=[2.,1.,1.],origin_lps_mm=[10.,20.,30.],orientation=[1,0,0,0,1,0])
        self.c.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False)); wait(self.c)
        self.assertIsNotNone(self.c.hu,self.c.status.get('error'))
        self.c.submit('connect',{}); wait(self.c)
    def tearDown(self): self.server.close(); self.tmp.cleanup()
    def enhance(self,mode='volume',backend='FPGA_LDCT',method='cnn'):
        self.c.submit('ldct',dict(case_id=self.c.status['case_id'],index=1,mode=mode,backend=backend,method=method))
        state=wait(self.c); self.assertNotEqual(state['phase'],'error',state.get('error')); return state

    def test_enhanced_hu_drives_basic_mpr_and_segmentation_then_original_restores(self):
        original_id=self.c.status['case_id']; original_geom=copy.deepcopy(self.c.geom)
        self.c.submit('basic',dict(case_id=original_id,index=1,backend='PC_REFERENCE',operation='threshold'))
        wait(self.c); original_result=copy.deepcopy(self.c.basic_result)
        self.enhance()
        source=self.c.snapshot()['analysis_sources'][-1]
        self.c.submit('analysis_source',dict(case_id=original_id,source_id=source['source_id']))
        self.assertNotEqual(self.c.status['case_id'],original_id)
        expected=np.rint(self.hu).astype(np.int16).astype(np.float32)+17
        np.testing.assert_array_equal(self.c.hu,expected)
        self.assertEqual(self.c.geom,original_geom); self.assertIsNone(self.c.basic_result)
        self.c.submit('basic',dict(case_id=self.c.status['case_id'],index=1,backend='PC_REFERENCE',operation='window'))
        wait(self.c)
        np.testing.assert_array_equal(self.c.basic_image,process(expected[1],parameters(dict(operation='window'))))
        self.c.submit('run',dict(mode='slice',index=1)); wait(self.c)
        np.testing.assert_array_equal(self.hw.capture,input_codes(expected[1],MODEL))
        enhanced_known=self.c.known.copy()
        enhanced_id=self.c.status['case_id']; enhanced_result=copy.deepcopy(self.c.basic_result)
        self.c.submit('analysis_source',dict(case_id=enhanced_id,source_id='original'))
        np.testing.assert_array_equal(self.c.hu,self.hu)
        self.assertEqual(self.c.status['case_id'],original_id)
        self.assertEqual(self.c.basic_result,original_result); self.assertFalse(self.c.known.any())
        self.c.submit('analysis_source',dict(case_id=original_id,source_id=source['source_id']))
        self.assertEqual(self.c.basic_result,enhanced_result)
        np.testing.assert_array_equal(self.c.known,enhanced_known)

    def test_partial_preview_cannot_be_analysis_source_and_original_is_retained(self):
        original=self.c.hu.copy(); self.enhance('slice')
        self.assertEqual(len(self.c.snapshot()['analysis_sources']),1)
        with self.assertRaises(ValueError):
            self.c.submit('analysis_source',dict(case_id=self.c.status['case_id'],source_id='enhanced:missing'))
        np.testing.assert_array_equal(self.c.hu,original)
        self.assertEqual(self.hw.ldct_calls,1)

    def test_real_cpu_reference_is_explicit_and_has_no_hardware_receipt(self):
        before=self.hw.ldct_calls; self.enhance(backend='CPU_REFERENCE')
        from ctdenoise.cnn_fixed import infer_fixed,load_quantized
        arrays,spec=load_quantized(RELEASE/'model')
        rows=self.c.ldct_groups[self.c.ldct_group]
        np.testing.assert_array_equal(rows[1]['hu'],infer_fixed(np.rint(self.hu[1]).astype(np.int16),arrays,spec)[0])
        self.assertEqual(self.hw.ldct_calls,before)
        self.assertTrue(all(r['metadata']['hardware_receipt'] is None for r in rows.values()))
        self.assertEqual(self.c.snapshot()['analysis_sources'][-1]['backend'],'CPU_REFERENCE')

    def test_fault_preserves_original_and_previous_results_and_no_enhanced_source(self):
        self.hw.fail=True; original=self.c.hu.copy()
        self.c.submit('ldct',dict(case_id=self.c.status['case_id'],index=1,mode='volume',backend='FPGA_LDCT',method='cnn'))
        state=wait(self.c)
        self.assertEqual(state['phase'],'error'); self.assertFalse(state['board_connected'])
        self.assertEqual(len(state['analysis_sources']),1); np.testing.assert_array_equal(self.c.hu,original)

    def test_new_ct_rejects_old_preview_and_source(self):
        self.enhance(); old_series=self.c.series_id; old_group=self.c.ldct_group
        self.c.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False)); wait(self.c)
        with self.assertRaises(ValueError): self.c.ldct_payload(old_series,old_group,1)
        self.assertEqual(len(self.c.snapshot()['analysis_sources']),1)

    def test_cancel_finishes_inflight_slice_and_never_activates_partial_volume(self):
        entered=threading.Event(); release=threading.Event(); original=self.client.denoise
        def held(image,method):
            entered.set(); release.wait(5); return original(image,method)
        with patch.object(self.client,'denoise',held):
            self.c.submit('ldct',dict(case_id=self.c.status['case_id'],index=1,mode='volume',backend='FPGA_LDCT',method='cnn'))
            self.assertTrue(entered.wait(3))
            with self.assertRaises(BlockingIOError): self.c.submit('basic',{})
            self.c.submit('ldct_cancel',{}); release.set(); state=wait(self.c)
        self.assertEqual(state['phase'],'cancelled')
        self.assertEqual(state['ldct']['completed_slices'],1)
        self.assertEqual(len(state['analysis_sources']),1)


class OwnershipTests(unittest.TestCase):
    def hardware(self,cls):
        events=[]
        class Buffer(np.ndarray):
            def __new__(kind,shape):
                obj=np.zeros(shape,np.uint32).view(kind); obj.freed=False; return obj
            def freebuffer(self): self.freed=True; events.append('free')
        class Processor:
            def __init__(self,**kw): self.poisoned=False; self.kw=kw; events.append('processor')
            def assert_idle(self):
                events.append('idle')
                if self.poisoned: raise RuntimeError('TEST DMA may access')
            def process_slice(self,image,method):
                self.assert_idle(); return dict(hu=image.copy(),total_ms=1.)
        dma=SimpleNamespace(mmio=SimpleNamespace(read=lambda _:2))
        gpio=SimpleNamespace(write=lambda *a:events.append('gpio'),channel1=SimpleNamespace(write=lambda *a:None))
        def overlay(path):
            events.append('load'); return SimpleNamespace(axi_dma_0=dma,axi_gpio_0=gpio,axi_gpio_1=gpio)
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        h=cls.__new__(cls); h.folder=Path(tmp.name); h.fault=None; h.mode='window'
        h.network=None; h.dma=None; h.buffers=[]; h.switches=0; h.timeout=1.
        basic=copy.deepcopy(HOST['basic'])
        for op,row in basic.items():
            for kind in ('bit','hwh'):
                p=h.folder/row[kind]; p.parent.mkdir(exist_ok=True); p.write_bytes((op+kind).encode())
                row[kind+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        h.manifest=dict(basic=basic,ldct=IDENTITY,basic_clocks_mhz={op:100. if cls is FastHardware else 50. for op in basic})
        h.clocks=SimpleNamespace(fclk0_mhz=50.,cpu_mhz=650.); h.cpu_mhz=650.
        h.pynq=SimpleNamespace(Overlay=overlay,allocate=lambda shape,dtype:Buffer(shape))
        return h,events,Processor

    def test_both_owners_restore_module_clocks_only_after_idle_and_keep_same_bit(self):
        for cls in (ClassicHardware,FastHardware):
            h,events,processor=self.hardware(cls)
            with patch('ct_studio_service.validate_package',return_value=h.manifest),patch('ct_studio_service.LDCTProcessor',processor):
                h.switch('ldct'); self.assertEqual(h.clocks.fclk0_mhz,100.)
                self.assertEqual([len(b) for b in h.buffers],[4096,4096])
                retained=list(h.buffers); loads=events.count('load')
                h.switch('ldct'); self.assertEqual(events.count('load'),loads)
                events.clear(); h.switch('window')
                self.assertTrue(all(b.freed for b in retained))
                self.assertLess(events.index('idle'),events.index('free'))
                self.assertLess(events.index('free'),events.index('load'))
                self.assertEqual(h.clocks.fclk0_mhz,100. if cls is FastHardware else 50.)

    def test_uncertain_dma_retains_buffers_and_refuses_bit_change(self):
        h,events,processor=self.hardware(ClassicHardware)
        with patch('ct_studio_service.validate_package',return_value=h.manifest),patch('ct_studio_service.LDCTProcessor',processor):
            h.switch('ldct'); h.ldct_processor.poisoned=True; retained=list(h.buffers); events.clear()
            with self.assertRaises(RuntimeError): h.denoise(np.zeros((1,1),np.int16),'cnn')
            with self.assertRaises(RuntimeError): h.switch('window')
            with self.assertRaises(RuntimeError): h.release_idle()
            self.assertTrue(all(not b.freed for b in retained)); self.assertNotIn('load',events)
        with self.assertRaises(RuntimeError): h._check_clock(float('nan'),100.,.5)


class PackageTests(unittest.TestCase):
    def test_both_new_packages_validate_without_modifying_frozen_assets(self):
        import ct_studio_runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ct_studio_runtime.install(root/'base','http://192.168.2.99:8765')
            for variant,name in ct_studio_runtime.VARIANTS.items():
                host=root/'base'/name[0]
                before=hashlib.sha256((host/name[1]/'platform_manifest.json').read_bytes()).hexdigest()
                generated=host/name[1]/'__pycache__'
                generated.mkdir(exist_ok=True)
                (generated/'developer.cpython-311.pyc').write_bytes(b'not a deployable source')
                target=package(host,variant,RELEASE,root/variant)
                manifest=validate_package(target/'ct_studio')
                self.assertFalse(manifest['integration']['board_verified'])
                self.assertEqual(manifest['ldct']['clock_mhz'],100.)
                self.assertEqual(set(manifest['basic_clocks_mhz'].values()),{100. if variant=='fast' else 50.})
                self.assertEqual(hashlib.sha256((host/name[1]/'platform_manifest.json').read_bytes()).hexdigest(),before)
                deployment=target/'ct_studio'
                self.assertFalse((deployment/'__pycache__').exists())
                self.assertTrue(all('__pycache__' not in name and not name.endswith('.pyc')
                                    for name in manifest['files_sha256']))
                for module in ('__init__.py','filters.py','cnn_fixed.py','pl_reference.py'):
                    self.assertEqual((deployment/'ctdenoise'/module).read_bytes(),
                                     (RELEASE/'python/ctdenoise'/module).read_bytes())
                # A fresh isolated interpreter must import only the generated
                # board package; inherited sys.path/modules previously hid this.
                code=('import sys; from pathlib import Path; '
                      'root=Path(sys.argv[1]); sys.path.insert(0,str(root)); '
                      'import ct_studio_service,ctdenoise.filters; '
                      'assert Path(ctdenoise.filters.__file__).parent==root/"ctdenoise"; '
                      'assert "pynq" not in sys.modules; '
                      'ct_studio_service.validate_package(root)')
                completed=subprocess.run([sys.executable,'-I','-B','-c',code,str(deployment)],
                                         cwd=root,capture_output=True,text=True,timeout=30)
                self.assertEqual(completed.returncode,0,completed.stderr)
                with self.assertRaises(ValueError): package(host,variant,RELEASE,target)

    def test_returned_array_auditor_rejects_missing_cases_wrong_clocks_and_pixels(self):
        import ct_studio_runtime
        from verify_ct_studio_board import audit
        from ldct_contract import sha
        from lung_continuous import unpack_words
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ct_studio_runtime.install(root/'base','http://192.168.2.99:8765')
            target=package(root/'base/ct_platform_20261017_v2','classic',RELEASE,root/'new')/'ct_studio'
            manifest=validate_package(target)
            cases=json.loads((target/'ldct/fixtures/cases.json').read_text())
            arrays={case['name']:np.load(target/'ldct'/case['expected'],allow_pickle=False) for case in cases}
            model=json.loads((target/'segmentation/model.json').read_text())
            with np.load(target/'segmentation/reference.npz',allow_pickle=False) as z:
                arrays['segmentation_after_ldct']=unpack_words(z['head'],model['nodes'][-1]['shape'])
            data=root/'TEST_ONLY_arrays.npz'; np.savez_compressed(data,**arrays)
            result=dict(passed=True,error=None,ldct=manifest['ldct'],arrays_file=data.name,
                arrays_sha256=sha(data),platform_manifest_sha256=sha(target/'platform_manifest.json'),
                checks=[dict(case=c['name'],stats=dict(actual_pl_mhz_before=100.,actual_pl_mhz_after=100.)) for c in cases],
                segmentation_after_ldct=dict(startup_nodes_checked=list(range(21)),startup_command_count=51,
                    bit_sha256=sha(target/'segmentation/lung_network.bit'),model_sha256=sha(target/'segmentation/model.json')))
            report=root/'TEST_ONLY_report.json'
            def save(): report.write_text(json.dumps(result))
            save(); self.assertEqual(audit(target,report)['ldct_cases'],15)
            result['checks'][0]['stats']['actual_pl_mhz_after']=50.; save()
            with self.assertRaises(ValueError): audit(target,report)
            result['checks'][0]['stats']['actual_pl_mhz_after']=100.
            removed=result['checks'].pop(); save()
            with self.assertRaises(ValueError): audit(target,report)
            result['checks'].append(removed)
            arrays[cases[0]['name']].flat[0]+=1; np.savez_compressed(data,**arrays)
            result['arrays_sha256']=sha(data); save()
            with self.assertRaises(AssertionError): audit(target,report)


if __name__=='__main__': unittest.main()
