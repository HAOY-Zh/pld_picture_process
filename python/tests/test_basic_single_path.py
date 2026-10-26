"""Verify single-path execution while preserving the existing explicit PC functions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
import ct_workspace_web as web
from ct_basic_processing import process,parameters,morphology

sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class SinglePathTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.controller=web.WorkspaceController(Mock(),{},self.folder/'results')
        self.controller.details={'analysis_source':{'source_id':'original'}}
        self.controller._update(case_id='case',board_connected=True)
        y,x=np.indices((512,512));self.hu=((x+y)%1600-1000).astype(np.float32)

    def tearDown(self):self.temp.cleanup()

    def mask_input(self):
        params=parameters(dict(operation='threshold',lower=-1000,upper=-400))
        mask=process(self.hu,params)
        source=dict(result_id='threshold',output_sha256=sha(mask),input_hu_sha256=sha(self.hu),
                    parameters=params,backend='FPGA_BASIC',index=0)
        return source,mask

    def run_work(self,op,backend='FPGA_BASIC',mask_input=None):
        params=parameters(dict(operation=op,lower=-1000,upper=-400))
        self.controller._basic_work(self.hu,0,0,'case',self.folder,params,backend,
                                    mask_input,'1'*32,time.perf_counter())
        return self.controller.basic_result

    def test_all_six_fpga_methods_skip_cpu_algorithm_and_pixel_comparison(self):
        for op in ['window','threshold','erosion','dilation','opening','closing']:
            with self.subTest(operation=op):
                selected=self.mask_input() if op in web.MORPHOLOGY else None
                actual=np.full((512,512),37,np.uint8)
                self.controller.client.basic=Mock(return_value=(actual,{'stats':{'dma_ms':2.5}}))
                with patch.object(web,'process',side_effect=AssertionError('CPU reference ran')), \
                     patch.object(web,'morphology',side_effect=AssertionError('CPU morphology ran')), \
                     patch.object(web.np,'array_equal',side_effect=AssertionError('Pixel comparison ran')):
                    result=self.run_work(op,mask_input=selected)
                self.assertEqual(self.controller.status['phase'],'done')
                self.controller.client.basic.assert_called_once()
                self.assertEqual(result['backend'],'FPGA_BASIC')
                self.assertIsNone(result['reference_ms'])
                self.assertIs(result['software_comparison_performed'],False)
                self.assertNotIn('reference_bit_exact',result)
                self.assertEqual(sha(self.controller.basic_image),sha(actual))

    def test_fpga_failure_does_not_run_pc_fallback(self):
        self.controller.client.basic.side_effect=RuntimeError('board failed')
        with patch.object(web,'process',side_effect=AssertionError('Fallback ran')):
            self.run_work('window')
        self.assertEqual(self.controller.status['phase'],'error')
        self.assertIn('board failed',self.controller.status['error'])
        self.assertFalse(self.controller.status['board_connected'])
        self.assertIsNone(self.controller.basic_result)
        self.controller.client.basic.assert_called_once()

    def test_saved_morphology_input_hash_change_is_rejected_without_recompute(self):
        source,mask=self.mask_input();mask[0,0]^=255
        with patch.object(web,'process',side_effect=AssertionError('Recomputed threshold')):
            self.run_work('erosion',mask_input=(source,mask))
        self.assertEqual(self.controller.status['phase'],'error')
        self.controller.client.basic.assert_not_called()
        self.assertIn('哈希变化',self.controller.status['error'])

    def test_selected_morphology_source_hu_change_is_rejected(self):
        selected=self.mask_input();self.hu[0,0]+=1
        self.run_work('dilation',mask_input=selected)
        self.assertEqual(self.controller.status['phase'],'error')
        self.controller.client.basic.assert_not_called()

    def test_explicit_pc_functions_remain_and_never_call_fpga(self):
        for op in ['window','threshold','histogram','pseudocolor']:
            with self.subTest(operation=op):
                expected=process(self.hu,parameters(dict(operation=op,lower=-1000,upper=-400)))
                result=self.run_work(op,'PC_REFERENCE')
                self.assertEqual(self.controller.status['phase'],'done')
                self.assertEqual(result['backend'],'PC_REFERENCE')
                np.testing.assert_array_equal(self.controller.basic_image,expected)
        self.controller.client.basic.assert_not_called()

    def test_explicit_pc_morphology_remains_and_never_calls_fpga(self):
        selected=self.mask_input()
        for op in ['erosion','dilation','opening','closing']:
            with self.subTest(operation=op):
                result=self.run_work(op,'PC_REFERENCE',selected)
                binary=selected[1]!=0
                if op=='erosion':expected=morphology(binary)
                elif op=='dilation':expected=morphology(binary,True)
                elif op=='opening':expected=morphology(morphology(binary),True)
                else:expected=morphology(morphology(binary,True))
                self.assertEqual(self.controller.status['phase'],'done')
                np.testing.assert_array_equal(self.controller.basic_image,expected.astype(np.uint8)*255)
        self.controller.client.basic.assert_not_called()

    def test_visible_reference_pass_claim_removed_without_removing_execution_choices(self):
        ui=ROOT/'python/lungseg/workspace'
        self.assertNotIn('实际 PL 输出 · 参考一致',(ui/'app.js').read_text('utf-8'))
        html=(ui/'index.html').read_text('utf-8')
        for value in ['FPGA_BASIC','PC_REFERENCE','FPGA_LDCT','CPU_REFERENCE']:
            self.assertIn(value,html)

if __name__=='__main__':unittest.main(verbosity=2)
