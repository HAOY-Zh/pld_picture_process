"""Verifier acceptance/rejection tests, explicitly synthetic saved outputs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
from verify_ct_platform_board import verify, startup_cases, BASIC_OPERATIONS, sha, parameters, process, unpack_words


class BoardEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.package=ROOT/'data/ct_platform_20261017_v2/ct_platform'
        if not self.package.exists():self.skipTest('Local pinned hardware package unavailable')
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.manifest=json.loads((self.package/'platform_manifest.json').read_text(encoding='utf-8'))
        self.model=json.loads((self.package/'segmentation/model.json').read_text(encoding='utf-8'))
    def tearDown(self):
        if hasattr(self,'temp'):self.temp.cleanup()

    def fixture(self):
        actual={};rows=[]
        for op in BASIC_OPERATIONS:
            for tag,hu in startup_cases():
                p=parameters(dict(operation=op))
                if op=='window' and tag=='patterned':p.update(wl=0,ww=5)
                key=op+'_'+tag;actual[key]=process(hu,p)
                rows.append(dict(key=key,parameters=p,input_sha256=sha(hu.astype('<i2').tobytes()),
                    output_sha256=sha(actual[key].tobytes()),bit_sha256=self.manifest['basic'][op]['bit_sha256'],
                    stats=dict(dma_ms=1.,preparation_ms=1.,overlay_switch_ms=1.,fclk0_mhz=100.,cpu_mhz=650.,
                        guard_verified=True,input_bytes=512*512*4,output_bytes=512*512*4)))
        with np.load(self.package/'segmentation/reference.npz') as refs:
            actual['segmentation_head']=unpack_words(refs['head'],self.model['nodes'][-1]['shape'])
        arrays=self.root/'synthetic_TEST_ONLY.npz';np.savez_compressed(arrays,**actual)
        proof=dict(schema='ct_platform_startup_v1',passed=True,basic=rows,
            platform_manifest_sha256=sha((self.package/'platform_manifest.json').read_bytes()),
            actual_arrays=arrays.name,actual_arrays_sha256=sha(arrays.read_bytes()),
            segmentation_after_switch=dict(bit_sha256=self.manifest['segmentation_bit_sha256'],
                model_sha256=self.manifest['model_sha256'],startup_command_count=51,
                startup_nodes_checked=[n['name'] for n in self.model['nodes']]))
        report=self.root/'synthetic_TEST_ONLY.json';report.write_text(json.dumps(proof),encoding='utf-8')
        return report,proof

    def test_complete_fixture_is_accepted_but_missing_or_duplicate_case_is_rejected(self):
        report,proof=self.fixture()
        missing=copy.deepcopy(proof);missing['basic'].pop()
        report.write_text(json.dumps(missing),encoding='utf-8')
        with self.assertRaises(ValueError):verify(self.package,report)
        duplicate=copy.deepcopy(proof);duplicate['basic'][-1]=duplicate['basic'][0]
        report.write_text(json.dumps(duplicate),encoding='utf-8')
        with self.assertRaises(ValueError):verify(self.package,report)
        report.write_text(json.dumps(proof),encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):result=verify(self.package,report)
        self.assertEqual(len(result['actual_basic_cases']),12)
        self.assertEqual(result['basic_output_pixels'],3145728)

    def test_actual_array_corruption_is_rejected_even_if_file_hash_is_recomputed(self):
        report,proof=self.fixture();arrays=self.root/proof['actual_arrays']
        with np.load(arrays) as a:data={k:a[k].copy() for k in a.files}
        data['window_patterned'][0,0]^=255
        np.savez_compressed(arrays,**data)
        proof['actual_arrays_sha256']=sha(arrays.read_bytes())
        proof['basic'][0]['output_sha256']=sha(data['window_patterned'].tobytes())
        report.write_text(json.dumps(proof),encoding='utf-8')
        with self.assertRaises(AssertionError):verify(self.package,report)


if __name__=='__main__':unittest.main()
