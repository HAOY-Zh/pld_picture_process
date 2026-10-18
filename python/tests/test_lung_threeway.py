import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
from lung_threeway_benchmark import validate,sha
from verify_lung_threeway_board import stats


class ThreewayTests(unittest.TestCase):
    def test_statistics_recomputed_not_trusted(self):
        row=dict(samples_ms=[1.,2.,4.],mean_ms=7/3,median_ms=2.,min_ms=1.,max_ms=4.)
        stats(row,3)
        for k in ['mean_ms','median_ms','min_ms','max_ms']:
            bad=dict(row);bad[k]+=1
            with self.assertRaises(ValueError):stats(bad,3)
        with self.assertRaises(ValueError):stats(row,2)
    def test_invalid_samples_rejected(self):
        for values in [[0.],[-1.],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError):stats(dict(samples_ms=values),1)
    def test_manifest_mutation_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);helper=Path(__file__).resolve().parents[2]/'pynq/lung_fp32.py'
            (folder/'lung_fp32.py').write_bytes(helper.read_bytes());(folder/'fixture.txt').write_text('original',encoding='utf-8')
            m=dict(schema='lung_original_fp32_threeway_v1',files_sha256={n:sha(folder/n) for n in ['lung_fp32.py','fixture.txt']})
            (folder/'fp32_manifest.json').write_text(json.dumps(m),encoding='utf-8');validate(folder)
            (folder/'fixture.txt').write_text('modified',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Changed'):validate(folder)
            m['files_sha256']={'../escape.txt':'0'*64}
            (folder/'fp32_manifest.json').write_text(json.dumps(m),encoding='utf-8')
            with self.assertRaises(ValueError):validate(folder)
    def test_no_fp32_quantization_or_cpu_fallback(self):
        p=Path(__file__).resolve().parents[2]/'pynq'
        cpp=(p/'lung_fp32_native.cpp').read_text(encoding='utf-8')
        self.assertNotIn('int8_t',cpp);self.assertIn('std::vector<float>',cpp)
        runner=(p/'lung_threeway_benchmark.py').read_text(encoding='utf-8')
        self.assertIn('network.features&3!=3',runner);self.assertIn('selected.run(args)',runner)
        # Inspect executable calls, not documentation describing export checks.
        calls=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else ''
               for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.Call)]
        for forbidden in ['clear','reset','quantize_per_tensor','Overlay']:self.assertNotIn(forbidden,calls)
    def test_bd_validation_before_reset_property_checks(self):
        p=Path(__file__).resolve().parents[1]/'lung_cache_return_prepare.py'
        text=p.read_text(encoding='utf-8')
        self.assertIn("'validate_bd_design\\nsource [file join $task_root fpga validate_lung_window_step.tcl]\\n'",text)


if __name__=='__main__':unittest.main()
