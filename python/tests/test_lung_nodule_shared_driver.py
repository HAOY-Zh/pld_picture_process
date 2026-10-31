"""Full board driver/report/verifier branches with EXPLICIT fake hardware.

This test does not establish CNN arithmetic or FPGA correctness.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
import lung_nodule_shared_benchmark as bench
from lung_nodule_shared import load_model
from lung_continuous import unpack_words
from lung_axil_smoke import feature_words
from nodule_plan import sha
from verify_lung_nodule_shared import verify


class DriverTests(unittest.TestCase):
    def test_full_size_switching_capture_and_independent_retention_and_tamper_checks(self):
        source=ROOT/'data/lung_nodule_shared_20261024_v1/lung_nodule_shared'
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);shutil.copytree(source/'models',folder/'models');out=folder/'returned'
            (folder/'lung_network.bit').write_bytes(b'explicit-fake-bit-for-driver-test')
            m=dict(schema='lung_nodule_single_engine_v1',clock_mhz=100,bit_sha256=sha(folder/'lung_network.bit'),
                files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()})
            (folder/'shared_manifest.json').write_text(json.dumps(m))
            class Owner:
                def __init__(self):
                    self.clocks=types.SimpleNamespace(fclk0_mhz=100.,cpu_mhz=650.)
                    self.build={};self.bank={};self.models={};self.previous=None;self.cma_bytes=0
                def register(self,label,model,params,kind):
                    self.models[label]=model;self.cma_bytes+=1000;return dict(cma_bytes=1000)
                def predict_codes(self,label,image):
                    model=self.models[label];p=folder/'models'/label
                    with np.load(p/'reference.npz') as z:
                        if label=='nodule':
                            cases=[i for i in range(4) if np.array_equal(image,z['case'+str(i)+'_input'])]
                            if len(cases)!=1:raise AssertionError('Wrong fake-case fixture')
                            refs={n['name']:z['case'+str(cases[0])+'_'+n['name']] for n in model['nodes']}
                        else:refs={n['name']:unpack_words(z[n['name']],n['shape']) for n in model['nodes']}
                    self.bank[label]={'input':feature_words(image,model['input_zero'])}
                    self.bank[label].update({n['name']:feature_words(refs[n['name']],n['zero']) for n in model['nodes']})
                    commands=[]
                    for n in model['nodes']:
                        for block in (range((n['shape'][0]+7)//8) if n['kind']=='conv' else [None]):
                            words=n['shape'][1]*n['shape'][2]*(1 if block is not None else (n['shape'][0]+7)//8)
                            commands.append(dict(node=n['name'],block=block,cycles=1000,written_words=words))
                    stats=dict(selected_model=label,previous_model=self.previous,model_changed=label!=self.previous,overlay_reloaded=False,
                        commands=commands,command_count=len(commands),pl_command_cycles=len(commands)*1000,host_total_ms=5.,
                        buffer_allocations_per_frame=0,weight_packets_repacked_per_frame=0)
                    self.previous=label;return refs['head'].copy(),stats
                def snapshot(self,label):return {n:a.copy() for n,a in self.bank[label].items()}
                def close(self):pass
            owner=Owner();loads=[]
            def opening(*a):loads.append(1);return owner,m
            with patch.object(bench,'__file__',str(folder/'lung_nodule_shared_benchmark.py')),patch.object(bench.platform,'machine',return_value='armv7l'),patch.object(bench,'open_shared',opening):
                with contextlib.redirect_stdout(io.StringIO()):
                    bench.run(types.SimpleNamespace(out=str(out),cycles=1,timeout=30.));v=verify(out,folder)
            self.assertEqual(len(loads),1);self.assertEqual(v['model_calls'],14);self.assertEqual(v['nodes_checked'],214)
            with np.load(out/'all_nodes.npz') as z:arrays={n:z[n] for n in z.files}
            key='stage006__head';arrays[key]=arrays[key].copy();arrays[key].flat[0]^=1
            np.savez_compressed(out/'all_nodes.npz',**arrays)
            r=json.loads((out/'result.json').read_text());r['all_nodes_sha256']=sha(out/'all_nodes.npz');(out/'result.json').write_text(json.dumps(r))
            with self.assertRaises(AssertionError):verify(out,folder)


if __name__=='__main__':unittest.main()
