import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_selected_benchmark as selected


class SelectionTests(unittest.TestCase):
    def folders(self,root):
        old=root/'old';new=root/'new';old.mkdir();new.mkdir()
        for folder in [old,new]:
            (folder/'lung_fast_benchmark.py').write_text('def run():\n    return __file__\n')
            (folder/'lung_clock_benchmark.py').write_text('import lung_fast_benchmark as benchmark\n')
        return old,new

    def test_reproduce_cached_root_then_pin_selected_root_without_deleting_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=self.folders(Path(tmp))
            cached=selected.load_file('old_bench',old/'lung_fast_benchmark.py')
            owner=SimpleNamespace(retained_buffers=['must stay alive'])
            with patch.dict(sys.modules,{'lung_fast_benchmark':cached,'hardware_owner':owner}):
                wrong=selected.load_file('wrong_clock',new/'lung_clock_benchmark.py')
                self.assertIs(wrong.benchmark,cached)
                self.assertEqual(Path(wrong.benchmark.run()).parent,old)
                with selected.selected_entrypoints(new,dict(files_sha256={})) as (bench,clocks,origins):
                    self.assertEqual(Path(bench.run()).parent,new)
                    self.assertIs(clocks.benchmark,bench)
                    self.assertIs(sys.modules['hardware_owner'],owner)
                self.assertIs(sys.modules['lung_fast_benchmark'],cached)
                self.assertEqual(owner.retained_buffers,['must stay alive'])

    def test_context_restores_bindings_and_path_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=self.folders(Path(tmp));cached=object();before=list(sys.path)
            with patch.dict(sys.modules,{'lung_fast_benchmark':cached}):
                with self.assertRaises(TimeoutError):
                    with selected.selected_entrypoints(new,dict(files_sha256={})):
                        raise TimeoutError('Uncertain hardware, no cache clearing')
                self.assertIs(sys.modules['lung_fast_benchmark'],cached)
            self.assertEqual(sys.path,before)

    def test_newly_imported_driver_owner_survives_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=self.folders(Path(tmp));name='_selection_test_new_owner'
            owner=SimpleNamespace(retained_buffers=['in-flight CMA'])
            try:
                with self.assertRaises(TimeoutError):
                    with selected.selected_entrypoints(new,dict(files_sha256={})):
                        sys.modules[name]=owner
                        raise TimeoutError('Keep all new driver modules alive too')
                self.assertIs(sys.modules[name],owner)
                self.assertEqual(owner.retained_buffers,['in-flight CMA'])
            finally: sys.modules.pop(name,None)

    def test_wrong_cached_helper_version_stops_before_loading_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=self.folders(Path(tmp))
            (old/'helper.py').write_text('different')
            with patch.dict(sys.modules,{'helper':SimpleNamespace(__file__=str(old/'helper.py'))}):
                with self.assertRaises(RuntimeError):
                    with selected.selected_entrypoints(new,dict(files_sha256={'helper.py':'expected'})): pass

    def test_download_guard_rejects_other_folder_same_basename_before_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=self.folders(Path(tmp));calls=[];loads=[]
            for folder in [old,new]:
                (folder/'lung_network.bit').write_bytes(folder.name.encode())
                (folder/'lung_network.hwh').write_bytes(b'hwh')
            expected={n:selected.sha(new/n) for n in ['lung_network.bit','lung_network.hwh']}
            def constructor(path,**kwargs):
                calls.append(path);return SimpleNamespace(bitfile_name=path)
            guarded=selected.overlay_guard(constructor,new,expected,loads)
            with self.assertRaises(RuntimeError): guarded(str(old/'lung_network.bit'))
            with self.assertRaises(RuntimeError): guarded(str(new/'lung_network.bit'),download=False)
            self.assertEqual(calls,[])
            guarded(str(new/'lung_network.bit'))
            self.assertEqual(len(calls),1);self.assertEqual(loads[0]['bit_sha256'],expected['lung_network.bit'])
            with self.assertRaises(RuntimeError): guarded(str(new/'lung_network.bit'))

    def test_tampered_bit_and_returned_overlay_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bit=folder/'lung_network.bit';hwh=bit.with_suffix('.hwh')
            bit.write_bytes(b'bit');hwh.write_bytes(b'hwh')
            expected={p.name:selected.sha(p) for p in [bit,hwh]};calls=[]
            def wrong(path):
                calls.append(path);return SimpleNamespace(bitfile_name=str(folder/'other.bit'))
            with self.assertRaises(RuntimeError): selected.overlay_guard(wrong,folder,expected,[])(str(bit))
            bit.write_bytes(b'changed')
            with self.assertRaises(ValueError): selected.overlay_guard(wrong,folder,expected,[])(str(bit))
            self.assertEqual(len(calls),1)

    def test_retained_unsafe_owners_forbid_bit_download(self):
        calls=[];owners=[object()]
        with patch.dict(sys.modules,{'lung_fast':SimpleNamespace(_UNSAFE=owners)}):
            guarded=selected.overlay_guard(lambda *a:calls.append(a),Path('.'),{},[])
            with self.assertRaisesRegex(RuntimeError,'Retained uncertain hardware owners'):
                guarded('must not be downloaded')
            self.assertEqual(calls,[])
            self.assertIs(sys.modules['lung_fast']._UNSAFE,owners)

    def test_complete_entry_records_selected_identity_despite_old_cached_benchmark(self):
        # Hardware is mocked; this proves selection/report plumbing, not PL speed.
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);script=folder/'lung_selected_benchmark.py'
            shutil.copy2(selected.__file__,script)
            (folder/'lung_network.bit').write_bytes(b'new bit')
            (folder/'lung_network.hwh').write_bytes(b'new hwh')
            (folder/'lung_fast_benchmark.py').write_text('''import json
import hashlib
from pathlib import Path
FastUNet=object
def run(args):
    from pynq import Overlay
    folder=Path(__file__).resolve().parent
    Overlay(str(folder/'lung_network.bit'))
    sha=lambda name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
    result=dict(bit_sha256=sha('lung_network.bit'),package_sha256=sha('fast_manifest.json'))
    Path(args.report).write_text(json.dumps(result),encoding='utf-8')
    return result
''')
            (folder/'lung_clock_benchmark.py').write_text('''import json
from pathlib import Path
import lung_fast_benchmark as benchmark
def run(args):
    benchmark.run(args)
    path=Path(args.report)
    path.with_name(path.stem+'_clock.json').write_text('{}')
    return dict(optimized_pl_mean_ms=123,cpu_over_pl=0.5)
''')
            def manifest(name,files,**extra):
                data=dict(files_sha256={n:selected.sha(folder/n) for n in files},**extra)
                (folder/name).write_text(json.dumps(data),encoding='utf-8')
            manifest('fast_manifest.json',['lung_fast_benchmark.py'])
            manifest('clock_manifest.json',['lung_clock_benchmark.py'])
            manifest('selection_manifest.json',['lung_selected_benchmark.py','fast_manifest.json',
                'clock_manifest.json','lung_network.bit','lung_network.hwh'],schema='lung_pinned_entrypoint_v1')
            downloaded=[]
            def overlay(path):
                downloaded.append(path);return SimpleNamespace(bitfile_name=path)
            cached=SimpleNamespace(run=lambda *a: self.fail('Old cached benchmark must not run'))
            args=SimpleNamespace(report=str(folder/'result.json'))
            with patch.dict(sys.modules,{'pynq':SimpleNamespace(Overlay=overlay),'lung_fast_benchmark':cached}):
                with patch.object(selected,'__file__',str(script)):
                    result=selected.run(args)
                self.assertIs(sys.modules['lung_fast_benchmark'],cached)
            self.assertEqual(downloaded,[str(folder/'lung_network.bit')])
            self.assertEqual(result['status'],'passed_pinned_real_board_benchmark')
            self.assertEqual(result['expected_bit_sha256'],selected.sha(folder/'lung_network.bit'))
            self.assertTrue((folder/'result_selection.json').exists())


if __name__=='__main__': unittest.main()
