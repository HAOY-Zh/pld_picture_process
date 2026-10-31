"""Check current bundle binding and protection of other localhost sessions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_Desktop_DualExact_20261026_02/app'
spec=importlib.util.spec_from_file_location('current_desktop_bootstrap',APP/'desktop_bootstrap.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class Tests(unittest.TestCase):
    def test_current_desktop_assets_and_both_v3_clients_validate_offline(self):
        code=f"import sys,json;sys.path.insert(0,{str(APP)!r});import desktop_bootstrap as b;c,s,m=b.validate_local();print(json.dumps([type(c).__name__,type(s).__name__,m['dual_io_interface'],s.bundle['service_bundle_sha256']==m['service_bundle_sha256']]))"
        value=subprocess.run([str(APP.parent/'runtime/python.exe'),'-B','-X','utf8','-c',code],
            capture_output=True,text=True,encoding='utf8',check=True)
        self.assertEqual(json.loads(value.stdout),['DualSegStudioClient','CachedCanvasClient','ct_dual_exact_io_v3',True])

    def test_changed_program_hash_is_rejected(self):
        original=b.sha
        def corrupted(path):return '0'*64 if path.name=='desktop_bootstrap.py'else original(path)
        with patch.object(b,'sha',side_effect=corrupted):
            with self.assertRaises(b.LaunchError)as exc:b.validate_local()
        self.assertEqual(exc.exception.code,'FILES')

    def _port(self,backend_id,marker_root=True):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/desktop_dual_integration_20261026')as folder:
            root=Path(folder);config=root/'config.json';config.write_text(json.dumps(dict(desktop_release_id='current')))
            settings=root/'data/desktop_launcher/last_port.json';settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps(dict(port=8099,release_id='current',
                launch_root=str(root.resolve())if marker_root else'Another package')))
            module=SimpleNamespace(web_info=Mock(side_effect=[(None,None,dict(desktop_release_id=backend_id)),None]))
            with patch.object(b,'ROOT',root),patch.object(b,'CONFIG',config),patch.dict('sys.modules',{'start_ct_studio_demo':module}):
                return b.choose_port({})

    def test_old_8099_backend_is_preserved_and_skipped(self):
        self.assertEqual(self._port(None),(8100,None))

    def test_matching_owned_backend_is_reused(self):
        self.assertEqual(self._port('current')[0],8099)

    def test_matching_release_from_other_directory_is_preserved(self):
        self.assertEqual(self._port('current',False),(8100,None))


if __name__=='__main__':unittest.main()
