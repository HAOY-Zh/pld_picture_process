"""Exercise the actual shipped archive, hash gates and fresh-install contract."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from lung_nodule_shared_runtime import ASSETS, install, verify


class RuntimeTests(unittest.TestCase):
    def test_real_archive_install_reverify_preserve_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'installed'
            result = install(dest)
            self.assertGreater(result['files_checked'], 30)
            self.assertEqual(result, verify(dest))
            with self.assertRaises(FileExistsError):
                install(dest)
            p = dest / 'lung_nodule_shared/models/nodule/parameters.npz'
            p.write_bytes(p.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                verify(dest)

    def test_corrupt_archive_is_rejected_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp) / 'assets'; shutil.copytree(ASSETS, assets)
            m = json.loads((assets / 'asset_manifest.json').read_text(encoding='utf-8'))
            p = assets / m['archive']; p.write_bytes(p.read_bytes() + b'changed')
            dest = Path(tmp) / 'installed'
            with self.assertRaises(ValueError):
                install(dest, assets)
            self.assertFalse(dest.exists())


if __name__ == '__main__':
    unittest.main()
