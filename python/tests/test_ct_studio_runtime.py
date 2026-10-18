"""Public asset installer rejects corruption and preserves existing deployments."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ct_studio_runtime as runtime


class InstallerTests(unittest.TestCase):
    def release(self, root, name='ct_studio_demo/readme.txt', expected=b'demo', actual=None):
        release=root/'release';release.mkdir()
        archive=release/'assets.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(name, expected if actual is None else actual)
        runtime.write(release/'asset_manifest.json',dict(archive=archive.name,
            archive_sha256=runtime.sha(archive),files_sha256={name:hashlib.sha256(expected).hexdigest()}))
        return release

    def test_member_hash_checked_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=self.release(root,actual=b'corrupted')
            with patch.object(runtime,'RELEASE',release), self.assertRaisesRegex(ValueError,'member SHA256'):
                runtime.install(root/'output','http://192.168.2.99:8765')
            self.assertFalse((root/'output').exists())

    def test_archive_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=self.release(root,name='../escape.txt')
            with patch.object(runtime,'RELEASE',release), self.assertRaisesRegex(ValueError,'Unsafe archive path'):
                runtime.install(root/'output','http://192.168.2.99:8765')
            self.assertFalse((root/'escape.txt').exists())

    def test_existing_package_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=self.release(root)
            existing=root/'output/ct_platform_20261017_v2';existing.mkdir(parents=True)
            marker=existing/'keep.txt';marker.write_text('keep')
            with patch.object(runtime,'RELEASE',release), self.assertRaises(FileExistsError):
                runtime.install(root/'output','http://192.168.2.99:8765')
            self.assertEqual(marker.read_text(),'keep')

    def test_public_or_credential_url_rejected(self):
        for url in ('http://8.8.8.8:8765','http://user:pass@192.168.2.99:8765','https://192.168.2.99:8765'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                runtime.validate_board_url(url)


if __name__=='__main__':
    unittest.main()
