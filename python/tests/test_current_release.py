"""Fresh-install checks for the public current snapshot; never accesses a board."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
import ct_studio_current_release as release
from install_ct_studio_current_boot import unit_text


class CurrentReleaseTests(unittest.TestCase):
    def test_archive_inventory_privacy_and_same_numerical_assets(self):
        meta=release.read(release.PUBLIC/'asset_manifest.json')
        archive=release.PUBLIC/meta['archive']
        self.assertEqual(release.sha(archive),meta['archive_sha256'])
        with zipfile.ZipFile(archive)as z:
            self.assertEqual(set(z.namelist()),set(meta['files_sha256']))
            for name in z.namelist():
                self.assertNotIn('token.txt',name)
                self.assertNotIn('/samples/',name)
                self.assertNotIn('C067',name);self.assertNotIn('C077',name)
                self.assertNotIn(Path(name).suffix,('.dcm','.nii','.pt','.pth','.jpg','.ipynb'))
            self.assertIn('nodule_scoring.json',z.namelist())
            self.assertIn('nodule_ranker.json',z.namelist())
            self.assertEqual(json.loads(z.read('nodule_scoring.json'))['threshold'],0.6418673229668569)
            m=json.loads(z.read('ct_studio/platform_manifest.json'))
            self.assertEqual(m['shared_segmentation']['model_sha256'],
                '70ad14405228ac08421c5d9a732a2825f46c1e364e4cf927488a90cdd0650dbd')
            self.assertEqual(m['shared_segmentation']['bit_sha256'],
                '21523a8034acf96b2b4c20615353a9a7804cc0c4f5a6f05f1c1592d020d47187')

    def test_fresh_install_complete_clients_and_frozen_scoring(self):
        from start_ct_studio_demo import configuration
        from nodule_detector import NoduleCaseProvider
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp)/'new'
            release.install(release.PUBLIC,folder,'http://192.168.2.99:8766')
            release.verify(folder)
            client,manifest=configuration(folder/'host_config.json')
            self.assertEqual(client.__class__.__name__,'DualSegStudioClient')
            provider=NoduleCaseProvider(folder,client)
            self.assertEqual(provider.scoring['threshold'],0.6418673229668569)
            self.assertEqual(provider.client.bundle['dual_io_interface'],'ct_dual_exact_io_v3')
            self.assertEqual(manifest['service_bundle_sha256'],release.sha(folder/'ct_studio/platform_manifest.json'))
            with self.assertRaises(FileExistsError):release.install(release.PUBLIC,folder,'http://192.168.2.99:8766')

    def test_boot_unit_uses_current_owner_and_generated_identity(self):
        text=unit_text('/home/xilinx/ct_test/ct_studio_current','a'*64)
        self.assertIn('ct_dual_io_owner.py',text)
        self.assertIn('--expected-manifest '+('a'*64),text)
        self.assertIn('SendSIGKILL=no',text)
        self.assertNotIn('Requires=ct-studio-shared-batch.service',text)
        with self.assertRaises(ValueError):unit_text('/tmp/a;bad','a'*64)


if __name__=='__main__':unittest.main()
