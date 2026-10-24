"""Public archive privacy, fresh credential binding and overwrite protection; no board."""
import json,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ct_studio_integration_release as release

class ReleaseTests(unittest.TestCase):
    def test_archive_has_only_declared_files_and_no_credentials_or_patient_fixtures(self):
        m=release.read(release.PUBLIC/'asset_manifest.json')
        self.assertFalse(m['credentials_included']);self.assertFalse(m['patient_data_included'])
        self.assertEqual(release.sha(release.PUBLIC/m['archive']),m['archive_sha256'])
        with zipfile.ZipFile(release.PUBLIC/m['archive'])as z:
            self.assertEqual(set(z.namelist()),set(m['files_sha256']))
            for name in z.namelist():
                self.assertNotIn('token.txt',name);self.assertNotIn('C067',name);self.assertNotIn('C077',name)
                self.assertNotIn(Path(name).suffix,('.dcm','.nii','.jpg','.pt','.pth','.ipynb'))
            cases=json.loads(z.read('ct_studio/ldct/fixtures/cases.json'))
            self.assertTrue(all(c['name'].startswith('synthetic_')for c in cases))
    def test_fresh_install_rebinds_contracts_generates_credentials_and_preserves_existing(self):
        with tempfile.TemporaryDirectory()as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            release.install(release.PUBLIC,a,'http://192.168.2.99:8766')
            release.verify(a)
            release.install(release.PUBLIC,b,'http://192.168.2.99:8766')
            self.assertNotEqual((a/'platform_token.txt').read_bytes(),(b/'platform_token.txt').read_bytes())
            self.assertNotEqual(release.sha(a/'ct_studio/platform_manifest.json'),release.sha(b/'ct_studio/platform_manifest.json'))
            protected=release.sha(a/'ct_studio/platform_manifest.json')
            with self.assertRaises(FileExistsError):release.install(release.PUBLIC,a,'http://192.168.2.99:8766')
            self.assertEqual(release.sha(a/'ct_studio/platform_manifest.json'),protected)
            self.assertIn('nodule_package',release.read(a/'host_config.json'))

if __name__=='__main__':unittest.main()
