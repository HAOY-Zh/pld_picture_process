import copy
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from luna16_archive_audit import parse_header
from lung_candidate_data import load_development_case
from lung_nodule_test_data import test_records,load_test_case,require_complete
from lung_nodule_cnn import DetectorConfig
from lung_nodule_heldout_eval import infer_volume


class FinalTestDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'source.zip'
        self.uid='1.2.3.4'
        self.hu=(np.arange(60,dtype=np.int16)-30).reshape(3,4,5)
        header=(f'NDims = 3\nBinaryData = True\nElementType = MET_SHORT\nElementDataFile = {self.uid}.raw\n'
                'DimSize = 5 4 3\nElementSpacing = 0.8 0.9 1.25\n'
                'TransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = -5 6 -7\n').encode('ascii')
        raw=self.hu.tobytes()
        nested=BytesIO()
        with ZipFile(nested,'w') as z:
            z.writestr(self.uid+'.mhd',header)
            z.writestr(self.uid+'.raw',raw)
        member=self.uid+'.mhd.zip'
        with ZipFile(self.path,'w') as z:
            z.writestr(member,nested.getvalue())
        self.record=dict(parse_header(header,self.uid),series_uid=self.uid,split='heldout',
            source_member=member,raw_sha256=hashlib.sha256(raw).hexdigest(),nodules=object())
        dev=dict(self.record,series_uid='1.2.3.5',split='development',raw_sha256='0'*64)
        stat=self.path.stat()
        self.split=dict(archive=str(self.path),archive_bytes=stat.st_size,archive_mtime_ns=stat.st_mtime_ns,
                        cases=[dev,self.record])

    def test_native_pixels_geometry_and_annotation_isolation(self):
        hu,geometry=load_test_case(self.split,self.uid,[self.uid])
        np.testing.assert_array_equal(hu,self.hu)
        self.assertEqual(geometry['spacing_zyx_mm'],[1.25,.9,.8])
        self.assertEqual(self.record['split'],'heldout')
        self.assertFalse(hu.flags.writeable)

    def test_no_development_identity_masquerading(self):
        with self.assertRaisesRegex(ValueError,'non-heldout'):
            load_test_case(self.split,'1.2.3.5',[self.uid])
        with self.assertRaisesRegex(ValueError,'Held-out image access refused'):
            load_development_case(self.split,self.record)

    def test_entire_cohort_required_and_raw_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete frozen'):
            test_records(self.split,[])
        changed=copy.deepcopy(self.split)
        changed['cases'][0]['raw_sha256']=self.record['raw_sha256']
        with self.assertRaisesRegex(ValueError,'Duplicate pixel'):
            test_records(changed)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            require_complete(['a','b'],['a'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            require_complete(['a','b'],['a','a','b'])
        require_complete(['a','b'],['b','a'])

    def test_changed_pixel_digest_or_native_geometry_rejected(self):
        changed=copy.deepcopy(self.split)
        changed['cases'][1]['raw_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'identity differs'):
            load_test_case(changed,self.uid,[self.uid])
        changed=copy.deepcopy(self.split)
        changed['cases'][1]['spacing_zyx_mm'][0]=2.
        with self.assertRaisesRegex(ValueError,'identity differs'):
            load_test_case(changed,self.uid,[self.uid])

    def test_empty_predicted_lungs_produce_empty_candidates(self):
        class UnusedNetwork:
            def __call__(self,x):
                raise AssertionError('Empty search must not execute CNN')
        hu=np.full((4,8,8),-1000,dtype=np.int16)
        body=np.zeros(hu.shape,bool)
        geometry=dict(spacing_zyx_mm=[1.,1.,1.],origin_lps_mm=[0.,0.,0.],
                      basis_lps_zyx=[[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])
        result=infer_volume(hu,body,geometry,UnusedNetwork(),DetectorConfig(patch=32,halo=8),object(),48)
        self.assertEqual(result[0],[])
        self.assertEqual(result[1].shape,(0,38))
        self.assertEqual(result[4]['lung_tiles'],0)
        self.assertFalse(result[3].any())


if __name__=='__main__':
    unittest.main()
