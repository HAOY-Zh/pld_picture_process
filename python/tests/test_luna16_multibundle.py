import copy
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from luna16_archive_audit import parse_header
from luna16_multibundle import (merge_audits,assign_roles,stratified_ids,cohort_summary,
                               load_development_volume)


def case(uid,positive=True):
    return dict(series_uid=uid,raw_sha256=hashlib.sha256(uid.encode()).hexdigest(),
        raw_bytes=80,dtype='<i2',shape_zyx=[2,4,5],spacing_zyx_mm=[1.25,.9,.8],
        origin_lps_mm=[0.,0.,0.],basis_lps_zyx=[[0.,0.,.8],[0.,.9,0.],[1.25,0.,0.]],
        nodules=[dict(diameter_mm=5.,center_lps_xyz_mm=[.8,.9,1.25],center_index_zyx=[1.,1.,1.])] if positive else [],
        source_member=uid+'.mhd.zip')


def audit(cases,name):
    return dict(status='PASS',errors=[],source_changed=False,annotations_sha256='same-version',
        archive_case_members=len(cases),cases=cases,source_archive=name,source_bytes=123,source_mtime_ns=456)


class ExpandedDataTests(unittest.TestCase):
    def test_exact_uid_duplicate_aliases_and_conflicts_fail(self):
        a,b=case('1.1'),case('1.2')
        merged,sources,aliases=merge_audits([audit([a],'first'),audit([a,b],'second')])
        self.assertEqual(len(merged),2)
        self.assertEqual(len(aliases),1)
        self.assertEqual(merged[0]['archive_id'],'bundle1')
        changed=copy.deepcopy(a)
        changed['spacing_zyx_mm'][0]=2.
        with self.assertRaisesRegex(ValueError,'conflicting'):
            merge_audits([audit([a],'first'),audit([changed],'second')])

    def test_same_pixels_different_uid_and_annotation_version_rejected(self):
        a,b=case('1.1'),case('1.2')
        b['raw_sha256']=a['raw_sha256']
        with self.assertRaisesRegex(ValueError,'different UIDs'):
            merge_audits([audit([a,b],'first')])
        x,y=audit([a],'first'),audit([case('1.3')],'second')
        y['annotations_sha256']='different'
        with self.assertRaisesRegex(ValueError,'annotation versions'):
            merge_audits([x,y])

    def test_preserve_old_roles_and_seal_only_new_unseen_cases(self):
        old_cases=[dict(case('1.'+str(i)),split='heldout' if i==4 else 'development') for i in range(1,5)]
        old_split=dict(cases=old_cases)
        plan=dict(fit_scans=['1.1'],calibration_scans=['1.2'],review_12=['1.3'],reused_6=[])
        added=[case('2.'+str(i),positive=i%2==0) for i in range(12)]
        full=[dict(c,archive_id='bundle1') for c in old_cases]+[dict(c,archive_id='bundle2') for c in added]
        result,allocation=assign_roles(full,old_split,plan,123)
        result2,_=assign_roles(full[::-1],old_split,plan,123)
        by_uid={r['series_uid']:r for r in result}
        self.assertEqual({r['series_uid']:r['role'] for r in result},{r['series_uid']:r['role'] for r in result2})
        self.assertEqual(by_uid['1.1']['role'],'train')
        self.assertEqual(by_uid['1.2']['role'],'calibration')
        self.assertEqual(by_uid['1.3']['role'],'review')
        self.assertEqual(by_uid['1.4']['role'],'historical_test')
        self.assertEqual(allocation,dict(added_unique=12,new_train=9,new_calibration=1,new_sealed_test=2))
        self.assertTrue(all(r['origin_cohort']=='new_download' for r in result if r['role']=='sealed_test'))
        self.assertEqual(sum(s['scans'] for s in cohort_summary(result).values()),16)

    def test_strata_reference_presence_balanced(self):
        records=[case('3.'+str(i),positive=i<10) for i in range(20)]
        chosen=stratified_ids(records,4,123)
        self.assertEqual(len(chosen),4)
        self.assertEqual(sum(bool(r['nodules']) for r in records if r['series_uid'] in chosen),2)

    def test_sealed_and_wrong_role_refused_before_archive_io(self):
        record=dict(case('4.1'),role='sealed_test',archive_id='missing')
        manifest=dict(cases=[record],archives={})
        with self.assertRaisesRegex(ValueError,'Sealed-test'):
            load_development_volume(manifest,'4.1','sealed_test')
        with self.assertRaisesRegex(ValueError,'sealed/mismatched'):
            load_development_volume(manifest,'4.1','train')

    def test_reader_resolves_correct_bundle_and_preserves_native_hu(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=dict(cases=[],archives={})
            expected={}
            for i in (1,2):
                uid=f'5.{i}'
                hu=(np.arange(60,dtype=np.int16)-1000+i).reshape(3,4,5)
                header=(f'NDims = 3\nBinaryData = True\nElementType = MET_SHORT\nElementDataFile = {uid}.raw\n'
                    'DimSize = 5 4 3\nElementSpacing = 0.8 0.9 1.25\n'
                    'TransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = -5 6 -7\n').encode('ascii')
                nested=BytesIO()
                with ZipFile(nested,'w') as z:
                    z.writestr(uid+'.mhd',header);z.writestr(uid+'.raw',hu.tobytes())
                path=Path(temp)/f'bundle{i}.zip'
                with ZipFile(path,'w') as z:
                    z.writestr(uid+'.mhd.zip',nested.getvalue())
                s=path.stat();key=f'bundle{i}'
                manifest['archives'][key]=dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
                manifest['cases'].append(dict(parse_header(header,uid),series_uid=uid,role='train',archive_id=key,
                    source_member=uid+'.mhd.zip',raw_sha256=hashlib.sha256(hu.tobytes()).hexdigest()))
                expected[uid]=hu
            actual,geometry=load_development_volume(manifest,'5.2','train')
            np.testing.assert_array_equal(actual,expected['5.2'])
            self.assertEqual(geometry['spacing_zyx_mm'],[1.25,.9,.8])
            self.assertFalse(actual.flags.writeable)


if __name__=='__main__':
    unittest.main()
