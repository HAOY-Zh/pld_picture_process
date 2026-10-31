import unittest
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from luna16_five_bundle import extend_roles,merge_reviewed_audits,current_summary


def record(uid,role='train'):
    return dict(series_uid=uid,role=role,origin_cohort='old',original_split=None,
        raw_sha256=uid,raw_bytes=8,dtype='<i2',shape_zyx=[2,2,1],spacing_zyx_mm=[1,1,1],
        origin_lps_mm=[0,0,0],basis_lps_zyx=[[0,0,1],[0,1,0],[1,0,0]],nodules=[])


class ExpansionTests(unittest.TestCase):
    def test_different_uid_pixels_alias_only_if_geometry_labels_agree(self):
        original=record('old');alias=record('new');alias['raw_sha256']=original['raw_sha256']
        def audit(rows):
            return dict(status='PASS',errors=[],source_changed=False,archive_case_members=len(rows),
                annotations_sha256='same',source_archive='unused.zip',source_bytes=10,source_mtime_ns=1,cases=rows)
        for r in (original,alias):r['source_member']=r['series_uid']+'.zip'
        merged,archives,duplicates=merge_reviewed_audits([audit([original]),audit([alias])],dict(cases=[original]))
        self.assertEqual([r['series_uid'] for r in merged],['old'])
        self.assertEqual(duplicates[0]['canonical_uid'],'old')
        alias['origin_lps_mm']=[1,0,0]
        with self.assertRaises(ValueError):merge_reviewed_audits([audit([original]),audit([alias])],dict(cases=[original]))

    def test_all_old_roles_preserved_and_new_roles_disjoint(self):
        old={'cases':[record('old_train'),record('old_test','sealed_test'),record('old_cal','calibration')]}
        cases=old['cases']+[record(str(i)) for i in range(30)]
        actual,allocation=extend_roles(cases,old)
        self.assertEqual(allocation,dict(added_unique=30,new_train=21,new_calibration=3,new_sealed_test=6))
        by={r['series_uid']:r for r in actual}
        for r in old['cases']:self.assertEqual(by[r['series_uid']]['role'],r['role'])
        self.assertEqual(actual,extend_roles(cases,old)[0])
        summary=current_summary(actual)
        for row in summary.values():
            self.assertEqual(row['original_cases']+row['new_cases'],row['scans'])
            self.assertEqual(sum(row['source_cohorts'].values()),row['scans'])

    def test_changed_old_reference_or_pixels_rejected(self):
        old={'cases':[record('old')]};cases=[record('old')]+[record(str(i)) for i in range(20)]
        cases[0]['raw_sha256']='changed'
        with self.assertRaises(ValueError):extend_roles(cases,old)

    def test_duplicate_and_missing_uid_rejected(self):
        old={'cases':[record('old')]}
        with self.assertRaises(ValueError):extend_roles([record('x') for _ in range(20)],old)
        with self.assertRaises(ValueError):extend_roles([record(str(i)) for i in range(20)],old)


if __name__=='__main__':unittest.main()
