"""Extend the frozen 413-scan split; preserve every old role, including sealed56."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse

from lung_nodule_cnn_balanced import read,save
from lung_candidate_data import sha_file
from luna16_multibundle import (ROOT,OUT as OLD,read_manifest,
    case_identity,stratified_ids,cohort_summary,load_development_volume,records_for)

OUT=ROOT/'data/luna16_expansion_20261021_v2'
AUDITS=[ROOT/'data'/f'luna16_ingest_20261021_ct4_part{p}/audit.json' for p in (4,5)]


def merge_reviewed_audits(audits,old):
    """Alias a different UID only after exact pixels, geometry and labels agree.

    Old audits precede the two additions, so an old UID remains canonical.
    Conflicting annotations/geometry are never silently merged.
    """
    by_uid={};by_raw={};archives={};duplicates=[]
    old_ids={r['series_uid'] for r in old['cases']}
    if len({a['annotations_sha256'] for a in audits})!=1:raise ValueError('Annotation versions differ')
    for index,a in enumerate(audits,1):
        if a['status']!='PASS' or a['errors'] or a['source_changed'] or a['archive_case_members']!=len(a['cases']):
            raise ValueError('Incomplete/failed audit')
        archive_id='bundle'+str(index)
        archives[archive_id]=dict(path=a['source_archive'],bytes=a['source_bytes'],mtime_ns=a['source_mtime_ns'],members=a['archive_case_members'])
        for original in a['cases']:
            r=dict(original,archive_id=archive_id);uid=r['series_uid'];digest=r['raw_sha256']
            if uid in by_uid:
                if case_identity(r)!=case_identity(by_uid[uid]):raise ValueError('Conflicting duplicate UID')
                duplicates.append(dict(series_uid=uid,duplicate_archive=archive_id,canonical_uid=uid,reason='identical_uid'))
                continue
            if digest in by_raw:
                canonical=by_raw[digest]
                if case_identity(r)!=case_identity(by_uid[canonical]):
                    raise ValueError('Same pixels but conflicting geometry/reference labels')
                if uid in old_ids:raise ValueError('Would discard an original UID; reorder/review sources')
                duplicates.append(dict(series_uid=uid,duplicate_archive=archive_id,canonical_uid=canonical,
                    raw_sha256=digest,source_member=r['source_member'],geometry_and_annotations_equal=True,
                    reason='different_UID_exact_pixel_geometry_label_alias_excluded_before_split'))
                continue
            by_uid[uid]=r;by_raw[digest]=uid
    return sorted(by_uid.values(),key=lambda r:r['series_uid']),archives,duplicates


def extend_roles(cases,old,seed=2026102101):
    before={r['series_uid']:r for r in old['cases']}
    after={r['series_uid']:r for r in cases}
    if len(after)!=len(cases) or not set(before)<=set(after):
        raise ValueError('Duplicate or missing original scan')
    for uid,r in before.items():
        if case_identity(r)!=case_identity(after[uid]):
            raise ValueError('Old case pixel/geometry/reference identity changed')
    added=[r for r in cases if r['series_uid'] not in before]
    if len(added)<10:raise ValueError('Need unique new cases, not old bundles')
    nt,nc=round(.2*len(added)),round(.1*len(added))
    test=stratified_ids(added,nt,f'{seed}:sealed')
    cal=stratified_ids([r for r in added if r['series_uid'] not in test],nc,f'{seed}:cal')
    result=[]
    for r in cases:
        uid=r['series_uid']
        if uid in before:
            result.append(dict(r,role=before[uid]['role'],origin_cohort=before[uid]['origin_cohort'],
                original_split=before[uid].get('original_split')))
        else:
            result.append(dict(r,role='sealed_test' if uid in test else 'calibration' if uid in cal else 'train',
                origin_cohort='new_20261021',original_split=None))
    return result,dict(added_unique=len(added),new_train=len(added)-nt-nc,new_calibration=nc,new_sealed_test=nt)


def current_summary(cases):
    summary=cohort_summary(cases)
    for role,row in summary.items():
        selected=[r for r in cases if r['role']==role]
        row['source_cohorts']=dict(Counter(r['origin_cohort'] for r in selected))
        row['new_cases']=sum(r['origin_cohort']!='original135' for r in selected)
        row['added_20261021_cases']=sum(r['origin_cohort']=='new_20261021' for r in selected)
    return summary


def freeze(out=OUT):
    if out.exists():raise FileExistsError('Preserve existing split; use read_manifest')
    old=read_manifest(OLD/'manifest.json')
    audit_paths=[Path(a['path']) for a in old['source_audits']]+AUDITS
    audits=[read(p) for p in audit_paths]
    cases,archives,duplicates=merge_reviewed_audits(audits,old)
    cases,allocation=extend_roles(cases,old)
    protected=[OLD/'manifest.json',OLD/'manifest.sha256',*audit_paths,
        Path(__file__).resolve(),ROOT/'python/tests/test_luna16_five_bundle.py']
    protected += [Path(a['relocation']['original_audit']) for a in audits if 'relocation' in a]
    protected += [ROOT/'python'/n for n in ('luna16_multibundle.py','luna16_archive_audit.py','lung_candidate_data.py')]
    manifest=dict(version='luna16_five_bundle_split_v1',created_utc=datetime.now(timezone.utc).isoformat(),seed=2026102101,
        archives=archives,cases=cases,summary=current_summary(cases),new_allocation=allocation,
        source_audits=[dict(path=str(p),sha256=sha_file(p),summary=a['summary']) for p,a in zip(audit_paths,audits)],
        annotations_sha256=old['annotations_sha256'],duplicate_locations=duplicates,
        protected_sha256={str(p):sha_file(p) for p in protected},
        partition='retain all original413 roles; new unique scans 70% train / 10% calibration / 20% sealed',
        old_sealed_uids=[r['series_uid'] for r in old['cases'] if r['role']=='sealed_test'],
        patient_independence_proven=False,official_tenfold_subset_mapping_verified=False,
        official_pixel_identity_verified=False,split_unit='scan UID and raw-pixel uniqueness',
        annotation_kind='reference centers and diameters, not voxel contours',new_test_scans_inferred=0)
    out.mkdir(parents=True)
    save(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(sha_file(out/'manifest.json')+'\n',encoding='utf-8')
    verified=read_manifest(out/'manifest.json')
    assert all(next(r for r in cases if r['series_uid']==o['series_uid'])['role']==o['role'] for o in old['cases'])
    save(out/'verification.json',dict(status='PASS',total=len(cases),reference_nodules=sum(len(r['nodules']) for r in cases),
        original_roles_preserved=len(old['cases']),allocation=allocation,old_sealed56_preserved=True,
        training_performed=False,sealed_test_scans_accessed=0))
    print('FIVE BUNDLES FROZEN',len(cases),allocation,manifest['summary'],flush=True)
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    freeze(p.parse_args().out.resolve())
