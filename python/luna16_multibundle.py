"""Merge verified mirror bundles and freeze a larger development/new-test split.

No model, lung-mask or nodule inference. Retain original cohort assignments.
The exposed old45 remains historical_test; new sealed_test images are refused
by the development reader. Mirror bundle numbering is not an official fold.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from lung_candidate_data import read_split,sha_file
from lung_nodule_cnn_balanced import read,save
from luna16_archive_audit import parse_header

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data/luna16_expansion_20261020_v1'
OLD_SPLIT=ROOT/'data/lung_candidate_split_20261018/split.json'
OLD_PLAN=ROOT/'data/lung_nodule_capacity_20261020_amp_safe_v2/plan.json'
OLD_TEST=ROOT/'data/lung_nodule_heldout_20261020_small_v1'
AUDITS=(ROOT/'data/luna16_ingest_20261018/audit.json',
        ROOT/'data/luna16_ingest_20261020_part2/audit.json',
        ROOT/'data/luna16_ingest_20261020_part3/audit.json')
ROLES=('train','calibration','review','historical_test','sealed_test')


def case_identity(case):
    keys=('raw_sha256','raw_bytes','dtype','shape_zyx','spacing_zyx_mm',
          'origin_lps_mm','basis_lps_zyx','nodules')
    return {k:case[k] for k in keys}


def merge_audits(audits):
    """Exact repeated UIDs alias one case; incompatible identities fail closed."""
    by_uid={};by_raw={};archives={};duplicates=[]
    annotations={a['annotations_sha256'] for a in audits}
    if len(annotations)!=1:
        raise ValueError('Different annotation versions across bundles')
    for i,audit in enumerate(audits,1):
        if audit['status']!='PASS' or audit['errors'] or audit['source_changed']:
            raise ValueError('Every source bundle must pass the complete audit')
        if audit['archive_case_members']!=len(audit['cases']):
            raise ValueError('Incomplete source case coverage')
        archive_id='bundle'+str(i)
        archives[archive_id]=dict(path=audit['source_archive'],bytes=audit['source_bytes'],
            mtime_ns=audit['source_mtime_ns'],members=audit['archive_case_members'])
        for original in audit['cases']:
            case=dict(original,archive_id=archive_id)
            uid=case['series_uid'];digest=case['raw_sha256']
            if uid in by_uid:
                if case_identity(by_uid[uid])!=case_identity(case):
                    raise ValueError('Same UID has conflicting pixels/geometry/annotations: '+uid)
                duplicates.append(dict(series_uid=uid,canonical_archive=by_uid[uid]['archive_id'],
                    duplicate_archive=archive_id,source_member=case['source_member']))
                continue
            if digest in by_raw:
                raise ValueError('Same pixels under different UIDs require identity review: '+uid+' / '+by_raw[digest])
            by_uid[uid]=case
            by_raw[digest]=uid
    return sorted(by_uid.values(),key=lambda c:c['series_uid']),archives,duplicates


def stratum(case):
    largest=max((n['diameter_mm'] for n in case['nodules']),default=0.)
    size=0 if largest<6 else 1 if largest<10 else 2
    return int(bool(case['nodules'])),size,int(case['spacing_zyx_mm'][0]>1.25)


def stratified_ids(cases,count,seed):
    if not 0<=count<=len(cases):
        raise ValueError('Invalid stratum sample count')
    if not count:
        return set()
    groups=defaultdict(list)
    for c in cases:
        groups[stratum(c)].append(c['series_uid'])
    targets={k:len(v)*count/len(cases) for k,v in groups.items()}
    quotas={k:int(target) for k,target in targets.items()}
    while sum(quotas.values())<count:
        eligible=[k for k in groups if quotas[k]<len(groups[k])]
        key=max(eligible,key=lambda k:(targets[k]-quotas[k],k))
        quotas[key]+=1
    selected=set()
    for key,values in groups.items():
        order=sorted(values,key=lambda uid:hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest())
        selected.update(order[:quotas[key]])
    return selected


def assign_roles(cases,old_split,old_plan,seed):
    old={r['series_uid']:r for r in old_split['cases']}
    by_uid={r['series_uid']:r for r in cases}
    if not set(old)<=set(by_uid):
        raise ValueError('Missing original cases')
    roles={}
    for key,role in [('fit_scans','train'),('calibration_scans','calibration'),
                     ('review_12','review'),('reused_6','review')]:
        for uid in old_plan[key]:
            if uid in roles or uid not in old or old[uid]['split']!='development':
                raise ValueError('Changed or overlapping old development role')
            roles[uid]=role
    for uid,case in old.items():
        if case_identity(case)!=case_identity(by_uid[uid]):
            raise ValueError('Old pixel/geometry/annotation identity changed')
        if case['split']=='heldout':
            if uid in roles:
                raise ValueError('Old heldout overlaps training')
            roles[uid]='historical_test'
    if set(roles)!=set(old):
        raise ValueError('Original roles do not partition original cohort')
    added=[c for c in cases if c['series_uid'] not in old]
    if len(added)<10:
        raise ValueError('Not enough unique new cases for train/calibration/new test')
    test_count=round(.2*len(added))
    cal_count=round(.1*len(added))
    sealed=stratified_ids(added,test_count,str(seed)+':sealed')
    remaining=[c for c in added if c['series_uid'] not in sealed]
    cal=stratified_ids(remaining,cal_count,str(seed)+':calibration')
    for case in added:
        uid=case['series_uid']
        roles[uid]='sealed_test' if uid in sealed else 'calibration' if uid in cal else 'train'
    result=[dict(c,role=roles[c['series_uid']],
                 origin_cohort='original135' if c['series_uid'] in old else 'new_download',
                 original_split=old[c['series_uid']]['split'] if c['series_uid'] in old else None) for c in cases]
    return result,dict(added_unique=len(added),new_train=len(added)-test_count-cal_count,
                       new_calibration=cal_count,new_sealed_test=test_count)


def cohort_summary(cases):
    summary={}
    for role in ROLES:
        selected=[r for r in cases if r['role']==role]
        summary[role]=dict(scans=len(selected),reference_nodules=sum(len(r['nodules']) for r in selected),
            with_reference=sum(bool(r['nodules']) for r in selected),
            without_reference=sum(not r['nodules'] for r in selected),
            original_cases=sum(r['origin_cohort']=='original135' for r in selected),
            new_cases=sum(r['origin_cohort']=='new_download' for r in selected),
            strata={str(k):v for k,v in Counter(stratum(r) for r in selected).items()})
    return summary


def read_manifest(path):
    path=Path(path).resolve(strict=True)
    if sha_file(path)!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Expanded split changed')
    manifest=read(path)
    for name,digest in manifest['protected_sha256'].items():
        if sha_file(name)!=digest:
            raise ValueError('Changed expanded-data input/source: '+name)
    for source in manifest['archives'].values():
        stat=Path(source['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(source['bytes'],source['mtime_ns']):
            raise ValueError('Archive changed after expanded split')
    return manifest


def records_for(manifest,role):
    if role not in ROLES:
        raise ValueError('Unknown cohort role')
    return [r for r in manifest['cases'] if r['role']==role]


def load_development_volume(manifest,uid,role):
    if role not in ('train','calibration','review','historical_test'):
        raise ValueError('Sealed-test image access refused by expanded development reader')
    records={r['series_uid']:r for r in records_for(manifest,role)}
    if uid not in records:
        raise ValueError('UID not in requested development role; sealed/mismatched UID refused')
    c=records[uid]
    source=manifest['archives'][c['archive_id']]
    stat=Path(source['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source['bytes'],source['mtime_ns']):
        raise ValueError('Source archive changed')
    with ZipFile(source['path']) as outer,ZipFile(BytesIO(outer.read(c['source_member']))) as inner:
        header=parse_header(inner.read(uid+'.mhd'),uid)
        raw=inner.read(uid+'.raw')
    if (len(raw)!=c['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=c['raw_sha256'] or
            header['shape_zyx']!=c['shape_zyx'] or header['dtype']!=c['dtype'] or
            not np.allclose(header['basis_lps_zyx'],c['basis_lps_zyx'],atol=1e-9,rtol=0) or
            not np.allclose(header['spacing_zyx_mm'],c['spacing_zyx_mm'],atol=1e-9,rtol=0) or
            not np.allclose(header['origin_lps_mm'],c['origin_lps_mm'],atol=1e-9,rtol=0)):
        raise ValueError('Expanded source pixel/geometry identity mismatch')
    hu=np.frombuffer(raw,dtype=header['dtype']).reshape(header['shape_zyx'])
    return hu,{k:header[k] for k in ('spacing_zyx_mm','origin_lps_mm','basis_lps_zyx')}


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve the first frozen expansion; do not reshuffle')
    audits=[read(p) for p in AUDITS]
    split=read_split(OLD_SPLIT)
    if sha_file(OLD_PLAN)!=OLD_PLAN.with_suffix('.sha256').read_text().strip():
        raise ValueError('Original model development plan changed')
    old_plan=read(OLD_PLAN)
    old_test=read(OLD_TEST/'audit.json')
    if (old_test['status']!='PASS' or old_test['all_test_scans']!=45 or
            old_test['protocol_sha256']!=sha_file(OLD_TEST/'plan.json') or
            old_test['evaluation_sha256']!=sha_file(OLD_TEST/'evaluation.json')):
        raise ValueError('Need documented old45 exposure before assigning historical_test')
    cases,archives,duplicates=merge_audits(audits)
    cases,allocation=assign_roles(cases,split,old_plan,2026102004)
    summary=cohort_summary(cases)
    annotation_path=Path(audits[0]['annotations'])
    if sha_file(annotation_path)!=audits[0]['annotations_sha256']:
        raise ValueError('Changed reference annotations')
    sources=[Path(__file__).resolve(),ROOT/'python/tests/test_luna16_multibundle.py',
             ROOT/'python/luna16_archive_audit.py',ROOT/'python/lung_candidate_data.py']
    protected_inputs=[*AUDITS,OLD_SPLIT,OLD_SPLIT.with_suffix('.sha256'),OLD_PLAN,
        OLD_PLAN.with_suffix('.sha256'),OLD_TEST/'audit.json',OLD_TEST/'evaluation.json',
        OLD_TEST/'plan.json',annotation_path,*sources]
    manifest=dict(version='luna16_multibundle_split_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seed=2026102004,archives=archives,cases=cases,summary=summary,new_allocation=allocation,
        source_audits=[dict(path=str(p),sha256=sha_file(p),summary=a['summary']) for p,a in zip(AUDITS,audits)],
        protected_sha256={str(p):sha_file(p) for p in protected_inputs},
        annotations_sha256=audits[0]['annotations_sha256'],duplicate_locations=duplicates,
        partition='new unique cases: 20% sealed test, 10% calibration, remainder training; original roles retained',
        strata='reference presence, largest reference diameter <6/6-to-<10/>=10 mm, native z spacing <=/>1.25mm',
        historical_test_policy='old45 stays separate for regression; already exposed, never called a new blind set',
        sealed_test_policy='metadata/CRC identity audit only; no visual preview, masks, detection, calibration or training',
        training_performed=False,segmentation_or_detection_performed=False,new_test_scans_inferred=0,
        split_unit='scan UID and raw-pixel uniqueness; original PatientID unavailable',
        patient_independence_proven=False,official_tenfold_subset_mapping_verified=False,
        official_pixel_identity_verified=False,annotation_kind='reference center and diameter, not voxel masks')
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in sources:
        (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(sha_file(out/'manifest.json')+'\n')
    loaded=read_manifest(out/'manifest.json')
    if len(loaded['cases'])!=sum(r['scans'] for r in summary.values()):
        raise ValueError('Cohort totals disagree')
    rows=['# LUNA16新增两包核验与冻结划分','',
          '| 角色 | CT病例 | 参考结节 | 有参考病例 | 无参考病例 |',
          '|---|---:|---:|---:|---:|']
    for role,s in summary.items():
        rows.append(f"| {role} | {s['scans']} | {s['reference_nodules']} | {s['with_reference']} | {s['without_reference']} |")
    rows+=['',f"全部唯一病例{len(cases)}，参考结节{sum(len(c['nodules']) for c in cases)}；新增唯一病例{allocation['added_unique']}。",
           '旧45例测试已经揭示结果，保持为historical_test；新的sealed_test在分割/检测训练与调参期间不访问影像。',
           '仅核验字节、几何、标注匹配并冻结角色，没有运行模型、训练、准备测试预览或写入原ZIP。',
           '镜像分包编号不对应官方十折；扫描UID去重不代表已经证明患者级独立。','',
           '[完整清单](manifest.json)']
    (out/'summary.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    save(out/'verification.json',dict(status='PASS',unique_cases=len(cases),
        reference_nodules=sum(len(c['nodules']) for c in cases),duplicate_locations=len(duplicates),
        raw_pixel_duplicates=0,original_roles_preserved=True,
        cohorts_disjoint=True,manifest_sha256=sha_file(out/'manifest.json'),
        input_audits_sha256={str(p):sha_file(p) for p in AUDITS},
        protected_files_unchanged=len(manifest['protected_sha256']),
        training_or_model_inference_performed=False,new_test_scans_inferred=0))
    print(json.dumps(dict(status='PASS',total=len(cases),new_allocation=allocation,summary=summary),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    args=parser.parse_args()
    freeze(args.outdir.resolve())
