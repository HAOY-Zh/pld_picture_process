"""Immutable scan split and verified native CT loading for candidate development."""
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from luna16_archive_audit import parse_header


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4*1024**2):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def stratified_pick(records, count, seed, forced_dev):
    groups = defaultdict(list)
    for r in records:
        diameter = max((n['diameter_mm'] for n in r['nodules']), default=0)
        size_bin = 0 if diameter < 6 else 1 if diameter < 10 else 2
        thickness_bin = int(r['spacing_zyx_mm'][0] > 1.25)
        groups[(size_bin, thickness_bin)].append(r['series_uid'])
    total = len(records)
    targets = {k: len(v)*count/total for k, v in groups.items()}
    quotas = {k: min(int(targets[k]), len(set(v)-forced_dev)) for k, v in groups.items()}
    while sum(quotas.values()) < count:
        eligible = [k for k, v in groups.items() if quotas[k] < len(set(v)-forced_dev)]
        if not eligible:
            raise ValueError('Insufficient uninspected scans for held-out quotas')
        k = max(eligible, key=lambda k: (targets[k]-quotas[k], k))
        quotas[k] += 1
    selected = set()
    for k, uids in groups.items():
        eligible = sorted(set(uids)-forced_dev,
            key=lambda uid: hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest())
        selected.update(eligible[:quotas[k]])
    return selected


def freeze_split(audit_path, outdir, seed=20261018):
    audit_path, output = Path(audit_path).resolve(strict=True), Path(outdir).resolve()
    if output.exists():
        raise ValueError('Split output exists; do not overwrite or reselect held-out scans')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or audit['errors'] or audit['source_changed']:
        raise ValueError('Need a successful integrity audit')
    records = audit['cases']
    if len({r['series_uid'] for r in records}) != len(records):
        raise ValueError('Duplicate scan UIDs')
    if len({r['raw_sha256'] for r in records}) != len(records):
        raise ValueError('Identical pixel payloads need grouped/manual identity review before splitting')
    forced_dev = {audit['prepared_case']['series_uid']} if audit.get('prepared_case') else set()
    positive = [r for r in records if r['nodules']]
    negative = [r for r in records if not r['nodules']]
    heldout = (stratified_pick(positive, round(len(positive)/3), seed, forced_dev) |
               stratified_pick(negative, round(len(negative)/3), seed, forced_dev))
    output.mkdir(parents=True)
    cases = [dict(r, split='heldout' if r['series_uid'] in heldout else 'development') for r in records]
    summary = {}
    for split in ('development', 'heldout'):
        selected = [r for r in cases if r['split'] == split]
        summary[split] = dict(scans=len(selected), with_reference=sum(bool(r['nodules']) for r in selected),
                              without_reference=sum(not r['nodules'] for r in selected),
                              reference_nodules=sum(len(r['nodules']) for r in selected))
    # Fixed small engineering cohort, chosen from metadata before candidate inference/scores.
    dev = [r for r in cases if r['split'] == 'development']
    pilot = [uid for uid in sorted(forced_dev) if uid in {r['series_uid'] for r in dev}]
    for required_positive, target in ((True, 8), (False, 4)):
        eligible = [r for r in dev if bool(r['nodules']) == required_positive and r['series_uid'] not in pilot]
        eligible.sort(key=lambda r: hashlib.sha256(f'pilot:{seed}:{r["series_uid"]}'.encode()).hexdigest())
        already = sum(bool(next(r for r in dev if r['series_uid'] == uid)['nodules']) == required_positive for uid in pilot)
        pilot.extend(r['series_uid'] for r in eligible[:max(0, target-already)])
    result = dict(version='luna16_scan_split_v1', seed=seed, audit_path=str(audit_path),
                  audit_sha256=sha_file(audit_path), archive=audit['source_archive'],
                  archive_bytes=audit['source_bytes'], archive_mtime_ns=audit['source_mtime_ns'],
                  annotations_sha256=audit['annotations_sha256'],
                  split_unit='scan UID; raw-identical scans rejected; original PatientID unavailable',
                  patient_independence_proven=False,
                  selection='pre-inference stratified by reference presence, largest diameter and slice spacing',
                  forced_development_uids=sorted(forced_dev), pilot_development_uids=pilot,
                  summary=summary, cases=cases, heldout_access_policy='development runner refuses heldout scans')
    write_json(output/'split.json', result)
    (output/'split.sha256').write_text(sha_file(output/'split.json')+'\n', encoding='ascii')
    return result


def read_split(path):
    p = Path(path).resolve(strict=True)
    if sha_file(p) != p.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen split changed')
    return json.loads(p.read_text(encoding='utf-8'))


def development_records(split, cohort='pilot'):
    if cohort not in ('pilot', 'check-development', 'all-development'):
        raise ValueError('No held-out execution mode before candidate protocol freeze')
    selected = set(split['pilot_development_uids']) if cohort == 'pilot' else {
        r['series_uid'] for r in split['cases'] if r['split'] == 'development'}
    if cohort == 'check-development':
        eligible = [r for r in split['cases'] if r['split'] == 'development' and
                    r['series_uid'] not in set(split['pilot_development_uids'])]
        selected = set()
        for positive, count in ((True, 8), (False, 4)):
            group = [r for r in eligible if bool(r['nodules']) == positive]
            group.sort(key=lambda r: hashlib.sha256(f'check:{split["seed"]}:{r["series_uid"]}'.encode()).hexdigest())
            selected.update(r['series_uid'] for r in group[:count])
    records = [r for r in split['cases'] if r['series_uid'] in selected]
    if not records or any(r['split'] != 'development' for r in records):
        raise ValueError('Held-out or missing case in development request')
    return records


def load_development_case(split, case):
    if case['split'] != 'development' or case['series_uid'] not in {
            r['series_uid'] for r in split['cases'] if r['split'] == 'development'}:
        raise ValueError('Held-out image access refused')
    archive = Path(split['archive'])
    stat = archive.stat()
    if (stat.st_size, stat.st_mtime_ns) != (split['archive_bytes'], split['archive_mtime_ns']):
        raise ValueError('Source archive changed since audit/split')
    uid = case['series_uid']
    with ZipFile(archive) as outer, ZipFile(BytesIO(outer.read(case['source_member']))) as inner:
        header = parse_header(inner.read(uid+'.mhd'), uid)
        raw = inner.read(uid+'.raw')
    if (len(raw) != case['raw_bytes'] or hashlib.sha256(raw).hexdigest() != case['raw_sha256'] or
            header['shape_zyx'] != case['shape_zyx'] or
            not np.allclose(header['basis_lps_zyx'], case['basis_lps_zyx'], atol=1e-9, rtol=0) or
            not np.allclose(header['origin_lps_mm'], case['origin_lps_mm'], atol=1e-9, rtol=0)):
        raise ValueError('Pixel/geometry identity differs from frozen audit')
    hu = np.frombuffer(raw, dtype=header['dtype']).reshape(header['shape_zyx'])
    geometry = {k: header[k] for k in ('spacing_zyx_mm', 'origin_lps_mm', 'basis_lps_zyx')}
    return hu, geometry
