"""Explicit final-test reader; existing development-only readers stay unchanged."""
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from luna16_archive_audit import parse_header


def test_records(split, expected_uids=None):
    records = split['cases']
    if any(r['split'] not in ('development', 'heldout') for r in records):
        raise ValueError('Unknown split')
    if len({r['series_uid'] for r in records}) != len(records):
        raise ValueError('Duplicate scan UID')
    if len({r['raw_sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate pixel payload across scan split')
    selected = sorted((r for r in records if r['split']=='heldout'),key=lambda r:r['series_uid'])
    actual = [r['series_uid'] for r in selected]
    if not actual:
        raise ValueError('No heldout scans')
    if expected_uids is not None and actual != list(expected_uids):
        raise ValueError('Must use the complete frozen heldout cohort in its declared order')
    return selected


def load_test_case(split, uid, expected_uids):
    records = {r['series_uid']:r for r in test_records(split,expected_uids)}
    if uid not in records:
        raise ValueError('Undeclared or non-heldout scan')
    # The caller supplies only UID, never a mutable case/split replacement.
    case = records[uid]
    archive = Path(split['archive'])
    stat = archive.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(split['archive_bytes'],split['archive_mtime_ns']):
        raise ValueError('Archive changed after split')
    with ZipFile(archive) as outer, ZipFile(BytesIO(outer.read(case['source_member']))) as inner:
        header = parse_header(inner.read(uid+'.mhd'),uid)
        if inner.getinfo(uid+'.raw').file_size != case['raw_bytes']:
            raise ValueError('Wrong native pixel byte count')
        raw = inner.read(uid+'.raw')
    if (len(raw)!=case['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=case['raw_sha256'] or
            header['shape_zyx']!=case['shape_zyx'] or header['dtype']!=case['dtype'] or
            not np.allclose(header['basis_lps_zyx'],case['basis_lps_zyx'],atol=1e-9,rtol=0) or
            not np.allclose(header['spacing_zyx_mm'],case['spacing_zyx_mm'],atol=1e-9,rtol=0) or
            not np.allclose(header['origin_lps_mm'],case['origin_lps_mm'],atol=1e-9,rtol=0)):
        raise ValueError('Test pixel/geometry identity differs from frozen audit')
    hu = np.frombuffer(raw,dtype=header['dtype']).reshape(header['shape_zyx'])
    geometry = {k:header[k] for k in ('spacing_zyx_mm','origin_lps_mm','basis_lps_zyx')}
    return hu,geometry


def require_complete(expected, completed):
    if len(completed)!=len(set(completed)) or set(expected)!=set(completed):
        raise ValueError('Incomplete/duplicate test cohort; do not publish as full test')
