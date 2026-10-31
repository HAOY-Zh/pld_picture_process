"""Boundary compatibility revision; reuse verified completed v1 evidence.

Keep all original frozen sources and artifacts untouched. This revision changes
only acceptance of centers that pass the original decoder's rounding gate.
"""
import argparse
from pathlib import Path

from lung_candidate_data import sha_file
from lung_nodule_cnn_balanced import read,save
import lung_nodule_cnn_experiment as first
import lung_nodule_continuity_experiment as experiment
from lung_nodule_continuity_boundary import evidence_for_packet

OLD = experiment.OUT
OUT = experiment.ROOT/'data/lung_nodule_continuity_20261019_boundary_v2'


def prepare(out):
    if out.exists():
        return
    old = first.verify(OLD)
    experiment.freeze(out)
    plan = read(out/'plan.json')
    files = [Path(__file__).resolve(),experiment.ROOT/'python/lung_nodule_continuity_boundary.py',
             experiment.ROOT/'python/tests/test_lung_nodule_continuity_boundary.py']
    for p in files:
        plan['protected_sha256'][str(p)] = sha_file(p)
        (out/'sources'/p.name).write_bytes(p.read_bytes())
    plan.update(version='native_slice_continuity_boundary_compatibility_v2',
        boundary_policy='accept exactly the original rounded native voxel gate; no moving centers or padding planes',
        recovery_parent=str(OLD),recovery_parent_plan_sha256=sha_file(OLD/'plan.json'),
        recovery_reason='v1 stopped at calibration case3 on strict edge-center validation; no method/threshold selection from calibration',
        recovery_fit_cache_sha256={str(OLD/'fit_scans'/(uid+'.json')):sha_file(OLD/'fit_scans'/(uid+'.json')) for uid in old['fit_scans']})
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    (out/'fit_scans').mkdir()
    for uid in old['fit_scans']:
        source = OLD/'fit_scans'/(uid+'.json')
        data = read(source)
        if data['plan_sha256'] != sha_file(OLD/'plan.json'):
            raise ValueError('Changed v1 training evidence')
        data.update(plan_sha256=sha_file(out/'plan.json'),reused_from=str(source),
                    reused_from_sha256=sha_file(source))
        save(out/'fit_scans'/(uid+'.json'),data)
    print('RECOVERY: 53 verified training evidence caches reused; no training CNN rerun',flush=True)


def reuse_completed_calibration(out):
    experiment.verify_ranker(out)
    folder = out/'calibration_scans'
    folder.mkdir(exist_ok=True)
    for source in sorted((OLD/'calibration_scans').glob('*.json')):
        dest = folder/source.name
        if dest.exists():
            continue
        data = read(source)
        if (data['plan_sha256'] != sha_file(OLD/'plan.json') or
                data['parent_sha256'] != sha_file(Path(data['parent_path']))):
            raise ValueError('Changed completed v1 calibration evidence')
        data.update(plan_sha256=sha_file(out/'plan.json'),reused_from=str(source),
                    reused_from_sha256=sha_file(source))
        save(dest,data)
    print('RECOVERY: completed calibration evidence reused without new CNN/CT passes',flush=True)


def run(out,review=False):
    experiment.evidence_for_packet = evidence_for_packet
    prepare(out)
    first.verify(out)
    experiment.fit(out)
    reuse_completed_calibration(out)
    experiment.cached_evidence(out,'calibration_scans')
    experiment.calibrate(out)
    if review:
        experiment.run(out,'review')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=OUT)
    p.add_argument('--review',action='store_true')
    a=p.parse_args();run(a.outdir,a.review)
