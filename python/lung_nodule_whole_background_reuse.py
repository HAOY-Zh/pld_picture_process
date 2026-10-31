"""Resume unchanged whole-background experiment with bitwise-equivalent sampling.

Existing complete volumes are reused; partial volumes are not used as training
evidence. Only XY interpolation caching changes, not CNN inputs or learning.
"""
import argparse
import gzip
import json
from pathlib import Path
from shutil import copyfile

from lung_candidate_data import sha_file
from lung_nodule_cnn_balanced import read,save
import lung_nodule_cnn_experiment as first
import lung_nodule_whole_background as experiment
from lung_nodule_sampling_cache import sample_patch_reuse

OLD=experiment.OUT
OUT=experiment.ROOT/'data/lung_nodule_whole_background_20261020_reuse_v2'


def prepare(out):
    if out.exists():
        return
    parent=first.verify(OLD)
    check=read(OLD/'sampling_reuse_check.json')
    if check['status']!='PASS' or not check['bitwise_identical']:
        raise ValueError('Need real CT sampling identity check')
    completed=[uid for uid in parent['fit_scans'] if (OLD/'fit_scans'/uid/'complete.json').exists()]
    experiment.freeze(out)
    plan=read(out/'plan.json')
    sources=[Path(__file__).resolve(),experiment.ROOT/'python/lung_nodule_sampling_cache.py',
             experiment.ROOT/'python/tests/test_lung_nodule_sampling_cache.py',OLD/'plan.json',
             OLD/'plan.sha256',OLD/'sampling_reuse_check.json']
    identity={}
    for uid in completed:
        for name in ('complete.json','candidates.json.gz','evidence.npz'):
            p=OLD/'fit_scans'/uid/name
            identity[str(p)]=sha_file(p)
    for p in sources:
        plan['protected_sha256'][str(p)]=sha_file(p)
        if p.suffix=='.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    plan.update(version='whole_background_identical_sampling_reuse_v2',
        sampling='same reference arithmetic; cache identical native-plane XY tiles, maximum1024 tiles',
        reuse_parent=str(OLD),reused_training_scans=completed,reused_artifacts_sha256=identity,
        sampling_identity_check_sha256=sha_file(OLD/'sampling_reuse_check.json'))
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    for uid in completed:
        packet,_,done=experiment.read_mining(OLD,uid)
        source=OLD/'fit_scans'/uid
        target=out/'fit_scans'/uid
        target.mkdir(parents=True)
        packet.update(plan_sha256=sha_file(out/'plan.json'),reused_from=str(source),
                      reused_packet_sha256=done['packet_sha256'])
        with gzip.open(target/'candidates.json.gz','wt',encoding='utf-8') as f:
            json.dump(packet,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        copyfile(source/'evidence.npz',target/'evidence.npz')
        done.update(plan_sha256=sha_file(out/'plan.json'),packet_sha256=sha_file(target/'candidates.json.gz'),
                    reused_from=str(source),reused_evidence_identical=True)
        save(target/'complete.json',done)
    print('REUSED_COMPLETE_VOLUMES',len(completed),'no repeated CNN inference on completed scans',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=OUT)
    p.add_argument('--stage',choices=['all','mine','select','calibrate','review'],default='all')
    a=p.parse_args()
    prepare(a.outdir)
    experiment.sample_patch=sample_patch_reuse
    experiment.run(a.outdir,a.stage)
