"""Freeze one fully completed development candidate while other candidates run.

This snapshots an entire candidate, never a selected test case. No test images,
labels or scores are consulted. Parent training outputs remain untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from lung_prune_refine import PLANS
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes


def snapshot(parent,candidate,out):
    parent=Path(parent).resolve()
    if candidate not in PLANS:raise ValueError('Undeclared candidate')
    folder=parent/candidate
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    if result['candidate']!=candidate or protocol['candidates'][candidate]!=PLANS[candidate]:
        raise ValueError('Candidate identity differs')
    fixed=json.loads((folder/'fixed_evaluation.json').read_text(encoding='utf-8'))
    if len(fixed['cases'])!=6 or fixed['summaries']['fixed_final']['mean_case_dice']!=result['mean_case_dice']:
        raise ValueError('Development result incomplete')
    check_hashes(protocol['input_sha256']);check_hashes(protocol['source_sha256'])
    files={str(p):file_sha256(p) for p in folder.rglob('*') if p.is_file()}
    output=fresh_output(out).resolve();shutil.copytree(folder,output/candidate)
    for old,h in files.items():
        if file_sha256(output/candidate/Path(old).relative_to(folder))!=h:
            raise ValueError('Candidate changed during snapshot')
    for name in ['protocol.json','baseline_workload.json','train_importance.json']:
        shutil.copy2(parent/name,output/('parent_protocol.json' if name=='protocol.json' else name))
    current=copy.deepcopy(protocol)
    current.update(status='frozen_completed_development_candidate_snapshot',
        candidates={candidate:PLANS[candidate]},parent_predeclared_candidates=protocol['candidates'],
        parent_experiment=str(parent),parent_protocol_sha256=file_sha256(parent/'protocol.json'),
        snapshot_candidate=candidate,snapshot_uses_test_scores=False)
    current['input_sha256'].update(files)
    current['input_sha256'][str(parent/'protocol.json')]=file_sha256(parent/'protocol.json')
    current['source_sha256'][str(Path(__file__).resolve())]=file_sha256(__file__)
    write_json(output/'protocol.json',current);write_json(output/'results.json',[result])
    write_json(output/'verification.json',dict(status='complete_pruning_development_experiment',
        protocol_sha256=file_sha256(output/'protocol.json'),original_inputs_unchanged=True,
        results=[result],test_or_covid_used=False,board_tested=False,
        scope='One completed development candidate snapshot; does not claim all parent candidates completed'))
    print('Completed development candidate snapshot:',candidate,output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True,choices=list(PLANS));p.add_argument('--outdir',required=True)
    a=p.parse_args();snapshot(a.parent,a.candidate,a.outdir)
