"""Evaluate the predeclared optional cleanup on six existing development cases."""
import hashlib
import json
from pathlib import Path
import sys
import argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_DualExact_20261026_02'
sys.path[:0]=[str(ROOT/'python'),str(APP/'python')]
from lung_structural_cleanup import clean_structural,POLICY
from lungseg.metrics import overlap_metrics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    old=ROOT/'data/lung_island_cleanup_20261014/screen'
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='structural_cleanup')
    args=parser.parse_args()
    if Path(args.output_name).name!=args.output_name:raise ValueError('Need a simple output name')
    out=ROOT/'data/exact_speed_iteration_20261026'/args.output_name;out.mkdir(exist_ok=False)
    protocol=json.loads((old/'protocol.json').read_text(encoding='utf8'))
    report=json.loads((old/'evaluation.json').read_text(encoding='utf8'))
    references={Path(p).parent.name:(Path(p),h)for p,h in protocol['input_sha256'].items()if p.endswith('reference_native.npy')}
    (out/'protocol.json').write_text(json.dumps(dict(policy=POLICY,rule_fixed_before_reference_read=True,
        scope='Six reused development cases, not independent test; original masks unchanged. V2 guard follows error review on the same development cases; not blind validation.',
        acceptance='No additional expert-positive voxel removed versus legacy cleanup on every case; some extra false voxels removed.'),indent=2),encoding='utf8')
    rows=[]
    for case in report['cases']:
        name=case['case'];path=old/name/'predictions_native.npz';before=sha(path)
        with np.load(path,allow_pickle=False)as z:original=z['original'];legacy=z['primary']
        candidate,info=clean_structural(original,case['spacing_zyx_mm'])
        # Geometric decisions are serialized before the expert reference is read.
        (out/(name+'_decisions.json')).write_text(json.dumps(info,indent=2),encoding='utf8')
        ref,expected=references[name];assert sha(ref)==expected
        expert=np.load(ref,mmap_mode='r',allow_pickle=False)!=0
        row=dict(case=name,legacy=overlap_metrics(legacy,expert),candidate=overlap_metrics(candidate,expert),
            additionally_removed_true_lung=int((legacy&~candidate&expert).sum()),
            additionally_removed_false_lung=int((legacy&~candidate&~expert).sum()),
            recovered_true_lung=int((~legacy&candidate&expert).sum()),
            removed_components=len(info.get('removed_component_ids',[])),
            original_masks_unchanged=sha(path)==before)
        rows.append(row)
        print('DEVELOPMENT_CLEANUP',name,'Dice',row['legacy']['dice'],'->',row['candidate']['dice'],
            'additional TP removed',row['additionally_removed_true_lung'],'FP removed',row['additionally_removed_false_lung'],flush=True)
        (out/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=6,rows=rows),indent=2),encoding='utf8')
    accept=all(r['additionally_removed_true_lung']==0 and r['candidate']['dice']>=r['legacy']['dice']for r in rows)
    accept=accept and sum(r['additionally_removed_false_lung']for r in rows)>0
    summary=dict(status='COMPLETE_DEVELOPMENT_ONLY',policy=POLICY,cases=rows,
        accepted_as_optional_development_view=bool(accept),adopted_as_production_default=False,
        additionally_removed_true_lung=sum(r['additionally_removed_true_lung']for r in rows),
        additionally_removed_false_lung=sum(r['additionally_removed_false_lung']for r in rows),
        original_masks_unchanged=all(r['original_masks_unchanged']for r in rows))
    (out/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('OPTIONAL_CLEANUP_DEVELOPMENT_ACCEPTED',accept,flush=True)


if __name__=='__main__':main()
