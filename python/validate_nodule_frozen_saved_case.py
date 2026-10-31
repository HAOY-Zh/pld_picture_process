"""Apply frozen scoring to existing raw PL candidates; never rerun the CNN."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
sys.path.insert(0,str(APP/'python'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    packet=json.loads(args.packet.read_text(encoding='utf8'))
    info=json.loads((args.packet.parents[2]/'input.json').read_text(encoding='utf8'))
    from lung_live_data import read_ct
    hu,g,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if details['hu_sha256']!=packet['source_hu_sha256']:raise ValueError('Saved CT hash changed')
    from nodule_frozen_scoring import load_scoring,score_candidates
    setup,ranker=load_scoring(APP/'data/ct_studio_integration_20261025')
    selected,record=score_candidates(hu,g['spacing_zyx_mm'],packet['candidates'],setup,ranker,args.out)
    # Cross-check the vendored scoring implementation against the original source.
    sys.path.insert(0,str(ROOT/'python'))
    from lung_nodule_continuity_boundary import evidence_for_packet
    from lung_nodule_continuity import rerank_scores
    features,_=evidence_for_packet(hu,g['spacing_zyx_mm'],packet['candidates'])
    scores=rerank_scores([r['score']for r in packet['candidates']],features,ranker)
    with np.load(args.out/'continuity_features.npz')as actual:
        np.testing.assert_array_equal(features,actual['features']);np.testing.assert_array_equal(scores,actual['scores'])
    final=dict(packet,candidates=selected,candidate_count=len(selected))
    final['detector_provenance']=dict(packet['detector_provenance'],**record,
        note='Offline scoring-only validation on saved raw PL output. No new FPGA canvas inference and no expert quality claim.')
    (args.out/'scored_prediction.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8')
    summary=dict(status='SAVED_RAW_PL_FROZEN_SCORE_REFERENCE_EXACT_PASS',raw_candidates=len(packet['candidates']),
        selected_candidates=len(selected),removed_candidates=len(packet['candidates'])-len(selected),
        cnn_rerun=False,canvas_physical_test=False,expert_labels_present=False,
        note='Removed candidates are not established false positives; no labels were used.',**record)
    (args.out/'validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items()if k not in ('note',)},indent=2),flush=True)


if __name__=='__main__':main()
