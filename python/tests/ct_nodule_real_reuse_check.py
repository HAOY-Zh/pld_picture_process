"""Explicit one-case software check. No U-Net forward or board operation."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ct_nodule_inference import ROOT, NoduleService, THRESHOLD
from luna16_multibundle import read_manifest, load_development_volume
from lung_candidate_data import sha_file

UID='1.3.6.1.4.1.14519.5.2.1.6279.6001.203425588524695836343069893813'


def run():
    out=ROOT/'data/ct_nodule_integration_20261023';out.mkdir(exist_ok=True)
    proof=out/'real_reuse_verification.json'
    if proof.exists():
        raise FileExistsError('Preserve prior verification; inspect instead of repeating inference')
    manifest=read_manifest(ROOT/'data/luna16_expansion_20261021_v2/manifest.json')
    hu,geometry=load_development_volume(manifest,UID,'calibration')
    mask_path=ROOT/'data/lung_nodule_context_20261020_v1/calibration_masks'/(UID+'.npz')
    with np.load(mask_path,allow_pickle=False) as a:
        body=np.unpackbits(a['body'],count=int(np.prod(a['shape_zyx']))).reshape(a['shape_zyx']).astype(bool)
    case=ROOT/'data/lung_nodule_five_bundle_20261021_v1/evaluation_cases'/UID
    done=json.loads((case/'complete.json').read_text(encoding='utf-8'))
    for n,h in done['artifacts_sha256'].items():
        assert sha_file(case/n)==h
    with gzip.open(case/'expanded.json.gz','rt',encoding='utf-8') as f:original=json.load(f)
    assert hashlib.sha256(body.tobytes()).hexdigest()==original['lung_mask_sha256']
    with np.load(case/'expanded.npz',allow_pickle=False) as a:scores=a['cross_scores'].copy()
    expected={c['candidate_id']:(c,float(scores[i])) for i,c in enumerate(original['candidates']) if scores[i]>=THRESHOLD}
    with patch('lung_candidate_run.predict_lungs',side_effect=AssertionError('U-Net forward forbidden')),\
         patch('lung_unet.make_model',side_effect=AssertionError('Lung model construction forbidden')),\
         threadpool_limits(limits=1):
        result=NoduleService().predict(hu,body,geometry,
            lambda stage,done,total:print(stage,done,'/',total,flush=True),threading.Event())
    actual={c['candidate_id']:c for c in result['candidates']}
    assert set(expected)==set(actual)
    maximum_score_error=0.
    for key,(previous,score) in expected.items():
        current=actual[key];maximum_score_error=max(maximum_score_error,abs(score-current['score']))
        assert np.allclose(previous['center_lps_xyz_mm'],current['center_lps_xyz_mm'],rtol=0,atol=1e-6)
        assert previous['bbox_zyx']==current['bbox_zyx']
        assert abs(score-current['score'])<=1e-6
    report=dict(status='PASS',series_uid=UID,native_shape_zyx=list(hu.shape),
        comparison='all selected candidate IDs, world centers, bounding boxes and cross scores vs saved445-model prediction',
        selected_candidates=len(actual),maximum_score_error=maximum_score_error,
        lung_model_constructed=False,lung_inference_rerun=False,hardware_calls=0,
        current_source='original FP32 cached development mask; not a new actual-PL mask validation',
        service_sha256=sha_file(ROOT/'python/ct_nodule_inference.py'),test_source_sha256=sha_file(Path(__file__)),
        model_sha256=result['model_sha256'],baseline_prediction_sha256=sha_file(case/'expanded.json.gz'),
        timing=result['timing'],device=result['device'])
    proof.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
