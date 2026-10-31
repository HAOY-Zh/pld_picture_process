"""Measure fresh case loading -> fresh lung prediction -> all candidates.

No cached lung predictions, candidate packets or descriptor tables are consumed
by the measured path. Models are loaded before timing, as in a running service.
One preselected development CT is processed once by each route.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','4')
import gc
import hashlib
import json
from time import perf_counter
import numpy as np

from lung_candidate_data import load_development_case,write_json,sha_file
from lung_candidate_rules_run import ROOT
from lung_nodule_cnn_experiment import verify,records_for,EXPAND
from lung_nodule_cnn import config_from_dict,model
from lung_nodule_speed_compare import cnn_volume
from lung_candidate_expand import memory_bounded_primary,PRIMARY,CHECKPOINT
from lung_candidate_paper import score
from lung_candidate_run import predict_lungs
from lung_unet import make_model


def main():
    import torch
    from threadpoolctl import threadpool_limits
    source=ROOT/'data/lung_nodule_cnn_20261019_v1';plan=verify(source);cfg=config_from_dict(plan['config'])
    split,records=records_for(plan,'calibration_scans');record=records[0];uid=record['series_uid']
    folder=ROOT/'data/lung_nodule_speed_compare_20261019';dest=folder/'new_case_report.json'
    if dest.exists():raise FileExistsError('Preserve first complete new-case measurement')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    tick=perf_counter();cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    lung_net=make_model(cp['base_channels'],1).cuda().eval();lung_net.load_state_dict(cp['model'])
    cnn_cp=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    cnn=model(cfg).cuda().eval();cnn.load_state_dict(cnn_cp['state'])
    svm_path=EXPAND/'training/trial_02/fold_0.json';svm=json.loads(svm_path.read_text('utf-8'))
    primary=json.loads(PRIMARY.read_text('utf-8'));cal=json.loads((source/'calibration/report.json').read_text('utf-8'))
    thresholds={name:next(p['threshold'] for p in cal[name]['operating_points'] if p['fp_budget']==4.) for name in ('cnn','svm_same_scans')}
    with torch.inference_mode():
        lung_net(torch.zeros(16,1,256,256,device='cuda'))
        cnn(torch.zeros(48,5,cfg.patch,cfg.patch,device='cuda'))
    torch.cuda.synchronize()
    report=dict(status='RUNNING',series_uid=uid,source_shape_zyx=record['shape_zyx'],
        model_initialization_and_warmup_ms=(perf_counter()-tick)*1000,
        scope='Local native CT loading -> fresh original FP32 CUDA lung segmentation -> complete detection; models resident',
        excluded='HTTP, browser rendering, PC-board transfer and bit switching; not FPGA measurements',
        repetitions_per_route=1,order=['rule_svm','cnn'],heldout_scans_run=0,
        lung_checkpoint_sha256=sha_file(CHECKPOINT),cnn_checkpoint_sha256=sha_file(source/'model.pt'),
        svm_checkpoint_sha256=sha_file(svm_path),measured_paths_use_cached_masks_or_features=False)
    write_json(dest,report);fresh_masks=[]
    with threadpool_limits(limits=1,user_api='blas'):
        for route in ('rule_svm','cnn'):
            print('NEW CASE',route,'START',flush=True);start=perf_counter()
            hu,geometry=load_development_case(split,record);loaded=perf_counter()
            raw,body,lung_ms=predict_lungs(hu,geometry,lung_net,'cuda',16);del raw
            lung_done=perf_counter()
            if route=='rule_svm':
                packet,features=memory_bounded_primary(hu,body,geometry,primary)
                feature_done=perf_counter();scores=score(features,svm)
                selected=[c for c,s in zip(packet['candidates'],scores) if s>=thresholds['svm_same_scans']]
                end=perf_counter();extra=dict(feature_ms=(feature_done-lung_done)*1000,score_filter_ms=(end-feature_done)*1000)
                del packet,features
            else:
                selected,extra=cnn_volume(hu,body,geometry,cnn,'cuda',cfg,thresholds['cnn']);end=perf_counter()
            report[route]=dict(load_ms=(loaded-start)*1000,lung_wall_ms=(lung_done-loaded)*1000,
                lung_internal_ms=lung_ms,detection_ms=(end-lung_done)*1000,total_ms=(end-start)*1000,
                selected_candidates=len(selected),detection_detail=extra)
            fresh_masks.append(hashlib.sha256(body.tobytes()).hexdigest())
            write_json(dest,report)
            print('NEW CASE',route,'DONE',round(report[route]['total_ms']/1000,3),'seconds',flush=True)
            del hu,body,selected;gc.collect()
    if fresh_masks[0]!=fresh_masks[1]:raise ValueError('Fresh lung predictions differ between routes')
    verify(source)
    report.update(status='COMPLETE',fresh_lung_predictions_identical=True,
        svm_over_cnn_time_ratio=report['rule_svm']['total_ms']/report['cnn']['total_ms'],
        limitation='One development case, one pass per route, fixed order. Reference CPU/GPU paths, not FPGA or all-case guarantees.')
    write_json(dest,report);print('NEW-CASE COMPARISON COMPLETE',flush=True)


if __name__=='__main__':main()
