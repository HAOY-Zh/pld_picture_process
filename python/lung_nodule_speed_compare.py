"""Controlled reference timings: same resident CT/lungs, complete pipelines.

One full-volume pass per backend; three warm per-plane CNN passes. The existing
SVM pipeline has no independent single-plane detection API. Cached browsing is
timed separately and must not be called inference.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
from time import perf_counter
import gc
import json
import numpy as np

from lung_candidate_data import load_development_case,write_json,sha_file
from lung_candidate_rules_run import ROOT
from lung_nodule_cnn_experiment import verify,records_for,load_mask,EXPAND
from lung_nodule_cnn import config_from_dict,model,search_mask,tiles,sample_patch,decode,consolidate
from lung_nodule_cnn_single_slice import plane_envelope
from lung_candidate_expand import memory_bounded_primary,PRIMARY
from lung_candidate_paper import score


def cnn_volume(hu,lung,geometry,net,device,cfg,threshold):
    import torch
    start=perf_counter();spacing=np.array(geometry['spacing_zyx_mm'])
    search=search_mask(lung,spacing,cfg);jobs,schedule=tiles(search,spacing,cfg);scheduled=perf_counter()
    groups=[];values=[];sizes=[];sampling=network=decoding=0.
    with torch.inference_mode():
        for begin in range(0,len(jobs),48):
            part=jobs[begin:begin+48];tick=perf_counter()
            batch=np.stack([sample_patch(hu,spacing,j,cfg) for j in part]);sampling+=perf_counter()-tick
            tick=perf_counter();tensor=torch.from_numpy(batch).to(device).float().clamp(-1000,400).add_(1000).div_(1400)
            prediction=net(tensor).cpu().numpy();network+=perf_counter()-tick;tick=perf_counter()
            p,s,d=decode(prediction,part,geometry,cfg,search);groups.append(p);values.append(s);sizes.append(d)
            decoding+=perf_counter()-tick
    tick=perf_counter();candidates=consolidate(np.concatenate(groups),np.concatenate(values),np.concatenate(sizes),geometry,cfg)
    selected=[c for c in candidates if c['score']>=threshold];merged=perf_counter()
    return selected,dict(total_ms=(merged-start)*1000,schedule_ms=(scheduled-start)*1000,
        sampling_ms=sampling*1000,network_and_transfer_ms=network*1000,decode_ms=decoding*1000,
        merge_filter_ms=(merged-tick)*1000,tiles=len(jobs),raw_candidates=len(candidates),selected_candidates=len(selected),
        schedule=schedule,backend=device+' FP32')


def cnn_plane(hu,lung,geometry,net,device,cfg,threshold,z,zm):
    import torch
    start=perf_counter();spacing=np.array(geometry['spacing_zyx_mm'])
    plane=plane_envelope(lung,spacing,z,cfg.margin_mm)
    jobs,_=tiles(plane[None],spacing,cfg);jobs=[(zm,y,x) for _,y,x in jobs];scheduled=perf_counter()
    image=np.stack([sample_patch(hu,spacing,j,cfg) for j in jobs]);sampled=perf_counter()
    with torch.inference_mode():
        prediction=net(torch.from_numpy(image).to(device).float().clamp(-1000,400).add_(1000).div_(1400)).cpu().numpy()
    network_done=perf_counter();points,scores,diameters=decode(prediction,jobs,geometry,cfg)
    if len(points):
        q=np.rint(points).astype(int);valid=(q>=0).all(1)&(q<hu.shape).all(1)
        valid &= plane[np.clip(q[:,1],0,plane.shape[0]-1),np.clip(q[:,2],0,plane.shape[1]-1)]
        points,scores,diameters=points[valid],scores[valid],diameters[valid]
    candidates=consolidate(points,scores,diameters,geometry,cfg)
    selected=[c for c in candidates if c['score']>=threshold];end=perf_counter()
    return prediction,dict(schedule_ms=(scheduled-start)*1000,sampling_ms=(sampled-scheduled)*1000,
        network_and_transfer_ms=(network_done-sampled)*1000,decode_merge_filter_ms=(end-network_done)*1000,
        total_ms=(end-start)*1000,tiles=len(jobs),selected_candidates=len(selected))


def browsing(candidates,z):
    times=[];count=0
    for _ in range(101):
        start=perf_counter()
        visible=[c for c in candidates if c['bbox_zyx'][0]<=z<c['bbox_zyx'][1]]
        count=len(visible);times.append((perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(times)),visible_candidates=count,repeats=len(times),
                scope='Python filtering of precomputed candidate boxes only; no inference, HTTP or rendering')


def main():
    import torch
    from threadpoolctl import threadpool_limits
    source=ROOT/'data/lung_nodule_cnn_20261019_v1';plan=verify(source);cfg=config_from_dict(plan['config'])
    folder=ROOT/'data/lung_nodule_speed_compare_20261019'
    if folder.exists():raise FileExistsError('Preserve previous benchmark')
    folder.mkdir();split,records=records_for(plan,'calibration_scans');record=records[0];uid=record['series_uid']
    svm_path=EXPAND/'training/trial_02/fold_0.json';svm=json.loads(svm_path.read_text('utf-8'))
    if set(svm['training_scans'])!=set(plan['fit_scans']):raise ValueError('Different fitting cohorts')
    old_report=json.loads((source/'calibration/report.json').read_text('utf-8'))
    thresholds={name:next(p['threshold'] for p in old_report[name]['operating_points'] if p['fp_budget']==4.) for name in ('cnn','svm_same_scans')}
    protected={str(p):sha_file(p) for p in [source/'plan.json',source/'model.pt',svm_path,PRIMARY,Path(__file__)]}
    result=dict(status='RUNNING',series_uid=uid,protected_sha256=protected,
        scope='Same resident native HU and cached original FP32 U-Net mask; lung segmentation, image/model loading, HTTP/rendering excluded',
        full_volume_repeats=1,single_plane_repeats=3,heldout_scans_run=0,
        fairness='Both models fitted on identical 53 scans. Fixed calibration thresholds. No accuracy reevaluation or retraining.',
        cpu_torch_threads=4,candidate_filter_workers=4,blas_threads=1,
        full_volume_svm='native 2D LoG + 3D regions/Hessian/local/shell features + RBF SVM',
        single_plane_svm='No existing standalone equivalent; full-volume preprocessing first, then cached browsing')
    write_json(folder/'report.json',result)
    started=perf_counter();hu,geometry=load_development_case(split,record);lung=load_mask(record,split)
    result['input_load_ms']=(perf_counter()-started)*1000;result['source_shape_zyx']=list(hu.shape)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    net=model(cfg).eval();net.load_state_dict(checkpoint['state'])
    raw_z=int(np.argmax(lung.sum(axis=(1,2))));zm=round(raw_z*geometry['spacing_zyx_mm'][0]/cfg.z_step_mm)*cfg.z_step_mm
    z=int(round(zm/geometry['spacing_zyx_mm'][0]));result['selected_plane']=dict(native_z=z,z_mm=zm,selection='largest predicted lung area, no reference labels')
    with threadpool_limits(limits=1,user_api='blas'):
        print('BENCH SVM full volume START',flush=True);tick=perf_counter()
        primary=json.loads(PRIMARY.read_text('utf-8'));packet,features=memory_bounded_primary(hu,lung,geometry,primary)
        features_done=perf_counter();margins=score(features,svm)
        svm_selected=[c for c,s in zip(packet['candidates'],margins) if s>=thresholds['svm_same_scans']];end=perf_counter()
        result['svm_cpu_volume']=dict(total_ms=(end-tick)*1000,primary_and_features_ms=(features_done-tick)*1000,
            score_filter_ms=(end-features_done)*1000,stage_timings_ms=packet['stage_timings_ms'],
            primary_candidates=len(features),selected_candidates=len(svm_selected),backend='CPU reference')
        previous=json.loads((EXPAND/'cases'/uid/'candidates.json').read_text('utf-8'))
        np.testing.assert_allclose([c['center_index_zyx'] for c in packet['candidates']],
                                   [c['center_index_zyx'] for c in previous['candidates']],atol=1e-8,rtol=0)
        with np.load(EXPAND/'cases'/uid/'features.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(features,a['features'],atol=1e-8,rtol=1e-8)
        result['svm_cached_output_equivalence_verified']=True
        write_json(folder/'report.json',result)
        print('BENCH SVM full volume DONE',round(result['svm_cpu_volume']['total_ms']/1000,3),'seconds',flush=True)
        result['cached_svm_browse']=browsing(svm_selected,z);del packet,features;gc.collect()
        raw_predictions={}
        for device in ('cpu','cuda'):
            net.to(device)
            with torch.inference_mode():net(torch.zeros(48,5,cfg.patch,cfg.patch,device=device))
            if device=='cuda':torch.cuda.synchronize()
            print('BENCH CNN',device,'full volume START',flush=True)
            selected,timing=cnn_volume(hu,lung,geometry,net,device,cfg,thresholds['cnn'])
            result['cnn_'+device+'_volume']=timing
            result['cached_cnn_'+device+'_browse']=browsing(selected,z)
            print('BENCH CNN',device,'full volume DONE',round(timing['total_ms']/1000,3),'seconds',flush=True)
            runs=[]
            # Warm the actual per-plane batch shape once before timing repeats.
            raw,_=cnn_plane(hu,lung,geometry,net,device,cfg,thresholds['cnn'],z,zm)
            for _ in range(3):
                raw,run=cnn_plane(hu,lung,geometry,net,device,cfg,thresholds['cnn'],z,zm);runs.append(run)
            result['cnn_'+device+'_plane']=dict(runs=runs,median_ms={k:float(np.median([r[k] for r in runs])) for k in runs[0] if k.endswith('_ms')})
            raw_predictions[device]=raw
            print('BENCH CNN',device,'single plane',round(result['cnn_'+device+'_plane']['median_ms']['total_ms'],2),'ms',flush=True)
            write_json(folder/'report.json',result)
        difference=float(np.max(np.abs(raw_predictions['cpu']-raw_predictions['cuda'])))
        np.testing.assert_allclose(raw_predictions['cpu'],raw_predictions['cuda'],rtol=1e-3,atol=1e-4)
        result['cnn_cpu_cuda_single_plane_max_abs_difference']=difference
    verify(source)
    for p,digest in protected.items():
        if sha_file(p)!=digest:raise ValueError('Protected benchmark identity changed')
    result.update(status='COMPLETE',single_plane_svm_independent_detection_measured=False,
        volume_speedup_cpu=result['svm_cpu_volume']['total_ms']/result['cnn_cpu_volume']['total_ms'],
        volume_speedup_cpu_vs_cuda=result['svm_cpu_volume']['total_ms']/result['cnn_cuda_volume']['total_ms'],
        limitation='One preselected case and one full-volume timing per backend; fixed order may have warm-cache effects. CNN accuracy remains below SVM in current calibration. Not FPGA measurements.')
    write_json(folder/'report.json',result);print('CONTROLLED SPEED COMPARISON COMPLETE',flush=True)


if __name__=='__main__':main()
