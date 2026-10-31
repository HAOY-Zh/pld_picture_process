"""User-authorized, frozen all45 final test of the continued 13k nodule CNN.

No training, calibration, checkpoint choice, case limit, or model-selection CLI.
Freeze the protocol before any test CT read. Inference receives no annotations.
"""
import argparse
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_candidate_data import read_split,sha_file
from lung_candidate_rules_run import ROOT,SPLIT,EXCLUDED
from lung_candidate_run import predict_lungs
from lung_nodule_cnn import config_from_dict,model,search_mask,tiles,decode,consolidate
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_continuity import rerank_scores
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_test_data import test_records,load_test_case,require_complete
import lung_nodule_cnn_experiment as first

OUT = ROOT/'data/lung_nodule_heldout_20261020_small_v1'
PARENT = ROOT/'data/lung_nodule_capacity_20261020_amp_safe_v2'
MODEL = PARENT/'small_continued/model.pt'
RANKER = ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json'
LUNG = ROOT/'data/lung_training_20261011/early_stop_run/best.pt'
PROFILES = ('1.0','4.0','8.0')


def freeze(out):
    if out.exists():
        raise FileExistsError('Resume existing frozen protocol instead')
    parent = first.verify(PARENT)
    split = read_split(SPLIT)
    records = test_records(split)
    if len(records)!=45 or sum(len(r['nodules']) for r in records)!=56:
        raise ValueError('Expected complete audited45/56 final test')
    test = {r['series_uid'] for r in records}
    development = set().union(*(set(parent[k]) for k in ('fit_scans','calibration_scans','review_12','reused_6')))
    if test & development:
        raise ValueError('Train/development/test overlap')
    f = read(MODEL.parent/'model_freeze.json')
    cal = read(PARENT/'calibration.json')
    cf = read(PARENT/'calibration_freeze.json')
    if (sha_file(MODEL)!=f['model_sha256'] or sha_file(PARENT/'calibration.json')!=cf['calibration_sha256'] or
            cf['model_sha256']['small_continued']!=f['model_sha256'] or sha_file(RANKER)!=cf['ranker_sha256']):
        raise ValueError('Changed chosen model or calibration/ranker')
    protected = dict(parent['protected_sha256'])
    sources = ['lung_nodule_heldout_eval.py','lung_nodule_test_data.py','lung_nodule_cnn.py',
        'lung_nodule_sampling_cache.py','lung_nodule_continuity.py','lung_nodule_continuity_boundary.py',
        'lung_nodule_capacity_froc_audit.py','lung_candidate_data.py','luna16_archive_audit.py',
        'lung_candidate_run.py','lung_candidates.py','lung_candidate_froc.py','lung_candidate_rules_run.py',
        'lung_body_gate.py','lung_body_crop.py','lung_unet.py',
        'tests/test_lung_nodule_test_data.py']
    extra = [ROOT/'python'/n for n in sources]
    extra += [MODEL,MODEL.parent/'model_freeze.json',LUNG,RANKER,EXCLUDED,
        PARENT/'plan.json',PARENT/'plan.sha256',PARENT/'calibration.json',PARENT/'calibration_freeze.json',
        PARENT/'audit.json',SPLIT,SPLIT.with_suffix('.sha256')]
    for p in extra:
        protected[str(p)]=sha_file(p)
    plan = dict(version='final45_small_continued_frozen_test_v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='2026-10-20: 用户要求“把测试集都测一下”；按当前讨论的小模型续训版执行全部45例',
        selected_model='small_continued',cnn_checkpoint=str(MODEL),cnn_sha256=f['model_sha256'],
        lung_checkpoint=str(LUNG),lung_sha256=sha_file(LUNG),ranker=str(RANKER),ranker_sha256=sha_file(RANKER),
        config=parent['configs']['small_continued'],
        test_uids=[r['series_uid'] for r in records],test_scans=45,reference_nodules=56,
        test_with_reference=29,test_without_reference=16,
        fit_scans=parent['fit_scans'],development_uids=sorted(development),split_sha256=sha_file(SPLIT),
        thresholds={mode:{p:cal['thresholds']['small_continued_'+key][p] for p in PROFILES}
                    for mode,key in [('cross','cross'),('raw','raw')]},
        primary=dict(score='cross',profile='4.0',metrics=['reference nodule recall','mean false positives per full CT']),
        secondary='pre-existing low/high profiles; raw CNN ablation; whole-test FROC is descriptive, no thresholds exported',
        lung_batch=16,cnn_batch=48,device='cuda',seed=2026102003,
        lung_policy='same original FP32 epoch12 256px model, float16 input roundtrip, >=.5 and same body gate',
        score_policy='same first cross-slice 38-feature ranker alpha=.25; no refitting',
        failure_policy='attempt all45; record failures; final report requires all45 verified complete; no performance exclusions',
        interpretation='scan-UID/raw-pixel independent split; original PatientID unavailable; not official LUNA16 ten-fold',
        test_use='once exposed, this45 cannot be described as a new blind set in later tuning',
        annotations_not_algorithm_input=True,training_allowed=False,threshold_tuning_allowed=False,
        fpga_validated=False,protected_sha256=protected)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in extra:
        if p.suffix=='.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FINAL TEST FROZEN: small13157, all45/56, fixed primary and low/high thresholds',flush=True)


def save_npz(path,**arrays):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**arrays)
    os.replace(temporary,path)


def save_packet(path,packet):
    temporary=path.with_name(path.name+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=3) as f:
        json.dump(packet,f,ensure_ascii=False,allow_nan=False)
    os.replace(temporary,path)


def load_packet(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return json.load(f)


def verify_case(out,uid,identity):
    folder=out/'cases'/uid
    done=read(folder/'complete.json')
    if done['series_uid']!=uid or done['plan_sha256']!=identity:
        raise ValueError('Wrong completed test identity')
    for name,digest in done['artifacts_sha256'].items():
        path=folder/name
        if path.parent!=folder or sha_file(path)!=digest:
            raise ValueError('Changed completed test artifact')
    return done


def infer_volume(hu,body,geometry,net,cfg,torch,batch_size):
    spacing=np.asarray(geometry['spacing_zyx_mm'])
    tick=perf_counter()
    search=search_mask(body,spacing,cfg)
    jobs,schedule=tiles(search,spacing,cfg)
    timing=dict(search_schedule_seconds=perf_counter()-tick,sampling_transfer_seconds=0.,
                forward_transfer_seconds=0.,decode_seconds=0.)
    cached=FloatPlaneCache(hu)
    accumulated=[[],[],[]]
    for begin in range(0,len(jobs),batch_size):
        batch=jobs[begin:begin+batch_size]
        tick=perf_counter()
        image=np.stack([sample_patch_reuse(cached,spacing,j,cfg) for j in batch])
        tensor=torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
        torch.cuda.synchronize()
        timing['sampling_transfer_seconds']+=perf_counter()-tick
        tick=perf_counter()
        prediction=net(tensor).cpu().numpy()
        timing['forward_transfer_seconds']+=perf_counter()-tick
        tick=perf_counter()
        for store,values in zip(accumulated,decode(prediction,batch,geometry,cfg,search)):
            store.append(values)
        timing['decode_seconds']+=perf_counter()-tick
    tick=perf_counter()
    candidates=consolidate(*(np.concatenate(a) for a in accumulated),geometry,cfg) if jobs else []
    for c in candidates:
        box=np.asarray(c['bbox_zyx']).reshape(3,2)
        box[:,0]=np.maximum(box[:,0],0)
        box[:,1]=np.minimum(box[:,1],hu.shape)
        c['bbox_zyx']=box.ravel().tolist()
    timing['consolidation_seconds']=perf_counter()-tick
    features,evidence=evidence_for_packet(hu,spacing,candidates)
    timing['local_evidence_seconds']=evidence['feature_seconds']
    return candidates,features,evidence,search,schedule,timing


def infer(out):
    plan=first.verify(out)
    split=read_split(SPLIT)
    records=test_records(split,plan['test_uids'])
    identity=sha_file(out/'plan.json')
    torch=setup_torch(plan['seed'])
    from lung_unet import make_model
    from threadpoolctl import threadpool_limits
    from lung_candidates import checked_inputs
    ckpt=torch.load(plan['cnn_checkpoint'],map_location='cpu',weights_only=True)
    if set(ckpt['training_scans'])!=set(plan['fit_scans']) or ckpt['config']!=plan['config']:
        raise ValueError('Chosen detector training/config differs')
    cfg=config_from_dict(plan['config'])
    net=model(cfg).cuda().eval()
    net.load_state_dict(ckpt['state'])
    cp=torch.load(plan['lung_checkpoint'],map_location='cpu',weights_only=True)
    if (cp['epoch']!=12 or cp['image_size']!=256 or cp.get('context_slices',1)!=1 or cp.get('threshold',.5)!=.5):
        raise ValueError('Wrong lung predictor')
    lung=make_model(cp['base_channels'],1).cuda().eval()
    lung.load_state_dict(cp['model'])
    ranker=read(Path(plan['ranker']))
    if ranker['alpha']!=.25:
        raise ValueError('Wrong original cross-slice score')
    if not (out/'test_access_started.json').exists():
        save(out/'test_access_started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            plan_sha256=identity,cnn_sha256=plan['cnn_sha256'],all_test_scans=45,
            authorization=plan['user_authorization']))
    successes=[];failures=[]
    with threadpool_limits(limits=1),torch.inference_mode():
        for i,r in enumerate(records,1):
            uid=r['series_uid']
            folder=out/'cases'/uid
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                verify_case(out,uid,identity)
                successes.append(uid)
                print('FINAL TEST REUSE',i,'/',len(records),flush=True)
                continue
            tick=perf_counter()
            try:
                hu,geometry=load_test_case(split,uid,plan['test_uids'])
                loaded=perf_counter()
                raw,body,lung_ms=predict_lungs(hu,geometry,lung,'cuda',plan['lung_batch'])
                lung_done=perf_counter()
                checked_inputs(hu,body,geometry)
                candidates,features,evidence,search,schedule,timing=infer_volume(
                    hu,body,geometry,net,cfg,torch,plan['cnn_batch'])
                original=np.array([c['score'] for c in candidates])
                adjusted=rerank_scores(original,features,ranker)
                detection_done=perf_counter()
                timing.update(load_seconds=loaded-tick,lung_wall_seconds=lung_done-loaded,
                    lung_internal_ms=lung_ms,detector_wall_seconds=detection_done-lung_done,
                    load_lung_detector_wall_seconds=detection_done-tick)
                packet=dict(series_uid=uid,split='heldout',schema_version='lung_candidate_result_v1',
                    algorithm_id='small_continued_first_cross_slice_final45',backend='FP32_CUDA_NOT_FPGA',
                    plan_sha256=identity,model_sha256=plan['cnn_sha256'],ranker_sha256=plan['ranker_sha256'],
                    lung_model_sha256=plan['lung_sha256'],input_raw_sha256=r['raw_sha256'],
                    lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    search_mask_sha256=hashlib.sha256(search.tobytes()).hexdigest(),
                    source_shape_zyx=list(hu.shape),geometry=geometry,parameters=plan['config'],
                    candidates=candidates,evidence=evidence,schedule=schedule,timing=timing,
                    score_policy='candidate.score is original CNN; npz cross_scores apply the frozen ranker',
                    timing_scope='resident models; native archive load + fresh FP32 lungs + complete detector; excludes saving/HTTP/board/3D')
                save_packet(folder/'candidates.json.gz',packet)
                save_npz(folder/'scores_features.npz',features=features,raw_scores=original,cross_scores=adjusted,
                    candidate_ids=np.asarray([c['candidate_id'] for c in candidates],dtype='U64'))
                save_npz(folder/'masks_packed.npz',shape_zyx=np.array(hu.shape),raw=np.packbits(raw),
                    body=np.packbits(body),search=np.packbits(search))
                names=['candidates.json.gz','scores_features.npz','masks_packed.npz']
                for profile,threshold in plan['thresholds']['cross'].items():
                    selected=[]
                    for c,score in zip(candidates,adjusted):
                        if score>=threshold:
                            selected.append(dict(c,cnn_score=c['score'],score=float(score),
                                score_kind='frozen_cross_slice_ranking_score_not_calibrated_probability'))
                    name='profile_'+profile+'.json'
                    save(folder/name,dict(series_uid=uid,plan_sha256=identity,threshold=threshold,
                        calibration_profile=profile,candidates=selected,geometry=geometry,
                        candidate_region_kind='estimated_bbox_not_expert_segmentation'))
                    names.append(name)
                save(folder/'complete.json',dict(series_uid=uid,plan_sha256=identity,input_raw_sha256=r['raw_sha256'],
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    artifacts_sha256={name:sha_file(folder/name) for name in names},
                    candidates=len(candidates),load_lung_detector_wall_seconds=detection_done-tick))
                successes.append(uid)
                print('FINAL TEST',i,'/',len(records),'tiles',schedule['lung_tiles'],'candidates',len(candidates),
                    'seconds',round(detection_done-tick,2),flush=True)
                del hu,raw,body,search,candidates,features,evidence
                gc.collect()
            except Exception as error:
                failure=dict(series_uid=uid,error=repr(error),plan_sha256=identity)
                failures.append(failure)
                save(folder/'failure.json',failure)
                print('FINAL TEST ERROR',i,'/',len(records),repr(error),flush=True)
            save(out/'inference_progress.json',dict(expected=45,completed=len(successes),failures=failures,
                completed_uids=successes,protocol_sha256=identity,partial_test_metrics_not_selected_or_used=True))
    require_complete(plan['test_uids'],successes)
    first.verify(out)
    save(out/'inference_complete.json',dict(status='COMPLETE',completed_uids=successes,completed=45,
        failures=failures,protocol_sha256=identity,annotations_used_for_inference=False))


def run(out):
    if not out.exists():
        freeze(out)
    try:
        if not (out/'inference_complete.json').exists():
            save(out/'pipeline_state.json',dict(status='RUNNING',stage='infer_all45'))
            infer(out)
        from lung_nodule_heldout_report import run as report
        if not (out/'audit.json').exists():
            save(out/'pipeline_state.json',dict(status='RUNNING',stage='independent_report'))
            report(out)
        first.verify(out)
        save(out/'pipeline_state.json',dict(status='COMPLETE_ALL45_FINAL_TEST',heldout_scans_run=45,
            user_authorized_unsealing=True,model_or_threshold_tuning_performed=False))
    except Exception as error:
        save(out/'pipeline_state.json',dict(status='FAILED',error=repr(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    run(parser.parse_args().outdir.resolve())
