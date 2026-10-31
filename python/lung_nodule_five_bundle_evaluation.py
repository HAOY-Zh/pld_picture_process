"""Paired development evaluation: original47, new calibration scans, combined.

Old47 baseline predictions are reused. No sealed-test image reader is exposed.
Thresholds calibrated on original47 are separately checked on new development
scans; none of these are called an independent final test or official tenfold.
"""
import gc
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

from lung_nodule_five_bundle_training import verify,INITIAL,LUNG,OLD,OUT
from lung_candidate_rules_run import ROOT,EXCLUDED
from lung_candidate_data import sha_file
from lung_candidate_run import predict_lungs,excluded_findings
from lung_nodule_cnn import model,config_from_dict
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_expanded_evaluation import paired_inference,verified_done,validate_cached
from lung_nodule_heldout_eval import save_packet,load_packet,save_npz
from lung_nodule_continuity import rerank_scores
from lung_candidate_froc import evaluate as froc
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts
from luna16_multibundle import read_manifest,records_for,load_development_volume,OUT as OLD_DATA

BASE_CAL=ROOT/'data/lung_nodule_expanded_calibration_20261020_v1'
RANKER=ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json'


def verify_rows(rows,scores,curve):
    """Independent threshold counting catches duplicates/exclusion mismatches."""
    for point in curve['operating_points']:
        scored=[dict(r,scores=scores[r['uid']]) for r in rows]
        count=independent_counts(scored,point['threshold'])
        if count['detected']!=point['detected'] or count['reference_nodules']!=curve['reference_nodules']:
            raise ValueError('FROC reference count mismatch')
        if abs(count['false_positive_candidates']/len(rows)-point['actual_fp_per_scan'])>1e-9:
            raise ValueError('FROC false-positive count mismatch')


def run(out=OUT):
    plan,manifest=verify(out);identity=sha_file(out/'plan.json');frozen=read(out/'model_freeze.json')
    if frozen['status']!='COMPLETE' or frozen['optimizer_updates']!=frozen['expected_updates'] or sha_file(out/'model.pt')!=frozen['model_sha256']:
        raise ValueError('New CNN is not a verified completed model')
    torch=setup_torch(plan['seed']);cfg=config_from_dict(plan['config'])
    records=records_for(manifest,'calibration');uids={r['series_uid'] for r in records}
    if uids&set(plan['training_scans']):raise ValueError('Calibration leaks into training')
    old_manifest=read_manifest(OLD_DATA/'manifest.json')
    old_cal={r['series_uid'] for r in records_for(old_manifest,'calibration')}
    if len(old_cal)!=47 or not old_cal<=uids:raise ValueError('Original47 changed')
    models=dict(baseline=dict(path=str(INITIAL),sha256=sha_file(INITIAL)),
                expanded=dict(path=str(out/'model.pt'),sha256=frozen['model_sha256']))
    ranker=read(RANKER)
    protocol=dict(plan_sha256=identity,models=models,ranker_sha256=sha_file(RANKER),ranker_refitted=False,
        calibration_uids=sorted(uids),common47=sorted(old_cal),new_development=sorted(uids-old_cal),
        calibration_is_independent_test=False,fpga_validated=False,sealed_test_scans_accessed=0)
    protocol_path=out/'evaluation_protocol.json'
    if protocol_path.exists():
        if read(protocol_path)!=protocol:raise ValueError('Evaluation protocol changed')
    else:save(protocol_path,protocol)
    nets={}
    for name,info in models.items():
        cp=torch.load(info['path'],map_location='cpu',weights_only=True)
        if set(cp['training_scans'])&uids or config_from_dict(cp['config'])!=cfg:raise ValueError('Wrong checkpoint/cohort')
        nets[name]=model(cfg).cuda().eval();nets[name].load_state_dict(cp['state'])
    from lung_unet import make_model
    cp=torch.load(LUNG,map_location='cpu',weights_only=True)
    lung=make_model(cp['base_channels'],cp.get('context_slices',1)).cuda().eval();lung.load_state_dict(cp['model'])
    folder=out/'evaluation_cases';folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=1),torch.inference_mode():
        for i,r in enumerate(records,1):
            uid=r['series_uid'];case=folder/uid;case.mkdir(exist_ok=True)
            if (case/'complete.json').exists():
                verified_done(case,identity)
                continue
            tick=perf_counter();hu,g=load_development_volume(manifest,uid,'calibration')
            baseline_packet=None
            if uid in old_cal:
                base=BASE_CAL/'cases'/uid
                verified_done(base,sha_file(BASE_CAL/'plan.json'))
                baseline_packet=load_packet(base/'expanded247.json.gz')
                with np.load(base/'expanded247.npz',allow_pickle=False) as a:
                    baseline_packet['features']=a['features'].tolist()
                validate_cached(baseline_packet,r,models['baseline']['sha256'])
                packed=ROOT/'data/lung_nodule_context_20261020_v1/calibration_masks'/(uid+'.npz')
                with np.load(packed,allow_pickle=False) as a:
                    body=np.unpackbits(a['body'],count=int(np.prod(a['shape_zyx']))).reshape(a['shape_zyx']).astype(bool)
                if hashlib.sha256(body.tobytes()).hexdigest()!=baseline_packet['lung_mask_sha256']:
                    raise ValueError('Old baseline lung mask mismatch')
            else:
                raw,body,_=predict_lungs(hu,g,lung,'cuda',16);del raw
            active={'expanded':nets['expanded']} if baseline_packet else nets
            results,schedule=paired_inference(hu,body,g,active,cfg,torch)
            if baseline_packet:
                results['baseline']=(baseline_packet['candidates'],np.asarray(baseline_packet['features']).reshape(-1,38),baseline_packet['evidence'])
            names=[]
            for name in models:
                candidates,features,evidence=results[name]
                scores=np.array([c['score'] for c in candidates]);cross=rerank_scores(scores,features,ranker)
                packet=dict(series_uid=uid,plan_sha256=identity,model_sha256=models[name]['sha256'],
                    input_raw_sha256=r['raw_sha256'],lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    source_shape_zyx=list(hu.shape),geometry=g,candidates=candidates,features=features.tolist(),evidence=evidence,
                    baseline_output_reused=name=='baseline' and baseline_packet is not None,schedule=schedule,
                    backend='FP32_CUDA_NOT_FPGA',annotations_used_by_inference=False)
                a,b=name+'.json.gz',name+'.npz'
                save_packet(case/a,packet);save_npz(case/b,raw_scores=scores,cross_scores=cross,features=features);names.extend((a,b))
            save(case/'complete.json',dict(series_uid=uid,plan_sha256=identity,
                artifacts_sha256={n:sha_file(case/n) for n in names},shared_seconds=perf_counter()-tick))
            save(out/'evaluation_progress.json',dict(status='RUNNING',completed=i,expected=len(records),sealed_test_scans_accessed=0))
            print('CNN EVAL',i,'/',len(records),'baseline_reused',baseline_packet is not None,'seconds',round(perf_counter()-tick,1),flush=True)
            del hu,body,results,candidates,features,evidence,packet,baseline_packet;gc.collect()
    excluded=excluded_findings(EXCLUDED)
    rows={name:[] for name in models};scores={name:{mode:{} for mode in ('raw','cross')} for name in models}
    for r in records:
        uid=r['series_uid'];case=folder/uid;verified_done(case,identity)
        for name in models:
            packet=load_packet(case/(name+'.json.gz'))
            if packet['model_sha256']!=models[name]['sha256'] or packet['input_raw_sha256']!=r['raw_sha256']:
                raise ValueError('Evaluated input/model changed')
            rows[name].append(dict(uid=uid,world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3),
                                  record=r,excluded=excluded.get(uid,[])))
            with np.load(case/(name+'.npz'),allow_pickle=False) as a:
                for mode in ('raw','cross'):scores[name][mode][uid]=a[mode+'_scores']
    results={};fixed={}
    for group,selected in [('common47',old_cal),('new_development',uids-old_cal),('all_calibration',uids)]:
        if not selected:continue
        results[group]={}
        for name in models:
            sub=[r for r in rows[name] if r['uid'] in selected]
            for mode in ('raw','cross'):
                curve=froc(sub,scores[name][mode]);verify_rows(sub,scores[name][mode],curve)
                results[group][name+'_'+mode]=curve
    for name in models:
        fixed[name]={}
        for point in results['common47'][name+'_cross']['operating_points']:
            if point['fp_budget'] not in (.5,1.,4.,8.):continue
            fresh=[dict(r,scores=scores[name]['cross'][r['uid']]) for r in rows[name] if r['uid'] not in old_cal]
            if fresh:fixed[name][str(point['fp_budget'])]=summarize_fixed(fresh,point['threshold'])
    report=dict(status='COMPLETE',models=models,results=results,new_development_at_original47_thresholds=fixed,
        cohorts=dict(common47=len(old_cal),new_development=len(uids-old_cal),all_calibration=len(uids)),
        independent_threshold_counting='PASS',ranker_refitted=False,calibration_is_independent_test=False,
        fpga_validated=False,sealed_test_scans_accessed=0,default_model_changed=False)
    save(out/'evaluation.json',report)
    save(out/'evaluation_progress.json',dict(status='COMPLETE',completed=len(records),expected=len(records),sealed_test_scans_accessed=0))
    print('CNN DEVELOPMENT EVALUATION COMPLETE',flush=True)


if __name__=='__main__':run()
