"""Frozen fit53 whole-volume mining, grouped local-ranker selection and review.

All training scans are already known to the parent CNN. Cross-fitting evaluates
only the new ranker, not an out-of-fold CNN. No sealed-image entrypoint exists.
Calibration/review reuse immutable predictions AND local evidence, with no new
CNN inference or CT loading there. No changes to LDCT/U-Net/RTL/web defaults.
"""
import argparse
from collections import OrderedDict
import gc
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from lung_candidate_data import load_development_case, sha_file
from lung_candidate_rules_run import ROOT,EXCLUDED,labels_for,reference_weights
from lung_candidate_run import excluded_findings
from lung_candidate_froc import evaluate
from lung_candidate_lowfp_audit import check_froc
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_cnn import config_from_dict,model,search_mask,tiles,sample_patch,decode,consolidate
import lung_nodule_cnn_experiment as first
from lung_nodule_continuity import ContinuityConfig,feature_names,rerank_scores
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_continuity_experiment import eligible,CNN
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts

PREVIOUS=ROOT/'data/lung_nodule_continuity_20261019_boundary_v2'
OUT=ROOT/'data/lung_nodule_whole_background_20261020_v1'
CFG=ContinuityConfig()


class FloatPlaneCache:
    """Cache only dtype conversion; sample_patch interpolation is unchanged."""
    def __init__(self,hu,limit=16):
        self.hu=hu
        self.limit=limit
        self.planes=OrderedDict()

    def __len__(self):
        return len(self.hu)

    def __getitem__(self,index):
        index=int(index)
        if index not in self.planes:
            self.planes[index]=np.asarray(self.hu[index],np.float32)
            if len(self.planes)>self.limit:
                self.planes.popitem(last=False)
        self.planes.move_to_end(index)
        return self.planes[index]


def freeze(out):
    if out.exists():
        return
    parent=first.verify(PREVIOUS)
    protected=dict(parent['protected_sha256'])
    files=[PREVIOUS/n for n in ('plan.json','plan.sha256','ranker.json','ranker_freeze.json',
                              'calibration.json','calibration_freeze.json','review.json')]
    files.extend([Path(__file__).resolve(),ROOT/'python/tests/test_lung_nodule_whole_background.py'])
    for cohort in ('calibration_scans','review_12','reused_6'):
        files.extend(PREVIOUS/cohort/(uid+'.json') for uid in parent[cohort])
    for f in files:
        protected[str(f)]=sha_file(f)
    plan={k:parent[k] for k in ('config','fit_scans','calibration_scans','review_12','reused_6','cnn_model_sha256')}
    for cohort in ('fit_scans','calibration_scans','review_12','reused_6'):
        eligible(plan,cohort)
    plan.update(version='whole_volume_training_background_v1',seed=2026102001,
        protected_sha256=protected,continuity=parent['continuity'],feature_names=feature_names(),
        parent_experiment=str(PREVIOUS),
        mining=dict(scope='all fit53 lung tiles; no annotation coordinates in inference',
            decode_floor=plan['config']['decode_floor'],feature_floor=CFG.score_floor,
            top_k=None,training_features='all whole-volume candidates with original score >= .1',
            label_policy='strict center in reference radius; excluded findings ignored, never healthy negatives'),
        selection=dict(folds=5,C_options=[.02,.1],negative_weight_options=['scan_balanced','score_hard'],
            alpha_options=[0.,.25,.5],
            score_hard='within each scan: negative weight proportional to max(original_score,.25)^2',
            constraint='ranker OOF detections at 4 and 8 FP/scan must not fall below original CNN baseline',
            objective='maximum empirical mean sensitivity at seven FP budgets; ties prefer lower alpha then C then uniform weighting',
            limitation='parent CNN saw fit53; OOF applies only to local ranker, not whole detector'),
        model_policy='one frozen ranker selected on training scans only before calibration/review scores are inspected',
        calibration_policy='19 scans determine 1/4/8 FP profiles; freeze before additional18 review',
        expert_voxel_masks_used=False,cnn_retrained=False,heldout_scans_run=0,
        fpga_validated=False,web_default_changed=False)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in files:
        if p.suffix=='.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN full fit53 mining; unchanged v3 CNN, unchanged 38 local features; sealed45 refused',flush=True)


def read_mining(out,uid):
    folder=out/'fit_scans'/uid
    done=read(folder/'complete.json')
    for name,key in [('candidates.json.gz','packet_sha256'),('evidence.npz','evidence_sha256')]:
        if sha_file(folder/name)!=done[key]:
            raise ValueError('Changed full-volume mining artifact')
    if done['plan_sha256']!=sha_file(out/'plan.json'):
        raise ValueError('Wrong mining plan')
    with gzip.open(folder/'candidates.json.gz','rt',encoding='utf-8') as f:
        packet=json.load(f)
    with np.load(folder/'evidence.npz',allow_pickle=False) as a:
        features=np.array(a['features'])
    return packet,features,done


def mine(out):
    plan=first.verify(out)
    torch=setup_torch(plan['seed'])
    cfg=config_from_dict(plan['config'])
    checkpoint=torch.load(CNN/'model.pt',map_location='cpu',weights_only=True)
    if set(checkpoint['training_scans'])!=set(plan['fit_scans']):
        raise ValueError('Wrong parent CNN training cohort')
    net=model(cfg).cuda().eval()
    net.load_state_dict(checkpoint['state'])
    split,records=eligible(plan,'fit_scans')
    ignored=excluded_findings(EXCLUDED)
    with threadpool_limits(limits=1),torch.inference_mode():
        for i,r in enumerate(records,1):
            uid=r['series_uid']
            folder=out/'fit_scans'/uid
            if (folder/'complete.json').exists():
                packet,_,_=read_mining(out,uid)
                if packet['input_raw_sha256']!=r['raw_sha256']:
                    raise ValueError('Changed completed CT identity')
                continue
            tick=perf_counter()
            folder.mkdir(parents=True,exist_ok=True)
            hu,geom=load_development_case(split,r)
            body=first.load_mask(r,split)
            spacing=np.asarray(geom['spacing_zyx_mm'])
            search=search_mask(body,spacing,cfg)
            jobs,schedule=tiles(search,spacing,cfg)
            loaded=perf_counter()
            cached=FloatPlaneCache(hu)
            result=[[],[],[]]
            for begin in range(0,len(jobs),48):
                group=jobs[begin:begin+48]
                images=np.stack([sample_patch(cached,spacing,j,cfg) for j in group])
                tensor=torch.from_numpy(images).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                prediction=net(tensor).cpu().numpy()
                for store,values in zip(result,decode(prediction,group,geom,cfg,search)):
                    store.append(values)
                if begin and begin % 1440 == 0:
                    print('MINING_TILES',i,'/',len(records),begin,'/',len(jobs),flush=True)
            values=[np.concatenate(a) for a in result] if jobs else [[],[],[]]
            candidates=consolidate(*values,geom,cfg)
            for c in candidates:
                box=np.asarray(c['bbox_zyx']).reshape(3,2)
                box[:,0]=np.maximum(box[:,0],0)
                box[:,1]=np.minimum(box[:,1],hu.shape)
                c['bbox_zyx']=box.ravel().tolist()
            inferred=perf_counter()
            features,evidence=evidence_for_packet(hu,spacing,candidates)
            world=np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3)
            # Annotation use starts only AFTER all inference and feature computation.
            labels=labels_for(world,r,ignored.get(uid,[]))
            active=np.array([c['score']>=CFG.score_floor for c in candidates],bool)
            packet=dict(schema_version='lung_candidate_result_v1',series_uid=uid,
                input_raw_sha256=r['raw_sha256'],lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                model_sha256=plan['cnn_model_sha256'],plan_sha256=sha_file(out/'plan.json'),
                source_shape_zyx=list(hu.shape),geometry=geom,candidates=candidates,schedule=schedule,
                parameters=plan['config'],backend='FP32_CUDA_REFERENCE_NOT_FPGA',
                candidate_limit=None,scope='fit53 whole-volume mining; parent CNN in-sample')
            temporary=folder/'candidates.tmp.gz'
            with gzip.open(temporary,'wt',encoding='utf-8') as f:
                json.dump(packet,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            temporary.replace(folder/'candidates.json.gz')
            np.savez_compressed(folder/'evidence.npz',features=features,labels=labels)
            done=dict(series_uid=uid,plan_sha256=sha_file(out/'plan.json'),
                packet_sha256=sha_file(folder/'candidates.json.gz'),evidence_sha256=sha_file(folder/'evidence.npz'),
                total_candidates=len(candidates),examined_candidates=int(active.sum()),
                matched_training_candidates=int(((labels==1)&active).sum()),
                background_training_candidates=int(((labels==0)&active).sum()),
                ignored_training_candidates=int(((labels<0)&active).sum()),
                feature_seconds=evidence['feature_seconds'],load_schedule_seconds=loaded-tick,
                inference_seconds=inferred-loaded,total_seconds=perf_counter()-tick,
                heldout_scans_run=0,top_k=None)
            save(folder/'complete.json',done)
            save(out/'progress.json',dict(stage='mining',completed=i,total=len(records),series_uid=uid,**{
                k:done[k] for k in ('total_seconds','examined_candidates','background_training_candidates')}))
            print('WHOLE_TRAIN',i,'/',len(records),'tiles',len(jobs),'background',done['background_training_candidates'],
                'positive',done['matched_training_candidates'],'seconds',round(done['total_seconds'],2),flush=True)
            del hu,body,search,cached,result,features
            gc.collect()


def training_rows(out):
    plan=first.verify(out)
    _,records=eligible(plan,'fit_scans')
    ignored=excluded_findings(EXCLUDED)
    rows=[]
    for r in records:
        packet,features,done=read_mining(out,r['series_uid'])
        if packet['input_raw_sha256']!=r['raw_sha256'] or packet['model_sha256']!=plan['cnn_model_sha256']:
            raise ValueError('Wrong mined input/model')
        world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
        rows.append(dict(uid=r['series_uid'],record=r,excluded=ignored.get(r['series_uid'],[]),
            world=world,scores=np.array([c['score'] for c in packet['candidates']]),
            labels=labels_for(world,r,ignored.get(r['series_uid'],[])),features=features,done=done))
    return rows


def fit_linear(rows,C,mode):
    xs=[];ys=[];weights=[]
    for row in rows:
        mask=(row['scores']>=CFG.score_floor)&(row['labels']>=0)
        x=row['features'][mask];y=row['labels'][mask]
        w=reference_weights(row['world'][mask],row['record'])
        neg=y==0
        if neg.any():
            nw=np.ones(neg.sum()) if mode=='scan_balanced' else np.maximum(row['scores'][mask][neg],.25)**2
            w[neg]=nw/nw.sum()
        xs.append(x);ys.append(y);weights.append(w)
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    if not (y==1).any() or not (y==0).any():
        raise ValueError('Insufficient training reference/background')
    w[y==0]*=w[y==1].sum()/w[y==0].sum()
    w*=len(w)/w.sum()
    scaler=StandardScaler().fit(x)
    classifier=LogisticRegression(C=C,max_iter=2000,solver='lbfgs')
    classifier.fit(scaler.transform(x),y,sample_weight=w)
    return dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        coef=classifier.coef_[0].tolist(),intercept=float(classifier.intercept_[0]),alpha=0.)


def select(out):
    plan=first.verify(out)
    if (out/'ranker.json').exists():
        return
    rows=training_rows(out)
    baseline=evaluate(rows,{r['uid']:r['scores'] for r in rows})
    baseline4=baseline['operating_points'][5]['detected']
    baseline8=baseline['operating_points'][6]['detected']
    folds=list(GroupKFold(5).split(np.zeros(len(rows)),groups=[r['uid'] for r in rows]))
    trials=[]
    (out/'selection').mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for C in plan['selection']['C_options']:
            for mode in plan['selection']['negative_weight_options']:
                predictions={str(a):{} for a in plan['selection']['alpha_options']}
                for k,(train_ids,val_ids) in enumerate(folds):
                    ranker=fit_linear([rows[j] for j in train_ids],C,mode)
                    for alpha in plan['selection']['alpha_options']:
                        ranker['alpha']=alpha
                        for j in val_ids:
                            r=rows[j]
                            predictions[str(alpha)][r['uid']]=rerank_scores(r['scores'],r['features'],ranker)
                for alpha in plan['selection']['alpha_options']:
                    curve=evaluate(rows,predictions[str(alpha)])
                    valid=(curve['operating_points'][5]['detected']>=baseline4 and
                           curve['operating_points'][6]['detected']>=baseline8)
                    trial=dict(C=C,negative_weight=mode,alpha=alpha,eligible=valid,curve=curve)
                    trials.append(trial)
                    name=f'C{C}_{mode}_a{alpha}'
                    np.savez_compressed(out/'selection'/(name+'.npz'),**predictions[str(alpha)])
                    print('RANKER_TRIAL',name,'eligible',valid,'empirical_CPM',round(curve['mean_empirical_sensitivity'],5),flush=True)
        best=max((t for t in trials if t['eligible']),key=lambda t:(t['curve']['mean_empirical_sensitivity'],
            -t['alpha'],-t['C'],t['negative_weight']=='scan_balanced'))
        ranker=fit_linear(rows,best['C'],best['negative_weight'])
    ranker.update(alpha=best['alpha'],feature_names=feature_names(),training_scans=plan['fit_scans'],
        cnn_model_sha256=plan['cnn_model_sha256'],plan_sha256=sha_file(out/'plan.json'),
        selected_C=best['C'],negative_weight=best['negative_weight'],
        score_kind='whole_background_continuity_rank_not_calibrated_probability')
    save(out/'selection.json',dict(baseline=baseline,trials=trials,selected={k:best[k] for k in ('C','negative_weight','alpha')},
        folds=[dict(training_scans=[rows[i]['uid'] for i in a],validation_scans=[rows[i]['uid'] for i in b]) for a,b in folds],
        training_candidates=sum(len(r['scores']) for r in rows),
        used_background=sum(r['done']['background_training_candidates'] for r in rows),
        used_matched=sum(r['done']['matched_training_candidates'] for r in rows),
        limitation='CNN saw all fit53; local-ranker grouped OOF only'))
    save(out/'ranker.json',ranker)
    artifacts=[out/'selection.json']+list((out/'selection').glob('*.npz'))
    for uid in plan['fit_scans']:
        artifacts.extend(out/'fit_scans'/uid/n for n in ('candidates.json.gz','evidence.npz','complete.json'))
    save(out/'ranker_freeze.json',dict(ranker_sha256=sha_file(out/'ranker.json'),plan_sha256=sha_file(out/'plan.json'),
        training_artifacts_sha256={str(p):sha_file(p) for p in artifacts},calibration_used_for_fit=False,
        review_used_for_fit=False,heldout_scans_run=0))
    print('RANKER_FROZEN',ranker['selected_C'],ranker['negative_weight'],'alpha',ranker['alpha'],flush=True)


def frozen_ranker(out):
    frozen=read(out/'ranker_freeze.json')
    if sha_file(out/'ranker.json')!=frozen['ranker_sha256'] or sha_file(out/'plan.json')!=frozen['plan_sha256']:
        raise ValueError('Changed frozen ranker')
    return read(out/'ranker.json')


def cached_rows(out,cohort):
    plan=first.verify(out)
    if cohort not in ('calibration_scans','review_12','reused_6'):
        raise ValueError('Only declared development evidence caches')
    _,records=eligible(plan,cohort)
    ignored=excluded_findings(EXCLUDED)
    rows=[]
    for r in records:
        packet=read(PREVIOUS/cohort/(r['series_uid']+'.json'))
        if packet['model_sha256']!=plan['cnn_model_sha256'] or packet['input_raw_sha256']!=r['raw_sha256']:
            raise ValueError('Wrong reused evidence identity')
        cs=packet['candidates']
        rows.append(dict(uid=r['series_uid'],record=r,excluded=ignored.get(r['series_uid'],[]),packet=packet,
            world=np.array([c['center_lps_xyz_mm'] for c in cs]).reshape(-1,3),
            scores=np.array([c['score'] for c in cs]),features=np.array(packet['features'])))
    return rows


def predictions(rows,ranker):
    return {r['uid']:rerank_scores(r['scores'],r['features'],ranker) for r in rows}


def calibrate(out):
    plan=first.verify(out)
    ranker=frozen_ranker(out)
    if (out/'calibration.json').exists():
        return
    rows=cached_rows(out,'calibration_scans')
    old=read(PREVIOUS/'ranker.json')
    models={'cnn':dict(old,alpha=0.),'previous':old,'whole_background':ranker}
    curves={}
    for name,model_info in models.items():
        score=predictions(rows,model_info)
        curves[name]=evaluate(rows,score)
        check_froc(rows,score,curves[name])
    save(out/'calibration.json',dict(curves=curves,
        thresholds={n:{str(p['fp_budget']):p['threshold'] for p in c['operating_points'] if p['fp_budget'] in (1.,4.,8.)} for n,c in curves.items()},
        ranker_sha256=sha_file(out/'ranker.json'),calibration_source=str(PREVIOUS/'calibration_scans'),
        new_ct_reads=0,new_cnn_forwards=0,heldout_scans_run=0))
    save(out/'calibration_freeze.json',dict(calibration_sha256=sha_file(out/'calibration.json'),
        ranker_sha256=sha_file(out/'ranker.json'),review_used_to_choose_thresholds=False))
    print('CALIBRATION_COMPLETE',{n:[p['detected'] for p in c['operating_points'] if p['fp_budget'] in (1.,4.,8.)] for n,c in curves.items()},flush=True)


def review(out):
    plan=first.verify(out)
    ranker=frozen_ranker(out)
    calibration=read(out/'calibration.json')
    if sha_file(out/'calibration.json')!=read(out/'calibration_freeze.json')['calibration_sha256']:
        raise ValueError('Changed calibration thresholds')
    if (out/'review.json').exists():
        return
    old=read(PREVIOUS/'ranker.json')
    models={'cnn':dict(old,alpha=0.),'previous':old,'whole_background':ranker}
    results={}
    for cohort in ('review_12','reused_6'):
        rows=cached_rows(out,cohort)
        result={}
        for name,ranker_info in models.items():
            score=predictions(rows,ranker_info)
            used=[dict(r,scores=score[r['uid']]) for r in rows]
            result[name]={}
            for profile,threshold in calibration['thresholds'][name].items():
                metric=summarize_fixed(used,threshold)
                for k,v in independent_counts(used,threshold).items():
                    if metric[k]!=v:
                        raise AssertionError('Independent count mismatch')
                result[name][profile]=metric
        results[cohort]=result
    save(out/'review.json',dict(results=results,ranker_sha256=sha_file(out/'ranker.json'),
        calibration_sha256=sha_file(out/'calibration.json'),review_thresholds_retuned=False,
        new_cnn_forwards=0,new_ct_reads=0,heldout_scans_run=0))
    print('REVIEW_COMPLETE',{k:{n:(m['4.0']['detected'],m['4.0']['actual_fp_per_scan']) for n,m in v.items()} for k,v in results.items()},flush=True)


def run(out,stage):
    if stage in ('all','freeze'):
        freeze(out)
    if stage in ('all','mine'):
        mine(out)
    if stage in ('all','select'):
        select(out)
    if stage in ('all','calibrate'):
        calibrate(out)
    if stage in ('all','review'):
        review(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=OUT)
    p.add_argument('--stage',choices=['all','freeze','mine','select','calibrate','review'],default='all')
    args=p.parse_args();run(args.outdir,args.stage)
