"""Frozen development experiment for lung-guided CNN candidate detection."""
import argparse
from dataclasses import asdict
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from lung_candidate_data import read_split,load_development_case,sha_file,write_json
from lung_candidate_rules_run import ROOT,SPLIT,EXCLUDED,ARCHIVE,BASELINES
from lung_candidate_run import excluded_findings
from lung_nodule_cnn import (DetectorConfig,config_from_dict,model,search_mask,tiles,sample_patch,
                              relative_labels,targets,loss,decode,consolidate)

EXPAND=ROOT/'data/lung_candidate_expand_20261018'


def verify(out):
    if sha_file(out/'plan.json')!=(out/'plan.sha256').read_text().strip():raise ValueError('Changed plan')
    plan=json.loads((out/'plan.json').read_text('utf-8'))
    for name,digest in plan['protected_sha256'].items():
        if sha_file(name)!=digest:raise ValueError('Changed protected input: '+name)
    return plan


def records_for(plan,key):
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']}
    if key not in ('fit_scans','calibration_scans','review_12','reused_6'):raise ValueError('No heldout mode')
    result=[records[u] for u in plan[key]]
    if any(r['split']!='development' for r in result):raise ValueError('Heldout refused')
    return split,result


def load_mask(record,split):
    uid=record['series_uid'];folder=EXPAND/'cases'/uid
    if folder.exists():
        meta=json.loads((folder/'mask.json').read_text('utf-8'))
        if sha_file(folder/'lung_mask.npz')!=meta['packed_sha256']:raise ValueError('Changed mask')
        with np.load(folder/'lung_mask.npz',allow_pickle=False) as a:
            shape=tuple(a['shape_zyx']);body=np.unpackbits(a['lung_body'],count=int(np.prod(shape))).reshape(shape).astype(bool)
    else:
        pilot=uid in split['pilot_development_uids'];base=BASELINES[0 if pilot else 1]
        old_six=ROOT/'data/lung_candidate_rules_20261018/new_development_check'/uid
        if old_six.exists():base=old_six.parent
        meta=json.loads((base/uid/'candidates.json').read_text('utf-8'))
        if pilot:
            with ZipFile(ARCHIVE) as z:
                payload=z.read(f'data/lung_candidates_pilot_20261018_v1/{uid}/masks_packed.npz')
            handle=BytesIO(payload)
        else:handle=base/uid/('lung_mask_packed.npz' if old_six.exists() else 'masks_packed.npz')
        with np.load(handle,allow_pickle=False) as a:
            shape=tuple(a['shape_zyx']);body=np.unpackbits(a['lung_body'],count=int(np.prod(shape))).reshape(shape).astype(bool)
    if (meta['input_raw_sha256']!=record['raw_sha256'] or tuple(record['shape_zyx'])!=shape or
            hashlib.sha256(body.tobytes()).hexdigest()!=meta['lung_mask_sha256']):raise ValueError('Mask identity mismatch')
    return body


def freeze(out):
    if out.exists():raise FileExistsError('Use fresh output; preserve existing experiment')
    expanded=json.loads((EXPAND/'plan.json').read_text('utf-8'));cfg=DetectorConfig().checked()
    fold=expanded['folds'][0]
    sources=[ROOT/'python'/n for n in ('lung_nodule_cnn.py','lung_nodule_cnn_experiment.py',
        'lung_candidate_data.py','lung_candidate_froc.py','lung_candidates.py','lung_candidate_run.py')]
    protected={str(p):sha_file(p) for p in [SPLIT,EXCLUDED,EXPAND/'plan.json',ARCHIVE,*sources]}
    fit=fold['training_scans'];cal=fold['validation_scans']
    if set(fit)&set(cal) or set(fit+cal)!=set(expanded['training_72']):raise ValueError('Invalid split')
    split=read_split(SPLIT)
    for uid in fit+cal+expanded['fresh_development_12']+expanded['reused_development_6']:
        folder=EXPAND/'cases'/uid
        if folder.exists():
            paths=[folder/'mask.json',folder/'lung_mask.npz']
        else:
            pilot=uid in split['pilot_development_uids'];base=BASELINES[0 if pilot else 1]
            old_six=ROOT/'data/lung_candidate_rules_20261018/new_development_check'/uid
            if old_six.exists():base=old_six.parent
            paths=[base/uid/'candidates.json']
            if not pilot:paths.append(base/uid/('lung_mask_packed.npz' if old_six.exists() else 'masks_packed.npz'))
        for p in paths:protected[str(p)]=sha_file(p)
    for p in (EXPAND/'training/selected_oof_scores.npz',EXPAND/'evaluation/report.json'):
        protected[str(p)]=sha_file(p)
    plan=dict(version='lung_guided_25d_center_v1',config=asdict(cfg),seed=20261019,
        fit_scans=fit,calibration_scans=cal,review_12=expanded['fresh_development_12'],
        reused_6=expanded['reused_development_6'],heldout_scans_run=0,
        training=dict(epochs=20,batch_size=24,learning_rate=.001,positive_crops_per_reference=16,
                      negative_crops_per_scan=24,checkpoint='fixed last epoch; no checkpoint selection on review data'),
        validation_policy='fit/calibration inside original training72; calibrate then freeze before review12+6',
        labels='reference center+diameter; heatmap and offset/size regression; no expert voxel segmentation',
        mask_source='cached original FP32 U-Net, NOT deployed pruned INT8 masks; board-mask robustness untested',
        inference='regular grid of lung-intersecting cores; no reference boxes, no LoG, no SVM, no top-K truncation',
        patient_independence_proven=False,formal_deployment=False,fpga_validated=False,
        exact_paper_reproduction=False,protected_sha256=protected)
    records_for(plan,'fit_scans');records_for(plan,'calibration_scans')
    out.mkdir(parents=True);(out/'sources').mkdir()
    for p in sources:(out/'sources'/p.name).write_bytes(p.read_bytes())
    write_json(out/'plan.json',plan);(out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('PLAN FROZEN fit',len(fit),'calibration',len(cal),'heldout untouched',flush=True)


def prepare(out):
    plan=verify(out);cfg=config_from_dict(plan['config']);split,records=records_for(plan,'fit_scans')
    ignored=excluded_findings(EXCLUDED);folder=out/'patches';folder.mkdir(exist_ok=True)
    rng=np.random.default_rng(plan['seed']);summary=[]
    for i,r in enumerate(records,1):
        uid=r['series_uid'];done=folder/(uid+'.json');path=folder/(uid+'.npy')
        # Reproduce the per-case RNG sequence even when resuming.
        rng=np.random.default_rng(int(hashlib.sha256((str(plan['seed'])+uid).encode()).hexdigest()[:16],16))
        if done.exists():
            meta=json.loads(done.read_text('utf-8'))
            if sha_file(path)!=meta['patch_sha256']:raise ValueError('Changed patch cache')
            summary.append(meta);continue
        start=perf_counter();hu,geometry=load_development_case(split,r);body=load_mask(r,split)
        spacing=np.asarray(geometry['spacing_zyx_mm']);search=search_mask(body,spacing,cfg)
        jobs,stats=tiles(search,spacing,cfg)
        if not jobs:raise ValueError('Empty lung search schedule; review lung prediction')
        coverage=[]
        for n in r['nodules']:
            q=np.rint(n['center_index_zyx']).astype(int)
            covered=bool(((q>=0)&(q<search.shape)).all() and search[tuple(q)])
            coverage.append(dict(center_in_search=covered,diameter_mm=n['diameter_mm']))
        requests=[]
        for n in r['nodules']:
            p=np.array(n['center_index_zyx'])*spacing
            zm=round(p[0]/cfg.z_step_mm)*cfg.z_step_mm
            # Same z grid as deployment. XY jitter prevents reference-centered inference assumptions.
            for _ in range(plan['training']['positive_crops_per_reference']):
                xy=p[1:]/cfg.xy_mm-cfg.patch/2+rng.uniform(-28,28,2)
                requests.append((float(zm),int(round(xy[0])),int(round(xy[1]))))
        for j in rng.choice(len(jobs),size=plan['training']['negative_crops_per_scan'],replace=len(jobs)<plan['training']['negative_crops_per_scan']):
            requests.append(jobs[int(j)])
        data=np.lib.format.open_memmap(path,mode='w+',dtype=np.int16,shape=(len(requests),5,cfg.patch,cfg.patch))
        labels=[]
        for j,job in enumerate(requests):
            data[j]=sample_patch(hu,spacing,job,cfg)
            pos,ignore=relative_labels(r['nodules'],ignored.get(uid,[]),geometry,job,cfg)
            labels.append(dict(job=list(job),positive=pos,ignored=ignore))
        data.flush();del data
        meta=dict(series_uid=uid,count=len(requests),labels=labels,patch_sha256=sha_file(path),
                  raw_sha256=r['raw_sha256'],lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                  schedule=stats,training_reference_search_coverage=coverage,elapsed_seconds=perf_counter()-start)
        write_json(done,meta);summary.append(meta)
        print('PREPARE',i,'/',len(records),'patches',len(requests),'seconds',round(meta['elapsed_seconds'],2),flush=True)
        del hu,body,search;gc.collect()
    verify(out)
    write_json(out/'preparation.json',dict(status='COMPLETE',fit_scans=len(summary),
        patches=sum(s['count'] for s in summary),bytes=sum((folder/(s['series_uid']+'.npy')).stat().st_size for s in summary),
        heldout_scans_run=0,lung_segmentation_rerun=False))


def train(out):
    import torch
    from torch.utils.data import Dataset,DataLoader
    plan=verify(out);cfg=config_from_dict(plan['config']);torch.set_num_threads(4)
    if (out/'model.pt').exists():raise FileExistsError('Preserve trained model')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this first experiment')
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False

    class Patches(Dataset):
        def __init__(self):
            self.maps=[];self.items=[]
            for uid in plan['fit_scans']:
                meta=json.loads((out/'patches'/(uid+'.json')).read_text('utf-8'))
                if sha_file(out/'patches'/(uid+'.npy'))!=meta['patch_sha256']:raise ValueError('Changed training cache')
                a=np.load(out/'patches'/(uid+'.npy'),mmap_mode='r',allow_pickle=False)
                self.maps.append(a)
                self.items.extend((len(self.maps)-1,j,label) for j,label in enumerate(meta['labels']))

        def __len__(self):return len(self.items)

        def __getitem__(self,index):
            k,j,label=self.items[index];image=np.array(self.maps[k][j],copy=True)
            pos=[list(p) for p in label['positive']];ignored=[list(p) for p in label['ignored']]
            for axis,coordinate in ((2,0),(1,1)):
                if random.random()<.5:
                    image=np.flip(image,axis).copy()
                    for p in pos+ignored:p[coordinate]=cfg.patch-1-p[coordinate]
            return torch.from_numpy(image),tuple(torch.from_numpy(a) for a in targets(pos,ignored,cfg))

    dataset=Patches();loader=DataLoader(dataset,batch_size=plan['training']['batch_size'],shuffle=True,
        num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(plan['seed']))
    net=model(cfg).cuda();optimizer=torch.optim.Adam(net.parameters(),lr=plan['training']['learning_rate'])
    scaler=torch.amp.GradScaler('cuda');history=[];start=perf_counter()
    for epoch in range(plan['training']['epochs']):
        net.train();totals=np.zeros(3);batches=0;epoch_start=perf_counter()
        for image,target in loader:
            image=image.cuda(non_blocking=True).float().clamp(-1000,400).add_(1000).div_(1400)
            target=tuple(t.cuda(non_blocking=True) for t in target);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                prediction=net(image);value,classification,regression=loss(prediction.float(),target)
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            scaler.scale(value).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            totals+=np.array([value.item(),classification.item(),regression.item()]);batches+=1
        item=dict(epoch=epoch+1,loss=(totals/batches).tolist(),seconds=perf_counter()-epoch_start)
        history.append(item);write_json(out/'training_progress.json',dict(status='RUNNING',epochs=history))
        print('TRAIN',epoch+1,'/',plan['training']['epochs'],'loss',round(item['loss'][0],4),'seconds',round(item['seconds'],2),flush=True)
    net.cpu();checkpoint=dict(state=net.state_dict(),config=plan['config'],plan_sha256=sha_file(out/'plan.json'),
        epochs=len(history),training_scans=plan['fit_scans'])
    torch.save(checkpoint,out/'model.pt')
    verify(out)
    write_json(out/'model_freeze.json',dict(model_sha256=sha_file(out/'model.pt'),plan_sha256=sha_file(out/'plan.json'),
        fit_scans=len(plan['fit_scans']),epochs=history,total_seconds=perf_counter()-start,
        parameters=sum(p.numel() for p in net.parameters()),calibration_not_run=True,
        review_12_not_run=True,heldout_scans_run=0,fpga_validated=False))
    print('TRAIN COMPLETE; model frozen before calibration',flush=True)


def infer(out,key):
    import torch
    from lung_candidates import checked_inputs
    plan=verify(out);cfg=config_from_dict(plan['config']);split,records=records_for(plan,key)
    frozen=json.loads((out/'model_freeze.json').read_text('utf-8'))
    if sha_file(out/'model.pt')!=frozen['model_sha256']:raise ValueError('Changed trained model')
    if key in ('review_12','reused_6') and not (out/'calibration/freeze.json').exists():raise ValueError('Calibrate thresholds first')
    torch.set_num_threads(4);checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    net=model(cfg).cuda().eval();net.load_state_dict(checkpoint['state']);folder=out/key;folder.mkdir(exist_ok=True)
    for i,r in enumerate(records,1):
        uid=r['series_uid'];dest=folder/(uid+'.json')
        if dest.exists():
            existing=json.loads(dest.read_text('utf-8'))
            if existing['model_sha256']!=frozen['model_sha256']:raise ValueError('Changed packet model')
            continue
        start=perf_counter();hu,geometry=load_development_case(split,r);body=load_mask(r,split)
        checked_inputs(hu,body,geometry);loaded=perf_counter();spacing=np.asarray(geometry['spacing_zyx_mm'])
        search=search_mask(body,spacing,cfg);jobs,stats=tiles(search,spacing,cfg);scheduled=perf_counter()
        all_points=[];all_scores=[];all_diameters=[];patch_ms=0.;network_ms=0.;decode_ms=0.
        with torch.inference_mode():
            for begin in range(0,len(jobs),48):
                batch_jobs=jobs[begin:begin+48];tick=perf_counter()
                image=np.stack([sample_patch(hu,spacing,j,cfg) for j in batch_jobs]);patch_ms+=(perf_counter()-tick)*1000
                tick=perf_counter();tensor=torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                prediction=net(tensor).cpu().numpy();network_ms+=(perf_counter()-tick)*1000
                tick=perf_counter();points,scores,diameters=decode(prediction,batch_jobs,geometry,cfg,search)
                all_points.append(points);all_scores.append(scores);all_diameters.append(diameters);decode_ms+=(perf_counter()-tick)*1000
        candidates=consolidate(np.concatenate(all_points) if all_points else [],np.concatenate(all_scores) if all_scores else [],
            np.concatenate(all_diameters) if all_diameters else [],geometry,cfg)
        for c in candidates:
            box=np.asarray(c['bbox_zyx']).reshape(3,2);box[:,0]=np.maximum(box[:,0],0);box[:,1]=np.minimum(box[:,1],hu.shape)
            c['bbox_zyx']=box.ravel().tolist()
        packet=dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_guided_25d_center_cnn_v1',
            backend='FP32_CUDA_REFERENCE_NOT_FPGA',status='success',series_uid=uid,input_raw_sha256=r['raw_sha256'],
            lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),model_sha256=frozen['model_sha256'],
            source_shape_zyx=list(hu.shape),geometry=geometry,candidates=candidates,schedule=stats,
            parameters=asdict(cfg),timings_ms=dict(load=(loaded-start)*1000,schedule=(scheduled-loaded)*1000,
                patch_sampling=patch_ms,network_and_transfer=network_ms,decode=decode_ms,
                total=(perf_counter()-start)*1000),
            scope='development; cached FP32 lung mask excluded from total time; not PL or whole-web latency')
        write_json(dest,packet)
        print('INFER',key,i,'/',len(records),'tiles',len(jobs),'candidates',len(candidates),
              'seconds',round(packet['timings_ms']['total']/1000,2),flush=True)
        del hu,body,search;gc.collect()
    verify(out)


def calibrate(out):
    from lung_candidate_froc import evaluate
    from lung_candidates import point_metrics
    plan=verify(out);split,records=records_for(plan,'calibration_scans');ignored=excluded_findings(EXCLUDED)
    folder=out/'calibration'
    if folder.exists():raise FileExistsError('Preserve first calibration')
    rows=[];scores={}
    for r in records:
        packet=json.loads((out/'calibration_scans'/(r['series_uid']+'.json')).read_text('utf-8'))
        if packet['input_raw_sha256']!=r['raw_sha256']:raise ValueError('Wrong case')
        world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
        rows.append(dict(uid=r['series_uid'],record=r,excluded=ignored.get(r['series_uid'],[]),world=world))
        scores[r['series_uid']]=np.array([c['score'] for c in packet['candidates']])
    froc=evaluate(rows,scores);thresholds={str(p['fp_budget']):p['threshold'] for p in froc['operating_points']}
    # Same fixed outer fold as SVM; its OOF scores are already saved.
    baseline=json.loads((EXPAND/'evaluation/report.json').read_text('utf-8'))
    from lung_candidate_expand_train import load_rows
    with np.load(EXPAND/'training/selected_oof_scores.npz',allow_pickle=False) as a:
        old_scores={r['series_uid']:a[r['series_uid']] for r in records}
    old_rows=load_rows(EXPAND,[r['series_uid'] for r in records]);old_froc=evaluate(old_rows,old_scores)
    folder.mkdir();write_json(folder/'report.json',dict(status='FIRST_CALIBRATION',cnn=froc,svm_same_scans=old_froc,
        interpretation='Development calibration fold, not final test. CNN fixed epoch and initial hyperparameters; SVM hyperparameters previously selected across training72.',
        heldout_scans_run=0,review_12_run=False))
    write_json(folder/'freeze.json',dict(model_sha256=sha_file(out/'model.pt'),thresholds=thresholds,
        report_sha256=sha_file(folder/'report.json'),scope='thresholds from calibration scans only; no review12 use'))
    print('CALIBRATION CPM',round(froc['cpm_interpolated'],4),'SVM same scans',round(old_froc['cpm_interpolated'],4),
          'sens@1/4/8',[(p['fp_budget'],round(p['sensitivity'],4)) for p in froc['operating_points'] if p['fp_budget'] in (1,4,8)],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','prepare','train','infer','calibrate'))
    parser.add_argument('--outdir',type=Path,default=ROOT/'data/lung_nodule_cnn_20261019_v1')
    parser.add_argument('--cohort',choices=('calibration_scans','review_12','reused_6'),default='calibration_scans')
    args=parser.parse_args();out=args.outdir.resolve()
    if args.command=='plan':freeze(out)
    elif args.command=='prepare':prepare(out)
    elif args.command=='train':train(out)
    elif args.command=='infer':infer(out,args.cohort)
    else:calibrate(out)


if __name__=='__main__':main()
