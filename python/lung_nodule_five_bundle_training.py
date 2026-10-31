"""Expand the unchanged 13k five-plane detector; refresh fit-only hard backgrounds.

Keeps original models/splits immutable, reuses old247 patches, and never exposes
sealed-test images. Added data, new teacher sampling and extra updates are a
combined development experiment, not a single-factor ablation.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from lung_candidate_data import sha_file,read_split
from lung_candidate_rules_run import ROOT,EXCLUDED,SPLIT
from lung_candidate_run import excluded_findings,predict_lungs
from lung_nodule_cnn import config_from_dict,model,search_mask,tiles,relative_labels,targets,loss
from lung_nodule_cnn_balanced import read,save,orient,balanced_weights
from lung_nodule_cnn_hardneg import setup_torch,background_mask
from lung_nodule_expanded_training import edge_details,label_group,requests_for,rng_for,atomic_torch_save,LUNG
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_whole_background import FloatPlaneCache
from luna16_multibundle import read_manifest,records_for,load_development_volume,OUT as OLD_DATA
from luna16_five_bundle import OUT as DATA

OUT=ROOT/'data/lung_nodule_five_bundle_20261021_v1'
OLD=ROOT/'data/lung_nodule_expanded_20261020_v1'
INITIAL=OLD/'model.pt'


def fit_records(manifest):
    records=records_for(manifest,'train')
    uids={r['series_uid'] for r in records}
    forbidden={r['series_uid'] for r in manifest['cases'] if r['role']!='train'}
    if len(uids)!=len(records) or uids&forbidden:raise ValueError('Training cohort overlap')
    return records


def diverse_hard(scores,points,count,distance=8.):
    """Select difficult *background* peaks at separated physical locations."""
    scores=np.asarray(scores,float);points=np.asarray(points,float).reshape(-1,3)
    if len(scores)!=len(points):raise ValueError('Peak scores/coordinates disagree')
    order=np.flatnonzero(np.isfinite(scores)&np.isfinite(points).all(1))
    order=order[np.argsort(-scores[order],kind='stable')]
    selected=[]
    for j in order:
        if not selected or (np.linalg.norm(points[selected]-points[j],axis=1)>=distance).all():
            selected.append(int(j))
            if len(selected)==count:break
    return selected


def freeze(out):
    if out.exists():raise FileExistsError('Preserve frozen experiment')
    manifest=read_manifest(DATA/'manifest.json');old=read_manifest(OLD_DATA/'manifest.json')
    records=fit_records(manifest);old_ids={r['series_uid'] for r in fit_records(old)}
    import torch
    cp=torch.load(INITIAL,map_location='cpu',weights_only=True)
    if set(cp['training_scans'])!=old_ids:raise ValueError('Teacher must be old247 training model')
    frozen=read(OLD/'model_freeze.json')
    if sha_file(INITIAL)!=frozen['model_sha256']:raise ValueError('Old model modified')
    paths=[INITIAL,OLD/'model_freeze.json',OLD/'training_catalogue.json',DATA/'manifest.json',LUNG,EXCLUDED,
        ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json',
        Path(__file__).resolve(),ROOT/'python/tests/test_lung_nodule_five_bundle_training.py']
    paths += [ROOT/'python'/n for n in ('luna16_five_bundle.py','luna16_multibundle.py','lung_nodule_cnn.py',
        'lung_nodule_cnn_balanced.py','lung_nodule_cnn_hardneg.py','lung_nodule_expanded_training.py',
        'lung_nodule_sampling_cache.py','lung_nodule_whole_background.py','lung_candidate_run.py','lung_unet.py',
        'lung_nodule_five_bundle_evaluation.py','lung_nodule_expanded_evaluation.py','lung_candidate_froc.py')]
    plan=dict(version='expanded_five_bundle_13k_v1',created_utc=datetime.now(timezone.utc).isoformat(),seed=2026102102,
        config=cp['config'],parameters=13157,initial_checkpoint=str(INITIAL),initial_sha256=sha_file(INITIAL),
        manifest=str(DATA/'manifest.json'),manifest_sha256=sha_file(DATA/'manifest.json'),
        training_scans=[r['series_uid'] for r in records],training_references=sum(len(r['nodules']) for r in records),
        reused_training_scans=sorted(old_ids),original_catalogue=str(OLD/'training_catalogue.json'),
        training=dict(epochs=30,batch_size=24,learning_rate=.0003,final_learning_rate=.00001,
            optimizer='Adam',checkpoint='fixed last epoch; no calibration checkpoint selection',positive_sampling_mass=.5,
            augmentation='XY right-angle rotations/reflections, Z reversal'),
        preparation=dict(sampled_mining_tiles=256,refresh_old_hard_patches=16,new_hard_patches=32,new_random_mined_patches=16,
            positive_crops_per_reference=16,random_base_crops_per_scan=24,minimum_peak_distance_mm=8.,
            teacher='frozen expanded247 five-plane model; background-only score mask excludes references and indeterminate findings'),
        inference_unchanged=True,ranker_refitted=False,architecture_unchanged=True,fpga_validated=False,
        calibration_scans_used_for_fit=0,sealed_test_scans_accessed=0,
        attribution_limit='Additional independent training scans + refreshed teacher/diverse background sampling + more updates; not pure data-only effect',
        protected_sha256={str(p):sha_file(p) for p in paths})
    out.mkdir(parents=True);(out/'sources').mkdir()
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n',encoding='utf-8')
    for p in paths:
        if p.suffix=='.py':(out/'sources'/p.name).write_bytes(p.read_bytes())
    print('FROZEN CNN',len(records),'train',plan['training_references'],'refs; old247 caches retained',flush=True)
    return plan,manifest


def verify(out):
    plan=read(out/'plan.json')
    if sha_file(out/'plan.json')!=(out/'plan.sha256').read_text().strip():raise ValueError('Plan modified')
    for p,h in plan['protected_sha256'].items():
        if sha_file(p)!=h:raise ValueError('Protected input changed: '+p)
    manifest=read_manifest(Path(plan['manifest']))
    if sha_file(plan['manifest'])!=plan['manifest_sha256'] or {r['series_uid'] for r in fit_records(manifest)}!=set(plan['training_scans']):
        raise ValueError('Training split changed')
    return plan,manifest


def load_body(r,hu,g,lung,torch,old_ids):
    uid=r['series_uid'];p=OLD/'patches'/(uid+'.mask.npz')
    if p.exists():
        info=read(p.with_name(uid+'.json'))
        if info['raw_sha256']!=r['raw_sha256'] or sha_file(p)!=info['mask_packed_sha256']:raise ValueError('Cached mask identity mismatch')
        with np.load(p,allow_pickle=False) as a:
            body=np.unpackbits(a['body'],count=int(np.prod(a['shape_zyx']))).reshape(a['shape_zyx']).astype(bool)
        if hashlib.sha256(body.tobytes()).hexdigest()!=info['lung_mask_sha256']:raise ValueError('Mask pixel mismatch')
        return body,'reused frozen expanded194 mask'
    if uid in old_ids:
        import lung_nodule_cnn_experiment as first
        return first.load_mask(r,read_split(SPLIT)),'reused original53 mask'
    raw,body,_=predict_lungs(hu,g,lung,'cuda',16);del raw
    return body,'frozen original U-Net; no lung retraining'


def prepare(out,limit=None):
    plan,manifest=verify(out);cfg=config_from_dict(plan['config']);torch=setup_torch(plan['seed'])
    from lung_unet import make_model
    cp=torch.load(LUNG,map_location='cpu',weights_only=True)
    lung=make_model(cp['base_channels'],cp.get('context_slices',1)).cuda().eval();lung.load_state_dict(cp['model'])
    teacher=model(cfg).cuda().eval();teacher.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True)['state'])
    oldcat=read(OLD/'training_catalogue.json');items=list(oldcat['items']);hashes=dict(oldcat['cache_sha256'])
    reused=set(plan['reused_training_scans']);records=fit_records(manifest)
    if limit is not None:records=records[:limit]
    ignored=excluded_findings(EXCLUDED);folder=out/'patches';folder.mkdir(exist_ok=True)
    summaries=[];identity=sha_file(out/'plan.json')
    with threadpool_limits(limits=1),torch.inference_mode():
        for count,r in enumerate(records,1):
            uid=r['series_uid'];path=folder/(uid+'.npy');meta_path=folder/(uid+'.json');is_old=uid in reused
            if meta_path.exists():
                meta=read(meta_path)
                if meta['plan_sha256']!=identity or meta['raw_sha256']!=r['raw_sha256'] or sha_file(path)!=meta['patch_sha256']:
                    raise ValueError('Changed completed training cache')
            else:
                tick=perf_counter();rng=rng_for(plan['seed'],uid)
                hu,g=load_development_volume(manifest,uid,'train')
                body,mask_source=load_body(r,hu,g,lung,torch,reused)
                spacing=np.asarray(g['spacing_zyx_mm']);search=search_mask(body,spacing,cfg)
                jobs,schedule=tiles(search,spacing,cfg);details=edge_details(body,r);cache=FloatPlaneCache(hu)
                requested=[] if is_old else requests_for(r,jobs,spacing,cfg,rng)
                base_labels=[];base_images=[]
                for job in requested:
                    pos,ex=relative_labels(r['nodules'],ignored.get(uid,[]),g,job,cfg)
                    base_labels.append(dict(job=list(job),positive=pos,ignored=ex,sampling_kind='base'))
                    base_images.append(sample_patch_reuse(cache,spacing,job,cfg))
                chosen=rng.choice(len(jobs),min(len(jobs),plan['preparation']['sampled_mining_tiles']),replace=False)
                mine_jobs=[jobs[int(j)] for j in chosen];mine_images=[];mine_labels=[]
                for job in mine_jobs:
                    pos,ex=relative_labels(r['nodules'],ignored.get(uid,[]),g,job,cfg)
                    mine_labels.append(dict(job=list(job),positive=pos,ignored=ex))
                    mine_images.append(sample_patch_reuse(cache,spacing,job,cfg))
                scores=np.full(len(chosen),-np.inf);points=np.full((len(chosen),3),np.nan)
                for begin in range(0,len(chosen),48):
                    x=torch.from_numpy(np.stack(mine_images[begin:begin+48])).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                    logits=teacher(x)[:,0].cpu().numpy()
                    for j,pred in enumerate(logits,begin):
                        available=background_mask(mine_labels[j],cfg)
                        if available.any():
                            iy,ix=np.unravel_index(np.where(available,pred,-np.inf).argmax(),pred.shape)
                            scores[j]=float(pred[iy,ix]);job=mine_jobs[j]
                            points[j]=[job[0],(job[1]+iy*cfg.output_stride)*cfg.xy_mm,(job[2]+ix*cfg.output_stride)*cfg.xy_mm]
                hard=diverse_hard(scores,points,plan['preparation']['refresh_old_hard_patches' if is_old else 'new_hard_patches'],
                                  plan['preparation']['minimum_peak_distance_mm'])
                remaining=np.setdiff1d(np.arange(len(chosen)),hard)
                random_ids=[] if is_old else rng.choice(remaining,min(len(remaining),16),replace=False).tolist()
                selected=hard+random_ids
                labels=base_labels+[dict(mine_labels[j],sampling_kind='diverse_hard' if j in hard else 'random_mined',
                    teacher_background_logit=float(scores[j]) if np.isfinite(scores[j]) else None,
                    mining_peak_mm=points[j].tolist() if np.isfinite(points[j]).all() else None) for j in selected]
                images=base_images+[mine_images[j] for j in selected]
                # An empty predicted ROI is not permission to drop an old scan.
                if not images:
                    job=(0.,0,0);pos,ex=relative_labels(r['nodules'],ignored.get(uid,[]),g,job,cfg)
                    labels=[dict(job=list(job),positive=pos,ignored=ex,sampling_kind='empty_roi_retained')]
                    images=[sample_patch_reuse(cache,spacing,job,cfg)]
                temp=path.with_suffix('.tmp.npy');np.save(temp,np.stack(images),allow_pickle=False);os.replace(temp,path)
                mask_path=folder/(uid+'.mask.npz')
                np.savez_compressed(mask_path,shape_zyx=np.array(body.shape),body=np.packbits(body))
                groups=[label_group(a,details,spacing,cfg) for a in labels]
                meta=dict(series_uid=uid,plan_sha256=identity,raw_sha256=r['raw_sha256'],patch_sha256=sha_file(path),
                    mask_packed_sha256=sha_file(mask_path),lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    count=len(labels),labels=labels,groups=groups,mask_source=mask_source,old_base_patches_reused=is_old,
                    hard_patches=len(hard),random_mined_patches=len(random_ids),sampled_mining_tiles=len(chosen),schedule=schedule,
                    teacher_sha256=plan['initial_sha256'],minimum_peak_distance_mm=8.,seconds=perf_counter()-tick)
                save(meta_path,meta);del images,mine_images,base_images,cache,hu,body,search;gc.collect()
            hashes[str(path)]=meta['patch_sha256'];hashes[str(meta_path)]=sha_file(meta_path)
            hashes[str(folder/(uid+'.mask.npz'))]=meta['mask_packed_sha256']
            items.extend(dict(series_uid=uid,image_path=str(path),patch_index=j,label=a,group=b)
                for j,(a,b) in enumerate(zip(meta['labels'],meta['groups'])))
            summaries.append(dict(series_uid=uid,added_patches=meta['count'],seconds=meta['seconds'],old=is_old))
            save(out/'preparation_progress.json',dict(status='RUNNING',completed=count,expected=len(records),
                old_training_scans_reused=len(reused),cases=summaries,sealed_test_scans_accessed=0))
            print('CNN PREP',count,'/',len(records),'old_refresh' if is_old else 'new_case',meta['count'],'patches',round(meta['seconds'],1),'s',flush=True)
    if limit is not None:return
    if {i['series_uid'] for i in items}!=set(plan['training_scans']):raise ValueError('Incomplete training catalogue')
    save(out/'training_catalogue.json',dict(plan_sha256=identity,items=items,cache_sha256=hashes,
        fit_scans=len(records),patches=len(items),groups=dict(Counter(i['group'] for i in items)),sealed_test_scans_accessed=0))
    save(out/'preparation_progress.json',dict(status='COMPLETE',completed=len(records),expected=len(records),
        patches=len(items),cases=summaries,sealed_test_scans_accessed=0))


def train(out):
    plan,_=verify(out);identity=sha_file(out/'plan.json');torch=setup_torch(plan['seed'])
    if (out/'model_freeze.json').exists():
        if sha_file(out/'model.pt')!=read(out/'model_freeze.json')['model_sha256']:raise ValueError('Completed model changed')
        return
    cat=read(out/'training_catalogue.json')
    if cat['plan_sha256']!=identity or {i['series_uid'] for i in cat['items']}!=set(plan['training_scans']):raise ValueError('Catalogue mismatch')
    for p,h in cat['cache_sha256'].items():
        if sha_file(p)!=h:raise ValueError('Changed patch data: '+p)
    cfg=config_from_dict(plan['config'])
    from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
    class Patches(Dataset):
        def __init__(self):self.maps={p:np.load(p,mmap_mode='r',allow_pickle=False) for p in cat['cache_sha256'] if p.endswith('.npy')}
        def __len__(self):return len(cat['items'])
        def __getitem__(self,index):
            item=cat['items'][index];a=item['label']
            x,pos,ex=orient(self.maps[item['image_path']][item['patch_index']],a['positive'],a['ignored'],
                rotation=random.randrange(4),flip_x=random.random()<.5,flip_y=random.random()<.5,flip_z=random.random()<.5)
            return torch.from_numpy(x),tuple(torch.from_numpy(t) for t in targets(pos,ex,cfg))
    data=Patches();generator=torch.Generator().manual_seed(plan['seed']);budget=plan['training']
    sampler=WeightedRandomSampler(balanced_weights([i['group'] for i in cat['items']]),len(data),replacement=True,generator=generator)
    loader=DataLoader(data,batch_size=budget['batch_size'],sampler=sampler,num_workers=0,pin_memory=True,
        generator=torch.Generator().manual_seed(plan['seed']))
    net=model(cfg).cuda();net.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True)['state'])
    if sum(p.numel() for p in net.parameters())!=13157:raise ValueError('Architecture changed')
    optimizer=torch.optim.Adam(net.parameters(),lr=budget['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=budget['epochs'],eta_min=budget['final_learning_rate'])
    scaler=torch.amp.GradScaler('cuda',init_scale=32.,growth_interval=2**31-1)
    history=[];updates=0
    if (out/'resume.pt').exists():
        cp=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        if cp['plan_sha256']!=identity or cp['catalogue_sha256']!=sha_file(out/'training_catalogue.json'):raise ValueError('Wrong resume identity')
        net.load_state_dict(cp['state']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
        history,updates=cp['history'],cp['updates'];random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng'])
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);generator.set_state(cp['sampler_rng']);loader.generator.set_state(cp['loader_rng'])
    with threadpool_limits(limits=1):
        for epoch in range(len(history),budget['epochs']):
            tick=perf_counter();totals=np.zeros(3);batches=0;net.train()
            for image,target in loader:
                x=image.cuda(non_blocking=True).float().clamp(-1000,400).add_(1000).div_(1400)
                target=tuple(t.cuda(non_blocking=True) for t in target);optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    value,classification,regression=loss(net(x).float(),target)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                scaler.scale(value).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()!=scale:raise FloatingPointError('AMP update skipped')
                updates+=1;batches+=1;totals+=[value.item(),classification.item(),regression.item()]
                if batches%200==0:
                    save(out/'training_progress.json',dict(status='RUNNING',epoch=epoch+1,epochs=history,batches=batches,
                        expected_batches=len(loader),optimizer_updates=updates,sealed_test_scans_accessed=0))
                    print('CNN BATCH',epoch+1,batches,'/',len(loader),flush=True)
            scheduler.step();history.append(dict(epoch=epoch+1,losses=(totals/batches).tolist(),seconds=perf_counter()-tick,batches=batches,cumulative_updates=updates))
            atomic_torch_save(torch,dict(plan_sha256=identity,catalogue_sha256=sha_file(out/'training_catalogue.json'),state=net.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,updates=updates,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                sampler_rng=generator.get_state(),loader_rng=loader.generator.get_state()),out/'resume.pt')
            save(out/'training_progress.json',dict(status='RUNNING',epochs=history,optimizer_updates=updates,
                expected_updates=len(loader)*budget['epochs'],sealed_test_scans_accessed=0))
            print('CNN EPOCH',epoch+1,'/',budget['epochs'],'seconds',round(history[-1]['seconds'],1),flush=True)
    if updates!=len(loader)*budget['epochs']:raise ValueError('Unexpected update count')
    net.cpu();atomic_torch_save(torch,dict(state=net.state_dict(),config=plan['config'],plan_sha256=identity,
        training_scans=plan['training_scans'],epochs=len(history),initialized_from_sha256=plan['initial_sha256'],
        variant='five_bundle_13k_diverse_background'),out/'model.pt')
    save(out/'model_freeze.json',dict(status='COMPLETE',model_sha256=sha_file(out/'model.pt'),plan_sha256=identity,
        catalogue_sha256=sha_file(out/'training_catalogue.json'),parameters=13157,epochs=history,optimizer_updates=updates,
        expected_updates=len(loader)*budget['epochs'],training_scans=len(plan['training_scans']),reference_nodules=plan['training_references'],
        amp_skipped_updates=0,sealed_test_scans_accessed=0,fpga_validated=False))
    save(out/'training_progress.json',dict(status='COMPLETE',epochs=history,optimizer_updates=updates,sealed_test_scans_accessed=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','prepare','train','run'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--limit',type=int);args=p.parse_args();out=args.out.resolve()
    if args.command=='freeze':freeze(out)
    elif args.command=='prepare':prepare(out,args.limit)
    elif args.command=='train':train(out)
    else:
        try:
            if not (out/'plan.json').exists():freeze(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='prepare',sealed_test_scans_accessed=0))
            if not (out/'training_catalogue.json').exists():prepare(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='train',sealed_test_scans_accessed=0))
            train(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='development_evaluation',sealed_test_scans_accessed=0))
            from lung_nodule_five_bundle_evaluation import run as evaluate
            evaluate(out)
            save(out/'pipeline.json',dict(status='COMPLETE',stage='complete',sealed_test_scans_accessed=0))
        except Exception as error:
            save(out/'pipeline.json',dict(status='FAILED',error=repr(error),sealed_test_scans_accessed=0));raise


if __name__=='__main__':main()
