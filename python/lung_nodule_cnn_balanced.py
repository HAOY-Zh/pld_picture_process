"""Frozen v3: matched-step control versus balanced, orientation-augmented fit.

Only training53 cached patches and predicted masks determine sample weights.
Both variants start from frozen v2, keep the same architecture and inference,
and finish a predeclared 30 epochs before looking at development19 results.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from lung_candidate_data import sha_file, write_json, load_development_case
from lung_candidate_rules_run import ROOT, EXCLUDED
from lung_candidate_run import excluded_findings
import lung_nodule_cnn_experiment as first
from lung_nodule_cnn_hardneg import BASE, DEFAULT as V2, setup_torch
from lung_nodule_cnn import (config_from_dict, model, targets, loss, search_mask,
                             tiles, sample_patch, decode, consolidate)

OUT = ROOT / 'data/lung_nodule_cnn_20261019_balanced_v3'
VARIANTS = ('continued_control', 'balanced_orientation')


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path, data):
    """Replace progress/report atomically; preserve completed frozen artifacts."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def orient(image, positive, ignored, rotation=0, flip_x=False, flip_y=False, flip_z=False):
    """Exact axis permutations only; transform labels with the same pixel map."""
    image = np.array(image, copy=True)
    positive = [list(p) for p in positive]
    ignored = [list(p) for p in ignored]
    n = image.shape[-1]
    if image.shape[-2] != n:
        raise ValueError('Square XY patches required')
    for _ in range(rotation % 4):
        image = np.rot90(image, axes=(1, 2))
        for p in positive + ignored:
            p[0], p[1] = p[1], n - 1 - p[0]
    for flag, axis, coord in ((flip_x, 2, 0), (flip_y, 1, 1)):
        if flag:
            image = np.flip(image, axis)
            for p in positive + ignored:
                p[coord] = n - 1 - p[coord]
    if flip_z:
        image = image[::-1]
        for p in positive:
            p[2] = -p[2]
    return np.ascontiguousarray(image), positive, ignored


def balanced_weights(groups, positive_fraction=.5):
    """Half mass to annotated patches, equal mass to nonempty positive strata."""
    groups = list(groups)
    positives = Counter(g for g in groups if g != 'background')
    backgrounds = groups.count('background')
    if not positives or not backgrounds or not 0 < positive_fraction < 1:
        raise ValueError('Need both reference and background patches')
    weights = np.array([(1-positive_fraction)/backgrounds if g == 'background' else
                        positive_fraction/len(positives)/positives[g] for g in groups], dtype=np.float64)
    return weights / weights.sum()


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve existing experiment')
    old = first.verify(V2)
    f = read(V2/'model_freeze.json')
    if sha_file(V2/'model.pt') != f['model_sha256']:
        raise ValueError('Changed v2 model')
    protected = dict(old['protected_sha256'])
    protected.update(read(V2/'mining_freeze.json')['cache_sha256'])
    extra = [V2/n for n in ('plan.json', 'plan.sha256', 'model.pt', 'model_freeze.json',
                            'mining_freeze.json', 'comparison.json', 'calibration/report.json')]
    extra += [Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_cnn_balanced.py',
              ROOT/'python/lung_nodule_cnn_experiment.py', ROOT/'python/lung_nodule_cnn_hardneg.py',
              ROOT/'python/lung_candidate_lowfp_audit.py']
    for p in extra:
        protected[str(p)] = sha_file(p)
    plan = {k: old[k] for k in ('config', 'fit_scans', 'calibration_scans', 'review_12', 'reused_6')}
    plan.update(version='matched_step_balanced_orientation_v3', seed=2026101903,
        variants=list(VARIANTS), protected_sha256=protected, initial_checkpoint=str(V2/'model.pt'),
        initial_checkpoint_sha256=f['model_sha256'],
        training=dict(epochs=30, batch_size=24, learning_rate=.0003, final_learning_rate=.00001,
            optimizer='Adam', patches_per_epoch=4840, checkpoint='fixed last epoch; no development checkpoint selection',
            control='uniform original v1 + v2 patch caches, original independent XY flips',
            treatment='same caches; positive probability .5; equal nonempty size/edge strata; XY rotations/flips and Z reversal',
            strata='diameter <6 / 6-to-<10 / >=10 mm crossed with predicted lung edge <=6 mm or interior',
            sampling_with_replacement=True, weighting_uses_fit_only=True, loss_unchanged=True),
        scope='Already-viewed development19 only; no review12/reused6 or sealed45',
        inference='same frozen sampling, tiling, geometry, decode and NMS; shared input batches for both variants',
        formal_deployment=False, fpga_validated=False, exact_paper_reproduction=False,
        architecture_unchanged=True, heldout_scans_run=0,
        attribution_limit='Treatment combines sampling and orientation augmentation; does not isolate their individual effects')
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in extra:
        if p.suffix == '.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN v3, two matched 30-epoch variants; unchanged 13157-parameter architecture', flush=True)


def verify(out):
    return first.verify(out)


def catalogue(out):
    plan = verify(out)
    cfg = config_from_dict(plan['config'])
    split, records = first.records_for(plan, 'fit_scans')
    edge_by_uid = {}
    for i, r in enumerate(records, 1):
        flags = []
        if r['nodules']:
            body = first.load_mask(r, split)
            spacing = np.asarray(r['spacing_zyx_mm'])
            cached_distance = {}
            for nodule in r['nodules']:
                q = np.rint(nodule['center_index_zyx']).astype(int)
                z, y, x = q
                if not ((q >= 0) & (q < body.shape)).all():
                    raise ValueError('Training reference outside native image')
                if z not in cached_distance:
                    filled = ndi.binary_fill_holes(body[z])
                    cached_distance[z] = ndi.distance_transform_edt(filled, sampling=spacing[1:])
                distance = float(cached_distance[z][y, x])
                flags.append(dict(diameter_mm=nodule['diameter_mm'], distance_to_predicted_lung_edge_mm=distance,
                                  edge=distance <= 6., center_index_zyx=nodule['center_index_zyx']))
            del body
        edge_by_uid[r['series_uid']] = flags
        print('CATALOGUE', i, '/', len(records), 'references', len(flags), flush=True)
    items, cache_paths = [], {}
    record_by = {r['series_uid']: r for r in records}
    for folder in (BASE/'patches', V2/'hard_patches'):
        for uid in plan['fit_scans']:
            metadata_path, image_path = folder/(uid+'.json'), folder/(uid+'.npy')
            meta = read(metadata_path)
            if sha_file(image_path) != meta['patch_sha256']:
                raise ValueError('Changed training cache')
            for j, label in enumerate(meta['labels']):
                positives = [p for p in label['positive'] if 0 <= round(p[0]/2) < 64 and 0 <= round(p[1]/2) < 64]
                group = 'background'
                if positives:
                    # Every positive patch of this cache has at most a few
                    # annotated centers. Use nearest annotation to its first
                    # valid center; all original labels stay in supervision.
                    p = positives[0]
                    job = label['job']
                    physical = np.array([job[0]+p[2]*cfg.z_step_mm,
                                         (job[1]+p[1])*cfg.xy_mm, (job[2]+p[0])*cfg.xy_mm])
                    spacing = np.asarray(record_by[uid]['spacing_zyx_mm'])
                    candidates = edge_by_uid[uid]
                    distances = [np.linalg.norm(np.asarray(n['center_index_zyx'])*spacing-physical) for n in candidates]
                    nearest = int(np.argmin(distances))
                    if distances[nearest] > 1e-4:
                        raise ValueError('Cached positive and training reference disagree')
                    n = candidates[nearest]
                    size = 'under6' if n['diameter_mm'] < 6 else '6to10' if n['diameter_mm'] < 10 else 'over10'
                    group = size + ('_edge' if n['edge'] else '_interior')
                items.append(dict(series_uid=uid, image_path=str(image_path), patch_index=j,
                                  label=label, group=group))
            cache_paths[str(image_path)] = sha_file(image_path)
            cache_paths[str(metadata_path)] = sha_file(metadata_path)
    groups = [r['group'] for r in items]
    weights = balanced_weights(groups)
    counts = Counter(groups)
    if len(items) != plan['training']['patches_per_epoch']:
        raise ValueError('Unexpected training sample count')
    save(out/'training_catalogue.json', dict(plan_sha256=sha_file(out/'plan.json'), items=items,
        reference_edge_details=edge_by_uid, groups=dict(counts),
        total_sampling_mass={g:float(weights[np.array(groups)==g].sum()) for g in counts},
        patches=len(items), cache_sha256=cache_paths, fit_scans=len(records),
        calibration_scans_used_for_weights=0, heldout_scans_run=0))


def train_variant(out, variant):
    plan = verify(out)
    cfg = config_from_dict(plan['config'])
    torch = setup_torch(plan['seed'])
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    cat = read(out/'training_catalogue.json')
    if cat['plan_sha256'] != sha_file(out/'plan.json'):
        raise ValueError('Wrong training catalogue')
    for p, digest in cat['cache_sha256'].items():
        if sha_file(p) != digest:
            raise ValueError('Changed training data')
    folder = out/variant
    folder.mkdir(exist_ok=True)
    if (folder/'model.pt').exists():
        raise FileExistsError('Preserve completed model')
    treatment = variant == 'balanced_orientation'

    class Patches(Dataset):
        def __init__(self):
            self.maps = {p:np.load(p, mmap_mode='r', allow_pickle=False) for p in cat['cache_sha256'] if p.endswith('.npy')}

        def __len__(self):
            return len(cat['items'])

        def __getitem__(self, index):
            item = cat['items'][index]
            label = item['label']
            image = self.maps[item['image_path']][item['patch_index']]
            image, positive, ignored = orient(image, label['positive'], label['ignored'],
                rotation=random.randrange(4) if treatment else 0,
                flip_x=random.random()<.5, flip_y=random.random()<.5,
                flip_z=random.random()<.5 if treatment else False)
            return torch.from_numpy(image), tuple(torch.from_numpy(t) for t in targets(positive, ignored, cfg))

    data = Patches()
    generator = torch.Generator().manual_seed(plan['seed'])
    sampler = WeightedRandomSampler(balanced_weights([i['group'] for i in cat['items']]),
        num_samples=len(data), replacement=True, generator=generator) if treatment else None
    loader = DataLoader(data, batch_size=plan['training']['batch_size'], sampler=sampler,
        shuffle=sampler is None, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(plan['seed']))
    net = model(cfg).cuda()
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    if set(initial['training_scans']) != set(plan['fit_scans']):
        raise ValueError('Initial model cohort mismatch')
    net.load_state_dict(initial['state'])
    optimizer = torch.optim.Adam(net.parameters(), lr=plan['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=plan['training']['epochs'], eta_min=plan['training']['final_learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    resume = folder/'resume.pt'
    history = []
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_sha256'] != sha_file(out/'plan.json') or state['variant'] != variant:
            raise ValueError('Wrong training resume identity')
        net.load_state_dict(state['state'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history = state['history']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['sampler_rng'])
        loader.generator.set_state(state['loader_rng'])
        print('RESUMED', variant, 'after epoch', len(history), flush=True)
    start = perf_counter()
    for epoch in range(len(history), plan['training']['epochs']):
        tick = perf_counter()
        net.train()
        totals = np.zeros(3)
        batches = 0
        for image, target in loader:
            image = image.cuda(non_blocking=True).float().clamp(-1000,400).add_(1000).div_(1400)
            target = tuple(t.cuda(non_blocking=True) for t in target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                value, classification, regression = loss(net(image).float(), target)
            if not torch.isfinite(value):
                raise ValueError('Nonfinite loss')
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            scaler.step(optimizer)
            scaler.update()
            totals += [value.item(), classification.item(), regression.item()]
            batches += 1
        history.append(dict(epoch=epoch+1, loss=(totals/batches).tolist(),
                            learning_rate=optimizer.param_groups[0]['lr'], seconds=perf_counter()-tick))
        scheduler.step()
        temporary = folder/'resume.tmp'
        torch.save(dict(plan_sha256=sha_file(out/'plan.json'), variant=variant, state=net.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            sampler_rng=generator.get_state(), loader_rng=loader.generator.get_state()), temporary)
        os.replace(temporary, resume)
        save(folder/'training_progress.json', dict(status='RUNNING', variant=variant, epochs=history))
        print('TRAIN', variant, epoch+1, '/', plan['training']['epochs'],
              'loss', round(history[-1]['loss'][0],4), 'seconds', round(history[-1]['seconds'],2), flush=True)
    net.cpu()
    temporary = folder/'model.tmp'
    torch.save(dict(state=net.state_dict(), config=plan['config'], plan_sha256=sha_file(out/'plan.json'),
        epochs=len(history), training_scans=plan['fit_scans'], variant=variant,
        initialized_from_sha256=plan['initial_checkpoint_sha256']), temporary)
    os.replace(temporary, folder/'model.pt')
    save(folder/'model_freeze.json', dict(model_sha256=sha_file(folder/'model.pt'),
        plan_sha256=sha_file(out/'plan.json'), catalogue_sha256=sha_file(out/'training_catalogue.json'),
        epochs=history, fit_scans=len(plan['fit_scans']), training_patches=len(data),
        training_batches=len(loader)*len(history), parameters=sum(p.numel() for p in net.parameters()),
        current_session_seconds=perf_counter()-start, variant=variant,
        heldout_scans_run=0, development_evaluation_not_started=True, fpga_validated=False))
    save(folder/'training_progress.json', dict(status='COMPLETE', variant=variant, epochs=history))


def infer_pair(out):
    plan = verify(out)
    torch = setup_torch(plan['seed'])
    cfg = config_from_dict(plan['config'])
    split, records = first.records_for(plan, 'calibration_scans')
    from threadpoolctl import threadpool_limits
    from lung_candidates import checked_inputs
    nets, freezes = {}, {}
    for variant in VARIANTS:
        folder = out/variant
        f = read(folder/'model_freeze.json')
        if sha_file(folder/'model.pt') != f['model_sha256']:
            raise ValueError('Changed model')
        checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
        nets[variant] = model(cfg).cuda().eval()
        nets[variant].load_state_dict(checkpoint['state'])
        freezes[variant] = f
        (folder/'calibration_scans').mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(records,1):
            uid = r['series_uid']
            pending = []
            for variant in VARIANTS:
                dest = out/variant/'calibration_scans'/(uid+'.json')
                if dest.exists():
                    packet = read(dest)
                    if packet['model_sha256'] != freezes[variant]['model_sha256'] or packet['plan_sha256'] != identity:
                        raise ValueError('Changed completed inference identity')
                else:
                    pending.append(variant)
            if not pending:
                continue
            tick = perf_counter()
            hu, geometry = load_development_case(split,r)
            body = first.load_mask(r,split)
            checked_inputs(hu,body,geometry)
            spacing = np.asarray(geometry['spacing_zyx_mm'])
            search = search_mask(body,spacing,cfg)
            jobs, schedule = tiles(search,spacing,cfg)
            result = {v:[[],[],[]] for v in pending}
            for begin in range(0,len(jobs),48):
                group = jobs[begin:begin+48]
                image = np.stack([sample_patch(hu,spacing,j,cfg) for j in group])
                tensor = torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                for v in pending:
                    prediction = nets[v](tensor).cpu().numpy()
                    for store, values in zip(result[v], decode(prediction,group,geometry,cfg,search)):
                        store.append(values)
            totals = {}
            for v in pending:
                values = [np.concatenate(a) for a in result[v]]
                candidates = consolidate(*values,geometry,cfg)
                for c in candidates:
                    box = np.asarray(c['bbox_zyx']).reshape(3,2)
                    box[:,0] = np.maximum(box[:,0],0)
                    box[:,1] = np.minimum(box[:,1],hu.shape)
                    c['bbox_zyx'] = box.ravel().tolist()
                packet = dict(schema_version='lung_candidate_result_v1', algorithm_id='lung_guided_25d_center_cnn_v3_'+v,
                    backend='FP32_CUDA_REFERENCE_NOT_FPGA', status='success', series_uid=uid,
                    input_raw_sha256=r['raw_sha256'], lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    model_sha256=freezes[v]['model_sha256'], plan_sha256=identity,
                    source_shape_zyx=list(hu.shape), geometry=geometry, parameters=plan['config'], schedule=schedule,
                    candidates=candidates, shared_pair_seconds=perf_counter()-tick,
                    scope='Development comparison with shared preprocessing; not per-model or FPGA latency')
                save(out/v/'calibration_scans'/(uid+'.json'),packet)
                totals[v] = len(candidates)
            print('INFER_PAIR',i,'/',len(records),'tiles',len(jobs),'candidates',totals,
                  'shared_seconds',round(perf_counter()-tick,2),flush=True)
            del hu,body,search,tensor,result
            gc.collect()


def summarize(out):
    plan = verify(out)
    from lung_candidate_froc import evaluate
    from lung_candidate_lowfp_audit import check_froc
    from lung_candidates import point_metrics
    split, records = first.records_for(plan,'calibration_scans')
    ignored = excluded_findings(EXCLUDED)
    curves = {}
    cases = {}
    for variant in VARIANTS:
        folder = out/variant
        if (folder/'calibration.json').exists():
            raise FileExistsError('Preserve first development calibration')
        rows, scores, packets = [], {}, {}
        f = read(folder/'model_freeze.json')
        import torch
        checkpoint = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        if (sha_file(folder/'model.pt') != f['model_sha256'] or
                set(checkpoint['training_scans']) != set(plan['fit_scans']) or
                checkpoint['plan_sha256'] != sha_file(out/'plan.json') or
                set(plan['fit_scans']) & set(plan['calibration_scans'])):
            raise ValueError('Wrong model/split identity')
        if not all(torch.isfinite(v).all() for v in checkpoint['state'].values()):
            raise ValueError('Nonfinite model weights')
        for r in records:
            uid=r['series_uid']
            packet=read(folder/'calibration_scans'/(uid+'.json'))
            if packet['input_raw_sha256']!=r['raw_sha256'] or packet['model_sha256']!=f['model_sha256']:
                raise ValueError('Wrong inference identity')
            world=np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
            index=np.asarray([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1,3)
            np.testing.assert_allclose(world,index@np.asarray(r['basis_lps_zyx']).T+r['origin_lps_mm'],atol=1e-6,rtol=0)
            rows.append(dict(uid=uid,record=r,excluded=ignored.get(uid,[]),world=world))
            scores[uid]=np.asarray([c['score'] for c in packet['candidates']])
            packets[uid]=packet
        curve=evaluate(rows,scores)
        check_froc(rows,scores,curve)
        threshold=next(p['threshold'] for p in curve['operating_points'] if p['fp_budget']==4.)
        cases[variant]=[]
        for row in rows:
            selected=row['world'][scores[row['uid']]>=threshold] if threshold is not None else np.empty((0,3))
            m=point_metrics(selected,row['record']['nodules'],row['excluded'])
            cases[variant].append(dict(series_uid=row['uid'],metric=m))
        save(folder/'calibration.json',dict(cnn=curve,model_sha256=f['model_sha256'],
            independent_froc_verified=True,geometry_verified=True,heldout_scans_run=0))
        curves[variant]=curve
    previous=read(V2/'comparison.json')
    v2metrics={c['series_uid']:c['v2'] for c in previous['cases']}
    changes={}
    for v in VARIANTS:
        gain=lost=0
        for c in cases[v]:
            for a,b in zip(c['metric']['reference_details'],v2metrics[c['series_uid']]['reference_details']):
                gain+=int(a['detected'] and not b['detected'])
                lost+=int(b['detected'] and not a['detected'])
        changes[v]=dict(gained_at4=gain,lost_at4=lost)
    save(out/'comparison.json',dict(status='COMPLETE_DEVELOPMENT_COMPARISON_NOT_PROMOTED',
        variants=curves,v1=previous['v1'],v2=previous['v2'],svm_same_scans=previous['svm_same_scans'],
        changes_vs_v2=changes,cases=cases,scans=len(records),reference_nodules=sum(len(r['nodules']) for r in records),
        independent_froc_verified=True,geometry_verified=True,protected_files_unchanged=len(plan['protected_sha256']),
        architecture_unchanged=True,matched_extra_epochs=30,heldout_scans_run=0,review12_run=False,
        timing_scope='Shared pair evaluation; no per-model speed claim',fpga_validated=False))
    for v in VARIANTS:
        print('RESULT',v,'CPM',curves[v]['cpm_interpolated'],
              '1/4/8',[(p['fp_budget'],p['detected']) for p in curves[v]['operating_points'] if p['fp_budget'] in (1.,4.,8.)],
              changes[v],flush=True)


def run(out):
    if not out.exists():
        freeze(out)
    stages=[('catalogue','training_catalogue.json',catalogue)]
    stages += [(v,v+'/model_freeze.json',lambda p,v=v:train_variant(p,v)) for v in VARIANTS]
    stages += [('infer',None,infer_pair),('summarize','comparison.json',summarize)]
    stage='verify'
    try:
        verify(out)
        for stage, marker, operation in stages:
            if marker and (out/marker).exists():
                continue
            save(out/'pipeline_state.json',dict(status='RUNNING',stage=stage))
            operation(out)
        verify(out)
        save(out/'pipeline_state.json',dict(status='COMPLETE_DEVELOPMENT_COMPARISON',heldout_scans_run=0))
    except Exception as error:
        save(out/'pipeline_state.json',dict(status='FAILED',stage=stage,error=repr(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    args=parser.parse_args()
    run(args.outdir.resolve())


if __name__=='__main__':
    main()
