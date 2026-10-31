"""Resumable fit-only training on the frozen 247-scan expansion.

No inference/calibration/test command is exposed here. Old caches are reused;
new scans get the same positive, random and sampled hard-background recipe.
"""
import argparse
from collections import Counter
import gc
import hashlib
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

from lung_candidate_data import sha_file
from lung_candidate_rules_run import ROOT, EXCLUDED
from lung_candidate_run import excluded_findings, predict_lungs
from lung_nodule_cnn import config_from_dict, model, search_mask, tiles, relative_labels, targets, loss
from lung_nodule_cnn_balanced import read, save, orient, balanced_weights
from lung_nodule_cnn_hardneg import setup_torch, background_mask, select_patches
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_whole_background import FloatPlaneCache
from luna16_multibundle import OUT as DATA, read_manifest, records_for, load_development_volume

OUT = ROOT/'data/lung_nodule_expanded_20261020_v1'
INITIAL = ROOT/'data/lung_nodule_capacity_20261020_amp_safe_v2/small_continued/model.pt'
LUNG = ROOT/'data/lung_training_20261011/early_stop_run/best.pt'
OLD_CAT = ROOT/'data/lung_nodule_cnn_20261019_balanced_v3/training_catalogue.json'


def fit_records(manifest):
    """This module accepts only training records, never metadata-selected test IDs."""
    records = records_for(manifest, 'train')
    if len({r['series_uid'] for r in records}) != len(records):
        raise ValueError('Duplicate training UID')
    return records


def rng_for(seed, uid):
    return np.random.default_rng(int(hashlib.sha256((str(seed)+uid).encode()).hexdigest()[:16], 16))


def edge_details(body, record):
    cached, result = {}, []
    spacing = np.asarray(record['spacing_zyx_mm'])
    for n in record['nodules']:
        q = np.rint(n['center_index_zyx']).astype(int)
        if not ((q >= 0) & (q < body.shape)).all():
            raise ValueError('Training reference outside image')
        z, y, x = q
        if z not in cached:
            cached[z] = ndi.distance_transform_edt(ndi.binary_fill_holes(body[z]), sampling=spacing[1:])
        distance = float(cached[z][y, x])
        result.append(dict(n, edge=distance <= 6., distance_to_predicted_lung_edge_mm=distance))
    return result


def label_group(label, details, spacing, cfg):
    positive = [p for p in label['positive'] if 0 <= round(p[0]/2) < 64 and 0 <= round(p[1]/2) < 64]
    if not positive:
        return 'background'
    p, job = positive[0], label['job']
    physical = np.array([job[0]+p[2]*cfg.z_step_mm,
                         (job[1]+p[1])*cfg.xy_mm, (job[2]+p[0])*cfg.xy_mm])
    distances = [np.linalg.norm(np.array(n['center_index_zyx'])*spacing-physical) for n in details]
    if not distances or min(distances) > 1e-4:
        raise ValueError('Positive patch and reference disagree')
    n = details[int(np.argmin(distances))]
    size = 'under6' if n['diameter_mm'] < 6 else '6to10' if n['diameter_mm'] < 10 else 'over10'
    return size + ('_edge' if n['edge'] else '_interior')


def requests_for(record, jobs, spacing, cfg, rng):
    result = []
    for n in record['nodules']:
        p = np.array(n['center_index_zyx'])*spacing
        zm = round(p[0]/cfg.z_step_mm)*cfg.z_step_mm
        for _ in range(16):
            xy = p[1:]/cfg.xy_mm-cfg.patch/2+rng.uniform(-28, 28, 2)
            result.append((float(zm), int(round(xy[0])), int(round(xy[1]))))
    if jobs:
        result.extend(jobs[int(j)] for j in rng.choice(len(jobs), 24, replace=len(jobs) < 24))
    else:
        # Empty predicted ROI is retained, including reference-positive scans.
        # A single air-padded origin patch supplies background supervision.
        result.append((0., 0, 0))
    return result


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve frozen experiment; use run to resume')
    manifest = read_manifest(DATA/'manifest.json')
    records = fit_records(manifest)
    import torch
    cp = torch.load(INITIAL, map_location='cpu', weights_only=True)
    uids = [r['series_uid'] for r in records]
    cat = read(OLD_CAT)
    old_uids = {i['series_uid'] for i in cat['items']}
    if set(cp['training_scans']) != old_uids or not old_uids < set(uids):
        raise ValueError('Original teacher/cache cohort is not contained in expanded training')
    protected = dict(cat['cache_sha256'])
    sources = [Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_expanded_training.py']
    sources += [ROOT/'python'/n for n in ('lung_nodule_cnn.py', 'lung_nodule_cnn_balanced.py',
        'lung_nodule_cnn_hardneg.py', 'lung_nodule_sampling_cache.py', 'lung_nodule_whole_background.py',
        'lung_candidate_run.py', 'lung_body_gate.py', 'lung_unet.py', 'luna16_multibundle.py')]
    for p in [INITIAL, LUNG, OLD_CAT, EXCLUDED, DATA/'manifest.json', *sources]:
        protected[str(p)] = sha_file(p)
    plan = dict(version='expanded_fit247_small13k_v1', seed=2026102005,
        manifest=str(DATA/'manifest.json'), manifest_sha256=sha_file(DATA/'manifest.json'),
        config=cp['config'], initial_checkpoint=str(INITIAL), initial_sha256=sha_file(INITIAL),
        lung_checkpoint=str(LUNG), lung_sha256=sha_file(LUNG),
        training_scans=uids, training_references=sum(len(r['nodules']) for r in records),
        reused_training_scans=sorted(old_uids), protected_sha256=protected,
        preparation=dict(positive_crops_per_reference=16, random_crops_per_scan=24,
            sampled_hard_mining_tiles=256, hard_patches_per_scan=32, random_mined_patches_per_scan=16,
            teacher='frozen original 13k continued model; no new network is deployed',
            original53='reuse exact v1 base and v2 hard caches; their historical teacher differs',
            empty_roi='keep scan; one fixed origin background patch plus positive crops'),
        training=dict(epochs=30, batch_size=24, learning_rate=.0003, final_learning_rate=.00001,
            positive_sampling_mass=.5, positive_strata='size <6, 6-to-<10, >=10 mm crossed with predicted edge <=6 mm',
            optimizer='Adam', augmentation='XY right-angle rotations/flips; Z channel reversal',
            amp_initial_scale=32., amp_growth_interval=2**31-1,
            checkpoint='fixed final epoch; no development checkpoint selection'),
        inference_unchanged=True, parameters=13157, fpga_validated=False,
        sealed_test_scans_accessed=0, calibration_scans_used_for_fit=0,
        attribution_limit='Expanded-data continuation with extra optimization; not a pure data-only causal ablation',
        post_training='separate development calibration; new56 remains sealed')
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in sources:
        (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN train247 / refs341 / epochs30 / params13157 / new56 SEALED', flush=True)


def verify(out):
    plan = read(out/'plan.json')
    if sha_file(out/'plan.json') != (out/'plan.sha256').read_text().strip():
        raise ValueError('Changed training plan')
    manifest = read_manifest(Path(plan['manifest']))
    if sha_file(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Changed manifest identity')
    if set(plan['training_scans']) != {r['series_uid'] for r in fit_records(manifest)}:
        raise ValueError('Training UID role mismatch')
    for p, digest in plan['protected_sha256'].items():
        if sha_file(p) != digest:
            raise ValueError('Changed protected training input: '+p)
    return plan, manifest


def prepare(out):
    plan, manifest = verify(out)
    cfg = config_from_dict(plan['config'])
    torch = setup_torch(plan['seed'])
    from lung_unet import make_model
    cp = torch.load(LUNG, map_location='cpu', weights_only=True)
    lung = make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model'])
    teacher = model(cfg).cuda().eval()
    teacher.load_state_dict(torch.load(INITIAL, map_location='cpu', weights_only=True)['state'])
    old_cat = read(OLD_CAT)
    items, hashes = list(old_cat['items']), dict(old_cat['cache_sha256'])
    records = fit_records(manifest)
    ignored = excluded_findings(EXCLUDED)
    new_records = [r for r in records if r['series_uid'] not in plan['reused_training_scans']]
    folder = out/'patches'
    folder.mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    summaries = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(new_records, 1):
            uid = r['series_uid']
            image_path, meta_path = folder/(uid+'.npy'), folder/(uid+'.json')
            if meta_path.exists():
                meta = read(meta_path)
                if (meta['plan_sha256'] != identity or meta['raw_sha256'] != r['raw_sha256'] or
                        sha_file(image_path) != meta['patch_sha256'] or
                        sha_file(folder/(uid+'.mask.npz')) != meta['mask_packed_sha256']):
                    raise ValueError('Changed completed preparation')
            else:
                tick = perf_counter()
                rng = rng_for(plan['seed'], uid)
                hu, geometry = load_development_volume(manifest, uid, 'train')
                print('LOAD_TRAIN', i, '/', len(new_records), uid, 'shape', hu.shape, flush=True)
                raw, body, lung_ms = predict_lungs(hu, geometry, lung, 'cuda', 16)
                del raw
                spacing = np.asarray(geometry['spacing_zyx_mm'])
                search = search_mask(body, spacing, cfg)
                jobs, schedule = tiles(search, spacing, cfg)
                details = edge_details(body, r)
                cache = FloatPlaneCache(hu)
                base_jobs = requests_for(r, jobs, spacing, cfg, rng)
                base_labels = []
                base_images = []
                for job in base_jobs:
                    p, e = relative_labels(r['nodules'], ignored.get(uid, []), geometry, job, cfg)
                    base_labels.append(dict(job=list(job), positive=p, ignored=e, sampling_kind='base'))
                    base_images.append(sample_patch_reuse(cache, spacing, job, cfg))
                chosen = rng.choice(len(jobs), min(len(jobs), 256), replace=False)
                mine_jobs = [jobs[int(j)] for j in chosen]
                mine_labels, mine_images = [], []
                for job in mine_jobs:
                    p, e = relative_labels(r['nodules'], ignored.get(uid, []), geometry, job, cfg)
                    mine_labels.append(dict(job=list(job), positive=p, ignored=e))
                    mine_images.append(sample_patch_reuse(cache, spacing, job, cfg))
                scores = np.full(len(mine_jobs), -np.inf)
                for begin in range(0, len(mine_jobs), 48):
                    tensor = torch.from_numpy(np.stack(mine_images[begin:begin+48])).cuda().float()
                    tensor = tensor.clamp(-1000, 400).add_(1000).div_(1400)
                    probabilities = teacher(tensor)[:, 0].sigmoid().cpu().numpy()
                    for j, probability in enumerate(probabilities, begin):
                        available = background_mask(mine_labels[j], cfg)
                        if available.any():
                            scores[j] = float(probability[available].max())
                hard, random_ids = select_patches(scores, 32, 16, rng)
                selected = hard + random_ids
                labels = base_labels + [dict(mine_labels[j],
                    mining_score=float(scores[j]) if np.isfinite(scores[j]) else None,
                    sampling_kind='hard' if j in hard else 'random_mined') for j in selected]
                groups = [label_group(label, details, spacing, cfg) for label in labels]
                temp = image_path.with_suffix('.tmp.npy')
                np.save(temp, np.stack(base_images+[mine_images[j] for j in selected]), allow_pickle=False)
                os.replace(temp, image_path)
                packed_path = folder/(uid+'.mask.npz')
                np.savez_compressed(packed_path, shape_zyx=np.array(body.shape), body=np.packbits(body))
                coverage = [bool(search[tuple(np.rint(n['center_index_zyx']).astype(int))]) for n in r['nodules']]
                meta = dict(series_uid=uid, plan_sha256=identity, raw_sha256=r['raw_sha256'],
                    patch_sha256=sha_file(image_path), mask_packed_sha256=sha_file(packed_path),
                    lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(), labels=labels, groups=groups,
                    reference_edge_details=details, count=len(labels), schedule=schedule,
                    reference_centers_in_search=coverage, lung_ms=lung_ms,
                    sampled_mining_tiles=len(mine_jobs), hard_patches=len(hard), random_mined_patches=len(random_ids),
                    seconds=perf_counter()-tick)
                save(meta_path, meta)
                del hu, body, search, cache, base_images, mine_images
                gc.collect()
            summaries.append(dict(series_uid=uid, count=meta['count'], seconds=meta['seconds']))
            items.extend(dict(series_uid=uid, image_path=str(image_path), patch_index=j, label=label, group=group)
                         for j, (label, group) in enumerate(zip(meta['labels'], meta['groups'])))
            hashes[str(image_path)] = meta['patch_sha256']
            hashes[str(meta_path)] = sha_file(meta_path)
            save(out/'preparation_progress.json', dict(status='RUNNING', new_completed=i, new_total=len(new_records),
                original_cases_reused=len(plan['reused_training_scans']), training_cases_ready=53+i,
                total_training_cases=len(records), cases=summaries, sealed_test_scans_accessed=0))
            print('PREPARED', i, '/', len(new_records), 'patches', meta['count'], 'seconds', round(meta['seconds'], 2), flush=True)
    if {i['series_uid'] for i in items} != set(plan['training_scans']):
        raise ValueError('Incomplete training catalogue')
    groups = [i['group'] for i in items]
    weights = balanced_weights(groups)
    save(out/'training_catalogue.json', dict(plan_sha256=identity, items=items, cache_sha256=hashes,
        patches=len(items), fit_scans=len(records), groups=dict(Counter(groups)),
        total_sampling_mass={g:float(weights[np.array(groups) == g].sum()) for g in set(groups)},
        sealed_test_scans_accessed=0, calibration_scans_used=0))
    save(out/'preparation_progress.json', dict(status='COMPLETE', new_completed=len(new_records),
        training_cases_ready=len(records), original_cases_reused=53, patches=len(items), cases=summaries,
        sealed_test_scans_accessed=0))
    print('ALL TRAINING PATCHES READY', len(items), flush=True)


def atomic_torch_save(torch, obj, path):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def train(out):
    plan, _ = verify(out)
    identity = sha_file(out/'plan.json')
    if (out/'model.pt').exists():
        frozen = read(out/'model_freeze.json')
        if frozen['model_sha256'] != sha_file(out/'model.pt') or frozen['plan_sha256'] != identity:
            raise ValueError('Changed completed model')
        print('Training already complete; preserving model', flush=True)
        return
    cat = read(out/'training_catalogue.json')
    if cat['plan_sha256'] != identity or {i['series_uid'] for i in cat['items']} != set(plan['training_scans']):
        raise ValueError('Wrong training catalogue')
    for p, digest in cat['cache_sha256'].items():
        if sha_file(p) != digest:
            raise ValueError('Changed patch cache: '+p)
    torch = setup_torch(plan['seed'])
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    cfg = config_from_dict(plan['config'])

    class Patches(Dataset):
        def __init__(self):
            self.maps = {p:np.load(p, mmap_mode='r', allow_pickle=False) for p in cat['cache_sha256'] if p.endswith('.npy')}
        def __len__(self):
            return len(cat['items'])
        def __getitem__(self, index):
            item = cat['items'][index]
            label = item['label']
            image, pos, ignored = orient(self.maps[item['image_path']][item['patch_index']],
                label['positive'], label['ignored'], rotation=random.randrange(4),
                flip_x=random.random() < .5, flip_y=random.random() < .5, flip_z=random.random() < .5)
            return torch.from_numpy(image), tuple(torch.from_numpy(a) for a in targets(pos, ignored, cfg))

    data = Patches()
    generator = torch.Generator().manual_seed(plan['seed'])
    sampler = WeightedRandomSampler(balanced_weights([i['group'] for i in cat['items']]),
        len(data), replacement=True, generator=generator)
    loader = DataLoader(data, batch_size=24, sampler=sampler, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(plan['seed']))
    net = model(cfg).cuda()
    net.load_state_dict(torch.load(INITIAL, map_location='cpu', weights_only=True)['state'])
    if sum(p.numel() for p in net.parameters()) != plan['parameters']:
        raise ValueError('Architecture changed')
    optimizer = torch.optim.Adam(net.parameters(), lr=plan['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=.00001)
    scaler = torch.amp.GradScaler('cuda', init_scale=32., growth_interval=2**31-1)
    history, updates = [], 0
    resume = out/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_sha256'] != identity or state['catalogue_sha256'] != sha_file(out/'training_catalogue.json'):
            raise ValueError('Wrong resume identity')
        net.load_state_dict(state['state'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, updates = state['history'], state['updates']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['sampler_rng'])
        loader.generator.set_state(state['loader_rng'])
        print('RESUMED after epoch', len(history), flush=True)
    with threadpool_limits(limits=1):
        for epoch in range(len(history), plan['training']['epochs']):
            tick, batches = perf_counter(), 0
            totals = np.zeros(3)
            net.train()
            for image, target in loader:
                image = image.cuda(non_blocking=True).float().clamp(-1000, 400).add_(1000).div_(1400)
                target = tuple(t.cuda(non_blocking=True) for t in target)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    value, classification, regression = loss(net(image).float(), target)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite loss; resume from last complete epoch')
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradients; no silent AMP update skip')
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() != before:
                    raise ValueError('AMP scale changed; no silent update skipping allowed')
                updates += 1
                batches += 1
                totals += [value.item(), classification.item(), regression.item()]
                if batches % 200 == 0:
                    print('BATCH', epoch+1, batches, '/', len(loader), flush=True)
            scheduler.step()
            history.append(dict(epoch=epoch+1, losses=(totals/batches).tolist(),
                seconds=perf_counter()-tick, batches=batches, cumulative_updates=updates,
                next_learning_rate=optimizer.param_groups[0]['lr']))
            atomic_torch_save(torch, dict(plan_sha256=identity, catalogue_sha256=sha_file(out/'training_catalogue.json'),
                state=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), history=history, updates=updates,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), sampler_rng=generator.get_state(),
                loader_rng=loader.generator.get_state()), resume)
            save(out/'training_progress.json', dict(status='RUNNING', epochs=history,
                optimizer_updates=updates, expected_updates=math.ceil(len(data)/24)*30, sealed_test_scans_accessed=0))
            print('TRAIN', epoch+1, '/30', 'loss', round(history[-1]['losses'][0], 4),
                'seconds', round(history[-1]['seconds'], 1), flush=True)
    net.cpu()
    atomic_torch_save(torch, dict(state=net.state_dict(), config=plan['config'], plan_sha256=identity,
        training_scans=plan['training_scans'], epochs=len(history), initialized_from_sha256=plan['initial_sha256'],
        variant='expanded_fit247_small13k'), out/'model.pt')
    save(out/'model_freeze.json', dict(model_sha256=sha_file(out/'model.pt'), plan_sha256=identity,
        catalogue_sha256=sha_file(out/'training_catalogue.json'), parameters=13157, epochs=history,
        training_scans=len(plan['training_scans']), reference_nodules=plan['training_references'],
        patches=len(data), optimizer_updates=updates, expected_updates=math.ceil(len(data)/24)*30,
        amp_scale=scaler.get_scale(), amp_skipped_updates=0, calibration_scans_used_for_fit=0,
        sealed_test_scans_accessed=0, fpga_validated=False))
    save(out/'training_progress.json', dict(status='COMPLETE', epochs=history,
        optimizer_updates=updates, sealed_test_scans_accessed=0))
    print('TRAINING COMPLETE; model frozen. Calibration and new56 test NOT RUN.', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('freeze', 'prepare', 'train', 'run'))
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    out = args.out.resolve()
    if args.stage == 'run':
        if not out.exists():
            freeze(out)
        if not (out/'training_catalogue.json').exists():
            prepare(out)
        train(out)
    else:
        globals()[args.stage](out)


if __name__ == '__main__':
    main()
