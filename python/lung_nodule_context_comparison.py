"""Matched-budget physical-context experiments, preserving all prior artifacts.

Only train247 images prepare patches. Only calibration47 images evaluate models.
The original five-plane model is reused, NOT continued. All three new models
start from its exact *pre-training* checkpoint and receive the same 29,070 steps.
"""
import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
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

from lung_candidate_data import sha_file, read_split
from lung_candidate_rules_run import ROOT, SPLIT, EXCLUDED
from lung_candidate_run import predict_lungs, excluded_findings
from lung_nodule_cnn import config_from_dict, model, targets, loss, search_mask, tiles, decode, consolidate
from lung_nodule_cnn_balanced import read, save, orient, balanced_weights
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_expanded_training import OUT as TRAIN, INITIAL, LUNG, atomic_torch_save
from lung_nodule_expanded_evaluation import OUT as CAL, RANKER, calibration_records, verified_done
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_continuity import rerank_scores
from lung_nodule_heldout_eval import save_npz, save_packet, load_packet
from luna16_multibundle import OUT as DATA, read_manifest, records_for, load_development_volume

OUT = ROOT/'data/lung_nodule_context_20261020_v1'
BASE_CONTEXT = (-2.5, -1.25, 0., 1.25, 2.5)
ALL_CONTEXT = (-5., -3.75, -2.5, -1.25, 0., 1.25, 2.5, 3.75, 5.)
VARIANTS = {
    'near_far5': (-5., -1.25, 0., 1.25, 5.),
    'context7': (-5., -2.5, -1.25, 0., 1.25, 2.5, 5.),
    'context9': ALL_CONTEXT,
}


def context_model(cfg, context, state=None):
    """Extra inputs start at zero; all pre-existing convolution weights survive."""
    import torch
    from torch import nn
    context = tuple(context)
    if context not in VARIANTS.values() and context != BASE_CONTEXT:
        raise ValueError('Unregistered context')
    net = model(cfg)
    if len(context) != 5:
        original = net.stem[0][0]
        net.stem[0][0] = nn.Conv2d(len(context), cfg.base, 3, stride=2, padding=1)
        if state is not None:
            expanded = dict(state)
            w = torch.zeros_like(net.stem[0][0].weight)
            for old, offset in enumerate(BASE_CONTEXT):
                w[:, context.index(offset)] = state['stem.0.0.weight'][:, old]
            expanded['stem.0.0.weight'] = w
            net.load_state_dict(expanded)
        else:
            nn.init.kaiming_uniform_(net.stem[0][0].weight, a=math.sqrt(5))
        del original
    elif state is not None:
        # Near/far five keeps channel order; outer channels now see +/-5 mm.
        net.load_state_dict(state)
    return net


def sample_context(cached, spacing, job, cfg, context):
    """Exact old XY/Z arithmetic, variable channels, geometry-keyed bounded LRU."""
    signature = (tuple(spacing), cfg.patch, cfg.xy_mm)
    if not hasattr(cached, 'context_tiles'):
        cached.context_tiles, cached.context_grids = OrderedDict(), {}
        cached.context_signature = signature
    if cached.context_signature != signature:
        raise ValueError('Sampling cache geometry changed')
    zm, y0, x0 = job
    if (y0, x0) not in cached.context_grids:
        y = (y0+np.arange(cfg.patch))*cfg.xy_mm/spacing[1]
        x = (x0+np.arange(cfg.patch))*cfg.xy_mm/spacing[2]
        cached.context_grids[y0, x0] = np.meshgrid(y, x, indexing='ij')
    grid = cached.context_grids[y0, x0]

    def plane(z):
        key = (z, y0, x0)
        if key not in cached.context_tiles:
            cached.context_tiles[key] = ndi.map_coordinates(np.asarray(cached[z], np.float32), grid,
                order=1, mode='constant', cval=-1000, prefilter=False)
            if len(cached.context_tiles) > 1024:
                cached.context_tiles.popitem(last=False)
        cached.context_tiles.move_to_end(key)
        return cached.context_tiles[key]

    output = np.empty((len(context), cfg.patch, cfg.patch), np.float32)
    for i, offset in enumerate(context):
        z = np.clip((zm+offset)/spacing[0], 0, len(cached)-1)
        lo, hi = int(np.floor(z)), int(np.ceil(z))
        a = plane(lo)
        if hi != lo:
            alpha = z-lo
            a = a*(1-alpha)+plane(hi)*alpha
        output[i] = a
    return np.rint(output).clip(-32768, 32767).astype(np.int16)


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve existing experiment; run resumes it')
    from lung_nodule_expanded_training import verify as verify_training
    from lung_nodule_expanded_evaluation import verify as verify_calibration
    old, manifest = verify_training(TRAIN)
    verify_calibration(CAL)
    cat = read(TRAIN/'training_catalogue.json')
    if len(cat['items']) != 23240 or len(old['training_scans']) != 247:
        raise ValueError('Wrong common training catalogue')
    paths = [INITIAL, TRAIN/'model.pt', TRAIN/'model_freeze.json', TRAIN/'training_catalogue.json',
        CAL/'plan.json', CAL/'evaluation.json', DATA/'manifest.json', LUNG, RANKER, EXCLUDED,
        Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_context_comparison.py']
    paths += [ROOT/'python'/n for n in ('lung_nodule_cnn.py', 'lung_nodule_cnn_balanced.py',
        'lung_nodule_sampling_cache.py', 'lung_nodule_expanded_training.py',
        'lung_nodule_expanded_evaluation.py', 'lung_nodule_continuity.py',
        'lung_nodule_continuity_boundary.py', 'lung_candidate_froc.py',
        'lung_nodule_capacity_froc_audit.py', 'lung_nodule_cnn_review18.py',
        'luna16_multibundle.py', 'lung_candidate_run.py', 'lung_unet.py')]
    protected = {str(p):sha_file(p) for p in paths}
    protected.update(cat['cache_sha256'])
    cal_records = calibration_records(manifest)
    ordered = sorted(cal_records, key=lambda r:(r['shape_zyx'][0], r['series_uid']))
    timing_uids = [ordered[i]['series_uid'] for i in (0, 15, 31, 46)]
    plan = dict(version='matched_30epoch_context_v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=old['seed'], config=old['config'], variants=VARIANTS, shared_context_mm=ALL_CONTEXT,
        training=old['training'], training_uids=old['training_scans'], patches=23240,
        initial_checkpoint=str(INITIAL), initial_sha256=sha_file(INITIAL),
        baseline_checkpoint=str(TRAIN/'model.pt'), baseline_sha256=sha_file(TRAIN/'model.pt'),
        baseline_evaluation=str(CAL/'evaluation.json'),
        calibration_uids=[r['series_uid'] for r in cal_records], timing_uids=timing_uids,
        timing='four predeclared calibration depth quantiles; 192 regularly spaced jobs, both orders; resident forward separately',
        initial_input_mapping='near_far5 channel order unchanged; 7/9 retain five physical offsets, added weights zero',
        budget='same pre-training checkpoint, seed, patch catalogue/order, targets, sampler, augmentation, optimizer, 30 epochs/29070 updates as baseline',
        raw_and_cross=True, ranker_refitted=False, sealed_test_scans_accessed=0, fpga_validated=False,
        attribution='One seed; near_far5 changes physical semantics of two pretrained channels; no extra baseline continuation needed',
        protected_sha256=protected)
    out.mkdir(parents=True)
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n', encoding='utf-8')
    (out/'sources').mkdir()
    for p in paths:
        if p.suffix == '.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    print('FROZEN: same initial checkpoint + 30 epochs; baseline NOT retrained; new56 SEALED', flush=True)


def verify(out, cache=False):
    plan = read(out/'plan.json')
    if sha_file(out/'plan.json') != (out/'plan.sha256').read_text().strip():
        raise ValueError('Changed context experiment plan')
    for p, digest in plan['protected_sha256'].items():
        if not cache and Path(p).suffix in ('.npy',) or not cache and str(Path(p).parent).endswith(('patches', 'hard_patches')):
            continue
        if sha_file(p) != digest:
            raise ValueError('Changed protected input: '+p)
    manifest = read_manifest(DATA/'manifest.json')
    if set(plan['training_uids']) != {r['series_uid'] for r in records_for(manifest, 'train')}:
        raise ValueError('Changed training cohort')
    if set(plan['calibration_uids']) != {r['series_uid'] for r in calibration_records(manifest)}:
        raise ValueError('Changed calibration cohort')
    return plan, manifest


def prepare(out):
    plan, manifest = verify(out, cache=True)
    cfg = config_from_dict(plan['config'])
    old_cat = read(TRAIN/'training_catalogue.json')
    grouped = defaultdict(list)
    for i, item in enumerate(old_cat['items']):
        grouped[item['series_uid']].append((i, item))
    record_by = {r['series_uid']:r for r in records_for(manifest, 'train')}
    folder = out/'patches'
    folder.mkdir(exist_ok=True)
    items = [None]*len(old_cat['items'])
    hashes, summaries = {}, []
    extra_context = tuple(c for c in ALL_CONTEXT if c not in BASE_CONTEXT)
    with threadpool_limits(limits=1):
        for count, (uid, entries) in enumerate(grouped.items(), 1):
            r = record_by[uid]
            path, meta_path = folder/(uid+'.npy'), folder/(uid+'.json')
            if meta_path.exists():
                meta = read(meta_path)
                if (meta['plan_sha256'] != sha_file(out/'plan.json') or meta['raw_sha256'] != r['raw_sha256'] or
                        meta['patch_sha256'] != sha_file(path) or meta['count'] != len(entries)):
                    raise ValueError('Changed shared context cache')
            else:
                tick = perf_counter()
                hu, geometry = load_development_volume(manifest, uid, 'train')
                cache = FloatPlaneCache(hu)
                spacing = np.asarray(geometry['spacing_zyx_mm'])
                maps = {p:np.load(p, mmap_mode='r', allow_pickle=False)
                        for p in {item['image_path'] for _, item in entries}}
                temp = path.with_suffix('.tmp.npy')
                images = np.lib.format.open_memmap(temp, mode='w+', dtype=np.int16,
                    shape=(len(entries), len(ALL_CONTEXT), cfg.patch, cfg.patch))
                for j, (_, item) in enumerate(entries):
                    old_image = maps[item['image_path']][item['patch_index']]
                    added = sample_context(cache, spacing, item['label']['job'], cfg, extra_context)
                    for k, offset in enumerate(ALL_CONTEXT):
                        images[j, k] = old_image[BASE_CONTEXT.index(offset)] if offset in BASE_CONTEXT else added[extra_context.index(offset)]
                    if j == 0:
                        # Real-input equality validates the mixed old/new cache construction.
                        direct = sample_context(cache, spacing, item['label']['job'], cfg, ALL_CONTEXT)
                        if not np.array_equal(images[j], direct):
                            raise AssertionError('Legacy cache and exact resampling disagree')
                images.flush()
                del images, maps, cache, hu
                os.replace(temp, path)
                meta = dict(plan_sha256=sha_file(out/'plan.json'), series_uid=uid, raw_sha256=r['raw_sha256'],
                    context_mm=ALL_CONTEXT, count=len(entries), patch_sha256=sha_file(path), seconds=perf_counter()-tick)
                save(meta_path, meta)
                gc.collect()
            for j, (original_index, item) in enumerate(entries):
                items[original_index] = dict(item, image_path=str(path), patch_index=j)
            hashes[str(path)] = meta['patch_sha256']
            hashes[str(meta_path)] = sha_file(meta_path)
            summaries.append(dict(series_uid=uid, patches=len(entries), seconds=meta['seconds']))
            save(out/'preparation_progress.json', dict(status='RUNNING', completed=count, expected=247,
                cases=summaries, sealed_test_scans_accessed=0))
            print('CONTEXT CACHE', count, '/247', 'patches', len(entries), 'seconds', round(meta['seconds'], 2), flush=True)
    if any(i is None for i in items):
        raise ValueError('Incomplete shared training catalogue')
    save(out/'training_catalogue.json', dict(plan_sha256=sha_file(out/'plan.json'), items=items,
        context_mm=ALL_CONTEXT, cache_sha256=hashes, patches=len(items), fit_scans=len(grouped)))
    save(out/'preparation_progress.json', dict(status='COMPLETE', completed=247, expected=247,
        cases=summaries, seconds=sum(r['seconds'] for r in summaries), sealed_test_scans_accessed=0))


def train(out, variant):
    plan, _ = verify(out)
    context = tuple(plan['variants'][variant])
    folder = out/variant
    folder.mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    if (folder/'model_freeze.json').exists():
        f = read(folder/'model_freeze.json')
        if sha_file(folder/'model.pt') != f['model_sha256'] or f['plan_sha256'] != identity:
            raise ValueError('Changed trained context model')
        return
    cat = read(out/'training_catalogue.json')
    if cat['plan_sha256'] != identity or len(cat['items']) != plan['patches']:
        raise ValueError('Wrong shared context catalogue')
    for p, digest in cat['cache_sha256'].items():
        if sha_file(p) != digest:
            raise ValueError('Changed shared training patch cache')
    torch = setup_torch(plan['seed'])
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    cfg = config_from_dict(plan['config'])
    channels = [ALL_CONTEXT.index(c) for c in context]

    class Patches(Dataset):
        def __init__(self):
            self.maps = {p:np.load(p, mmap_mode='r', allow_pickle=False)
                         for p in cat['cache_sha256'] if p.endswith('.npy')}
        def __len__(self):
            return len(cat['items'])
        def __getitem__(self, index):
            item = cat['items'][index]
            label = item['label']
            image = self.maps[item['image_path']][item['patch_index']][channels]
            image, pos, ignored = orient(image, label['positive'], label['ignored'], rotation=random.randrange(4),
                flip_x=random.random() < .5, flip_y=random.random() < .5, flip_z=random.random() < .5)
            return torch.from_numpy(image), tuple(torch.from_numpy(a) for a in targets(pos, ignored, cfg))

    data = Patches()
    generator = torch.Generator().manual_seed(plan['seed'])
    sampler = WeightedRandomSampler(balanced_weights([i['group'] for i in cat['items']]), len(data),
        replacement=True, generator=generator)
    loader = DataLoader(data, batch_size=24, sampler=sampler, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(plan['seed']))
    initial = torch.load(INITIAL, map_location='cpu', weights_only=True)
    net = context_model(cfg, context, initial['state']).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=plan['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=.00001)
    scaler = torch.amp.GradScaler('cuda', init_scale=32., growth_interval=2**31-1)
    history, updates = [], 0
    catalogue_sha = sha_file(out/'training_catalogue.json')
    resume = folder/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if (state['plan_sha256'] != identity or state['catalogue_sha256'] != catalogue_sha or
                state['variant'] != variant):
            raise ValueError('Wrong context resume identity')
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
    with threadpool_limits(limits=1):
        for epoch in range(len(history), 30):
            tick, batches, totals = perf_counter(), 0, np.zeros(3)
            net.train()
            for image, target in loader:
                image = image.cuda(non_blocking=True).float().clamp(-1000, 400).add_(1000).div_(1400)
                target = tuple(t.cuda(non_blocking=True) for t in target)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    value, classification, regression = loss(net(image).float(), target)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite context loss')
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite context gradient')
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() != scale:
                    raise ValueError('AMP skipped update or changed scale')
                updates += 1
                batches += 1
                totals += [value.item(), classification.item(), regression.item()]
            scheduler.step()
            history.append(dict(epoch=epoch+1, losses=(totals/batches).tolist(), batches=batches,
                seconds=perf_counter()-tick, cumulative_updates=updates))
            atomic_torch_save(torch, dict(plan_sha256=identity, variant=variant, catalogue_sha256=catalogue_sha,
                state=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), history=history, updates=updates, python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                sampler_rng=generator.get_state(), loader_rng=loader.generator.get_state()), resume)
            save(folder/'training_progress.json', dict(status='RUNNING', variant=variant, epochs=history,
                optimizer_updates=updates, expected_updates=29070, sealed_test_scans_accessed=0))
            print('TRAIN', variant, epoch+1, '/30', 'seconds', round(history[-1]['seconds'], 2), flush=True)
    if updates != 29070:
        raise AssertionError('Mismatched training update budget')
    net.cpu()
    atomic_torch_save(torch, dict(state=net.state_dict(), config=plan['config'], input_context_mm=context,
        variant=variant, plan_sha256=identity, training_scans=plan['training_uids'], epochs=30,
        initialized_from_sha256=plan['initial_sha256']), folder/'model.pt')
    save(folder/'model_freeze.json', dict(model_sha256=sha_file(folder/'model.pt'), plan_sha256=identity,
        variant=variant, context_mm=context, parameters=sum(p.numel() for p in net.parameters()),
        epochs=history, optimizer_updates=updates, expected_updates=29070, amp_skipped_updates=0,
        training_seconds=sum(h['seconds'] for h in history), sealed_test_scans_accessed=0, fpga_validated=False))
    save(folder/'training_progress.json', dict(status='COMPLETE', variant=variant, epochs=history,
        optimizer_updates=updates, sealed_test_scans_accessed=0))
    print('MODEL COMPLETE', variant, flush=True)


def forward_jobs(hu, geometry, search, jobs, net, cfg, context, torch):
    """Times real sampling, H2D/forward/D2H and decode; no annotations accepted."""
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    cached, accum = FloatPlaneCache(hu), [[], [], []]
    seconds = dict(sampling=0., transfer_forward=0., decode=0.)
    for begin in range(0, len(jobs), 48):
        group = jobs[begin:begin+48]
        tick = perf_counter()
        array = np.stack([sample_context(cached, spacing, j, cfg, context) for j in group])
        seconds['sampling'] += perf_counter()-tick
        tick = perf_counter()
        tensor = torch.from_numpy(array).cuda().float().clamp(-1000, 400).add_(1000).div_(1400)
        prediction = net(tensor).cpu().numpy()
        seconds['transfer_forward'] += perf_counter()-tick
        tick = perf_counter()
        for dest, value in zip(accum, decode(prediction, group, geometry, cfg, search)):
            dest.append(value)
        seconds['decode'] += perf_counter()-tick
    values = [np.concatenate(a) for a in accum] if jobs else [np.empty((0, 3)), np.empty(0), np.empty(0)]
    return values, seconds


def timing_probe(hu, geometry, search, jobs, baseline, net, cfg, context, torch):
    # A subset benchmark, explicitly not claimed to be measured full-CT latency.
    indices = np.unique(np.linspace(0, len(jobs)-1, min(192, len(jobs))).astype(int)) if jobs else []
    chosen = [jobs[int(i)] for i in indices]
    measurements = {'baseline5':[], 'variant':[]}
    for order in (('baseline5', 'variant'), ('variant', 'baseline5')):
        for name in order:
            n, c = (baseline, BASE_CONTEXT) if name == 'baseline5' else (net, context)
            _, times = forward_jobs(hu, geometry, search, chosen, n, cfg, c, torch)
            measurements[name].append(times)
    resident = {}
    for name, n, c in (('baseline5', baseline, BASE_CONTEXT), ('variant', net, context)):
        tensor = torch.zeros((48, len(c), cfg.patch, cfg.patch), device='cuda')
        for _ in range(3):
            n(tensor)
        torch.cuda.synchronize()
        tick = perf_counter()
        for _ in range(30):
            n(tensor)
        torch.cuda.synchronize()
        resident[name] = (perf_counter()-tick)/30/48
    mean = {name:float(np.mean([sum(t.values()) for t in runs])) for name, runs in measurements.items()}
    return dict(sampled_jobs=len(chosen), total_jobs=len(jobs), timings=measurements,
        mean_sample_pipeline_seconds=mean, resident_forward_seconds_per_patch=resident,
        pipeline_ratio=mean['variant']/mean['baseline5'] if mean['baseline5'] else None,
        estimated_full_ct_pipeline_extra_seconds=(mean['variant']-mean['baseline5'])*len(jobs)/len(chosen) if chosen else None,
        note='192 regular jobs, two orders, native CT in RAM, supplied lungs; full-CT difference extrapolated, not measured; not FPGA')


def infer(out, variant):
    plan, manifest = verify(out)
    folder = out/variant
    frozen = read(folder/'model_freeze.json')
    if frozen['model_sha256'] != sha_file(folder/'model.pt') or frozen['optimizer_updates'] != 29070:
        raise ValueError('Incomplete or changed context model')
    torch = setup_torch(plan['seed'])
    cfg, context = config_from_dict(plan['config']), tuple(plan['variants'][variant])
    cp = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if tuple(cp['input_context_mm']) != context or set(cp['training_scans']) != set(plan['training_uids']):
        raise ValueError('Wrong context checkpoint')
    net = context_model(cfg, context).cuda().eval()
    net.load_state_dict(cp['state'])
    baseline = model(cfg).cuda().eval()
    baseline.load_state_dict(torch.load(TRAIN/'model.pt', map_location='cpu', weights_only=True)['state'])
    from lung_unet import make_model
    import lung_nodule_cnn_experiment as first
    lung_cp = torch.load(LUNG, map_location='cpu', weights_only=True)
    lung = make_model(lung_cp['base_channels'], lung_cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(lung_cp['model'])
    split, ranker = read_split(SPLIT), read(RANKER)
    masks = out/'calibration_masks'
    masks.mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    completed = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(calibration_records(manifest), 1):
            uid = r['series_uid']
            case_dir = folder/'cases'/uid
            case_dir.mkdir(parents=True, exist_ok=True)
            if (case_dir/'complete.json').exists():
                verified_done(case_dir, identity)
                completed.append(uid)
                continue
            tick = perf_counter()
            previous = CAL/'cases'/uid
            verified_done(previous, sha_file(CAL/'plan.json'))
            baseline_packet = load_packet(previous/'expanded247.json.gz')
            hu, geometry = load_development_volume(manifest, uid, 'calibration')
            load_seconds = perf_counter()-tick
            if baseline_packet['input_raw_sha256'] != r['raw_sha256'] or baseline_packet['geometry'] != geometry:
                raise ValueError('Baseline pixel or geometry identity mismatch')
            mask_path = masks/(uid+'.npz')
            if mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as a:
                    body = np.unpackbits(a['body'], count=int(np.prod(hu.shape))).reshape(hu.shape).astype(bool)
            elif r['origin_cohort'] == 'original135':
                body = first.load_mask(r, split)
                save_npz(mask_path, body=np.packbits(body), shape_zyx=np.asarray(hu.shape))
            else:
                raw, body, _ = predict_lungs(hu, geometry, lung, 'cuda', 16)
                del raw
                save_npz(mask_path, body=np.packbits(body), shape_zyx=np.asarray(hu.shape))
            mask_sha = hashlib.sha256(body.tobytes()).hexdigest()
            if mask_sha != baseline_packet['lung_mask_sha256']:
                raise ValueError('Different lung mask from baseline evaluation')
            spacing = np.asarray(geometry['spacing_zyx_mm'])
            stage_tick = perf_counter()
            search = search_mask(body, spacing, cfg)
            jobs, schedule = tiles(search, spacing, cfg)
            if schedule != baseline_packet['schedule']:
                raise ValueError('Different job grid from baseline')
            grid_seconds = perf_counter()-stage_tick
            values, seconds = forward_jobs(hu, geometry, search, jobs, net, cfg, context, torch)
            stage_tick = perf_counter()
            candidates = consolidate(*values, geometry, cfg) if jobs else []
            for c in candidates:
                box = np.asarray(c['bbox_zyx']).reshape(3, 2)
                box[:, 0] = np.maximum(box[:, 0], 0)
                box[:, 1] = np.minimum(box[:, 1], hu.shape)
                c['bbox_zyx'] = box.ravel().tolist()
            seconds['consolidate'] = perf_counter()-stage_tick
            stage_tick = perf_counter()
            features, evidence = evidence_for_packet(hu, spacing, candidates)
            raw_scores = np.asarray([c['score'] for c in candidates])
            cross_scores = rerank_scores(raw_scores, features, ranker)
            seconds['evidence_and_rerank'] = perf_counter()-stage_tick
            timing = None
            if uid in plan['timing_uids']:
                timing = timing_probe(hu, geometry, search, jobs, baseline, net, cfg, context, torch)
            packet = dict(series_uid=uid, variant=variant, plan_sha256=identity,
                model_sha256=frozen['model_sha256'], input_raw_sha256=r['raw_sha256'],
                lung_mask_sha256=mask_sha, source_shape_zyx=list(hu.shape), geometry=geometry,
                candidates=candidates, evidence=evidence, schedule=schedule, input_context_mm=context,
                annotations_used_by_inference=False, backend='FP32_CUDA_NOT_FPGA',
                seconds=seconds, load_seconds=load_seconds, grid_seconds=grid_seconds, timing_probe=timing)
            save_packet(case_dir/'prediction.json.gz', packet)
            save_npz(case_dir/'scores.npz', raw_scores=raw_scores, cross_scores=cross_scores, features=features)
            save(case_dir/'complete.json', dict(plan_sha256=identity, series_uid=uid,
                artifacts_sha256={n:sha_file(case_dir/n) for n in ('prediction.json.gz', 'scores.npz')},
                seconds=perf_counter()-tick))
            completed.append(uid)
            save(folder/'inference_progress.json', dict(status='RUNNING', completed=len(completed), expected=47,
                sealed_test_scans_accessed=0))
            print('CAL', variant, i, '/47', 'candidates', len(candidates), 'seconds', round(perf_counter()-tick, 2), flush=True)
            del hu, body, search, values, features, evidence, candidates, baseline_packet, packet
            gc.collect()
    save(folder/'inference_progress.json', dict(status='COMPLETE', completed=47, expected=47,
        sealed_test_scans_accessed=0))


def summarize(out, variant):
    from lung_candidate_froc import evaluate
    from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
    from lung_nodule_capacity_froc_audit import check_froc_stable
    plan, manifest = verify(out)
    folder, rows, predictions, timings, stage_times = out/variant, [], {'raw':{}, 'cross':{}}, [], []
    excluded = excluded_findings(EXCLUDED)
    frozen = read(folder/'model_freeze.json')
    for r in calibration_records(manifest):
        uid = r['series_uid']
        case_dir = folder/'cases'/uid
        verified_done(case_dir, sha_file(out/'plan.json'))
        packet = load_packet(case_dir/'prediction.json.gz')
        if (packet['model_sha256'] != frozen['model_sha256'] or packet['input_raw_sha256'] != r['raw_sha256'] or
                tuple(packet['input_context_mm']) != tuple(plan['variants'][variant])):
            raise ValueError('Wrong reported model/input')
        rows.append(dict(uid=uid, world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3),
            record=r, excluded=excluded.get(uid, [])))
        with np.load(case_dir/'scores.npz', allow_pickle=False) as a:
            for mode in predictions:
                predictions[mode][uid] = a[mode+'_scores'].copy()
        if packet['timing_probe'] is not None:
            timings.append(dict(series_uid=uid, **packet['timing_probe']))
        stage_times.append(packet['seconds'])
    results = {}
    for mode, scores in predictions.items():
        scored = [dict(r, scores=scores[r['uid']]) for r in rows]
        report = evaluate(scored, scores)
        audit = check_froc_stable(scored, scores, report)
        profiles = {}
        for point in report['operating_points']:
            if point['fp_budget'] in (1., 4., 8.):
                value = summarize_fixed(scored, point['threshold'])
                independent = independent_counts(scored, point['threshold'])
                for key in ('detected', 'reference_nodules', 'false_positive_candidates'):
                    if value[key] != independent[key]:
                        raise AssertionError('Independent TP/FP mismatch')
                profiles[str(point['fp_budget'])] = dict(value, threshold=point['threshold'])
        results[mode] = dict(froc=report, audit=audit, profiles=profiles)
    record = dict(status='COMPLETE', variant=variant, cohort='calibration47_not_independent_test',
        scans=47, references=56, model_sha256=frozen['model_sha256'], plan_sha256=sha_file(out/'plan.json'),
        input_context_mm=plan['variants'][variant], parameters=frozen['parameters'],
        training_seconds=frozen['training_seconds'], results=results, timing_probes=timings,
        mean_ct_stage_seconds={k:float(np.mean([t[k] for t in stage_times])) for k in stage_times[0]},
        baseline_results_reused=True, ranker_refitted=False, sealed_test_scans_accessed=0, fpga_validated=False)
    save(folder/'evaluation.json', record)
    baseline = read(CAL/'evaluation.json')['results']
    lines = ['# 三种物理上下文的同预算比较', '',
        '247例训练，47例/56个结节开发校准；新56例仍封存。原5层已有输出直接复用。',
        '所有新版从原5层那次训练的同一个初始权重出发，训练30轮/29,070次更新，未额外续训旧基线。', '',
        '|模型|评分|误报预算/CT|检出/56|检出率|实际误报/CT|', '|---|---|---:|---:|---:|---:|']
    for mode in ('raw', 'cross'):
        for profile, value in baseline['expanded247_'+mode]['profiles'].items():
            lines.append(f"|原5层|{mode}|{profile}|{value['detected']}/56|{value['sensitivity']:.2%}|{value['actual_fp_per_scan']:.2f}|")
    completed = []
    for name in VARIANTS:
        p = out/name/'evaluation.json'
        if not p.exists():
            continue
        result = read(p)
        completed.append(result)
        for mode in ('raw', 'cross'):
            for profile, value in result['results'][mode]['profiles'].items():
                lines.append(f"|{name}|{mode}|{profile}|{value['detected']}/56|{value['sensitivity']:.2%}|{value['actual_fp_per_scan']:.2f}|")
        lines += ['', f"{name}：训练{result['training_seconds']/60:.2f}分钟，参数{result['parameters']}。"]
        for t in result['timing_probes']:
            lines.append(f"- 原生{t['total_jobs']}图块病例：192图块采样/网络/解码比值{t['pipeline_ratio']:.3f}；按图块数外推整卷额外{t['estimated_full_ct_pipeline_extra_seconds']:.2f}秒（估计，非实测整卷差值）。")
    lines += ['', '同一冻结跨层评分同时用于各版，原始CNN另列。阈值分别由47例校准选择；不是独立测试结果。',
        'GPU计时不代表PYNQ-Z2；输入采样与首层卷积成本变化分开报告。仅一个随机种子，差一两个结节不代表稳定领先。', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    save(out/'comparison.json', dict(status='COMPLETE' if len(completed) == 3 else 'PARTIAL',
        completed_variants=[r['variant'] for r in completed], baseline_evaluation=str(CAL/'evaluation.json'),
        variant_evaluations={r['variant']:str(out/r['variant']/'evaluation.json') for r in completed},
        sealed_test_scans_accessed=0))
    for mode in ('raw', 'cross'):
        p = record['results'][mode]['profiles']['4.0']
        print('RESULT', variant, mode, f"{p['detected']}/56", 'FP/CT', p['actual_fp_per_scan'], flush=True)
    print('VARIANT COMPLETE', variant, 'RESULT READY', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'prepare', 'train', 'infer', 'summarize', 'run'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == 'run':
        if not out.exists():
            freeze(out)
        if not (out/'training_catalogue.json').exists():
            prepare(out)
        for variant in ([args.variant] if args.variant else VARIANTS):
            train(out, variant)
            infer(out, variant)
            summarize(out, variant)
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    elif args.stage in ('freeze', 'prepare'):
        globals()[args.stage](out)
    else:
        if args.variant is None:
            parser.error('--variant required')
        globals()[args.stage](out, args.variant)


if __name__ == '__main__':
    main()
