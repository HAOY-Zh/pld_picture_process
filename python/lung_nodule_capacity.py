"""Paired width-only continuation of the FIRST cross-slice CNN baseline.

Both arms see the exact same augmented minibatch and take the same 6060 updates.
The original historical model is a third, untrained reference. No heldout mode.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from lung_candidate_data import sha_file, load_development_case
from lung_candidate_rules_run import ROOT, EXCLUDED
from lung_candidate_run import excluded_findings
from lung_candidate_froc import evaluate
from lung_candidate_lowfp_audit import check_froc
from lung_nodule_cnn import (config_from_dict, model, targets, loss, search_mask,
                             tiles, decode, consolidate)
from lung_nodule_cnn_balanced import (OUT as V3, read, save, orient, balanced_weights)
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
from lung_nodule_continuity import rerank_scores
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_sampling_cache import sample_patch_reuse
import lung_nodule_cnn_experiment as first

OUT = ROOT/'data/lung_nodule_capacity_20261020_v1'
PREVIOUS = ROOT/'data/lung_nodule_continuity_20261019_boundary_v2'
INITIAL = V3/'balanced_orientation/model.pt'
VARIANTS = ('small_continued', 'wide_continued')
COHORTS = ('calibration_scans', 'review_12', 'reused_6')


def doubled_state(state, seed):
    """Duplicate hidden channels; split outgoing weights unequally with pair sum w.

    Identical incoming activations preserve the initial function. Unequal outgoing
    weights allow duplicated channels to receive different gradients. Concatenated
    skip paths retain their order under repeat_interleave, including both decoders.
    """
    import torch
    rng = torch.Generator().manual_seed(seed)
    result = {}
    for key, value in state.items():
        first_layer = key.startswith('stem.0.0.')
        head = key.startswith('head.')
        if key.endswith('.bias'):
            result[key] = value.clone() if head else value.repeat_interleave(2, 0)
        elif key.endswith('.weight'):
            expanded = value.clone()
            if not first_layer:
                fraction = .35 + .30*torch.rand(value.shape, generator=rng, dtype=value.dtype)
                left = value*fraction
                expanded = torch.stack((left, value-left), dim=2).flatten(1, 2)
            result[key] = expanded if head else expanded.repeat_interleave(2, 0)
        else:
            raise ValueError('Unsupported model state: '+key)
    return result


def architecture_counts(cfg):
    """Parameter counts and Conv2d MACs only; excludes interpolation and activation."""
    import torch
    net = model(cfg).eval()
    macs = []
    hooks = []
    for layer in net.modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(lambda m, a, y:
                macs.append(int(y.numel()*m.in_channels*m.kernel_size[0]*m.kernel_size[1]/m.groups))))
    with torch.inference_mode():
        net(torch.zeros(1, 5, cfg.patch, cfg.patch))
    for hook in hooks:
        hook.remove()
    return dict(parameters=sum(p.numel() for p in net.parameters()), conv_macs_per_patch=sum(macs),
                fp32_weight_bytes=sum(p.numel()*p.element_size() for p in net.parameters()))


def eligible(plan, cohort):
    if cohort not in ('fit_scans', *COHORTS):
        raise ValueError('No heldout entrypoint')
    groups = [set(plan[k]) for k in ('fit_scans', *COHORTS)]
    if any(len(plan[k]) != len(set(plan[k])) for k in ('fit_scans', *COHORTS)):
        raise ValueError('Duplicate scan')
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('Cohort overlap')
    return first.records_for(plan, cohort)


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve completed experiments')
    old = first.verify(PREVIOUS)
    protected = dict(old['protected_sha256'])
    catalogue = V3/'training_catalogue.json'
    cat = read(catalogue)
    protected.update(cat['cache_sha256'])
    extra = [catalogue, INITIAL, INITIAL.parent/'model_freeze.json', PREVIOUS/'plan.json',
        PREVIOUS/'plan.sha256', PREVIOUS/'ranker.json', PREVIOUS/'ranker_freeze.json',
        PREVIOUS/'calibration.json', PREVIOUS/'calibration_freeze.json', PREVIOUS/'review.json',
        Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_capacity.py',
        ROOT/'python/lung_nodule_continuity_boundary.py', ROOT/'python/lung_nodule_sampling_cache.py',
        ROOT/'python/lung_nodule_whole_background.py']
    for cohort in COHORTS:
        extra += [PREVIOUS/cohort/(uid+'.json') for uid in old[cohort]]
    for p in extra:
        protected[str(p)] = sha_file(p)
    cfg = config_from_dict(old['config'])
    plan = {k:old[k] for k in ('config', 'fit_scans', *COHORTS)}
    for cohort in ('fit_scans', *COHORTS):
        eligible(plan, cohort)
    plan.update(version='paired_function_preserving_width_capacity_v1', seed=2026102002,
        variants=list(VARIANTS), protected_sha256=protected,
        configs={v:asdict(replace(cfg, base=8 if v==VARIANTS[0] else 16)) for v in VARIANTS},
        architectures={v:architecture_counts(replace(cfg, base=8 if v==VARIANTS[0] else 16)) for v in VARIANTS},
        initial_checkpoint=str(INITIAL), initial_checkpoint_sha256=sha_file(INITIAL),
        training_catalogue=str(catalogue),
        training=dict(epochs=30, batch_size=24, patches_per_epoch=len(cat['items']),
            learning_rate=.0003, final_learning_rate=.00001, optimizer='Adam',
            gradient_clip=10., mixed_precision=True, initial_grad_scale=1024.,
            sampling='same v3 weighted replacement: half background, half equal positive size/edge strata',
            augmentation='identical shared XY rotations/flips and context Z reversal',
            initialization='small: frozen v3; wide: function-preserving duplication with seeded unequal outgoing splits',
            checkpoint='fixed last epoch; no development epoch selection',
            minibatches='one tensor and label tuple shared by both arms; sampler and augmentation stream hashed'),
        cross_slice_rule='same FIRST frozen 38-feature ranker, alpha=.25, no refit for either arm',
        evaluation='raw CNN and original cross-slice scores; calibrate 1/4/8 on19 then freeze before18',
        interpretation='paired width continuation; historical reference has 30 fewer extra epochs; single seed',
        excluded_work='no whole-volume-background ranker changes, no LDCT/U-Net/default-web changes',
        heldout_scans_run=0, fpga_validated=False, expert_nodule_voxel_masks=False)
    if len(cat['items']) != 4840 or read(PREVIOUS/'ranker.json')['alpha'] != .25:
        raise ValueError('Unexpected first cross-slice baseline')
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in extra:
        if p.suffix == '.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN CAPACITY', plan['architectures'], flush=True)


def train(out):
    plan = first.verify(out)
    torch = setup_torch(plan['seed'])
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    cfg = config_from_dict(plan['config'])
    cat = read(Path(plan['training_catalogue']))
    fit = set(plan['fit_scans'])
    if any(item['series_uid'] not in fit for item in cat['items']):
        raise ValueError('Non-fit patch')
    training = plan['training']
    stream = hashlib.sha256()

    class Patches(Dataset):
        def __init__(self):
            self.maps = {p:np.load(p, mmap_mode='r', allow_pickle=False)
                         for p in cat['cache_sha256'] if p.endswith('.npy')}

        def __len__(self):
            return len(cat['items'])

        def __getitem__(self, index):
            item = cat['items'][index]
            label = item['label']
            aug = (random.randrange(4), random.random()<.5, random.random()<.5, random.random()<.5)
            stream.update(np.asarray((index, *aug), dtype='<i8').tobytes())
            image, pos, ign = orient(self.maps[item['image_path']][item['patch_index']],
                label['positive'], label['ignored'], *aug)
            return torch.from_numpy(image), tuple(torch.from_numpy(t) for t in targets(pos, ign, cfg))

    data = Patches()
    generator = torch.Generator().manual_seed(plan['seed'])
    sampler = WeightedRandomSampler(balanced_weights([i['group'] for i in cat['items']]),
        len(data), replacement=True, generator=generator)
    loader = DataLoader(data, batch_size=training['batch_size'], sampler=sampler,
        num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(plan['seed']))
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    if set(initial['training_scans']) != fit:
        raise ValueError('Initial training cohort mismatch')
    nets, optimizers, schedulers, scalers = {}, {}, {}, {}
    for variant in VARIANTS:
        folder = out/variant
        folder.mkdir(exist_ok=True)
        net = model(config_from_dict(plan['configs'][variant])).cuda()
        net.load_state_dict(initial['state'] if variant==VARIANTS[0] else doubled_state(initial['state'], plan['seed']))
        nets[variant] = net
        optimizer = torch.optim.Adam(net.parameters(), lr=training['learning_rate'])
        optimizers[variant] = optimizer
        schedulers[variant] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=training['epochs'], eta_min=training['final_learning_rate'])
        scalers[variant] = torch.amp.GradScaler('cuda', init_scale=training['initial_grad_scale'])
    # Real fit-only cached patches, FP32/TF32-off, before any update or development CT.
    images = []
    for item in cat['items'][:8]:
        images.append(np.array(data.maps[item['image_path']][item['patch_index']], copy=True))
    x = torch.from_numpy(np.stack(images)).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
    with torch.inference_mode():
        a, b = [nets[v].eval()(x) for v in VARIANTS]
        difference = (a-b).abs()
        initial_check = dict(max_absolute_error=float(difference.max()),
            rms_error=float(torch.sqrt((a-b).square().mean())), fit_patches=8,
            unchanged_initial_function_with_fp32_roundoff=True)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-4)
    save(out/'initial_function_check.json', initial_check)
    print('INITIAL FUNCTION', initial_check, flush=True)
    del x, a, b
    history = []
    resume = out/'paired_resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_sha256'] != sha_file(out/'plan.json'):
            raise ValueError('Wrong resume identity')
        for v in VARIANTS:
            nets[v].load_state_dict(state['arms'][v]['state'])
            optimizers[v].load_state_dict(state['arms'][v]['optimizer'])
            schedulers[v].load_state_dict(state['arms'][v]['scheduler'])
            scalers[v].load_state_dict(state['arms'][v]['scaler'])
        history = state['history']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['sampler_rng'])
        loader.generator.set_state(state['loader_rng'])
    for epoch in range(len(history), training['epochs']):
        tick = perf_counter()
        stream = hashlib.sha256()
        totals = {v:np.zeros(3) for v in VARIANTS}
        skipped = {v:0 for v in VARIANTS}
        batches = 0
        for net in nets.values():
            net.train()
        for image, target in loader:
            image = image.cuda(non_blocking=True).float().clamp(-1000,400).add_(1000).div_(1400)
            target = tuple(t.cuda(non_blocking=True) for t in target)
            for v in VARIANTS:
                optimizer, scaler, net = optimizers[v], scalers[v], nets[v]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    value, classification, regression = loss(net(image).float(), target)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite paired loss')
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), training['gradient_clip'])
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped[v] += int(scaler.get_scale() < scale)
                totals[v] += [value.item(), classification.item(), regression.item()]
            batches += 1
        if any(skipped.values()):
            raise ValueError('AMP overflow would break equal-update condition: '+str(skipped))
        history.append(dict(epoch=epoch+1, batches=batches, shared_sample_augmentation_sha256=stream.hexdigest(),
            loss={v:(totals[v]/batches).tolist() for v in VARIANTS}, skipped_updates=skipped,
            learning_rate=optimizers[VARIANTS[0]].param_groups[0]['lr'], seconds=perf_counter()-tick))
        for scheduler in schedulers.values():
            scheduler.step()
        temporary = out/'paired_resume.tmp'
        torch.save(dict(plan_sha256=sha_file(out/'plan.json'), history=history,
            arms={v:dict(state=nets[v].state_dict(), optimizer=optimizers[v].state_dict(),
                scheduler=schedulers[v].state_dict(), scaler=scalers[v].state_dict()) for v in VARIANTS},
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), sampler_rng=generator.get_state(),
            loader_rng=loader.generator.get_state()), temporary)
        os.replace(temporary, resume)
        save(out/'training_progress.json', dict(status='RUNNING', epochs=history))
        print('PAIRED TRAIN', epoch+1, '/', training['epochs'],
            {v:round(history[-1]['loss'][v][0],4) for v in VARIANTS},
            'seconds', round(history[-1]['seconds'],2), flush=True)
    for v in VARIANTS:
        folder = out/v
        net = nets[v].cpu()
        temporary = folder/'model.tmp'
        torch.save(dict(state=net.state_dict(), config=plan['configs'][v], training_scans=plan['fit_scans'],
            plan_sha256=sha_file(out/'plan.json'), epochs=len(history), variant=v,
            initialized_from_sha256=plan['initial_checkpoint_sha256']), temporary)
        os.replace(temporary, folder/'model.pt')
        save(folder/'model_freeze.json', dict(model_sha256=sha_file(folder/'model.pt'),
            plan_sha256=sha_file(out/'plan.json'), architecture=plan['architectures'][v],
            training_batches=sum(h['batches'] for h in history), epochs=len(history),
            shared_sample_augmentation_sha256=[h['shared_sample_augmentation_sha256'] for h in history],
            skipped_updates=sum(h['skipped_updates'][v] for h in history), heldout_scans_run=0))
    save(out/'training_complete.json', dict(epochs=history, initialization=initial_check,
        heldout_scans_run=0, exact_shared_minibatches=True, fixed_final_checkpoint=True))


def infer_pair(out, cohort):
    plan = first.verify(out)
    if cohort not in COHORTS:
        raise ValueError('No heldout entrypoint')
    if cohort != 'calibration_scans':
        f = read(out/'calibration_freeze.json')
        if sha_file(out/'calibration.json') != f['calibration_sha256']:
            raise ValueError('Freeze calibration before review')
    torch = setup_torch(plan['seed'])
    from threadpoolctl import threadpool_limits
    from lung_candidates import checked_inputs
    cfg = config_from_dict(plan['config'])
    split, records = eligible(plan, cohort)
    nets, identities = {}, {}
    for v in VARIANTS:
        folder = out/v
        f = read(folder/'model_freeze.json')
        if sha_file(folder/'model.pt') != f['model_sha256']:
            raise ValueError('Changed capacity model')
        ckpt = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
        if set(ckpt['training_scans']) != set(plan['fit_scans']):
            raise ValueError('Wrong training cohort')
        nets[v] = model(config_from_dict(plan['configs'][v])).cuda().eval()
        nets[v].load_state_dict(ckpt['state'])
        identities[v] = f['model_sha256']
        (folder/cohort).mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(records, 1):
            uid = r['series_uid']
            pending = []
            for v in VARIANTS:
                path = out/v/cohort/(uid+'.json')
                if path.exists():
                    p = read(path)
                    if p['model_sha256'] != identities[v] or p['plan_sha256'] != identity:
                        raise ValueError('Wrong completed inference identity')
                else:
                    pending.append(v)
            if not pending:
                continue
            start = perf_counter()
            hu, geometry = load_development_case(split, r)
            body = first.load_mask(r, split)
            checked_inputs(hu, body, geometry)
            spacing = np.asarray(geometry['spacing_zyx_mm'])
            search = search_mask(body, spacing, cfg)
            jobs, schedule = tiles(search, spacing, cfg)
            preprocess_seconds = perf_counter()-start
            cached = FloatPlaneCache(hu)
            accumulated = {v:[[], [], []] for v in pending}
            timers = {v:dict(forward_transfer_seconds=0., decode_seconds=0.) for v in pending}
            sample_seconds = 0.
            for begin in range(0, len(jobs), 48):
                batch = jobs[begin:begin+48]
                tick = perf_counter()
                image = np.stack([sample_patch_reuse(cached, spacing, j, cfg) for j in batch])
                tensor = torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                torch.cuda.synchronize()
                sample_seconds += perf_counter()-tick
                # Alternate order to reduce a systematic first/second-arm timing bias.
                order = pending if (begin//48)%2==0 else pending[::-1]
                for v in order:
                    tick = perf_counter()
                    prediction = nets[v](tensor).cpu().numpy()
                    timers[v]['forward_transfer_seconds'] += perf_counter()-tick
                    tick = perf_counter()
                    for store, values in zip(accumulated[v], decode(prediction, batch, geometry, cfg, search)):
                        store.append(values)
                    timers[v]['decode_seconds'] += perf_counter()-tick
            counts = {}
            for v in pending:
                tick = perf_counter()
                values = [np.concatenate(a) for a in accumulated[v]]
                candidates = consolidate(*values, geometry, cfg)
                for candidate in candidates:
                    box = np.asarray(candidate['bbox_zyx']).reshape(3, 2)
                    box[:,0] = np.maximum(box[:,0], 0)
                    box[:,1] = np.minimum(box[:,1], hu.shape)
                    candidate['bbox_zyx'] = box.ravel().tolist()
                timers[v]['consolidate_seconds'] = perf_counter()-tick
                features, evidence = evidence_for_packet(hu, spacing, candidates)
                packet = dict(schema_version='lung_candidate_result_v1', algorithm_id='capacity_'+v,
                    backend='FP32_CUDA_REFERENCE_NOT_FPGA', status='success', series_uid=uid,
                    input_raw_sha256=r['raw_sha256'], lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    model_sha256=identities[v], plan_sha256=identity, source_shape_zyx=list(hu.shape),
                    geometry=geometry, parameters=plan['configs'][v], schedule=schedule,
                    candidates=candidates, features=features.tolist(), evidence=evidence,
                    timing=dict(shared_preprocess_seconds=preprocess_seconds,
                        shared_sampling_transfer_seconds=sample_seconds, **timers[v]),
                    timing_scope='paired desktop experiment; excludes fresh lung inference; NOT board or independent latency')
                save(out/v/cohort/(uid+'.json'), packet)
                counts[v] = len(candidates)
            print('CAPACITY INFER', cohort, i, '/', len(records), 'tiles', len(jobs), counts,
                'paired_seconds', round(perf_counter()-start,2), flush=True)
            del hu, body, search, cached, tensor, accumulated
            gc.collect()


def rows_for(out, cohort, variant):
    plan = read(out/'plan.json')
    _, records = eligible(plan, cohort)
    excluded = excluded_findings(EXCLUDED)
    rows = []
    parent = variant == 'historical'
    ranker = read(PREVIOUS/'ranker.json')
    for r in records:
        uid = r['series_uid']
        path = (PREVIOUS if parent else out/variant)/cohort/(uid+'.json')
        p = read(path)
        if p['input_raw_sha256'] != r['raw_sha256']:
            raise ValueError('Wrong CT identity')
        if not parent:
            if p['model_sha256'] != read(out/variant/'model_freeze.json')['model_sha256']:
                raise ValueError('Wrong capacity model identity')
            if p['plan_sha256'] != sha_file(out/'plan.json'):
                raise ValueError('Wrong capacity inference plan')
        index = np.array([c['center_index_zyx'] for c in p['candidates']]).reshape(-1, 3)
        world = np.array([c['center_lps_xyz_mm'] for c in p['candidates']]).reshape(-1, 3)
        np.testing.assert_allclose(world, index@np.asarray(r['basis_lps_zyx']).T+r['origin_lps_mm'], atol=1e-6, rtol=0)
        scores = np.array([c['score'] for c in p['candidates']])
        features = np.asarray(p['features'])
        rows.append(dict(uid=uid, record=r, excluded=excluded.get(uid, []), world=world,
            raw_scores=scores, scores=rerank_scores(scores, features, ranker), features=features, packet=p))
    return rows


def calibrated(out):
    plan = first.verify(out)
    curves, thresholds = {}, {}
    for variant in ('historical', *VARIANTS):
        rows = rows_for(out, 'calibration_scans', variant)
        for mode in ('raw', 'cross'):
            name = variant+'_'+mode
            scores = {r['uid']:r['raw_scores' if mode=='raw' else 'scores'] for r in rows}
            curve = evaluate(rows, scores)
            check_froc(rows, scores, curve)
            curves[name] = curve
            thresholds[name] = {str(p['fp_budget']):p['threshold'] for p in curve['operating_points']
                                if p['fp_budget'] in (1., 4., 8.)}
            print('CAPACITY CAL', name, 'CPM', round(curve['cpm_interpolated'],4),
                '1/4/8', [(p['fp_budget'],p['detected']) for p in curve['operating_points']
                           if p['fp_budget'] in (1.,4.,8.)], flush=True)
    previous = read(PREVIOUS/'calibration.json')
    for mode, key in [('raw','baseline'), ('cross','continuity')]:
        for a, b in zip(curves['historical_'+mode]['operating_points'], previous['curves'][key]['operating_points']):
            if (a['detected'],a['threshold'],a['actual_fp_per_scan']) != (b['detected'],b['threshold'],b['actual_fp_per_scan']):
                raise ValueError('Historical calibration failed to reproduce')
    save(out/'calibration.json', dict(curves=curves, thresholds=thresholds, scans=len(plan['calibration_scans']),
        independent_froc_verified=True, heldout_scans_run=0))
    save(out/'calibration_freeze.json', dict(calibration_sha256=sha_file(out/'calibration.json'),
        ranker_sha256=sha_file(PREVIOUS/'ranker.json'),
        model_sha256={v:sha_file(out/v/'model.pt') for v in VARIANTS},
        next_stage='review12 and reused6; no retuning from their results', heldout_scans_run=0))


def reviewed(out):
    first.verify(out)
    cal = read(out/'calibration.json')
    if sha_file(out/'calibration.json') != read(out/'calibration_freeze.json')['calibration_sha256']:
        raise ValueError('Changed calibration')
    results = {}
    for cohort in ('review_12', 'reused_6'):
        results[cohort] = {}
        for variant in ('historical', *VARIANTS):
            rows = rows_for(out, cohort, variant)
            for mode in ('raw', 'cross'):
                name = variant+'_'+mode
                used = [dict(r,scores=r['raw_scores']) if mode=='raw' else r for r in rows]
                results[cohort][name] = {}
                for profile, threshold in cal['thresholds'][name].items():
                    m = summarize_fixed(used, threshold)
                    for k, v in independent_counts(used, threshold).items():
                        if m[k] != v:
                            raise ValueError('Independent fixed count mismatch')
                    results[cohort][name][profile] = m
                print('CAPACITY REVIEW', cohort, name,
                    [(p,m['detected'],round(m['actual_fp_per_scan'],3))
                     for p,m in results[cohort][name].items()], flush=True)
    save(out/'review.json', dict(results=results, thresholds_frozen_before_review=True,
        all_development_previously_viewed=True, heldout_scans_run=0, fpga_validated=False))


def run(out):
    if not out.exists():
        freeze(out)
    stages = [('train','training_complete.json',train),
        ('infer_calibration',None,lambda p:infer_pair(p,'calibration_scans')),
        ('calibration','calibration_freeze.json',calibrated),
        ('infer_review12',None,lambda p:infer_pair(p,'review_12')),
        ('infer_reused6',None,lambda p:infer_pair(p,'reused_6')),
        ('review','review.json',reviewed)]
    stage = 'verify'
    try:
        for stage, marker, operation in stages:
            if marker and (out/marker).exists():
                continue
            save(out/'pipeline_state.json', dict(status='RUNNING', stage=stage))
            operation(out)
        first.verify(out)
        save(out/'pipeline_state.json', dict(status='COMPLETE_CAPACITY_COMPARISON', heldout_scans_run=0))
    except Exception as error:
        save(out/'pipeline_state.json', dict(status='FAILED', stage=stage, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=OUT)
    run(parser.parse_args().outdir.resolve())
