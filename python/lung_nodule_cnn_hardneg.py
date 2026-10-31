"""One frozen training-only hard-background experiment; preserve CNN v1.

Reuses the architecture, loss, labels, geometry and inference of v1. Only the
training sample distribution and fixed fine-tuning schedule change. This is
an adaptation of hard-example mining, not a reproduction of DeepLung/YOLO.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from lung_candidate_data import load_development_case, sha_file, write_json
from lung_candidate_run import excluded_findings
from lung_candidate_rules_run import ROOT, EXCLUDED
import lung_nodule_cnn_experiment as first
from lung_nodule_cnn import (config_from_dict, model, search_mask, tiles,
                             sample_patch, relative_labels, targets, loss)

BASE = ROOT / 'data/lung_nodule_cnn_20261019_v1'
DEFAULT = ROOT / 'data/lung_nodule_cnn_20261019_hardneg_v2'


def background_mask(label, cfg):
    # Reference nodules (including neighboring-z findings) and excluded findings
    # are unavailable as negative mining targets. Positive labels remain intact
    # if a selected patch contains a reference nodule elsewhere.
    regions = label['ignored'] + [[p[0], p[1], p[3]] for p in label['positive']]
    _, valid, _, _ = targets([], regions, cfg)
    core = np.zeros_like(valid, dtype=bool)
    lo = cfg.halo // cfg.output_stride
    hi = (cfg.patch - cfg.halo) // cfg.output_stride
    core[lo:hi, lo:hi] = True
    return core & valid.astype(bool)


def select_patches(scores, hard_count, random_count, rng):
    scores = np.asarray(scores, dtype=float)
    eligible = np.flatnonzero(np.isfinite(scores))
    hard = eligible[np.argsort(-scores[eligible], kind='stable')[:hard_count]]
    remaining = np.setdiff1d(np.arange(len(scores)), hard)
    count = min(random_count, len(remaining))
    random_ids = rng.choice(remaining, size=count, replace=False)
    return hard.tolist(), random_ids.tolist()


def setup_torch(seed):
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the existing CUDA environment')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve existing experiments; resume completed stages')
    original = first.verify(BASE)
    original_freeze = json.loads((BASE / 'model_freeze.json').read_text('utf-8'))
    if sha_file(BASE / 'model.pt') != original_freeze['model_sha256']:
        raise ValueError('Changed v1 checkpoint')
    protected = dict(original['protected_sha256'])
    extra = [BASE / n for n in ('plan.json', 'plan.sha256', 'model.pt',
                               'model_freeze.json', 'calibration/report.json')]
    extra += [ROOT / 'python/lung_nodule_cnn_hardneg.py',
              ROOT / 'python/tests/test_lung_nodule_cnn_hardneg.py']
    for uid in original['fit_scans']:
        extra += [BASE / 'patches' / (uid + suffix) for suffix in ('.json', '.npy')]
    for path in extra:
        protected[str(path)] = sha_file(path)
    plan = dict(original)
    plan.update(version='lung_guided_25d_center_hard_background_v2', seed=2026101902,
        protected_sha256=protected, initial_checkpoint=str(BASE / 'model.pt'),
        initial_checkpoint_sha256=original_freeze['model_sha256'],
        parent_experiment=str(BASE), architecture_unchanged=True,
        exact_paper_reproduction=False,
        mining=dict(cohort='fit_scans_only', sampled_grid_tiles_per_scan=256,
                    hard_patches_per_scan=32, random_patches_per_scan=16,
                    method='v1 maximum center score over valid background in inference core',
                    exclusions='ignore excluded findings and reference nodule disks; preserve positive labels',
                    full_volume_hard_mining=False),
        training=dict(epochs=30, batch_size=24, learning_rate=.0003,
                      final_learning_rate=.00001, optimizer='Adam',
                      checkpoint='fixed last epoch; no validation checkpoint selection',
                      reuse_all_v1_training_patches=True, loss_unchanged=True,
                      augmentation='original independent XY flips only'),
        evaluation_policy='Same already-viewed development19; no review12/6 or sealed45; not independent test',
        ablation_limit='Negative sampling plus more optimization steps; not an isolated sampling ablation',
        references=[
            'https://github.com/wentaozhu/DeepLung/blob/master/detector/layers.py',
            'https://arxiv.org/abs/1904.07850',
            'https://raw.githubusercontent.com/ultralytics/yolov5/master/models/hub/yolov5-p2.yaml'])
    out.mkdir(parents=True)
    (out / 'sources').mkdir()
    for path in extra:
        if path.suffix == '.py':
            (out / 'sources' / path.name).write_bytes(path.read_bytes())
    write_json(out / 'plan.json', plan)
    (out / 'plan.sha256').write_text(sha_file(out / 'plan.json') + '\n')
    print('FROZEN v2: same 53 fit / 19 development; sampled hard-background mining', flush=True)


def mine(out):
    plan = first.verify(out)
    cfg = config_from_dict(plan['config'])
    torch = setup_torch(plan['seed'])
    split, records = first.records_for(plan, 'fit_scans')
    ignored = excluded_findings(EXCLUDED)
    checkpoint = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    if set(checkpoint['training_scans']) != set(plan['fit_scans']):
        raise ValueError('Mining teacher must have exactly the same training cohort')
    net = model(cfg).cuda().eval()
    net.load_state_dict(checkpoint['state'])
    folder = out / 'hard_patches'
    folder.mkdir(exist_ok=True)
    summaries = []
    for i, record in enumerate(records, 1):
        uid = record['series_uid']
        path, meta_path = folder / (uid + '.npy'), folder / (uid + '.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text('utf-8'))
            if (sha_file(path) != meta['patch_sha256'] or
                    meta['plan_sha256'] != sha_file(out / 'plan.json')):
                raise ValueError('Changed mined training cache')
            summaries.append(meta)
            continue
        start = perf_counter()
        rng = np.random.default_rng(int(hashlib.sha256((str(plan['seed']) + uid).encode()).hexdigest()[:16], 16))
        hu, geometry = load_development_case(split, record)
        body = first.load_mask(record, split)
        spacing = np.asarray(geometry['spacing_zyx_mm'])
        search = search_mask(body, spacing, cfg)
        jobs, schedule = tiles(search, spacing, cfg)
        count = min(len(jobs), plan['mining']['sampled_grid_tiles_per_scan'])
        if count == 0:
            raise ValueError('No predicted lung search grid')
        chosen = rng.choice(len(jobs), size=count, replace=False)
        jobs = [jobs[int(j)] for j in chosen]
        labels = []
        for job in jobs:
            positive, ignore = relative_labels(record['nodules'], ignored.get(uid, []), geometry, job, cfg)
            labels.append(dict(job=list(job), positive=positive, ignored=ignore))
        images = np.empty((count, 5, cfg.patch, cfg.patch), dtype=np.int16)
        scores = np.full(count, -np.inf)
        with torch.inference_mode():
            for begin in range(0, count, 48):
                group = jobs[begin:begin + 48]
                images[begin:begin + len(group)] = np.stack([sample_patch(hu, spacing, j, cfg) for j in group])
                tensor = torch.from_numpy(images[begin:begin + len(group)]).cuda().float().clamp(-1000, 400).add_(1000).div_(1400)
                probability = net(tensor)[:, 0].sigmoid().cpu().numpy()
                for j, p in enumerate(probability, begin):
                    available = background_mask(labels[j], cfg)
                    if available.any():
                        scores[j] = float(p[available].max())
        hard, random_ids = select_patches(scores, plan['mining']['hard_patches_per_scan'],
                                         plan['mining']['random_patches_per_scan'], rng)
        selected = hard + random_ids
        np.save(path, images[selected], allow_pickle=False)
        selected_labels = [dict(labels[j], mining_score=float(scores[j]) if np.isfinite(scores[j]) else None,
                                sampling_kind='hard' if j in hard else 'random') for j in selected]
        meta = dict(series_uid=uid, count=len(selected), labels=selected_labels,
            patch_sha256=sha_file(path), plan_sha256=sha_file(out / 'plan.json'),
            raw_sha256=record['raw_sha256'], teacher_model_sha256=plan['initial_checkpoint_sha256'],
            lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
            sampled_tiles=count, schedule=schedule, hard_patches=len(hard), random_patches=len(random_ids),
            hard_score_mean=float(np.mean(scores[hard])) if hard else None,
            random_score_mean=float(np.mean(scores[random_ids][np.isfinite(scores[random_ids])])) if
                              len(random_ids) and np.isfinite(scores[random_ids]).any() else None,
            elapsed_seconds=perf_counter() - start)
        write_json(meta_path, meta)
        summaries.append(meta)
        print('MINE', i, '/', len(records), 'sampled', count, 'kept', len(selected),
              'seconds', round(meta['elapsed_seconds'], 2), flush=True)
        del hu, body, search, images
        gc.collect()
    first.verify(out)
    identity = {str(p): sha_file(p) for uid in plan['fit_scans']
                for p in (folder / (uid + '.json'), folder / (uid + '.npy'))}
    write_json(out / 'mining_freeze.json', dict(status='COMPLETE',
        fit_scans=len(summaries), patches=sum(s['count'] for s in summaries),
        sampled_tiles=sum(s['sampled_tiles'] for s in summaries),
        bytes=sum((folder / (s['series_uid'] + '.npy')).stat().st_size for s in summaries),
        cache_sha256=identity, plan_sha256=sha_file(out / 'plan.json'),
        calibration_scans_mined=0, heldout_scans_run=0, lung_segmentation_rerun=False))


def train(out):
    plan = first.verify(out)
    cfg = config_from_dict(plan['config'])
    torch = setup_torch(plan['seed'])
    from torch.utils.data import Dataset, DataLoader
    if (out / 'model.pt').exists():
        raise FileExistsError('Do not overwrite trained model')
    mining = json.loads((out / 'mining_freeze.json').read_text('utf-8'))
    if mining['plan_sha256'] != sha_file(out / 'plan.json'):
        raise ValueError('Wrong mining plan')
    for path, digest in mining['cache_sha256'].items():
        if sha_file(path) != digest:
            raise ValueError('Changed mining cache')

    class Patches(Dataset):
        def __init__(self):
            self.maps, self.items = [], []
            for folder in (BASE / 'patches', out / 'hard_patches'):
                for uid in plan['fit_scans']:
                    meta = json.loads((folder / (uid + '.json')).read_text('utf-8'))
                    if sha_file(folder / (uid + '.npy')) != meta['patch_sha256']:
                        raise ValueError('Changed training patch')
                    self.maps.append(np.load(folder / (uid + '.npy'), mmap_mode='r', allow_pickle=False))
                    self.items.extend((len(self.maps) - 1, j, label) for j, label in enumerate(meta['labels']))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            k, j, label = self.items[index]
            image = np.array(self.maps[k][j], copy=True)
            positive = [list(p) for p in label['positive']]
            ignored = [list(p) for p in label['ignored']]
            for axis, coordinate in ((2, 0), (1, 1)):
                if random.random() < .5:
                    image = np.flip(image, axis).copy()
                    for p in positive + ignored:
                        p[coordinate] = cfg.patch - 1 - p[coordinate]
            return torch.from_numpy(image), tuple(torch.from_numpy(a) for a in targets(positive, ignored, cfg))

    dataset = Patches()
    loader = DataLoader(dataset, batch_size=plan['training']['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(plan['seed']))
    net = model(cfg).cuda()
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    net.load_state_dict(initial['state'])
    optimizer = torch.optim.Adam(net.parameters(), lr=plan['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=plan['training']['epochs'], eta_min=plan['training']['final_learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    history = []
    start = perf_counter()
    for epoch in range(plan['training']['epochs']):
        net.train()
        total = np.zeros(3)
        batches = 0
        tick = perf_counter()
        for image, target in loader:
            image = image.cuda(non_blocking=True).float().clamp(-1000, 400).add_(1000).div_(1400)
            target = tuple(t.cuda(non_blocking=True) for t in target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                value, classification, regression = loss(net(image).float(), target)
            if not torch.isfinite(value):
                raise ValueError('Nonfinite loss')
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            total += [value.item(), classification.item(), regression.item()]
            batches += 1
        item = dict(epoch=epoch + 1, loss=(total / batches).tolist(),
                    learning_rate=optimizer.param_groups[0]['lr'], seconds=perf_counter() - tick)
        history.append(item)
        scheduler.step()
        write_json(out / 'training_progress.json', dict(status='RUNNING', epochs=history))
        print('TRAIN', epoch + 1, '/', plan['training']['epochs'], 'loss', round(item['loss'][0], 4),
              'seconds', round(item['seconds'], 2), flush=True)
    net.cpu()
    torch.save(dict(state=net.state_dict(), config=plan['config'],
        plan_sha256=sha_file(out / 'plan.json'), epochs=len(history), training_scans=plan['fit_scans'],
        initialized_from_sha256=plan['initial_checkpoint_sha256']), out / 'model.pt')
    first.verify(out)
    write_json(out / 'model_freeze.json', dict(model_sha256=sha_file(out / 'model.pt'),
        plan_sha256=sha_file(out / 'plan.json'), mining_freeze_sha256=sha_file(out / 'mining_freeze.json'),
        fit_scans=len(plan['fit_scans']), epochs=history, total_seconds=perf_counter() - start,
        parameters=sum(p.numel() for p in net.parameters()), training_patches=len(dataset),
        calibration_not_run=True, review_12_not_run=True, heldout_scans_run=0, fpga_validated=False))


def audit(out):
    from lung_candidate_lowfp_audit import check_froc
    from lung_candidates import point_metrics
    plan = first.verify(out)
    report = json.loads((out / 'calibration/report.json').read_text('utf-8'))
    original = json.loads((BASE / 'calibration/report.json').read_text('utf-8'))
    frozen = json.loads((out / 'calibration/freeze.json').read_text('utf-8'))
    if sha_file(out / 'model.pt') != frozen['model_sha256'] or sha_file(out / 'calibration/report.json') != frozen['report_sha256']:
        raise ValueError('Changed calibrated model/report')
    split, records = first.records_for(plan, 'calibration_scans')
    ignored = excluded_findings(EXCLUDED)
    rows, scores, cases = [], {}, []
    gain = lost = references = maximum_hits = 0
    t = next(p['threshold'] for p in report['cnn']['operating_points'] if p['fp_budget'] == 4.)
    old_t = next(p['threshold'] for p in original['cnn']['operating_points'] if p['fp_budget'] == 4.)
    for record in records:
        uid = record['series_uid']
        packet = json.loads((out / 'calibration_scans' / (uid + '.json')).read_text('utf-8'))
        old = json.loads((BASE / 'calibration_scans' / (uid + '.json')).read_text('utf-8'))
        if packet['model_sha256'] != frozen['model_sha256'] or packet['input_raw_sha256'] != record['raw_sha256']:
            raise ValueError('Wrong inference identity')
        world = np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3)
        index = np.asarray([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1, 3)
        np.testing.assert_allclose(world, index @ np.asarray(record['basis_lps_zyx']).T + record['origin_lps_mm'], atol=1e-6, rtol=0)
        rows.append(dict(uid=uid, record=record, excluded=ignored.get(uid, []), world=world))
        scores[uid] = np.array([c['score'] for c in packet['candidates']])
        chosen = world[scores[uid] >= t] if t is not None else np.empty((0, 3))
        before = [c['center_lps_xyz_mm'] for c in old['candidates'] if old_t is not None and c['score'] >= old_t]
        m = point_metrics(chosen, record['nodules'], ignored.get(uid, []))
        previous = point_metrics(before, record['nodules'], ignored.get(uid, []))
        for a, b in zip(m['reference_details'], previous['reference_details']):
            gain += int(a['detected'] and not b['detected'])
            lost += int(b['detected'] and not a['detected'])
        references += len(record['nodules'])
        maximum_hits += point_metrics(world, record['nodules'], ignored.get(uid, []))['detected']
        cases.append(dict(series_uid=uid, v2=m, v1=previous))
    check_froc(rows, scores, report['cnn'])
    import torch
    ckpt = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    if (set(ckpt['training_scans']) != set(plan['fit_scans']) or set(plan['fit_scans']) & set(plan['calibration_scans']) or
            ckpt['plan_sha256'] != sha_file(out / 'plan.json') or not all(torch.isfinite(v).all() for v in ckpt['state'].values())):
        raise ValueError('Invalid training provenance')
    write_json(out / 'comparison.json', dict(status='COMPLETE_DEVELOPMENT_CALIBRATION_NOT_PROMOTED',
        scans=len(records), reference_nodules=references, v1=original['cnn'], v2=report['cnn'],
        svm_same_scans=report['svm_same_scans'], gained_at4=gain, lost_at4=lost,
        maximum_candidate_hits=maximum_hits, cases=cases, independent_froc_verified=True,
        geometry_verified=True, protected_files_unchanged=len(plan['protected_sha256']),
        model_sha256=sha_file(out / 'model.pt'), heldout_scans_run=0, review12_run=False,
        architecture_unchanged=True, runtime_speed_rebenchmarked=False, fpga_validated=False))
    print('RESULT', json.dumps(dict(cpm_v1=original['cnn']['cpm_interpolated'],
        cpm_v2=report['cnn']['cpm_interpolated'], gained_at4=gain, lost_at4=lost,
        operating_points=[p for p in report['cnn']['operating_points'] if p['fp_budget'] in (1., 4., 8.)])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'plan', 'mine', 'train', 'infer', 'calibrate', 'audit'))
    parser.add_argument('--outdir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.outdir.resolve()
    if args.command == 'run':
        if not out.exists():
            freeze(out)
        stages = [('mine', 'mining_freeze.json', mine), ('train', 'model_freeze.json', train),
                  ('infer', None, lambda p: first.infer(p, 'calibration_scans')),
                  ('calibrate', 'calibration/freeze.json', first.calibrate), ('audit', 'comparison.json', audit)]
        try:
            for name, marker, operation in stages:
                if marker and (out / marker).exists():
                    continue
                write_json(out / 'pipeline_state.json', dict(status='RUNNING', stage=name))
                setup_torch(first.verify(out)['seed'])
                operation(out)
            write_json(out / 'pipeline_state.json', dict(status='COMPLETE_INTERNAL_CALIBRATION', heldout_scans_run=0))
        except Exception as error:
            write_json(out / 'pipeline_state.json', dict(status='FAILED', stage=name, error=repr(error)))
            raise
    else:
        {'plan': freeze, 'mine': mine, 'train': train,
         'infer': lambda p: first.infer(p, 'calibration_scans'),
         'calibrate': first.calibrate, 'audit': audit}[args.command](out)


if __name__ == '__main__':
    main()
