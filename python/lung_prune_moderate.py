"""One moderate, HWC8-aligned candidate; original TRAIN/development split only.

Old source files, checkpoints and frozen experiments remain untouched. Test/COVID
data are never loaded here, including for checkpoint selection or calibration.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np

from lung_prune import (BASELINE, DATASET, ORIGINAL, WORKSPACE, evaluate,
    export_contract, prune_model, restore_candidate, restore_pruned_quantized,
    workload)
from lung_fixed_network import FixedUNet
from lung_int8_final import final_masks, score_mask, summarize_rows
from lung_quantize import (calibrate_training, convert_static, export_core,
    fold_batchnorm, make_quantizable, prepare_static)
from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, _case_prediction, context_batch, file_sha256,
    load_cached_source, make_model, mapped_npy, read_manifest, segmentation_loss,
    torch_module)
from verify_lung_axil import check_hashes

WIDTHS = {'enc3.0': 24, 'neck.0': 48, 'neck.3': 48,
          'dec3.0': 24, 'dec3.3': 24}
NAME = 'moderate'
SEED = 20261019


def development_rank(rows, summary, baseline):
    """Pair every development patient before ranking; never hide a case loss."""
    old = {r['case']: r['scores']['fp32_final'] for r in baseline['cases']}
    if len(rows) != len(old) or {r['case'] for r in rows} != set(old):
        raise ValueError('Missing, duplicate or extra development cases')
    deltas = [r['scores']['final']['metrics']['dice'] -
              old[r['case']]['metrics']['dice'] for r in rows]
    base = baseline['summaries']['fp32_final']
    quality = summary['slice_quality']; previous = base['slice_quality']
    small_delta = (quality['groups']['small']['mean_slice_dice'] -
                   previous['groups']['small']['mean_slice_dice'])
    passed = bool(summary['mean_case_dice'] >= base['mean_case_dice'] - .01
                  and min(deltas) >= -.01 - 1e-12 and small_delta >= -.03
                  and quality['empty_false_positive_pixels'] <=
                  previous['empty_false_positive_pixels'] + 1000)
    return (passed, summary['mean_case_dice']), dict(
        passes_development_selection=passed, worst_case_dice_delta=min(deltas),
        small_slice_mean_dice_delta=small_delta,
        empty_false_positive_pixels=quality['empty_false_positive_pixels'],
        per_case_dice_delta={r['case']: d for r, d in zip(rows, deltas)})


def run(outdir, epochs=8):
    if not 1 <= epochs <= 8:
        raise ValueError('Bounded experiment: one to eight epochs')
    torch = torch_module(); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise ValueError('Explicit CUDA experiment; no CPU fallback')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cache, manifest = read_manifest(DATASET)
    train = [r for r in manifest['cases'] if r['split'] == 'train']
    validation = [r for r in manifest['cases'] if r['split'] == 'validation']
    if len(train) != 18 or len(validation) != 6:
        raise ValueError('Need frozen original 18/6 patient split')
    baseline = json.loads((BASELINE/'evaluation.json').read_text(encoding='utf-8'))
    protected = dict(baseline['protocol']['input_sha256'])
    sampling = []; sampling_rows = []
    for record in train:
        for name in ['images.npy', 'labels.npy', 'reference_native.npy']:
            path = cache/record['cache_folder']/name
            protected[str(path)] = file_sha256(path)
        with mapped_npy(cache/record['cache_folder']/'reference_native.npy') as ref:
            area = (ref != 0).sum((1, 2))
            weights = np.where(area == 0, 2., np.where(area <= 1000, 4., 1.))
        sampling.extend(weights.tolist())
        sampling_rows.append(dict(case=record['case'], empty=int((area == 0).sum()),
            small=int(((area > 0) & (area <= 1000)).sum())))
    for path in [ORIGINAL, DATASET, BASELINE/'evaluation.json']:
        protected[str(path)] = file_sha256(path)
    check_hashes(protected)
    original = make_model(8, 1).eval()
    original.load_state_dict(torch.load(ORIGINAL, map_location='cpu',
                                      weights_only=True)['model'])
    model, kept = prune_model(original, WIDTHS)
    original_work = workload(original); work = workload(model)
    output = fresh_output(outdir).resolve(); folder = output/NAME; folder.mkdir()
    source_files = [Path(__file__).resolve()] + [WORKSPACE/'python'/n for n in
        ['lung_prune.py', 'lung_prune_recover.py', 'lung_unet.py', 'lung_quantize.py',
         'lung_int8_final.py', 'lung_fixed_network.py', 'lung_body_gate.py',
         'lung_island_cleanup.py']]
    protocol = dict(status='predeclared_moderate_pruning_development_protocol',
        source_checkpoint_sha256=file_sha256(ORIGINAL), seed=SEED, epochs=epochs,
        candidates={NAME: WIDTHS}, learning_rate=.00005, batch_size=12,
        sampling=dict(empty_weight=2., small_weight=4., other_weight=1.,
            native_small_area_max=1000, rows=sampling_rows,
            draws_per_epoch=len(sampling), with_replacement=True),
        loss='BCE+softDice + 0.5 teacher BCE + 0.1 teacher-logit SmoothL1(beta=1)',
        bn_running_stats='frozen; affine parameters trainable',
        checkpoint_selection='Prefer all development gates, then highest FP32 final mean Dice; includes epoch zero',
        max_mean_dice_drop=.01, max_each_case_dice_drop=.01,
        max_small_slice_dice_drop=.03, max_extra_empty_fp_pixels=1000,
        final_acceptance='Exact INT8 each cohort mean and each case Dice loss <=0.01; inspect small/empty errors; real speed separately',
        min_structural_work_reduction=.10, desired_board_latency_reduction=.15,
        train_cases=[r['case'] for r in train],
        validation_cases=[r['case'] for r in validation],
        validation_for_selection=True, test_or_covid_used=False,
        threshold=.5, image_size=256, input_sha256=protected,
        source_sha256={str(p): file_sha256(p) for p in source_files},
        warning='Architecture proposed after earlier test review; iterative development, not an untouched blind trial')
    write_json(output/'protocol.json', protocol)
    write_json(output/'baseline_workload.json', original_work)
    write_json(folder/'workload.json', work); write_json(folder/'kept.json', kept)
    teacher = copy.deepcopy(original).cuda().eval(); del original
    hu = {r['case']: load_cached_source(r) for r in validation}
    saved = dict(schema='lung_pruned_fp32_v1', widths=WIDTHS, kept=kept,
        source_checkpoint_sha256=protocol['source_checkpoint_sha256'],
        dataset_fingerprint=manifest['fingerprint'])
    model.cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.00005)
    scaler = torch.amp.GradScaler('cuda'); history = []; best_rank = (False, -float('inf'))
    generator = torch.Generator().manual_seed(SEED)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sampling, dtype=torch.double),
        len(sampling), replacement=True, generator=generator)
    with SliceDataset(cache, train, augment=True) as dataset:
        loader = torch.utils.data.DataLoader(dataset, batch_size=12, sampler=sampler,
            num_workers=0, pin_memory=True)
        for epoch in range(epochs + 1):
            start = time.perf_counter(); total_loss = 0.; seen = 0
            if epoch:
                model.train()
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d): module.eval()
                for image, target in loader:
                    image = image.cuda(non_blocking=True); target = target.cuda(non_blocking=True)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                        teacher_logits = teacher(image)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(image)
                        loss = segmentation_loss(logits, target)
                        loss = loss + .5*torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, torch.sigmoid(teacher_logits))
                        loss = loss + .1*torch.nn.functional.smooth_l1_loss(logits, teacher_logits, beta=1.)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    scaler.step(optimizer); scaler.update()
                    total_loss += loss.detach().item()*len(image); seen += len(image)
            model.eval(); rows = []
            for record in validation:
                raw, _ = _case_prediction(model, cache, record, 'cuda', 8)
                _, final, _, _, _ = final_masks(hu[record['case']], raw, record['spacing_zyx_mm'])
                with mapped_npy(cache/record['cache_folder']/'reference_native.npy') as ref:
                    rows.append(dict(case=record['case'], scores={'final': score_mask(final, ref)}))
            summary = summarize_rows(rows, ['final'])['final']
            rank, gate = development_rank(rows, summary, baseline)
            history.append(dict(epoch=epoch, loss=total_loss/seen if seen else None,
                wall_seconds=time.perf_counter()-start, summary=summary, cases=rows, **gate))
            write_json(folder/'training_history.json', history)
            if rank > best_rank:
                best_rank = rank
                torch.save(dict(saved, epoch=epoch,
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()}), folder/'best.pt')
            print('MODERATE epoch', epoch, 'Dice', summary['mean_case_dice'],
                'worst', gate['worst_case_dice_delta'], 'small', gate['small_slice_mean_dice_delta'],
                'emptyFP', gate['empty_false_positive_pixels'], 'gate', rank[0],
                'seconds', round(history[-1]['wall_seconds'], 1), flush=True)
    del hu, model, teacher, optimizer, scaler; torch.cuda.empty_cache()
    model, saved = restore_candidate(folder/'best.pt'); model.cuda().eval()
    write_json(folder/'fp32_evaluation.json', evaluate(model, cache, validation, 'cuda', folder, 'fp32'))
    prepared = prepare_static(fold_batchnorm(make_quantizable(model.cpu().eval())))
    calibration = calibrate_training(prepared, model, cache, train, 'cuda', 8, 256)
    write_json(folder/'calibration.json', calibration)
    quantized = convert_static(prepared); core = export_core(quantized, 256, folder)
    torch.save(dict(schema='lung_pruned_ptq_v1', widths=WIDTHS, kept=kept,
        state_dict=quantized.state_dict(), source_checkpoint_sha256=file_sha256(folder/'best.pt'),
        original_checkpoint_sha256=protocol['source_checkpoint_sha256'],
        dataset_fingerprint=manifest['fingerprint']), folder/'int8_model.pt')
    restored, _ = restore_pruned_quantized(folder/'int8_model.pt')
    with mapped_npy(cache/train[0]['cache_folder']/'images.npy') as images, torch.inference_mode():
        probe = torch.from_numpy(context_batch(images, [0, len(images)//2, len(images)-1], 1))
        torch.testing.assert_close(restored(probe), quantized(probe), rtol=0, atol=0)
    contract = export_contract(core, folder, file_sha256(folder/'best.pt'))
    del restored, prepared, model; torch.cuda.empty_cache()
    write_json(folder/'ptq_evaluation.json', evaluate(quantized, cache, validation, 'cpu', folder, 'ptq'))
    del quantized
    with np.load(folder/'parameters.npz') as parameters:
        fixed = FixedUNet(contract, parameters, 'cuda')
    report = evaluate(fixed, cache, validation, 'cpu', folder, 'fixed')
    write_json(folder/'fixed_evaluation.json', report)
    dice = report['summaries']['fixed_final']['mean_case_dice']
    base = baseline['summaries']['fixed_final']['mean_case_dice']
    pairs = [dict(case=r['case'], dice_delta=r['scores']['fixed_final']['metrics']['dice'] -
        next(b for b in baseline['cases'] if b['case'] == r['case'])['scores']['fixed_final']['metrics']['dice'])
        for r in report['cases']]
    reduction = 1-work['padded_input_beats']/original_work['padded_input_beats']
    result = dict(candidate=NAME, best_epoch=saved['epoch'], mean_case_dice=dice,
        baseline_mean_case_dice=base, mean_dice_drop=base-dice, per_case=pairs,
        padded_work_reduction=reduction,
        passes_mean_and_work_screen=dice >= base-.01 and reduction >= .10,
        passes_each_case_dice_screen=min(r['dice_delta'] for r in pairs) >= -.01 - 1e-12,
        boundary_and_small_slice_review_required=True, board_tested=False)
    write_json(folder/'result.json', result); write_json(output/'results.json', [result])
    check_hashes(protected); check_hashes(protocol['source_sha256'])
    write_json(output/'verification.json', dict(status='complete_pruning_development_experiment',
        original_inputs_unchanged=True, protocol_sha256=file_sha256(output/'protocol.json'),
        results=[result], test_or_covid_used=False, board_tested=False))
    print('MODERATE RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', default='data/lung_prune_moderate_20261017')
    p.add_argument('--epochs', type=int, default=8)
    a = p.parse_args(); run(a.outdir, a.epochs)
