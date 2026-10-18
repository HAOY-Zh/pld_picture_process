"""Single-slice, soft coarse-level context experiment; CPU/GPU, not FPGA inference.

Coarse labels are relative thirds of the EXPERT lung extent, not clinical
anatomical landmarks. They supervise training and stratify validation only.
Inference sees one normalized CT image, never a level/position/reference mask.
"""
import argparse
from bisect import bisect_right
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pydicom

from lung_body_crop import check_hashes
from lung_outer_contour import OuterValidator
from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, _case_prediction, audit_context_geometry,
    context_batch, file_sha256, load_cached_source, make_model, mapped_npy,
    model_summary, read_manifest, resize_stack, save_checkpoint, save_learned_preview,
    segmentation_loss, select_device, torch_module)
from lungseg.metrics import overlap_metrics

LEVELS = ['no_reference_lung', 'relative_lower', 'relative_middle', 'relative_upper']


def coarse_tags(reference, z_mm):
    """Annotation-derived targets, NOT an inference function or a lung-lobe label."""
    reference = np.asarray(reference)
    z = np.asarray(z_mm, dtype=float)
    if (reference.ndim != 3 or not len(reference) or z.shape != (len(reference),)
            or not np.isfinite(z).all() or not np.isin(reference, [0, 1, 255]).all()):
        raise ValueError('Need binary ZHW expert masks and finite physical z coordinates')
    if len(z) > 1 and not (np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)):
        raise ValueError('Physical z order must be strictly monotonic')
    positive = np.any(reference != 0, axis=(1, 2))
    tags = np.zeros(len(z), dtype=np.int64)
    if positive.any():
        low, high = float(z[positive].min()), float(z[positive].max())
        if high == low:
            tags[positive] = 2  # One lung-containing plane: level is ambiguous.
        else:
            relative = (z[positive] - low) / (high - low)
            tags[positive] = 1 + np.minimum((relative * 3).astype(int), 2)
    return tags


def make_layer_model(base_channels=8, soft_context=True):
    torch = torch_module()
    parent = type(make_model(base_channels, 1))

    class LayerUNet(parent):
        def __init__(self):
            super().__init__()
            self.soft_context = soft_context
            channels = 8 * base_channels
            self.level_head = torch.nn.Linear(channels, 4)
            self.context_film = torch.nn.Linear(4, channels * 2)
            # Preserve the exact retained segmentation function at initialization.
            torch.nn.init.zeros_(self.context_film.weight)
            torch.nn.init.zeros_(self.context_film.bias)

        def forward(self, image, return_level=False):
            e1 = self.enc1(image)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            x = self.neck(self.pool(e3))
            level_logits = self.level_head(x.mean((2, 3)))
            if self.soft_context:
                # No argmax, no GT routing and no 'none -> erase mask' rule.
                gamma, beta = self.context_film(level_logits.softmax(1)).chunk(2, 1)
                x = x * (1 + gamma.tanh()[:, :, None, None]) + beta.tanh()[:, :, None, None]
            for skip, decoder in ((e3, self.dec3), (e2, self.dec2), (e1, self.dec1)):
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='nearest')
                x = decoder(torch.cat((x, skip), 1))
            segmentation = self.head(x)
            return (segmentation, level_logits) if return_level else segmentation

    return LayerUNet()


def initialize(model, original_state):
    result = model.load_state_dict(original_state, strict=False)
    expected = {'level_head.weight', 'level_head.bias', 'context_film.weight', 'context_film.bias'}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError('Initialization must match every retained segmentation parameter')


def predict_slice(model, hu, device, size=256):
    """Unknown single CT slice -> native mask and soft context; no labels/order required."""
    torch = torch_module()
    image = np.asarray(hu)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all() or size < 32 or size % 8:
        raise ValueError('Need a finite 2-D HU image and network size divisible by 8')
    normalized = (resize_stack(np.clip(image, -1000, 400)[None], size, 1) + 1000) / 1400
    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(normalized[:, None], device=device, dtype=torch.float32)
        logits, levels = model(tensor, return_level=True)
        probability = torch.nn.functional.interpolate(logits.sigmoid(), size=image.shape,
                                                       mode='bilinear', align_corners=False)
    return dict(mask=(probability[0, 0] >= .5).cpu().numpy(),
                level_probability=levels.softmax(1)[0].cpu().numpy())


class LevelDataset(SliceDataset):
    def __init__(self, root, records, tags, augment=True):
        if any(r['split'] != 'train' or not r['case'].startswith('LCTSC-Train-') for r in records):
            raise ValueError('Training accepts only original TRAIN patients, never validation/Test')
        super().__init__(root, records, augment=augment, context_slices=1)
        self.tags = {}
        for r in records:
            values = np.asarray(tags[r['case']], dtype=np.int64)
            if values.shape != (r['slices'],) or not np.isin(values, range(4)).all():
                raise ValueError('Coarse targets must match this training case')
            self.tags[r['case']] = values

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        case_index = bisect_right(self.ends, index)
        local = index - (self.ends[case_index - 1] if case_index else 0)
        target = int(self.tags[self.records[case_index]['case']][local])
        return image, label, torch_module().tensor(target, dtype=torch_module().long)


def prepare_tags(root, records):
    audit = audit_context_geometry(records)
    tags = {}
    for r in records:
        headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in r['source_ct_files_sorted']]
        orientation = np.asarray(headers[0].ImageOrientationPatient, float)
        normal = np.cross(orientation[:3], orientation[3:])
        if abs(normal[2]) < .99:
            raise ValueError('Relative superior/inferior experiment requires nearly axial LPS planes')
        z = [float(ds.ImagePositionPatient[2]) for ds in headers]
        with mapped_npy(root / r['cache_folder'] / 'reference_native.npy') as reference:
            tags[r['case']] = coarse_tags(reference, z).tolist()
    return tags, audit


def training_class_weights(records, tags):
    counts = np.bincount(np.concatenate([tags[r['case']] for r in records]), minlength=4)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights /= weights.mean()
    return counts.tolist(), weights.astype(np.float32)


def selection_key(summary):
    """Predeclared contour selection; NO previous 0.5% Dice floor."""
    outer = summary['outer']
    return (outer['main_missing_foreground'], outer['mean_case_bad5_fraction'],
            outer['worst5_percent_bad5_mean'], -summary['mean_case_dice'])


def level_diagnostics(model, root, records, tags, device, batch_size, rows, output=None):
    torch = torch_module()
    model.eval()
    confusion = np.zeros((4, 4), np.int64)
    per_case, grouped = [], {i: [] for i in range(4)}
    for r, row in zip(records, rows):
        if r['case'] != row['case']:
            raise ValueError('Validation rows reordered')
        probability = []
        with mapped_npy(root / r['cache_folder'] / 'images.npy') as images, torch.inference_mode():
            for begin in range(0, len(images), batch_size):
                centers = np.arange(begin, min(begin + batch_size, len(images)))
                image = torch.from_numpy(context_batch(images, centers, 1)).to(device)
                _, levels = model(image, return_level=True)
                probability.extend(levels.softmax(1).cpu().numpy().tolist())
        probability = np.asarray(probability)
        target = np.asarray(tags[r['case']])
        guessed = probability.argmax(1)
        matrix = np.zeros((4, 4), np.int64)
        np.add.at(matrix, (target, guessed), 1)
        confusion += matrix
        item = dict(case=r['case'], accuracy=float(np.mean(target == guessed)),
                    confusion=matrix.tolist(), reference_level=target.tolist(),
                    predicted_probability=probability.tolist(), groups={})
        # All segmentation metrics below use actual predictions, not level guesses.
        prediction, _ = _case_prediction(model, root, r, device, batch_size)
        with mapped_npy(root / r['cache_folder'] / 'reference_native.npy') as reference:
            for level, name in enumerate(LEVELS):
                selected = target == level
                if not selected.any():
                    continue
                metrics = overlap_metrics(prediction[selected], reference[selected])
                main = [s for s, include in zip(row['outer_slices'], selected)
                        if include and s['reference_pixels'] > 1000]
                group = dict(slices=int(selected.sum()), metrics=metrics,
                    main_bad5=float(np.mean([s['bad5_fraction'] for s in main])) if main else None,
                    main_slices=len(main))
                item['groups'][name] = group
                grouped[level].append(group)
        if output:
            folder = Path(output) / r['case']
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'prediction_native.npz', mask=prediction)
        per_case.append(item)
    totals = confusion.sum(1)
    summary = dict(classes=LEVELS, confusion_rows_reference_columns_prediction=confusion.tolist(),
        accuracy=float(np.trace(confusion) / confusion.sum()),
        per_level_recall=[float(confusion[i, i] / totals[i]) if totals[i] else None for i in range(4)],
        by_reference_level={})
    for level, name in enumerate(LEVELS):
        groups = grouped[level]
        contour = [g['main_bad5'] for g in groups if g['main_bad5'] is not None]
        summary['by_reference_level'][name] = dict(slices=sum(g['slices'] for g in groups),
            mean_case_dice=float(np.mean([g['metrics']['dice'] for g in groups])) if groups and level else None,
            false_negative=sum(g['metrics']['false_negative'] for g in groups),
            false_positive=sum(g['metrics']['false_positive'] for g in groups),
            mean_case_main_bad5=float(np.mean(contour)) if contour else None)
    return dict(summary=summary, cases=per_case,
        warning='GT-derived coarse labels only evaluate classification/stratify errors; they NEVER enter forward(). '
                'Relative lung-extent thirds are not verified clinical body landmarks.')


def previews(root, manifest, output):
    saved = []
    for case, index in [('LCTSC-Train-S2-006', 39), ('LCTSC-Train-S2-006', 54),
                        ('LCTSC-Train-S3-008', 79), ('LCTSC-Train-S3-008', 72)]:
        records = [r for r in manifest['cases'] if r['case'] == case]
        if not records or index >= records[0]['slices']:
            continue
        record = records[0]
        hu = load_cached_source(record)[index]
        with mapped_npy(root / record['cache_folder'] / 'reference_native.npy') as expert:
            reference = expert[index] != 0
        with np.load(output / 'baseline' / case / 'prediction_native.npz') as archive:
            original = archive['mask'][index]
        for arm in ['auxiliary_only', 'soft_context']:
            for kind in ['best_contour', 'best_dice']:
                with np.load(output / arm / kind / case / 'prediction_native.npz') as archive:
                    prediction = archive['mask'][index]
                path = output / f'{arm}_{kind}_{case}_slice_{index:03d}.png'
                save_learned_preview(hu, reference, prediction, original, path, case, index,
                    baseline_title='Retained original (yellow)', learned_title=f'{arm}/{kind} (green), CPU/GPU')
                saved.append(str(path))
    return saved


def paired(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or
            not 0 < args.lr < 1 or not 0 < args.level_weight <= 1):
        raise ValueError('Invalid bounded experiment settings')
    root, manifest = read_manifest(args.dataset)
    torch = torch_module()
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (original['dataset_fingerprint'] != manifest['fingerprint'] or original.get('context_slices', 1) != 1 or
            original['image_size'] != manifest['image_size'] or original['hu_clip'] != [-1000, 400] or
            original['threshold'] != .5):
        raise ValueError('Original full-field single-slice checkpoint/data required')
    train = [r for r in manifest['cases'] if r['split'] == 'train']
    val = [r for r in manifest['cases'] if r['split'] == 'validation']
    paths = [args.dataset, args.checkpoint, __file__]
    paths += [Path(__file__).with_name(n) for n in ['lung_unet.py', 'lung_outer_contour.py', 'lung_body_crop.py']]
    for r in manifest['cases']:
        paths += [root / r['cache_folder'] / n for n in ['images.npy', 'labels.npy', 'reference_native.npy']]
        paths += [Path(p) for p in r['source_ct_files_sorted']]
    tracked = {str(Path(p).resolve()): file_sha256(p) for p in paths}
    tags, geometry = prepare_tags(root, manifest['cases'])
    counts, weights = training_class_weights(train, tags)
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial = make_layer_model(original['base_channels'], True)
    initialize(initial, original['model'])
    initial_state = copy.deepcopy(initial.state_dict())
    architecture = model_summary(initial, manifest['image_size'])
    channels, spatial = 8 * original['base_channels'], manifest['image_size'] // 8
    architecture.update(extra_parameters=channels * 4 + 4 + 4 * channels * 2 + channels * 2,
        linear_MACs_per_slice=channels * 4 + 4 * channels * 2,
        context_feature_multiplies=channels * spatial ** 2, context_feature_adds=channels * spatial ** 2,
        global_pool_inputs=channels * spatial ** 2,
        extra_operators=['global average pool', '4-class softmax', '128 tanh', 'soft FiLM'],
        hardware_status='Computer experiment only; INT8/softmax/tanh RTL, BRAM/DDR/timing not validated')
    output = fresh_output(args.outdir)
    protocol = dict({k: v for k, v in vars(args).items() if k != 'func'}, input_sha256=tracked,
        architecture=architecture, train_cases=[r['case'] for r in train], validation_cases=[r['case'] for r in val],
        held_out_test_cases=manifest.get('held_out_test_cases', []), train_class_counts=counts,
        train_class_weights=weights.tolist(), geometry=geometry,
        labels='0=no expert lung; 1/2/3=physical LPS-z thirds of each reference lung extent. '
               'Train tags supervise only TRAIN. Validation tags used only for diagnostics, not inputs/loss.',
        inference='One normalized full-frame CT slice. Predicted probabilities modulate shared features; '
                  'NO provided level, slice index, GT, neighbor images, hard routing or forced empty mask.',
        paired_difference='Both arms have identical initial network weights/level supervision/order/flips. '
                          'Auxiliary-only disables FiLM; soft-context enables predicted-probability FiLM.',
        loss='Original BCE+Dice + 0.1 weighted coarse CE by default; no teacher or outer-distance loss in either arm.',
        selection='Save best native mean-patient Dice AND best outer-contour rank separately. '
                  'Contour rank: missing main slices, bad5, worst5 tail, negative Dice. No 0.5% gate.',
        warning='Exploratory repeated use of the same six validation patients; not a new blind test. '
                'Original model never overwritten. No Test/COVID data accessed; CPU/GPU only.')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'coarse_labels.json', dict(classes=LEVELS, cases=tags, definition=protocol['labels']))
    report = dict(status='running', protocol=protocol, arms={}, original_model_retained=True)
    write_json(output / 'evaluation.json', report)
    validator = OuterValidator(root, val)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        baseline = make_model(original['base_channels'], 1).to(device)
        baseline.load_state_dict(original['model'])
        rows, summary = validator.evaluate(baseline, device, args.batch_size)
        for r in val:
            prediction, _ = _case_prediction(baseline, root, r, device, args.batch_size)
            folder = output / 'baseline' / r['case']
            folder.mkdir(parents=True)
            np.savez_compressed(folder / 'prediction_native.npz', mask=prediction)
        report.update(baseline=summary, baseline_cases=rows)
        write_json(output / 'evaluation.json', report)
        del baseline
        class_weight = torch.as_tensor(weights, device=device)
        for arm in ['auxiliary_only', 'soft_context']:
            model = make_layer_model(original['base_channels'], arm == 'soft_context').to(device)
            model.load_state_dict(initial_state)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            folder = output / arm
            folder.mkdir()
            history, best_contour, best_dice = [], None, -float('inf')
            with LevelDataset(root, train, tags) as dataset:
                for epoch in range(1, args.epochs + 1):
                    torch.manual_seed(args.seed + epoch)
                    random.seed(args.seed + epoch)
                    np.random.seed(args.seed + epoch)
                    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(args.seed + epoch))
                    start, total, ce_total, correct, count = perf_counter(), 0., 0., 0, 0
                    model.train()
                    for step, batch in enumerate(loader, 1):
                        images, labels, level = [x.to(device) for x in batch]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                            logits, level_logits = model(images, return_level=True)
                            ce = torch.nn.functional.cross_entropy(level_logits.float(), level, weight=class_weight)
                            loss = segmentation_loss(logits.float(), labels) + args.level_weight * ce
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        scaler.step(optimizer)
                        scaler.update()
                        total += float(loss.detach()) * len(images)
                        ce_total += float(ce.detach()) * len(images)
                        correct += int((level_logits.argmax(1) == level).sum())
                        count += len(images)
                        if step % 100 == 0:
                            print(f'{arm} epoch {epoch}: batch {step}/{len(loader)}', flush=True)
                    rows, summary = validator.evaluate(model, device, args.batch_size)
                    # Cheap classification/stratification only at selected checkpoints below.
                    history.append(dict(epoch=epoch, loss=total / count, coarse_ce=ce_total / count,
                        train_coarse_accuracy=correct / count, summary=summary, cases=rows,
                        seconds_including_validation=perf_counter() - start))
                    payload = {k: copy.deepcopy(original[k]) for k in ['base_channels', 'image_size', 'hu_clip', 'threshold']}
                    payload.update(model=model.state_dict(), epoch=epoch, context_slices=1,
                        dataset_fingerprint=manifest['fingerprint'], kind='soft_level_v1', soft_context=model.soft_context,
                        level_names=LEVELS, history=history, protocol_file=str(output / 'protocol.json'))
                    save_checkpoint(folder / 'last.pt', payload)
                    key = selection_key(summary)
                    if best_contour is None or key < best_contour:
                        best_contour = key
                        save_checkpoint(folder / 'best_contour.pt', payload)
                    if summary['mean_case_dice'] > best_dice:
                        best_dice = summary['mean_case_dice']
                        save_checkpoint(folder / 'best_dice.pt', payload)
                    write_json(folder / 'history.json', history)
                    print(f'{arm} epoch {epoch}: Dice={summary["mean_case_dice"]:.6f}, '
                          f'outer bad5={summary["outer"]["mean_case_bad5_fraction"]:.6f}, '
                          f'train level accuracy={correct / count:.3f}', flush=True)
                selected = {}
                for kind in ['best_contour', 'best_dice']:
                    checkpoint = torch.load(folder / f'{kind}.pt', map_location='cpu', weights_only=True)
                    model.load_state_dict(checkpoint['model'])
                    rows, summary = validator.evaluate(model, device, args.batch_size)
                    diagnostics = level_diagnostics(model, root, val, tags, device, args.batch_size, rows, folder / kind)
                    selected[kind] = dict(epoch=checkpoint['epoch'], cases=rows, summary=summary, levels=diagnostics)
                report['arms'][arm] = dict(history=history, selected=selected)
                write_json(output / 'evaluation.json', report)
            del model, optimizer
        report['previews'] = previews(root, manifest, output)
        check_hashes(tracked)
        report.update(status='complete_paired_layer_context_screen', protected_inputs_unchanged=True,
                      decision='exploratory_only_original_retained')
        write_json(output / 'evaluation.json', report)
        print('Layer-context screen complete. Original model retained.', flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete', error=str(error))
        write_json(output / 'evaluation.json', report)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', default='data/lung_learning_20261009/dataset.json')
    result.add_argument('--checkpoint', default='data/lung_training_20261011/early_stop_run/best.pt')
    result.add_argument('--outdir', required=True)
    result.add_argument('--epochs', type=int, default=6)
    result.add_argument('--batch-size', type=int, default=8)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--seed', type=int, default=20261013)
    result.add_argument('--lr', type=float, default=.0001)
    result.add_argument('--level-weight', type=float, default=.1)
    result.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    result.set_defaults(func=paired)
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    args.func(args)
