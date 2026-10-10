"""Small 2-D U-Net development: cache -> patient split -> train -> evaluate.

Computer CPU/GPU only. No FPGA inference or clinical use is claimed.
The prepare command works without PyTorch; all learned-model commands need it.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pydicom
from scipy import ndimage as ndi

from lung_segmentation import fresh_output, write_json
from lungseg.baseline import segment_volume
from lungseg.dicom_io import discover, load_ct, rasterize_lungs, referenced_series
from lungseg.metrics import overlap_metrics


DEVELOPMENT_SEEN = {'LCTSC-Train-S2-001', 'LCTSC-Train-S3-001'}


def patient_split(names, seed=20261009, validation_per_site=3):
    """Stratify case folders by institution; previously inspected cases stay in train."""
    if len(set(names)) != len(names) or any(not n.startswith('LCTSC-Train-') for n in names):
        raise ValueError('Split accepts unique official training cases only, never test cases')
    groups = defaultdict(list)
    for name in names:
        groups[name.split('-')[2]].append(name)
    rng, result = random.Random(seed), {}
    for site in sorted(groups):
        candidates = sorted(set(groups[site]) - DEVELOPMENT_SEEN)
        if validation_per_site < 1 or len(candidates) <= validation_per_site:
            raise ValueError(f'Not enough independent train/validation patients at {site}')
        rng.shuffle(candidates)
        validation = set(candidates[:validation_per_site])
        for name in sorted(groups[site]):
            result[name] = 'validation' if name in validation else 'train'
    return result


def resize_stack(stack, size, order):
    """Pixel-center resize, no cropping. Masks use nearest; HU uses linear."""
    stack = np.asarray(stack)
    if stack.ndim != 3 or size < 16 or order not in (0, 1):
        raise ValueError('Need a 3-D stack, size >=16 and nearest/linear interpolation')
    rows = (np.arange(size) + .5) * stack.shape[1] / size - .5
    cols = (np.arange(size) + .5) * stack.shape[2] / size - .5
    coordinates = np.array(np.meshgrid(rows, cols, indexing='ij'))
    resized = np.empty((len(stack), size, size), dtype=np.float32)
    for i, image in enumerate(stack):
        resized[i] = ndi.map_coordinates(image.astype(np.float32), coordinates,
                                        order=order, mode='nearest', prefilter=False)
    return resized


def _load_source(name, folder, prepared_root=None):
    series, structures, _ = discover(folder)
    if len(series) != 1 or len(structures) != 1:
        raise ValueError('Need one CT series and one RTSTRUCT')
    uid, entries = next(iter(series.items()))
    rt = structures[0][1]
    if uid not in referenced_series(rt):
        raise ValueError('RTSTRUCT series association does not match')
    cached = Path(prepared_root) / name / 'prepared' / 'case.npz' if prepared_root else None
    if cached and cached.is_file():
        with np.load(cached, allow_pickle=False) as item:
            if str(item['series_uid']) != uid:
                raise ValueError('Existing prepared case belongs to another CT series')
            hu = item['hu']
            reference = (item['reference_left'] != 0) | (item['reference_right'] != 0)
            overlap = int(np.count_nonzero((item['reference_left'] != 0) & (item['reference_right'] != 0)))
            spacing, sop_uids = item['spacing_zyx'], tuple(item['sop_uids'].tolist())
            frame_uid = str(item['frame_uid'])
    else:
        volume = load_ct(entries)
        # This task predicts the UNION. Keep both source labels unchanged even
        # when their rasterized boundaries overlap; record the overlap visibly.
        masks, info = rasterize_lungs(volume, rt, allow_overlap=True)
        hu, reference = volume.hu, masks['left'] | masks['right']
        spacing, sop_uids, frame_uid = volume.spacing, volume.sop_uids, volume.frame_uid
        overlap = info['overlap_check']['voxels']
    path_map = {str(ds.SOPInstanceUID): str(path.resolve()) for path, ds in entries}
    if set(sop_uids) != set(path_map):
        raise ValueError('Prepared volume and downloaded CT slice set differ')
    patient_ids = {str(ds.PatientID) for _, ds in entries}
    if len(patient_ids) != 1 or not next(iter(patient_ids)).strip():
        raise ValueError('Need one nonempty anonymized PatientID per case')
    patient_key = hashlib.sha256(next(iter(patient_ids)).encode()).hexdigest()
    return hu, reference, spacing, [path_map[uid] for uid in sop_uids], patient_key, frame_uid, overlap


def prepare_cache(args):
    if args.size < 32 or args.size % 8:
        raise ValueError('Network size must be >=32 and divisible by 8')
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    names = sorted(audit['available_train_cases'])
    splits = patient_split(names, args.seed, args.validation_per_site)
    existing = {}
    if args.resume_cache:
        output = Path(args.outdir)
        previous = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
        if previous['seed'] != args.seed or previous['image_size'] != args.size or previous['case_split'] != splits:
            raise ValueError('Resume cache seed/size/patient split differs')
        existing = {r['case']: r for r in previous['cases']}
    else:
        output = fresh_output(args.outdir)
    records, failures, patients, frames = [], [], set(), set()
    for i, name in enumerate(names, 1):
        print(f'[{i}/{len(names)}] Preparing {name} ({splits[name]})', flush=True)
        try:
            if name in existing:
                record = existing[name]
                folder = output / record['cache_folder']
                images = np.load(folder / 'images.npy', mmap_mode='r')
                labels = np.load(folder / 'labels.npy', mmap_mode='r')
                reference = np.load(folder / 'reference_native.npy', mmap_mode='r')
                if images.shape != (record['slices'], args.size, args.size) or labels.shape != images.shape:
                    raise ValueError('Existing cached image/label shapes differ')
                if list(reference.shape) != record['native_shape']:
                    raise ValueError('Existing native reference shape differs')
                if any(not Path(p).is_file() for p in record['source_ct_files_sorted']):
                    raise ValueError('Existing cache source CT files no longer exist')
                if record['patient_key'] in patients or record['frame_uid'] in frames:
                    raise ValueError('Duplicate patient/frame in existing cache')
                patients.add(record['patient_key'])
                frames.add(record['frame_uid'])
                records.append(record)
                del images, labels, reference
                print('  Reusing previously generated case cache', flush=True)
                continue
            hu, reference, spacing, files, patient, frame, overlap = _load_source(
                name, Path(args.input_root) / name, args.prepared_root)
            if patient in patients or frame in frames:
                raise ValueError('Duplicate patient/frame across case folders: split would leak')
            patients.add(patient)
            frames.add(frame)
            folder = output / 'cases' / name
            folder.mkdir(parents=True)
            images = resize_stack(np.clip(hu, -1000, 400), args.size, 1)
            images = ((images + 1000) / 1400).astype(np.float16)
            labels = (resize_stack(reference, args.size, 0) >= .5).astype(np.uint8)
            np.save(folder / 'images.npy', images)
            np.save(folder / 'labels.npy', labels)
            np.save(folder / 'reference_native.npy', reference.astype(np.uint8))
            record = {'case': name, 'split': splits[name], 'patient_key': patient,
                      'frame_uid': frame, 'slices': len(hu), 'native_shape': list(hu.shape),
                      'spacing_zyx_mm': np.asarray(spacing).tolist(),
                      'cache_folder': str(folder.relative_to(output)),
                      'source_ct_files_sorted': files, 'reference_overlap_voxels': overlap}
            records.append(record)
            write_json(folder / 'case.json', record)
            # Reference contours alone, with no predicted labels used for alignment review.
            j = int(np.argmax(reference.sum(axis=(1, 2))))
            # Separate compact alignment plot avoids presenting reference as a prediction.
            save_alignment(hu[j], reference[j], folder / 'reference_alignment.png', name, j)
            if overlap:
                print(f'  NOTE: reference labels overlap at {overlap} voxels; target remains their union.', flush=True)
            del hu, reference, images, labels
        except (ValueError, KeyError, AttributeError, OSError) as error:
            failures.append({'case': name, 'error': str(error), 'assigned_split': splits[name]})
            print(f'  FAILED: {error}', flush=True)
    payload = {'schema': 1, 'status': 'complete' if not failures else 'incomplete',
               'seed': args.seed, 'image_size': args.size, 'hu_clip': [-1000, 400],
               'input_normalization': '(clip(HU,-1000,400)+1000)/1400',
               'target': 'Union of expert left/right lung masks; nearest resize for training only',
               'evaluation_target': 'Untouched native-resolution reference mask',
               'case_split': splits, 'development_seen_kept_in_train': sorted(DEVELOPMENT_SEEN & set(names)),
               'cases': records, 'failures': failures,
               'held_out_test_cases': audit['held_out_test_cases'],
               'warnings': ['CPU/GPU research only, no PL deployment or clinical validation.',
                            'Review reference_alignment.png images before interpreting training scores.',
                            'Missing S1 training CT limits institution coverage.',
                            'All slices, including empty-lung slices, are retained.']}
    fingerprint = {'seed': args.seed, 'size': args.size, 'splits': splits,
                   'patients': sorted(patients), 'frames': sorted(frames), 'clip': [-1000, 400]}
    payload['fingerprint'] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    write_json(output / 'dataset.json', payload)
    print(f'Cache: {output / "dataset.json"} | successful {len(records)} | failed {len(failures)}', flush=True)
    if failures:
        raise ValueError('Cache is incomplete: training is disabled; inspect dataset.json failures')


def save_alignment(hu, reference, path, case, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(hu, cmap='gray', vmin=-1000, vmax=400)
    axes[0].contour(reference, levels=[.5], colors=['cyan'], linewidths=.7)
    axes[0].set_title('Expert lung boundary (cyan)')
    axes[1].imshow(reference, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Expert reference mask, NOT prediction')
    for axis in axes:
        axis.axis('off')
    figure.suptitle(f'{case} | slice {index} | alignment review')
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def torch_module():
    try:
        import torch
    except ImportError as error:
        raise ValueError('PyTorch not installed; use the project training environment') from error
    return torch


def make_model(base_channels=8):
    if base_channels < 2:
        raise ValueError('Base channels must be >=2')
    torch = torch_module()
    nn = torch.nn

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            def block(cin, cout):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
                                     nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=False))
            b = base_channels
            self.enc1, self.enc2, self.enc3 = block(1, b), block(b, 2*b), block(2*b, 4*b)
            self.neck = block(4*b, 8*b)
            self.dec3, self.dec2, self.dec1 = block(12*b, 4*b), block(6*b, 2*b), block(3*b, b)
            self.pool = nn.MaxPool2d(2)
            self.head = nn.Conv2d(b, 1, 1)

        def forward(self, image):
            e1 = self.enc1(image)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            x = self.neck(self.pool(e3))
            for skip, decoder in ((e3, self.dec3), (e2, self.dec2), (e1, self.dec1)):
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='nearest')
                x = decoder(torch.cat((x, skip), dim=1))
            return self.head(x)  # logits, not a hard mask
    return SmallUNet()


def segmentation_loss(logits, target):
    torch = torch_module()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    dice = (2 * (probability * target).sum(dims) + 1) / (probability.sum(dims) + target.sum(dims) + 1)
    return bce + 1 - dice.mean()


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('status') != 'complete' or data.get('failures'):
        raise ValueError('Dataset cache must be complete before training')
    cases = data['cases']
    if not cases or len({r['patient_key'] for r in cases}) != len(cases):
        raise ValueError('Empty or duplicate-patient dataset')
    if len({r['case'] for r in cases}) != len(cases) or any('Test-' in r['case'] for r in cases):
        raise ValueError('Duplicate cases or official test case leakage')
    if {r['split'] for r in cases} != {'train', 'validation'}:
        raise ValueError('Need both training and validation patients')
    return path.parent, data


class SliceDataset:
    def __init__(self, root, records, augment=False):
        self.root, self.records, self.augment = Path(root), records, augment
        self.ends = np.cumsum([r['slices'] for r in records]).tolist()
        self.arrays = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        case_index = bisect_right(self.ends, index)
        row = index - (self.ends[case_index-1] if case_index else 0)
        if case_index not in self.arrays:
            folder = self.root / self.records[case_index]['cache_folder']
            self.arrays[case_index] = (np.load(folder / 'images.npy', mmap_mode='r'),
                                       np.load(folder / 'labels.npy', mmap_mode='r'))
        images, labels = self.arrays[case_index]
        image, label = np.array(images[row], dtype=np.float32), np.array(labels[row], dtype=np.float32)
        if self.augment and random.random() < .5:
            image, label = image[:, ::-1].copy(), label[:, ::-1].copy()
        torch = torch_module()
        return torch.from_numpy(image[None]), torch.from_numpy(label[None])


def select_device(name):
    torch = torch_module()
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; do not silently change training hardware')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else
                        ('cpu' if name == 'auto' else name))


def model_summary(model, size):
    torch = torch_module()
    macs, shapes, handles = [], [], []
    def hook(module, inputs, output):
        shapes.append(list(output.shape))
        if isinstance(module, torch.nn.Conv2d):
            macs.append(int(output.numel() * (module.in_channels // module.groups) *
                            module.kernel_size[0] * module.kernel_size[1]))
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, size, size))
    for handle in handles:
        handle.remove()
    count = sum(p.numel() for p in model.parameters())
    return {'parameters': count, 'conv_MACs_per_slice': sum(macs), 'input_shape': [1, 1, size, size],
            'float32_parameter_bytes': count * 4, 'output_shapes': shapes,
            'operators': ['Conv2D 3x3/1x1', 'BatchNorm', 'ReLU', 'MaxPool2D', 'nearest resize', 'concat'],
            'hardware_status': 'Not quantized, not deployed; BN folding, buffers/bandwidth and RTL remain to be designed'}


def _case_prediction(model, root, record, device, batch_size, threshold=.5):
    torch = torch_module()
    folder = Path(root) / record['cache_folder']
    images = np.load(folder / 'images.npy', mmap_mode='r')
    native = tuple(record['native_shape'])
    prediction = np.empty(native, dtype=bool)
    start = perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(images), batch_size):
            image = torch.from_numpy(np.array(images[begin:begin+batch_size], dtype=np.float32)[:, None]).to(device)
            probability = torch.sigmoid(model(image))
            probability = torch.nn.functional.interpolate(probability, size=native[1:],
                                                           mode='bilinear', align_corners=False)
            prediction[begin:begin+batch_size] = (probability[:, 0] >= threshold).cpu().numpy()
    # CPU copies above synchronize CUDA; includes cached-input loading and mask resize/copy.
    return prediction, (perf_counter() - start) * 1000


def validate_model(model, root, records, device, batch_size):
    model.eval()
    rows = []
    for record in records:
        prediction, elapsed = _case_prediction(model, root, record, device, batch_size)
        reference = np.load(Path(root) / record['cache_folder'] / 'reference_native.npy', mmap_mode='r')
        scores = overlap_metrics(prediction, reference)
        rows.append({'case': record['case'], 'metrics': scores, 'inference_ms': elapsed})
    return rows, float(np.mean([r['metrics']['dice'] for r in rows]))


def save_checkpoint(path, payload):
    torch = torch_module()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)  # only updates this run's generated checkpoint


def train_model(args):
    torch = torch_module()
    root, manifest = read_manifest(args.dataset)
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1 or args.threads < 1:
        raise ValueError('Invalid epochs, batch size, learning rate or CPU thread count')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    model = make_model(args.base_channels)
    summary = model_summary(model, manifest['image_size'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_records = [r for r in manifest['cases'] if r['split'] == 'train']
    val_records = [r for r in manifest['cases'] if r['split'] == 'validation']
    dataset = SliceDataset(root, train_records, augment=True)
    history, start_epoch, best_dice = [], 0, -1.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['dataset_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Resume dataset fingerprint does not match')
        if checkpoint['base_channels'] != args.base_channels or checkpoint['seed'] != args.seed:
            raise ValueError('Resume network size/seed does not match')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, start_epoch, best_dice = checkpoint['history'], checkpoint['epoch'], checkpoint['best_dice']
        if args.epochs <= start_epoch:
            raise ValueError('--epochs is total target epochs; must exceed resumed epoch')
    output = fresh_output(args.outdir)
    settings = {k: v for k, v in vars(args).items() if k != 'func'}
    settings.update(device_resolved=str(device), torch_version=str(torch.__version__),
                    train_cases=[r['case'] for r in train_records],
                    validation_cases=[r['case'] for r in val_records],
                    train_slices=len(dataset), dataset_fingerprint=manifest['fingerprint'],
                    threshold=.5, loss='BCEWithLogits + soft Dice',
                    warning='Research CPU/GPU run, not FPGA or clinical validation')
    write_json(output / 'run_config.json', settings)
    write_json(output / 'model_summary.json', summary)
    print(f'Device {device} | parameters {summary["parameters"]:,} | '
          f'train patients {len(train_records)}, validation patients {len(val_records)} | '
          f'train slices {len(dataset)}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Per-epoch seeds make sample order and augmentations reproducible on resume.
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0,
                                            pin_memory=device.type == 'cuda', generator=generator)
        model.train()
        total_loss, samples, start = 0., 0, perf_counter()
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                logits = model(images)
                loss = segmentation_loss(logits.float(), labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; stop instead of saving invalid weights')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            samples += len(images)
            if step % 100 == 0:
                print(f'  epoch {epoch}: batch {step}/{len(loader)} | loss {total_loss/samples:.4f}', flush=True)
        rows, mean_dice = validate_model(model, root, val_records, device, args.batch_size)
        epoch_row = {'epoch': epoch, 'train_loss': total_loss/samples,
                     'mean_validation_case_dice_native': mean_dice,
                     'epoch_seconds_including_validation': perf_counter() - start,
                     'validation_cases': rows}
        history.append(epoch_row)
        improved = mean_dice > best_dice
        best_dice = max(best_dice, mean_dice)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scaler': scaler.state_dict(), 'epoch': epoch, 'best_dice': best_dice,
                   'history': history, 'base_channels': args.base_channels, 'seed': args.seed,
                   'image_size': manifest['image_size'], 'hu_clip': manifest['hu_clip'],
                   'dataset_fingerprint': manifest['fingerprint'], 'threshold': .5}
        save_checkpoint(output / 'last.pt', payload)
        if improved:
            save_checkpoint(output / 'best.pt', payload)
        write_json(output / 'history.json', history)
        print(f'Epoch {epoch}/{args.epochs} | loss {total_loss/samples:.4f} | '
              f'mean validation case Dice (native): {mean_dice:.4f} | '
              f'{epoch_row["epoch_seconds_including_validation"]:.1f}s', flush=True)
    print(f'Completed. Best validation mean Dice: {best_dice:.4f}; not independent-test accuracy.', flush=True)


def read_native_hu(record):
    images = []
    for path in record['source_ct_files_sorted']:
        ds = pydicom.dcmread(path)
        images.append(ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
    return np.stack(images)


def save_learned_preview(hu, reference, prediction, baseline, path, name, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    titles = ['Expert boundary (cyan)', 'Rule baseline (yellow)',
              'Learned U-Net (green), CPU/GPU', 'Red extra / blue missed / gray correct']
    for axis, mask, title, color in zip(axes[:3], [reference, baseline, prediction], titles, ['cyan', 'yellow', 'lime']):
        axis.imshow(hu, cmap='gray', vmin=-1000, vmax=400)
        if mask.any():
            axis.contour(mask, levels=[.5], colors=[color], linewidths=.7)
        axis.set_title(title)
    errors = np.zeros((*prediction.shape, 3), dtype=float)
    errors[prediction & reference] = [.6, .6, .6]
    errors[prediction & ~reference] = [1, 0, 0]
    errors[~prediction & reference] = [0, .4, 1]
    axes[3].imshow(errors)
    axes[3].set_title(titles[3])
    for axis in axes:
        axis.axis('off')
    figure.suptitle(f'{name} | slice {index} | native resolution | education only')
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_learned(args):
    torch = torch_module()
    root, manifest = read_manifest(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['dataset_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Checkpoint and dataset fingerprint differ')
    if checkpoint['image_size'] != manifest['image_size']:
        raise ValueError('Checkpoint and input image size differ')
    if args.batch_size < 1 or not 0 < args.threshold < 1:
        raise ValueError('Invalid batch size or probability threshold')
    if args.case:
        records = [r for r in manifest['cases'] if r['case'] == args.case]
        if len(records) != 1:
            raise ValueError('Requested case is not in this development dataset')
    else:
        records = [r for r in manifest['cases'] if r['split'] == 'validation']
    device = select_device(args.device)
    model = make_model(checkpoint['base_channels']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    output = fresh_output(args.outdir)
    rows = []
    for record in records:
        print(f'Evaluating {record["case"]} ({record["split"]})', flush=True)
        prediction, elapsed = _case_prediction(model, root, record, device, args.batch_size, args.threshold)
        reference = np.load(root / record['cache_folder'] / 'reference_native.npy', mmap_mode='r') != 0
        scores = overlap_metrics(prediction, reference)
        hu = read_native_hu(record)
        baseline, _, baseline_ms = segment_volume(hu, record['spacing_zyx_mm'][1:])
        baseline_scores = overlap_metrics(baseline, reference)
        case_dir = output / record['case']
        case_dir.mkdir()
        np.save(case_dir / 'prediction_native.npy', prediction.astype(np.uint8) * 255)
        per_slice = [overlap_metrics(prediction[i], reference[i]) for i in range(len(reference))]
        nonempty = [i for i in range(len(reference)) if reference[i].sum() > 100]
        indices = [int(np.argmax(reference.sum(axis=(1, 2))))]
        if nonempty:
            indices.append(min(nonempty, key=lambda i: per_slice[i]['dice']))
        # Also show the worst false-positive-only empty-lung slice, if any.
        empty = [i for i in range(len(reference)) if not reference[i].any() and prediction[i].any()]
        if empty:
            indices.append(max(empty, key=lambda i: int(prediction[i].sum())))
        for index in sorted(set(indices)):
            save_learned_preview(hu[index], reference[index], prediction[index], baseline[index],
                                 case_dir / f'comparison_slice_{index:03d}.png', record['case'], index)
        row = {'case': record['case'], 'split': record['split'], 'learned_metrics_native': scores,
               'baseline_metrics_native': baseline_scores, 'learned_execution': str(device),
               'cached_input_to_native_mask_ms': elapsed, 'cpu_baseline_algorithm_ms': baseline_ms,
               'timing_warning': 'Different timing scopes; not a fair speedup claim or FPGA/system FPS',
               'preview_slices': sorted(set(indices)), 'per_slice_metrics': per_slice}
        rows.append(row)
        write_json(case_dir / 'metrics.json', row)
        print(f'  learned Dice {scores["dice"]:.4f} | baseline {baseline_scores["dice"]:.4f}', flush=True)
        del prediction, reference, hu, baseline
    report = {'checkpoint_epoch': checkpoint['epoch'], 'probability_threshold': args.threshold,
              'mean_case_dice': float(np.mean([r['learned_metrics_native']['dice'] for r in rows])),
              'mean_baseline_case_dice': float(np.mean([r['baseline_metrics_native']['dice'] for r in rows])),
              'cases': rows, 'status': 'Development evaluation only; not official held-out test or FPGA inference'}
    write_json(output / 'evaluation.json', report)
    print(f'Results: {output / "evaluation.json"}', flush=True)


def smoke_model(args):
    torch = torch_module()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    device = select_device(args.device)
    model = make_model(4).to(device).train()
    image = torch.rand(2, 1, 64, 64, device=device)
    target = (image > .5).float()
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = segmentation_loss(model(image), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss) or torch.equal(before, next(model.parameters()).detach()):
        raise ValueError('Smoke test failed: nonfinite loss or weights did not update')
    print(f'Forward/backward smoke PASS | device {device} | loss {float(loss.detach()):.4f}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description='Lightweight 2-D lung U-Net development; computer only')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Create patient-level split and image/reference caches')
    prep.add_argument('--input-root', required=True)
    prep.add_argument('--audit', required=True)
    prep.add_argument('--outdir', required=True)
    prep.add_argument('--prepared-root', help='Reuse existing matched CT/reference preparation')
    prep.add_argument('--size', type=int, default=256)
    prep.add_argument('--seed', type=int, default=20261009)
    prep.add_argument('--validation-per-site', type=int, default=3)
    prep.add_argument('--resume-cache', action='store_true',
                      help='Reuse this pipeline generated cases and retry failures without changing the patient split')
    prep.set_defaults(func=prepare_cache)
    train = commands.add_parser('train', help='Train with native-resolution patient-level validation')
    train.add_argument('--dataset', required=True)
    train.add_argument('--outdir', required=True)
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--base-channels', type=int, default=8)
    train.add_argument('--lr', type=float, default=.001)
    train.add_argument('--seed', type=int, default=20261009)
    train.add_argument('--threads', type=int, default=4)
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train.add_argument('--resume', help='Trusted last.pt from an earlier run; output must be a new directory')
    train.set_defaults(func=train_model)
    evaluate = commands.add_parser('evaluate', help='Compare learned and rule outputs on development cases')
    evaluate.add_argument('--dataset', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--outdir', required=True)
    evaluate.add_argument('--case', help='One development case; default all validation cases')
    evaluate.add_argument('--batch-size', type=int, default=8)
    evaluate.add_argument('--threshold', type=float, default=.5)
    evaluate.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluate.set_defaults(func=evaluate_learned)
    smoke = commands.add_parser('smoke', help='Synthetic forward/backward check, not an accuracy test')
    smoke.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    smoke.set_defaults(func=smoke_model)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError, AttributeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
