"""Small 2-D/2.5-D U-Net development: cache -> patient split -> train -> evaluate.

Computer CPU/GPU only. No FPGA inference or clinical use is claimed.
The prepare command works without PyTorch; all learned-model commands need it.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import pydicom
from scipy import ndimage as ndi

from lung_segmentation import fresh_output, write_json
from lungseg.baseline import segment_volume
from lungseg.dicom_io import discover, load_ct, orientation_vectors, rasterize_lungs, referenced_series
from lungseg.metrics import overlap_metrics


DEVELOPMENT_SEEN = {'LCTSC-Train-S2-001', 'LCTSC-Train-S3-001'}


@contextmanager
def mapped_npy(path):
    """Own one .npy mapping; callers must copy any arrays that outlive this scope."""
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if not isinstance(array, np.memmap):
        close = getattr(array, 'close', None)
        if close is not None:
            close()
        raise ValueError('Expected a memory-mapped .npy array')
    try:
        yield array
    finally:
        # NumPy memmap has no public close method. Close only our owned root
        # mapping after its consumers have copied their data, not arbitrary views.
        array._mmap.close()


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
                with mapped_npy(folder / 'images.npy') as images, \
                        mapped_npy(folder / 'labels.npy') as labels, \
                        mapped_npy(folder / 'reference_native.npy') as reference:
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


def make_model(base_channels=8, context_slices=1):
    if base_channels < 2:
        raise ValueError('Base channels must be >=2')
    if context_slices not in (1, 3):
        raise ValueError('Context slices must be 1 (2D) or 3 (2.5D)')
    torch = torch_module()
    nn = torch.nn

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_slices = context_slices
            def block(cin, cout):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
                                     nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=False))
            b = base_channels
            self.enc1, self.enc2, self.enc3 = block(context_slices, b), block(b, 2*b), block(2*b, 4*b)
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


def read_manifest(path, allow_body_crop=False):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('preprocessing', {}).get('kind') == 'body_crop_v1' and not allow_body_crop:
        raise ValueError('Body-cropped inputs need lung_body_crop.py native restoration; full-frame readers are unsafe')
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cached_source(record):
    """Re-read original CT, checking its identity/order against the saved label grid."""
    files = record['source_ct_files_sorted']
    if len(files) != record['slices']:
        raise ValueError('Saved CT file count differs from the reference slice count')
    entries = [(Path(path), pydicom.dcmread(path, stop_before_pixels=True)) for path in files]
    patient_keys = {hashlib.sha256(str(ds.PatientID).encode()).hexdigest() for _, ds in entries}
    if patient_keys != {record['patient_key']}:
        raise ValueError('Source CT patient identity no longer matches this cache')
    volume = load_ct(entries)
    if tuple(str(ds.SOPInstanceUID) for _, ds in entries) != volume.sop_uids:
        raise ValueError('Saved source file order differs from physical CT order; reference association is unsafe')
    if volume.frame_uid != record['frame_uid'] or list(volume.hu.shape) != record['native_shape']:
        raise ValueError('Source CT frame/shape differs from the saved reference grid')
    if not np.allclose(volume.spacing, record['spacing_zyx_mm'], rtol=0, atol=1e-5):
        raise ValueError('Source CT spacing differs from the saved reference grid')
    return volume.hu


def rebuild_resolution(args):
    """Change input size only: original CT pixels plus byte-identical native references."""
    root, original = read_manifest(args.dataset)
    if args.size < 32 or args.size % 8 or args.size == original['image_size']:
        raise ValueError('New size must be >=32, divisible by 8, and different from the existing size')
    if original['hu_clip'] != [-1000, 400]:
        raise ValueError('Resolution rebuild expects the existing [-1000,400] HU normalization')
    output = fresh_output(args.outdir)
    records, failures = [], []
    for index, record in enumerate(original['cases'], 1):
        print(f'[{index}/{len(original["cases"])}] Rebuilding {record["case"]} '
              f'({record["split"]}) at {args.size}x{args.size}', flush=True)
        try:
            source_reference = root / record['cache_folder'] / 'reference_native.npy'
            reference_hash = file_sha256(source_reference)
            hu = load_cached_source(record)
            folder = output / 'cases' / record['case']
            folder.mkdir(parents=True)
            images = ((resize_stack(np.clip(hu, -1000, 400), args.size, 1) + 1000) / 1400).astype(np.float16)
            with mapped_npy(source_reference) as reference:
                if list(reference.shape) != record['native_shape'] or not np.isin(reference, [0, 1]).all():
                    raise ValueError('Native reference shape or binary values are invalid')
                labels = (resize_stack(reference, args.size, 0) >= .5).astype(np.uint8)
                preview_index = int(np.argmax(reference.sum(axis=(1, 2))))
                save_alignment(hu[preview_index], reference[preview_index],
                               folder / 'reference_alignment.png', record['case'], preview_index)
            np.save(folder / 'images.npy', images)
            np.save(folder / 'labels.npy', labels)
            shutil.copyfile(source_reference, folder / 'reference_native.npy')
            if file_sha256(folder / 'reference_native.npy') != reference_hash:
                raise ValueError('Copied native reference differs; do not compare against changed labels')
            rebuilt = dict(record, cache_folder=str(folder.relative_to(output)),
                           reference_native_sha256=reference_hash)
            records.append(rebuilt)
            write_json(folder / 'case.json', rebuilt)
            del hu, images, labels
        except (ValueError, KeyError, AttributeError, OSError) as error:
            failures.append({'case': record['case'], 'assigned_split': record['split'], 'error': str(error)})
            print(f'  FAILED: {error}', flush=True)
    manifest = dict(original, image_size=args.size, cases=records, failures=failures,
                    status='incomplete' if failures else 'complete',
                    resolution_rebuild={'source_dataset': str(Path(args.dataset).resolve()),
                                        'source_fingerprint': original['fingerprint'],
                                        'previous_image_size': original['image_size'],
                                        'input': 'Re-read original DICOM HU; NOT upsampled low-resolution images',
                                        'reference': 'Native reference files copied and checked with SHA256'})
    fingerprint = {'seed': original['seed'], 'size': args.size, 'splits': original['case_split'],
                   'patients': sorted(r['patient_key'] for r in records),
                   'frames': sorted(r['frame_uid'] for r in records), 'clip': [-1000, 400]}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    write_json(output / 'dataset.json', manifest)
    print(f'Cache: {output / "dataset.json"} | successful {len(records)} | failed {len(failures)}', flush=True)
    if failures:
        raise ValueError('Resolution cache is incomplete; training is disabled, no patients silently dropped')


def context_batch(images, centers, context_slices=1):
    """Owned NCHW CT input; clamp neighbors within this ONE case, never across cases."""
    if context_slices not in (1, 3):
        raise ValueError('Context slices must be 1 or 3')
    centers = np.asarray(centers)
    if (images.ndim != 3 or len(images) < 1 or centers.ndim != 1 or not centers.size
            or not np.issubdtype(centers.dtype, np.integer)):
        raise ValueError('Context input needs nonempty ZHW images and integer center indices')
    if np.any(centers < 0) or np.any(centers >= len(images)):
        raise IndexError('Center slice is outside this case')
    offsets = np.array([0] if context_slices == 1 else [-1, 0, 1])
    indices = np.clip(centers[:, None] + offsets[None], 0, len(images) - 1)
    return np.array(images[indices], dtype=np.float32, copy=True)


def audit_context_geometry(records):
    """Check saved CT order/identity and uniform geometry from headers, no labels/pixels."""
    summaries, seen_patients, seen_frames = [], set(), set()
    for record in records:
        files = record['source_ct_files_sorted']
        if len(files) != record['slices'] or not files:
            raise ValueError('Context CT file count differs from cached slices')
        headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in files]
        first = headers[0]
        patient = hashlib.sha256(str(first.PatientID).encode()).hexdigest()
        if patient != record['patient_key'] or patient in seen_patients or record['frame_uid'] in seen_frames:
            raise ValueError('Context patient/frame identity differs or repeats across cases')
        seen_patients.add(patient)
        seen_frames.add(record['frame_uid'])
        orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
        _, _, normal = orientation_vectors(orientation)
        spacing = np.asarray(record['spacing_zyx_mm'], dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError('Context slice/pixel spacing must be positive and finite')
        positions, seen_uids = [], set()
        for ds in headers:
            if (str(ds.PatientID) != str(first.PatientID) or str(ds.SeriesInstanceUID) != str(first.SeriesInstanceUID)
                    or str(ds.FrameOfReferenceUID) != record['frame_uid']
                    or int(getattr(ds, 'NumberOfFrames', 1)) != 1):
                raise ValueError('Mixed context CT patient/series/frame or unsupported multiframe')
            if ([int(ds.Rows), int(ds.Columns)] != record['native_shape'][1:]
                    or not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-5, rtol=0)
                    or not np.allclose(ds.PixelSpacing, spacing[1:], atol=1e-5, rtol=0)):
                raise ValueError('Context CT shape/orientation/pixel spacing differs')
            uid = str(ds.SOPInstanceUID)
            if uid in seen_uids:
                raise ValueError('Duplicate context CT slice UID')
            seen_uids.add(uid)
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Invalid context CT position')
            positions.append(position)
        if len(positions) > 1:
            deltas = np.diff(np.asarray(positions), axis=0)
            gaps = deltas @ normal
            if (np.any(gaps <= 0) or not np.allclose(gaps, spacing[0], atol=.05, rtol=.01)
                    or np.max(np.linalg.norm(deltas - gaps[:, None] * normal, axis=1)) > .05):
                raise ValueError('Context CT planes unordered, missing, nonuniform or sheared; resample explicitly')
        summaries.append({'case': record['case'], 'slices': len(files), 'slice_spacing_mm': float(spacing[0])})
    return {'status': 'passed', 'cases': summaries,
            'input': 'Previous/current/next physical planes in each cached case; repeat edge planes',
            'target': 'Center slice only; untouched expert masks',
            'spacing_policy': 'Native adjacent planes; no inter-patient fixed-mm resampling'}


class SliceDataset:
    def __init__(self, root, records, augment=False, context_slices=1):
        if context_slices not in (1, 3):
            raise ValueError('Context slices must be 1 or 3')
        self.context_slices = context_slices
        self.root, self.records, self.augment = Path(root), records, augment
        self.ends = np.cumsum([r['slices'] for r in records]).tolist()
        self.arrays = {}
        self._mappings = ExitStack()
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Dataset has already been closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release owned mappings, including on training errors; safe to call twice."""
        self._closed = True
        self.arrays.clear()
        self._mappings.close()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if self._closed:
            raise RuntimeError('Dataset has already been closed')
        if not 0 <= index < len(self):
            raise IndexError(index)
        case_index = bisect_right(self.ends, index)
        row = index - (self.ends[case_index-1] if case_index else 0)
        if case_index not in self.arrays:
            folder = self.root / self.records[case_index]['cache_folder']
            # If labels cannot be opened, release the image map immediately.
            with ExitStack() as opened:
                images = opened.enter_context(mapped_npy(folder / 'images.npy'))
                labels = opened.enter_context(mapped_npy(folder / 'labels.npy'))
                if (images.ndim != 3 or images.shape != labels.shape
                        or len(images) != self.records[case_index]['slices']):
                    raise ValueError('Cached case image/label shape or slice count differs')
                self.arrays[case_index] = (images, labels)
                self._mappings.callback(opened.pop_all().close)
        images, labels = self.arrays[case_index]
        image = context_batch(images, [row], self.context_slices)[0]
        label = np.array(labels[row], dtype=np.float32)
        if self.augment and random.random() < .5:
            image, label = image[:, :, ::-1].copy(), label[:, ::-1].copy()
        torch = torch_module()
        return torch.from_numpy(image), torch.from_numpy(label[None])


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
        model(torch.zeros(1, model.context_slices, size, size))
    for handle in handles:
        handle.remove()
    count = sum(p.numel() for p in model.parameters())
    return {'parameters': count, 'conv_MACs_per_slice': sum(macs), 'input_shape': [1, model.context_slices, size, size],
            'float32_parameter_bytes': count * 4, 'output_shapes': shapes,
            'operators': ['Conv2D 3x3/1x1', 'BatchNorm', 'ReLU', 'MaxPool2D', 'nearest resize', 'concat'],
            'hardware_status': 'Not quantized, not deployed; BN folding, buffers/bandwidth and RTL remain to be designed'}


def _case_prediction(model, root, record, device, batch_size, threshold=.5):
    torch = torch_module()
    folder = Path(root) / record['cache_folder']
    native = tuple(record['native_shape'])
    prediction = np.empty(native, dtype=bool)
    start = perf_counter()
    with mapped_npy(folder / 'images.npy') as images, torch.inference_mode():
        if images.ndim != 3 or len(images) != native[0]:
            raise ValueError('Cached input shape/slice count differs from native prediction grid')
        for begin in range(0, len(images), batch_size):
            centers = np.arange(begin, min(begin + batch_size, len(images)))
            image = torch.from_numpy(context_batch(images, centers, model.context_slices)).to(device)
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
        with mapped_npy(Path(root) / record['cache_folder'] / 'reference_native.npy') as reference:
            scores = overlap_metrics(prediction, reference)
            quality = slice_quality_metrics(prediction, reference)
        rows.append({'case': record['case'], 'metrics': scores, 'inference_ms': elapsed,
                     'slice_quality': quality})
    return rows, float(np.mean([r['metrics']['dice'] for r in rows]))


def slice_quality_metrics(prediction, reference):
    """Separate native-resolution small-lung and empty-lung slice failures."""
    prediction, reference = np.asarray(prediction) != 0, np.asarray(reference) != 0
    if prediction.shape != reference.shape or prediction.ndim != 3 or not prediction.size:
        raise ValueError('Slice quality needs matching nonempty 3-D masks')
    area = reference.sum(axis=(1, 2))
    predicted = prediction.sum(axis=(1, 2))
    intersection = (prediction & reference).sum(axis=(1, 2))
    dice = np.divide(2. * intersection, area + predicted,
                     out=np.zeros(len(area), dtype=float), where=(area + predicted) != 0)
    groups = {}
    for name, selected in (('small', (area > 0) & (area <= 1000)),
                           ('medium', (area > 1000) & (area <= 10000)),
                           ('large', area > 10000)):
        count = int(selected.sum())
        groups[name] = {'slices': count,
                        'mean_slice_dice': float(dice[selected].mean()) if count else None}
    empty = area == 0
    return {'grouping': 'Native expert mask pixels: small 1..1000, medium 1001..10000, large >10000',
            'groups': groups, 'empty_reference_slices': int(empty.sum()),
            'empty_slices_with_prediction': int(np.count_nonzero(predicted[empty])),
            'empty_false_positive_pixels': int(predicted[empty].sum()),
            'warning': 'Slice means differ from patient-volume Dice; any foreground is not a clinical false-positive rate'}


def aggregate_slice_quality(qualities):
    result = {'groups': {}}
    for name in ('small', 'medium', 'large'):
        count = sum(q['groups'][name]['slices'] for q in qualities)
        total = sum(q['groups'][name]['slices'] * q['groups'][name]['mean_slice_dice']
                    for q in qualities if q['groups'][name]['slices'])
        result['groups'][name] = {'slices': count, 'mean_slice_dice': total / count if count else None}
    for name in ('empty_reference_slices', 'empty_slices_with_prediction', 'empty_false_positive_pixels'):
        result[name] = sum(q[name] for q in qualities)
    return result


class EarlyStopping:
    """Monitor validation improvement, not training loss. Zero patience disables stopping."""
    def __init__(self, patience=8, min_delta=.001):
        if patience < 0 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Early stopping patience/delta must be nonnegative and finite')
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_epoch, self.bad_epochs = None, None, 0

    @property
    def stopped(self):
        return self.patience > 0 and self.bad_epochs >= self.patience

    def update(self, score, epoch):
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Validation Dice must be finite and between zero and one')
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score, self.best_epoch, self.bad_epochs = float(score), epoch, 0
        else:
            self.bad_epochs += 1
        return self.stopped

    def state_dict(self):
        return {'best_score': self.best_score, 'best_epoch': self.best_epoch, 'bad_epochs': self.bad_epochs}

    def load_state_dict(self, state):
        self.best_score, self.best_epoch, self.bad_epochs = (
            state['best_score'], state['best_epoch'], state['bad_epochs'])


def training_controls(args):
    return {'early_stopping_patience': args.early_stopping_patience, 'min_delta': args.min_delta,
            'lr_patience': args.lr_patience, 'lr_factor': args.lr_factor, 'min_lr': args.min_lr,
            'small_lung_weight': args.small_lung_weight}


def restore_training_controls(scheduler, stopper, checkpoint, config):
    previous = checkpoint.get('training_controls')
    if previous is not None:
        saved_config = dict(previous['config'])
        # Earlier checkpoints used uniform sampling and had no sampling option.
        if 'small_lung_weight' in config:
            saved_config.setdefault('small_lung_weight', 1.)
        if saved_config != config:
            raise ValueError('Resume training controls differ; use the saved patience/delta/LR settings')
        scheduler.load_state_dict(previous['scheduler'])
        stopper.load_state_dict(previous['early_stopping'])
        return 'Restored learning-rate and early-stopping state'
    if config.get('small_lung_weight', 1.) != checkpoint.get('sampling', {}).get('small_lung_weight', 1.):
        raise ValueError('Resume sampling controls differ from the legacy uniform run')
    # Legacy checkpoints had a fixed LR. Start LR monitoring here without
    # retroactively changing that LR; initialize early stopping from their history.
    scheduler.best = checkpoint['best_dice']
    scheduler.last_epoch = checkpoint['epoch']
    for row in checkpoint['history']:
        stopper.update(row['mean_validation_case_dice_native'], row['epoch'])
    return 'Legacy checkpoint: LR monitoring starts here; early stopping initialized from history'


def small_lung_sampling(root, records, factor):
    """Boost rare small-lung slices using TRAINING references only, never validation."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('Small-lung sampling weight must be finite and >=1')
    if not records or any(r['split'] != 'train' for r in records):
        raise ValueError('Sampling weights may use training patients only')
    total = sum(r['slices'] for r in records)
    if total < 1:
        raise ValueError('Sampling needs at least one training slice')
    info = {'training_slices': total, 'small_lung_weight': float(factor),
            'strategy': 'uniform_shuffle_no_replacement' if factor == 1 else 'small_lung_weighted_with_replacement'}
    if factor == 1:
        return None, info
    weights, small_count, boosted_count, label_lost_count = [], 0, 0, 0
    for record in records:
        folder = Path(root) / record['cache_folder']
        with mapped_npy(folder / 'reference_native.npy') as reference, mapped_npy(folder / 'labels.npy') as labels:
            if (list(reference.shape) != record['native_shape'] or len(reference) != record['slices']
                    or labels.ndim != 3 or len(labels) != record['slices']
                    or not np.isin(labels, [0, 1]).all() or not np.isin(reference, [0, 1]).all()):
                raise ValueError('Sampling reference/training-label shape or values differ from the cache')
            area = reference.sum(axis=(1, 2))
            small = (area > 0) & (area <= 1000)
            boost = small & (labels.sum(axis=(1, 2)) > 0)
            case_weights = np.ones(record['slices'], dtype=np.float64)
            case_weights[boost] = factor
            weights.append(case_weights)
            small_count += int(small.sum())
            boosted_count += int(boost.sum())
            label_lost_count += int((small & ~boost).sum())
    weights = np.concatenate(weights)
    info.update(native_small_lung_slices=small_count, boosted_slices=boosted_count,
                small_slices_without_low_resolution_foreground=label_lost_count,
                expected_boosted_draw_fraction=float(factor * boosted_count / weights.sum()),
                original_boosted_fraction=boosted_count / total,
                note='Each epoch draws the original number of training samples, with replacement; no validation/test labels used')
    return weights, info


def recover_best_checkpoint(resume_path, checkpoint):
    """Carry historical best weights into a new run even if resume never improves."""
    history = checkpoint['history']
    best_row = max(history, key=lambda row: row['mean_validation_case_dice_native'])
    if not np.isclose(best_row['mean_validation_case_dice_native'], checkpoint['best_dice'], rtol=0, atol=1e-12):
        raise ValueError('Resume best Dice and saved validation history disagree')
    if checkpoint['epoch'] == best_row['epoch']:
        return checkpoint
    path = Path(resume_path).resolve().parent / 'best.pt'
    if not path.is_file():
        raise ValueError('Resume needs the original sibling best.pt to preserve earlier best weights')
    best = torch_module().load(path, map_location='cpu', weights_only=True)
    for key in ('dataset_fingerprint', 'base_channels', 'seed', 'image_size'):
        if best[key] != checkpoint[key]:
            raise ValueError('Sibling best.pt does not match the resumed run')
    if best.get('context_slices', 1) != checkpoint.get('context_slices', 1):
        raise ValueError('Sibling best.pt context input does not match the resumed run')
    if best['epoch'] != best_row['epoch'] or not np.isclose(
            best['history'][-1]['mean_validation_case_dice_native'], checkpoint['best_dice'], rtol=0, atol=1e-12):
        raise ValueError('Sibling best.pt is not the historical best model')
    return best


def save_training_curves(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    epochs = [row['epoch'] for row in history]
    axes[0, 0].plot(epochs, [row['train_loss'] for row in history], marker='.')
    axes[0, 0].set_title('Training loss (lower is better)')
    axes[0, 1].plot(epochs, [row['mean_validation_case_dice_native'] for row in history], marker='.')
    best = max(history, key=lambda row: row['mean_validation_case_dice_native'])
    axes[0, 1].axvline(best['epoch'], color='green', linestyle='--', label=f'Best model: epoch {best["epoch"]}')
    axes[0, 1].set_title('Validation mean patient-volume Dice (native)')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    for name in ('small', 'medium', 'large'):
        rows = [r for r in history if r.get('validation_quality', {}).get('groups', {}).get(name, {}).get('mean_slice_dice') is not None]
        if rows:
            axes[1, 0].plot([r['epoch'] for r in rows],
                            [r['validation_quality']['groups'][name]['mean_slice_dice'] for r in rows], label=name)
    axes[1, 0].set_title('Validation slice Dice by expert lung area')
    axes[1, 0].set_ylim(0, 1)
    if axes[1, 0].lines:
        axes[1, 0].legend()
    rows = [r for r in history if 'learning_rate' in r]
    if rows:
        axes[1, 1].plot([r['epoch'] for r in rows], [r['learning_rate'] for r in rows], marker='.')
        axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Learning rate used during each epoch')
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.25)
    figure.suptitle('Development validation only; lower train loss alone does NOT imply better generalization')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_checkpoint(path, payload):
    torch = torch_module()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)  # only updates this run's generated checkpoint


def train_model(args):
    root, manifest = read_manifest(args.dataset)
    records = [r for r in manifest['cases'] if r['split'] == 'train']
    with SliceDataset(root, records, augment=True, context_slices=args.context_slices) as dataset:
        _train_with_dataset(args, root, manifest, dataset)


def _train_with_dataset(args, root, manifest, dataset):
    torch = torch_module()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1 or args.threads < 1:
        raise ValueError('Invalid epochs, batch size, learning rate or CPU thread count')
    if (args.lr_patience < 0 or not 0 < args.lr_factor < 1 or not 0 < args.min_lr <= args.lr
            or args.early_stopping_patience < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0):
        raise ValueError('Invalid early-stopping or learning-rate settings')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    model = make_model(args.base_channels, args.context_slices)
    summary = model_summary(model, manifest['image_size'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode='abs', min_lr=args.min_lr)
    stopper = EarlyStopping(args.early_stopping_patience, args.min_delta)
    controls = training_controls(args)
    train_records = [r for r in manifest['cases'] if r['split'] == 'train']
    val_records = [r for r in manifest['cases'] if r['split'] == 'validation']
    sampling_weights, sampling_info = small_lung_sampling(root, train_records, args.small_lung_weight)
    history, start_epoch, best_dice, previous_best = [], 0, -1., None
    resume_note = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['dataset_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Resume dataset fingerprint does not match')
        if checkpoint['base_channels'] != args.base_channels or checkpoint['seed'] != args.seed:
            raise ValueError('Resume network size/seed does not match')
        if checkpoint['image_size'] != manifest['image_size']:
            raise ValueError('Resume input image size does not match')
        if checkpoint.get('context_slices', 1) != args.context_slices:
            raise ValueError('Resume context slices differ; 2D and 2.5D weights cannot be mixed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, start_epoch, best_dice = checkpoint['history'], checkpoint['epoch'], checkpoint['best_dice']
        if args.epochs <= start_epoch:
            raise ValueError('--epochs is total target epochs; must exceed resumed epoch')
        previous_best = recover_best_checkpoint(args.resume, checkpoint)
        resume_note = restore_training_controls(scheduler, stopper, checkpoint, controls)
        if stopper.stopped:
            raise ValueError('This checkpoint already exhausted early-stopping patience; do not silently restart its counter')
    context_geometry = audit_context_geometry(manifest['cases']) if args.context_slices == 3 else None
    output = fresh_output(args.outdir)
    settings = {k: v for k, v in vars(args).items() if k != 'func'}
    settings.update(device_resolved=str(device), torch_version=str(torch.__version__),
                    train_cases=[r['case'] for r in train_records],
                    validation_cases=[r['case'] for r in val_records],
                    train_slices=len(dataset), dataset_fingerprint=manifest['fingerprint'],
                    threshold=.5, loss='BCEWithLogits + soft Dice',
                    resume_control_note=resume_note,
                    sampling=sampling_info,
                    context_geometry=context_geometry,
                    selection='Exact highest validation mean patient-volume Dice; min_delta only controls patience/LR',
                    warning='Research CPU/GPU run, not FPGA or clinical validation')
    write_json(output / 'run_config.json', settings)
    write_json(output / 'model_summary.json', summary)
    if previous_best is not None:
        save_checkpoint(output / 'best.pt', previous_best)
        print(resume_note, flush=True)
    print(f'Device {device} | parameters {summary["parameters"]:,} | '
          f'train patients {len(train_records)}, validation patients {len(val_records)} | '
          f'train slices {len(dataset)}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    print(f'Sampling: {sampling_info}', flush=True)
    print(f'Input mode: {"2.5D" if args.context_slices == 3 else "2D"} | context slices {args.context_slices}', flush=True)
    stop_reason = 'epoch_limit'
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Per-epoch seeds make sample order and augmentations reproducible on resume.
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        sampler = (torch.utils.data.WeightedRandomSampler(sampling_weights, len(dataset), replacement=True,
                                                         generator=generator)
                   if sampling_weights is not None else None)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=sampler is None, sampler=sampler, num_workers=0,
                                            pin_memory=device.type == 'cuda', generator=generator)
        model.train()
        learning_rate = optimizer.param_groups[0]['lr']
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
        should_stop = stopper.update(mean_dice, epoch)
        scheduler.step(mean_dice)
        quality = aggregate_slice_quality([row['slice_quality'] for row in rows])
        epoch_row = {'epoch': epoch, 'train_loss': total_loss/samples,
                     'mean_validation_case_dice_native': mean_dice,
                     'learning_rate': learning_rate, 'next_learning_rate': optimizer.param_groups[0]['lr'],
                     'early_stopping_bad_epochs': stopper.bad_epochs, 'validation_quality': quality,
                     'epoch_seconds_including_validation': perf_counter() - start,
                     'validation_cases': rows}
        history.append(epoch_row)
        improved = mean_dice > best_dice
        best_dice = max(best_dice, mean_dice)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scaler': scaler.state_dict(), 'epoch': epoch, 'best_dice': best_dice,
                   'history': history, 'base_channels': args.base_channels, 'seed': args.seed,
                   'image_size': manifest['image_size'], 'hu_clip': manifest['hu_clip'],
                   'context_slices': args.context_slices, 'context_geometry': context_geometry,
                   'dataset_fingerprint': manifest['fingerprint'], 'threshold': .5,
                   'sampling': sampling_info,
                   'training_controls': {'config': controls, 'scheduler': scheduler.state_dict(),
                                         'early_stopping': stopper.state_dict()}}
        save_checkpoint(output / 'last.pt', payload)
        if improved:
            save_checkpoint(output / 'best.pt', payload)
        write_json(output / 'history.json', history)
        save_training_curves(history, output / 'training_curves.png')
        stop_reason = 'early_stopping' if should_stop else 'epoch_limit'
        best_row = max(history, key=lambda row: row['mean_validation_case_dice_native'])
        write_json(output / 'training_summary.json', {
            'status': 'complete' if should_stop or epoch == args.epochs else 'running',
            'start_epoch': start_epoch, 'completed_epoch': epoch, 'max_epochs': args.epochs,
            'best_epoch': best_row['epoch'], 'best_validation_mean_case_dice': best_dice,
            'stop_reason': stop_reason if should_stop or epoch == args.epochs else None,
            'early_stopping_bad_epochs': stopper.bad_epochs,
            'warning': 'Validation selects this model; independent test and FPGA inference remain unperformed'})
        print(f'Epoch {epoch}/{args.epochs} | loss {total_loss/samples:.4f} | '
              f'mean validation case Dice (native): {mean_dice:.4f} | '
              f'LR {learning_rate:.2g} -> {optimizer.param_groups[0]["lr"]:.2g} | '
              f'no significant improvement {stopper.bad_epochs}/{stopper.patience} | '
              f'{epoch_row["epoch_seconds_including_validation"]:.1f}s', flush=True)
        print(f'  Small-lung slice Dice: {quality["groups"]["small"]["mean_slice_dice"]} | '
              f'empty-lung slices with foreground: {quality["empty_slices_with_prediction"]}/'
              f'{quality["empty_reference_slices"]} | FP pixels {quality["empty_false_positive_pixels"]}', flush=True)
        if should_stop:
            print('Early stopping: validation has not improved significantly; retaining best.pt, not last.pt.', flush=True)
            break
    print(f'Completed ({stop_reason}). Best validation mean Dice: {best_dice:.4f}; not independent-test accuracy.', flush=True)


def read_native_hu(record):
    images = []
    for path in record['source_ct_files_sorted']:
        ds = pydicom.dcmread(path)
        images.append(ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
    return np.stack(images)


def save_learned_preview(hu, reference, prediction, baseline, path, name, index,
                         baseline_title='Rule baseline (yellow)', learned_title='Learned U-Net (green), CPU/GPU'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    titles = ['Expert boundary (cyan)', baseline_title,
              learned_title, 'Red extra / blue missed / gray correct']
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


def load_comparison_prediction(root, row, reference):
    with mapped_npy(Path(root) / row['case'] / 'prediction_native.npy') as array:
        scores = overlap_metrics(array, reference)
        prediction = array != 0  # owned copy after mapping closes
    for key in ('true_positive', 'false_positive', 'false_negative'):
        if scores[key] != row['learned_metrics_native'][key]:
            raise ValueError('Previous prediction/report does not match the current expert reference')
    return prediction, scores


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
    previous_rows, previous_root, previous_report = {}, None, None
    if args.compare_evaluation:
        previous_path = Path(args.compare_evaluation).resolve()
        previous_report = json.loads(previous_path.read_text(encoding='utf-8'))
        if previous_report['probability_threshold'] != args.threshold:
            raise ValueError('Previous evaluation threshold differs; this is not a controlled comparison')
        previous_rows = {row['case']: row for row in previous_report['cases']}
        if len(previous_rows) != len(previous_report['cases']) or any(r['case'] not in previous_rows for r in records):
            raise ValueError('Previous evaluation must contain each selected case exactly once')
        expected = manifest.get('resolution_rebuild', {}).get('source_fingerprint', manifest['fingerprint'])
        if previous_report.get('dataset_fingerprint', expected) != expected:
            raise ValueError('Previous evaluation belongs to a different source dataset')
        previous_root = previous_path.parent
    device = select_device(args.device)
    context_slices = checkpoint.get('context_slices', 1)
    context_geometry = audit_context_geometry(records) if context_slices == 3 else None
    model = make_model(checkpoint['base_channels'], context_slices).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    output = fresh_output(args.outdir)
    rows = []
    for record in records:
        print(f'Evaluating {record["case"]} ({record["split"]})', flush=True)
        prediction, elapsed = _case_prediction(model, root, record, device, args.batch_size, args.threshold)
        with mapped_npy(root / record['cache_folder'] / 'reference_native.npy') as native_reference:
            reference = native_reference != 0  # owned copy, valid after closing the map
        scores = overlap_metrics(prediction, reference)
        hu = read_native_hu(record)
        baseline, _, baseline_ms = segment_volume(hu, record['spacing_zyx_mm'][1:])
        baseline_scores = overlap_metrics(baseline, reference)
        case_dir = output / record['case']
        case_dir.mkdir()
        np.save(case_dir / 'prediction_native.npy', prediction.astype(np.uint8) * 255)
        per_slice = [overlap_metrics(prediction[i], reference[i]) for i in range(len(reference))]
        previous_prediction, previous_scores = None, None
        if previous_root is not None:
            previous_prediction, previous_scores = load_comparison_prediction(
                previous_root, previous_rows[record['case']], reference)
        nonempty = [i for i in range(len(reference)) if reference[i].sum() > 100]
        indices = [int(np.argmax(reference.sum(axis=(1, 2))))]
        if nonempty:
            indices.append(min(nonempty, key=lambda i: per_slice[i]['dice']))
        # Also show the worst false-positive-only empty-lung slice, if any.
        empty = [i for i in range(len(reference)) if not reference[i].any() and prediction[i].any()]
        if empty:
            indices.append(max(empty, key=lambda i: int(prediction[i].sum())))
        if previous_root is not None:
            indices.extend(previous_rows[record['case']].get('preview_slices', []))
        for index in sorted(set(indices)):
            save_learned_preview(hu[index], reference[index], prediction[index], baseline[index],
                                 case_dir / f'comparison_slice_{index:03d}.png', record['case'], index,
                                 learned_title=f'{"2.5D" if context_slices == 3 else "2D"} U-Net (green), CPU/GPU')
            if previous_prediction is not None:
                save_learned_preview(hu[index], reference[index], prediction[index], previous_prediction[index],
                    case_dir / f'model_comparison_slice_{index:03d}.png', record['case'], index,
                    baseline_title='Previous model (yellow), CPU/GPU',
                    learned_title=f'Current {"2.5D" if context_slices == 3 else "2D"} '
                                  f'{manifest["image_size"]}x{manifest["image_size"]} U-Net (green), CPU/GPU')
        row = {'case': record['case'], 'split': record['split'], 'learned_metrics_native': scores,
               'slice_quality': slice_quality_metrics(prediction, reference),
               'baseline_metrics_native': baseline_scores, 'learned_execution': str(device),
               'cached_input_to_native_mask_ms': elapsed, 'cpu_baseline_algorithm_ms': baseline_ms,
               'timing_warning': 'Different timing scopes; not a fair speedup claim or FPGA/system FPS',
               'preview_slices': sorted(set(indices)), 'per_slice_metrics': per_slice}
        if previous_prediction is not None:
            row['previous_model_metrics_native'] = previous_scores
            row['previous_model_slice_quality'] = slice_quality_metrics(previous_prediction, reference)
        rows.append(row)
        write_json(case_dir / 'metrics.json', row)
        print(f'  learned Dice {scores["dice"]:.4f} | baseline {baseline_scores["dice"]:.4f}', flush=True)
        del prediction, reference, hu, baseline
        del previous_prediction
    report = {'checkpoint_epoch': checkpoint['epoch'], 'probability_threshold': args.threshold,
              'dataset_fingerprint': manifest['fingerprint'], 'image_size': manifest['image_size'],
              'context_slices': context_slices, 'context_geometry': context_geometry,
              'mean_case_dice': float(np.mean([r['learned_metrics_native']['dice'] for r in rows])),
              'mean_baseline_case_dice': float(np.mean([r['baseline_metrics_native']['dice'] for r in rows])),
              'slice_quality': aggregate_slice_quality([r['slice_quality'] for r in rows]),
              'cases': rows, 'status': 'Development evaluation only; not official held-out test or FPGA inference'}
    if previous_report is not None:
        report['previous_evaluation'] = {'path': str(Path(args.compare_evaluation).resolve()),
            'checkpoint_epoch': previous_report['checkpoint_epoch'],
            'context_slices': previous_report.get('context_slices', 1),
            'mean_case_dice': float(np.mean([row['previous_model_metrics_native']['dice'] for row in rows])),
            'slice_quality': aggregate_slice_quality([row['previous_model_slice_quality'] for row in rows])}
    write_json(output / 'evaluation.json', report)
    print(f'Results: {output / "evaluation.json"}', flush=True)


def smoke_model(args):
    torch = torch_module()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    if args.size < 32 or args.size % 8 or args.batch_size < 1:
        raise ValueError('Smoke size must be >=32 and divisible by 8; batch size must be positive')
    device = select_device(args.device)
    model = make_model(args.base_channels, args.context_slices).to(device).train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    image = torch.rand(args.batch_size, args.context_slices, args.size, args.size, device=device)
    target = (image[:, args.context_slices // 2:args.context_slices // 2 + 1] > .5).float()
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
        print(f'Synthetic FP32 peak allocated memory: {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MiB; '
              'not actual AMP training memory, FPGA resources or medical accuracy', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description='Lightweight 2-D/2.5-D lung U-Net development; computer only')
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
    rebuild = commands.add_parser('rebuild-resolution', help='Re-read original CT at another size; preserve split/native references')
    rebuild.add_argument('--dataset', required=True)
    rebuild.add_argument('--outdir', required=True)
    rebuild.add_argument('--size', type=int, default=512)
    rebuild.set_defaults(func=rebuild_resolution)
    train = commands.add_parser('train', help='Train with native-resolution patient-level validation')
    train.add_argument('--dataset', required=True)
    train.add_argument('--outdir', required=True)
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--base-channels', type=int, default=8)
    train.add_argument('--context-slices', type=int, choices=[1, 3], default=1,
                       help='1: existing 2D; 3: previous/center/next CT input, center mask target (2.5D)')
    train.add_argument('--lr', type=float, default=.001)
    train.add_argument('--early-stopping-patience', type=int, default=8,
                       help='Stop after this many validations without significant improvement; 0 disables')
    train.add_argument('--min-delta', type=float, default=.001,
                       help='Absolute Dice improvement for patience/LR; best.pt still uses the exact highest score')
    train.add_argument('--lr-patience', type=int, default=3,
                       help='Reduce LR after more than this many validations without significant improvement')
    train.add_argument('--lr-factor', type=float, default=.5)
    train.add_argument('--min-lr', type=float, default=.00001)
    train.add_argument('--small-lung-weight', type=float, default=1.,
                       help='Training-only sampling weight for native lung area 1..1000 pixels; 1 keeps uniform sampling')
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
    evaluate.add_argument('--compare-evaluation', help='Previous evaluation.json with native predictions; save paired model previews')
    evaluate.set_defaults(func=evaluate_learned)
    smoke = commands.add_parser('smoke', help='Synthetic forward/backward check, not an accuracy test')
    smoke.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    smoke.add_argument('--size', type=int, default=64)
    smoke.add_argument('--base-channels', type=int, default=4)
    smoke.add_argument('--batch-size', type=int, default=2)
    smoke.add_argument('--context-slices', type=int, choices=[1, 3], default=1)
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
