"""Frozen COVID CT external evaluation; workstation only, not clinical/FPGA.

Read the downloaded ZIPs without extracting paths. The official lung-mask ZIP
is the reference. The predeclared primary cohort is the ten Coronacases HU
volumes; the ten windowed Radiopaedia volumes are audited but never treated as
HU. No training, calibration, threshold tuning, case selection or model search.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile

import nibabel as nib
import numpy as np

from lung_segmentation import fresh_output, write_json
from lung_quantize import restore_quantized, quality_summary
from lung_unet import (_case_prediction, file_sha256, make_model, mapped_npy,
                       read_manifest, resize_stack, save_alignment,
                       save_learned_preview, select_device, slice_quality_metrics,
                       torch_module)
from lungseg.metrics import overlap_metrics


OFFICIAL_MASK_MD5 = '972b3d9b4c7b64d2c518b936bd98cb47'
PRIMARY = tuple(f'coronacases_{i:03d}' for i in range(1, 11))


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def case_id(name):
    stem = Path(name).name
    stem = re.sub(r'\.nii(?:\.gz)?$', '', stem)
    if re.fullmatch(r'coronacases_(?:org_)?\d{3}', stem):
        return stem.replace('coronacases_org_', 'coronacases_')
    match = re.fullmatch(r'radiopaedia_org_covid-19-pneumonia-(\d+_\d+_\d+)-dcm', stem)
    if match:
        return 'radiopaedia_' + match[1]
    if re.fullmatch(r'radiopaedia_\d+_\d+_\d+', stem):
        return stem
    raise ValueError(f'Unknown COVID case identifier: {name}')


def archive_members(archive, ct=False):
    result = {}
    for info in archive.infolist():
        if not info.filename.endswith(('.nii', '.nii.gz')):
            continue
        if ct and '/ct_scans/' not in '/' + info.filename:
            continue
        key = case_id(info.filename)
        if key in result:
            raise ValueError(f'Duplicate case/member in archive: {key}')
        result[key] = info.filename
    if not result:
        raise ValueError('No CT/reference NIfTI members in archive')
    return result


def read_member(archive, name):
    # Reading to EOF verifies the ZIP member CRC; gzip also verifies its own CRC.
    payload = archive.read(name)
    digest = hashlib.sha256(payload).hexdigest()
    if name.endswith('.gz'):
        payload = gzip.decompress(payload)
    image = nib.Nifti1Image.from_bytes(payload)
    return image, digest


def lps_zyx(image):
    """Lossless axis permutation/flips ONLY; no interpolation or image rot90 guess.

    NIfTI world coordinates are RAS. Make columns/rows advance towards L/P and
    slices towards S, matching the usual axial LCTSC DICOM input convention.
    Returned affine still maps XYZ voxel indices to RAS world millimeters.
    """
    if len(image.shape) != 3 or any(n < 1 for n in image.shape):
        raise ValueError('Need a nonempty 3-D NIfTI volume')
    sform, scode = image.get_sform(coded=True)
    qform, qcode = image.get_qform(coded=True)
    if not scode and not qcode:
        raise ValueError('No coded physical affine; refuse filename-only alignment')
    if scode and qcode and not np.allclose(sform, qform, rtol=0, atol=1e-3):
        raise ValueError('Conflicting NIfTI sform/qform')
    affine = np.asarray(image.affine)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError('Nonfinite or singular NIfTI affine')
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                               nib.orientations.axcodes2ornt(('L', 'P', 'S')))
    aligned = image.as_reoriented(transform)
    spacing_xyz = nib.affines.voxel_sizes(aligned.affine)
    direction = aligned.affine[:3, :3] / spacing_xyz
    if not np.allclose(direction, np.diag([-1., -1., 1.]), rtol=0, atol=1e-4):
        raise ValueError('Oblique/sheared volume needs a separately validated resampling protocol')
    array = aligned.get_fdata(dtype=np.float32, caching='unchanged').transpose(2, 1, 0)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite NIfTI voxels')
    metadata = {'source_shape_xyz': list(image.shape),
                'source_axis_codes': list(nib.aff2axcodes(image.affine)),
                'native_shape_zyx': list(array.shape),
                'axis_transform': transform.tolist(),
                'lps_xyz_to_ras_world_affine': aligned.affine.tolist(),
                'spacing_zyx': spacing_xyz[::-1].tolist(),
                'header_units': list(image.header.get_xyzt_units()),
                'source_dtype': str(image.get_data_dtype())}
    return array, metadata


def align_pair(ct, label):
    hu, ct_meta = lps_zyx(ct)
    values, label_meta = lps_zyx(label)
    if (hu.shape != values.shape or not np.allclose(ct_meta['lps_xyz_to_ras_world_affine'],
            label_meta['lps_xyz_to_ras_world_affine'], rtol=0, atol=1e-3)):
        raise ValueError('CT/reference physical voxel grids differ; no automatic label resampling')
    units = [ct_meta['header_units'][0], label_meta['header_units'][0]]
    if any(unit not in ('mm', 'unknown') for unit in units):
        raise ValueError('Unsupported spatial unit; do not mix coordinate scales')
    labels, counts = np.unique(values, return_counts=True)
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Official lung reference must have labels 0/1/2, not infection or probabilities')
    reference = values != 0
    if not reference.any():
        raise ValueError('Whole-volume expert lung reference is empty')
    return hu, reference, {'ct_geometry': ct_meta, 'reference_geometry': label_meta,
        'label_counts': {str(int(k)): int(v) for k, v in zip(labels, counts)},
        'geometry': 'Same physical grid after lossless permutation/flips; expert voxels not interpolated'}


def require_hu(name, hu, dtype):
    if name not in PRIMARY:
        raise ValueError('Not in predeclared Coronacases HU cohort')
    if np.dtype(dtype).kind == 'u' or hu.min() >= 0:
        raise ValueError('HU sanity check failed; do not interpret windowed grayscale as HU')
    if hu.shape[1] != hu.shape[2]:
        raise ValueError('Primary axial HU volume must be square; refuse aspect-ratio distortion')


def normalize_hu(hu, size):
    return ((resize_stack(np.clip(hu, -1000, 400), size, 1) + 1000) / 1400).astype(np.float16)


def prior_image_hashes(dataset, prior_test):
    """Exact preprocessed-image duplicate check, NOT proof of patient identity."""
    sources = [Path(dataset), Path(prior_test)]
    result = {}
    for source in sources:
        manifest = json.loads(source.read_text(encoding='utf-8'))
        for record in manifest['cases']:
            path = source.parent / record['cache_folder'] / 'images.npy'
            digest = file_sha256(path)
            if 'images_sha256' in record and digest != record['images_sha256']:
                raise ValueError('Previously evaluated CT cache changed')
            result.setdefault(digest, []).append(record['case'])
    return result


def freeze(args, manifest, checkpoint, integer):
    if (checkpoint['dataset_fingerprint'] != manifest['fingerprint'] or
        checkpoint['image_size'] != manifest['image_size'] or checkpoint.get('context_slices', 1) != 1 or
        checkpoint.get('threshold', .5) != .5 or manifest['hu_clip'] != [-1000, 400] or
        manifest['input_normalization'] != '(clip(HU,-1000,400)+1000)/1400'):
        raise ValueError('Require retained 2D model and unchanged HU preprocessing/threshold')
    if (integer['source_checkpoint_sha256'] != file_sha256(args.checkpoint) or
        integer['dataset_fingerprint'] != manifest['fingerprint'] or
        integer['image_size'] != checkpoint['image_size'] or
        integer['base_channels'] != checkpoint['base_channels'] or
        integer['context_slices'] != 1 or integer['threshold'] != .5 or
        integer['engine'] != 'x86' or integer['quantization'] != 'static_ptq_x86_w8_uint8'):
        raise ValueError('INT8 model does not belong to the frozen FP32 model/protocol')
    if md5_file(args.masks) != OFFICIAL_MASK_MD5:
        raise ValueError('Lung_Mask.zip does not match the official published MD5')
    paths = {key: str(Path(getattr(args, key)).resolve()) for key in
             ('dataset', 'prior_test', 'checkpoint', 'int8_model', 'ct', 'masks')}
    root = Path(__file__).resolve().parent
    return {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_paths': paths, 'source_sha256': {key: file_sha256(path) for key, path in paths.items()},
        'official_mask_md5': OFFICIAL_MASK_MD5, 'image_size': checkpoint['image_size'],
        'base_channels': checkpoint['base_channels'], 'checkpoint_epoch': checkpoint['epoch'],
        'primary_cohort': list(PRIMARY), 'threshold': .5,
        'excluded_cohort': 'Radiopaedia: windowed UINT8 [0,255]; original HU lost; audit only',
        'input_normalization': '(clip(HU,-1000,400)+1000)/1400',
        'orientation': 'Lossless LPS XYZ reorientation then ZYX; native expert labels not resized',
        'input_resize': 'Original pixel-center linear HU resize; normalized FP16 input cache',
        'output_resize': 'Sigmoid then bilinear probability to native grid, threshold >=0.5',
        'target': 'Unchanged official Lung_Mask labels 1 and 2 union, NOT infection mask',
        'policy': 'Freeze before external voxel reads. Attempt all archives cases. Evaluate all valid '
                  'primary volumes and all their slices including empty layers. No training, '
                  'recalibration, checkpoint/threshold search or performance exclusions.',
        'identity_limit': 'NIfTI lacks source PatientID/SOP; exact normalized-image duplicate checks '
                          'cannot prove patient independence or detect all transformed duplicates.',
        'scope': 'Cross-dataset computer FP32/CPU INT8 evaluation; NOT clinical/FPGA',
        'code_sha256': {str((root/name).resolve()): file_sha256(root/name) for name in
                        ('lung_external.py', 'lung_unet.py', 'lung_quantize.py', 'lungseg/metrics.py')}}


def check_frozen(protocol):
    for key, expected in protocol['source_sha256'].items():
        if file_sha256(protocol['source_paths'][key]) != expected:
            raise ValueError(f'Frozen input/model changed: {key}')
    for path, expected in protocol['code_sha256'].items():
        if file_sha256(path) != expected:
            raise ValueError(f'Frozen inference code changed: {path}')


def prepare(args):
    torch = torch_module()
    _, manifest = read_manifest(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    integer = torch.load(args.int8_model, map_location='cpu', weights_only=True)
    protocol = freeze(args, manifest, checkpoint, integer)
    known_hashes = prior_image_hashes(args.dataset, args.prior_test)
    protocol['prior_exact_input_hashes'] = known_hashes
    output = fresh_output(args.outdir)
    write_json(output/'protocol.json', protocol)  # BEFORE any external voxels.
    report = {'status': 'preparing', 'primary_expected': 10, 'cases': [], 'failures': [],
              'primary_prepared': 0, 'cohort_exclusions': [], 'primary_slices': 0}
    native_seen, input_seen = {}, {}
    with zipfile.ZipFile(args.ct) as ct_zip, zipfile.ZipFile(args.masks) as mask_zip:
        images, labels = archive_members(ct_zip, ct=True), archive_members(mask_zip)
        expected = sorted(set(images) | set(labels) | set(PRIMARY))
        report['archive_case_identifiers'] = len(expected)
        for number, name in enumerate(expected, 1):
            print(f'Auditing [{number}/{len(expected)}] {name}', flush=True)
            try:
                if name not in images or name not in labels:
                    raise ValueError('Missing matched CT or expert lung reference')
                ct, ct_hash = read_member(ct_zip, images[name])
                label, label_hash = read_member(mask_zip, labels[name])
                hu, reference, metadata = align_pair(ct, label)
                metadata.update({'case': name, 'split': 'external_test', 'ct_member': images[name],
                    'mask_member': labels[name], 'ct_member_sha256': ct_hash,
                    'mask_member_sha256': label_hash, 'slices': len(hu),
                    'native_shape': list(hu.shape), 'hu_or_gray_min': float(hu.min()),
                    'hu_or_gray_max': float(hu.max())})
                image_hash = hashlib.sha256(np.ascontiguousarray(hu).tobytes()).hexdigest()
                if image_hash in native_seen:
                    raise ValueError(f'Duplicate native image volume: {native_seen[image_hash]}')
                native_seen[image_hash] = name
                metadata['native_scaled_pixel_sha256'] = image_hash
                if name not in PRIMARY:
                    metadata['status'] = 'excluded_predeclared_windowed_cohort'
                    metadata['reason'] = protocol['excluded_cohort']
                    report['cohort_exclusions'].append({'case': name, 'reason': metadata['reason']})
                    print(f'  AUDIT ONLY: gray {hu.min():g}..{hu.max():g}, {len(hu)} layers', flush=True)
                else:
                    require_hu(name, hu, metadata['ct_geometry']['source_dtype'])
                    images_small = normalize_hu(hu, protocol['image_size'])
                    folder = output/'cache'/name
                    folder.mkdir(parents=True)
                    np.save(folder/'images.npy', images_small)
                    digest = file_sha256(folder/'images.npy')
                    if digest in known_hashes or digest in input_seen:
                        raise ValueError('Exact normalized-image duplicate of another external or prior LCTSC case')
                    input_seen[digest] = name
                    np.save(folder/'reference_native.npy', reference.astype(np.uint8))
                    # Preserve HU exactly; use int16 only when lossless for this volume.
                    integer_hu = (hu.min() >= -32768 and hu.max() <= 32767 and np.equal(hu, np.rint(hu)).all())
                    np.save(folder/'hu_native.npy', hu.astype(np.int16) if integer_hu else hu)
                    metadata.update({'status': 'ready_primary_hu', 'cache_folder': str(folder.relative_to(output)),
                        'images_sha256': digest,
                        'reference_sha256': file_sha256(folder/'reference_native.npy'),
                        'hu_cache_sha256': file_sha256(folder/'hu_native.npy')})
                    index = int(np.argmax(reference.sum(axis=(1, 2))))
                    save_alignment(hu[index], reference[index], folder/'alignment.png', name, index)
                    write_json(folder/'case.json', metadata)
                    report['primary_prepared'] += 1
                    report['primary_slices'] += len(hu)
                    print(f'  READY: HU {hu.min():g}..{hu.max():g}, {len(hu)} layers', flush=True)
                    del images_small
                report['cases'].append(metadata)
                del ct, label, hu, reference
            except (ValueError, KeyError, OSError, zipfile.BadZipFile, EOFError) as error:
                report['failures'].append({'case': name, 'primary': name in PRIMARY, 'error': str(error)})
                print(f'  DATA FAILURE: {error}', flush=True)
            write_json(output/'external_dataset.json', report)
    check_frozen(protocol)
    report['status'] = 'complete_audit' if not report['failures'] else 'audit_with_data_failures'
    report['identity_warning'] = protocol['identity_limit']
    write_json(output/'external_dataset.json', report)
    print(f'Prepared {report["primary_prepared"]}/10 HU volumes, {report["primary_slices"]} layers; '
          f'audited/excluded {len(report["cohort_exclusions"])} windowed volumes. Review alignment plots before evaluation.', flush=True)


def save_report(output, report):
    write_json(output/'evaluation.json', report)
    with (output/'case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'slices', 'fp32_dice', 'int8_dice', 'fp32_iou', 'int8_iou', 'different_pixels'])
        for row in report['cases']:
            writer.writerow([row['case'], row['slices'], row['fp32']['dice'], row['int8']['dice'],
                             row['fp32']['iou'], row['int8']['iou'], row['different_pixels']])


def evaluate(args):
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and CPU threads must be positive')
    root, output = Path(args.prepared), fresh_output(args.outdir)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    dataset = json.loads((root/'external_dataset.json').read_text(encoding='utf-8'))
    check_frozen(protocol)
    if dataset['status'] not in ('complete_audit', 'audit_with_data_failures'):
        raise ValueError('Data audit not complete')
    records = [r for r in dataset['cases'] if r['status'] == 'ready_primary_hu']
    if not records or any(r['case'] not in PRIMARY or r['split'] != 'external_test' for r in records):
        raise ValueError('Invalid or empty primary cohort')
    write_json(output/'protocol.json', protocol)
    report = {'status': 'evaluating_external', 'scope': protocol['scope'], 'cases': [],
        'expected_primary_volumes': 10, 'total_archive_case_identifiers': dataset['archive_case_identifiers'],
        'cohort_exclusions': dataset['cohort_exclusions'], 'data_failures': dataset['failures'],
        'prepared_manifest_sha256': file_sha256(root/'external_dataset.json'),
        'identity_warning': protocol['identity_limit'],
        'warning': 'External domain shift test, not comparable to published scores on other cohorts. '
                   'Further tuning after seeing results consumes this hold-out. No clinical/FPGA claim.'}
    torch = torch_module()
    torch.set_num_threads(args.threads)
    old_engine, old_tf32 = torch.backends.quantized.engine, torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = select_device(args.device)
        checkpoint = torch.load(protocol['source_paths']['checkpoint'], map_location='cpu', weights_only=True)
        original = make_model(checkpoint['base_channels'], 1).to(device).eval()
        original.load_state_dict(checkpoint['model'])
        quantized, _ = restore_quantized(protocol['source_paths']['int8_model'])
        save_report(output, report)
        for record in records:
            print(f'External {record["case"]}, all {record["slices"]} layers', flush=True)
            folder = root/record['cache_folder']
            for filename, key in [('images.npy','images_sha256'), ('reference_native.npy','reference_sha256'),
                                  ('hu_native.npy','hu_cache_sha256')]:
                if file_sha256(folder/filename) != record[key]:
                    raise ValueError('Prepared image/reference cache changed')
            fp32, fp32_ms = _case_prediction(original, root, record, device, args.batch_size, .5)
            int8, int8_ms = _case_prediction(quantized, root, record, torch.device('cpu'), args.batch_size, .5)
            with mapped_npy(folder/'reference_native.npy') as array:
                reference = array != 0
            scores, q_scores = overlap_metrics(fp32, reference), overlap_metrics(int8, reference)
            fp_slices = [overlap_metrics(fp32[i], reference[i]) for i in range(len(reference))]
            q_slices = [overlap_metrics(int8[i], reference[i]) for i in range(len(reference))]
            nonempty = np.flatnonzero(reference.any(axis=(1,2))).tolist()
            indices = {int(np.argmax(reference.sum(axis=(1,2)))),
                min(nonempty, key=lambda i: fp_slices[i]['dice']),
                min(nonempty, key=lambda i: q_slices[i]['dice']),
                int(np.argmax((fp32 != int8).sum(axis=(1,2))))}
            empty_fp = [i for i in range(len(reference)) if not reference[i].any() and int8[i].any()]
            if empty_fp:
                indices.add(max(empty_fp, key=lambda i: int(int8[i].sum())))
            case_dir = output/record['case']
            case_dir.mkdir()
            np.save(case_dir/'float_prediction_native.npy', fp32.astype(np.uint8)*255)
            np.save(case_dir/'int8_prediction_native.npy', int8.astype(np.uint8)*255)
            with mapped_npy(folder/'hu_native.npy') as hu:
                for index in sorted(indices):
                    save_learned_preview(hu[index], reference[index], int8[index], fp32[index],
                        case_dir/f'comparison_slice_{index:03d}.png', record['case'], index,
                        baseline_title='Frozen FP32 (yellow), computer',
                        learned_title='Frozen INT8 (green), CPU; FP32 postprocess')
            row = {'case': record['case'], 'split': 'external_test', 'slices': len(reference),
                'fp32': scores, 'int8': q_scores, 'fp32_slice_quality': slice_quality_metrics(fp32, reference),
                'int8_slice_quality': slice_quality_metrics(int8, reference),
                'fp32_per_slice': fp_slices, 'int8_per_slice': q_slices,
                'different_pixels': int((fp32 != int8).sum()), 'total_pixels': int(reference.size),
                'preview_slices': sorted(indices), 'fp32_cached_input_to_native_mask_ms': fp32_ms,
                'int8_cached_input_to_native_mask_ms': int8_ms,
                'timing_warning': 'FP32 on '+str(device)+' vs CPU INT8; not a fair speedup or board FPS'}
            write_json(case_dir/'metrics.json', row)
            report['cases'].append(row)
            save_report(output, report)
            print(f'  Dice FP32 {scores["dice"]:.6f} | INT8 {q_scores["dice"]:.6f}', flush=True)
            del fp32, int8, reference
        check_frozen(protocol)
        if file_sha256(root/'external_dataset.json') != report['prepared_manifest_sha256']:
            raise ValueError('Prepared manifest changed during evaluation')
        report.update({'status': 'complete_primary_external_test' if len(records)==10 else 'complete_available_primary_subset',
            'evaluated_volumes': len(records), 'evaluated_slices': sum(r['slices'] for r in records),
            'fp32': quality_summary(report['cases'], 'fp32'), 'int8': quality_summary(report['cases'], 'int8'),
            'frozen_sources_unchanged': True})
        save_report(output, report)
        print(f'Completed {len(records)}/10 HU volumes, {report["evaluated_slices"]} layers: '
              f'FP32 {report["fp32"]["mean_case_dice"]:.6f}; INT8 {report["int8"]["mean_case_dice"]:.6f}', flush=True)
    except Exception as error:
        report['status'] = 'aborted_incomplete_external_evaluation'
        report['execution_failure'] = str(error)
        save_report(output, report)
        raise
    finally:
        torch.backends.quantized.engine = old_engine
        torch.backends.cudnn.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze models then audit all data; no predictions')
    for key in ('ct','masks','dataset','prior-test','checkpoint','int8-model','outdir'):
        prep.add_argument('--'+key, required=True)
    run = sub.add_parser('evaluate', help='Only evaluate the previously frozen primary cohort')
    run.add_argument('--prepared', required=True)
    run.add_argument('--outdir', required=True)
    run.add_argument('--batch-size', type=int, default=8)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)
