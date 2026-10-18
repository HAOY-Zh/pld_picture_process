"""Frozen, patient-independent LCTSC evaluation. Computer only, not clinical use.

This entry point cannot train, select checkpoints, tune thresholds, or select
individual test cases. The development manifest still rejects official tests.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom

from lung_segmentation import fresh_output, write_json
from lung_unet import (_load_source, _case_prediction, aggregate_slice_quality,
                       audit_context_geometry, file_sha256, make_model,
                       read_manifest, resize_stack, save_learned_preview,
                       select_device, slice_quality_metrics, torch_module)
from lungseg.baseline import segment_volume
from lungseg.metrics import overlap_metrics


def source_identities(files):
    """Read headers only; reject ambiguous patient/series/frame identities."""
    result = {key: set() for key in ('patients', 'frames', 'series', 'sops')}
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        patient = str(ds.PatientID)
        if not patient.strip():
            raise ValueError('Empty source PatientID')
        result['patients'].add(hashlib.sha256(patient.encode()).hexdigest())
        result['frames'].add(str(ds.FrameOfReferenceUID))
        result['series'].add(str(ds.SeriesInstanceUID))
        sop = str(ds.SOPInstanceUID)
        if sop in result['sops']:
            raise ValueError('Duplicate source SOP Instance UID')
        result['sops'].add(sop)
    if any(len(result[k]) != 1 for k in ('patients', 'frames', 'series')) or not result['sops']:
        raise ValueError('Need exactly one patient/frame/series per case')
    return result


def add_independent(identities, used):
    """Do not accept a duplicate even under a different case-folder name."""
    for key in used:
        if used[key] & identities[key]:
            raise ValueError(f'Patient independence audit failed: overlapping {key}')
    for key in used:
        used[key].update(identities[key])


def freeze_protocol(args, manifest, checkpoint):
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    reserved = manifest['held_out_test_cases']
    expected = sorted(r['case'] for r in audit['cases'] if r['split'] == 'test')
    if (not reserved or len(set(reserved)) != len(reserved) or
            len(set(expected)) != len(expected) or not set(reserved) <= set(expected) or
            any(not n.startswith('LCTSC-Test-') or '/' in n or '\\' in n for n in expected)):
        raise ValueError('Audit and reserved official test case lists do not match')
    if checkpoint['dataset_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Checkpoint does not belong to the development dataset')
    if checkpoint['image_size'] != manifest['image_size']:
        raise ValueError('Checkpoint image size differs from development preprocessing')
    if (manifest['hu_clip'] != [-1000, 400] or
            manifest['input_normalization'] != '(clip(HU,-1000,400)+1000)/1400' or
            checkpoint.get('threshold', .5) != .5):
        raise ValueError('This frozen protocol requires original HU normalization and threshold 0.5')
    if checkpoint.get('context_slices', 1) != 1:
        raise ValueError('This held-out protocol is for the retained 2D model only')
    script_root = Path(__file__).resolve().parent
    return {'schema': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(args.checkpoint),
            'checkpoint_epoch': checkpoint['epoch'],
            'development_dataset': str(Path(args.dataset).resolve()),
            'development_manifest_sha256': file_sha256(args.dataset),
            'development_fingerprint': manifest['fingerprint'],
            'audit_sha256': file_sha256(args.audit),
            'image_size': manifest['image_size'], 'context_slices': 1,
            'hu_clip': manifest['hu_clip'], 'input_normalization': manifest['input_normalization'],
            'probability_threshold': .5, 'mask_resize': 'bilinear probability to native grid before >=0.5',
            'input_resize': 'pixel-center linear HU resize; float16 cached normalized input',
            'target': 'Union of unchanged rasterized expert left/right RTSTRUCT lung masks',
            'reference_rasterization': 'integer pixel centers, boundary included, closed planar union/XOR',
            'reserved_available_candidates': sorted(reserved),
            'all_test_folders_to_attempt': expected,
            'policy': 'Freeze before test reads. Attempt all audited test folders, including download failures. '
                      'No performance-based exclusions, retraining, threshold tuning or checkpoint selection.',
            'slices': 'All CT slices, including empty-reference slices; no lung-only filtering',
            'code_sha256': {name: file_sha256(script_root / name) for name in
                           ('lung_heldout.py', 'lung_unet.py', 'lung_segmentation.py',
                            'lungseg/dicom_io.py', 'lungseg/metrics.py', 'lungseg/baseline.py')}}


def prepare_test_cache(args, manifest, protocol, output):
    used = {key: set() for key in ('patients', 'frames', 'series', 'sops')}
    for row in manifest['cases']:
        identity = source_identities(row['source_ct_files_sorted'])
        if (identity['patients'] != {row['patient_key']} or
                identity['frames'] != {row['frame_uid']} or
                len(identity['sops']) != row['slices']):
            raise ValueError('Development source identities changed; independence cannot be established')
        add_independent(identity, used)
    records, failures = [], []
    expected = protocol['all_test_folders_to_attempt']
    for number, name in enumerate(expected, 1):
        print(f'Preparing held-out [{number}/{len(expected)}] {name}', flush=True)
        try:
            hu, reference, spacing, files, patient, frame, overlap = _load_source(
                name, Path(args.input_root) / name)
            identity = source_identities(files)
            # Header geometry also verifies sorted source planes against label-grid metadata.
            folder = output / 'cache' / name
            row = {'case': name, 'split': 'held_out_test', 'patient_key': patient,
                   'frame_uid': frame, 'slices': len(hu), 'native_shape': list(hu.shape),
                   'spacing_zyx_mm': np.asarray(spacing).tolist(),
                   'cache_folder': str(folder.relative_to(output)),
                   'source_ct_files_sorted': files, 'reference_overlap_voxels': overlap}
            audit_context_geometry([row])
            if not reference.any():
                raise ValueError('Expert whole-volume lung mask is empty')
            # Do not mutate seen identities if this case fails preparation.
            for key in used:
                if used[key] & identity[key]:
                    raise ValueError(f'Patient independence audit failed: overlapping {key}')
            images = ((resize_stack(np.clip(hu, -1000, 400), protocol['image_size'], 1)
                       + 1000) / 1400).astype(np.float16)
            if not np.isfinite(images).all():
                raise ValueError('Nonfinite normalized CT input')
            folder.mkdir(parents=True)
            np.save(folder / 'images.npy', images)
            np.save(folder / 'reference_native.npy', reference.astype(np.uint8))
            row['images_sha256'] = file_sha256(folder / 'images.npy')
            row['reference_sha256'] = file_sha256(folder / 'reference_native.npy')
            write_json(folder / 'case.json', row)
            add_independent(identity, used)
            records.append(row)
            print(f'  OK: {len(hu)} slices; native expert grid {list(hu.shape)}', flush=True)
            del hu, reference, images
        except (ValueError, KeyError, AttributeError, OSError) as error:
            failures.append({'case': name, 'stage': 'data_preparation_or_independence',
                             'error': str(error), 'reserved_candidate': name in
                             protocol['reserved_available_candidates']})
            print(f'  UNAVAILABLE: {error}', flush=True)
        write_json(output / 'test_dataset.json', {
            'purpose': 'Held-out evaluation only; never pass to training',
            'status': 'preparing', 'protocol_sha256': file_sha256(output / 'protocol.json'),
            'cases': records, 'failures': failures, 'attempted': number, 'expected': len(expected)})
    payload = {'purpose': 'Held-out evaluation only; never pass to training',
               'status': 'complete' if not failures else 'partial_coverage',
               'protocol_sha256': file_sha256(output / 'protocol.json'),
               'cases': records, 'failures': failures, 'attempted': len(expected), 'expected': len(expected),
               'independence': 'No shared PatientID hash, frame UID, series UID or CT SOP UID '
                               'with development cases or other accepted test cases'}
    write_json(output / 'test_dataset.json', payload)
    return payload


def summarize(rows):
    result = {'evaluated_cases': len(rows), 'evaluated_slices': sum(r['slices'] for r in rows),
              'slice_quality': aggregate_slice_quality([r['slice_quality'] for r in rows])}
    for metric in ('dice', 'iou', 'precision', 'recall'):
        values = [r['learned_metrics_native'][metric] for r in rows
                  if r['learned_metrics_native'][metric] is not None]
        result[metric] = {'defined_cases': len(values), 'mean': float(np.mean(values)) if values else None,
                          'std_population': float(np.std(values)) if values else None,
                          'min': float(min(values)) if values else None,
                          'max': float(max(values)) if values else None}
    return result


def save_summary(output, report):
    write_json(output / 'evaluation.json', report)
    rows = report['cases']
    with (output / 'case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'slices', 'dice', 'iou', 'precision', 'recall', 'tp', 'fp', 'fn'])
        for row in rows:
            scores = row['learned_metrics_native']
            writer.writerow([row['case'], row['slices']] + [scores[key] for key in
                            ('dice', 'iou', 'precision', 'recall', 'true_positive',
                             'false_positive', 'false_negative')])


def run_heldout(args):
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    _, manifest = read_manifest(args.dataset)
    torch = torch_module()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    protocol = freeze_protocol(args, manifest, checkpoint)
    device = select_device(args.device)
    model = make_model(checkpoint['base_channels'], 1).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    output = fresh_output(args.outdir)
    write_json(output / 'protocol.json', protocol)  # Before any held-out CT/label reads.
    print(f'Frozen epoch {checkpoint["epoch"]}, 2D, threshold 0.5; protocol: {output / "protocol.json"}', flush=True)
    cache = prepare_test_cache(args, manifest, protocol, output)
    report = {'status': 'evaluating', 'scope': 'Independent LCTSC held-out evaluation; CPU/GPU, not FPGA or clinical',
              'checkpoint_epoch': checkpoint['epoch'], 'probability_threshold': .5,
              'checkpoint_sha256': protocol['checkpoint_sha256'],
              'protocol_sha256': file_sha256(output / 'protocol.json'),
              'context_slices': 1, 'image_size': protocol['image_size'],
              'development_fingerprint': manifest['fingerprint'],
              'expected_test_cases': len(protocol['all_test_folders_to_attempt']),
              'reserved_candidates': len(protocol['reserved_available_candidates']),
              'data_failures': cache['failures'], 'cases': [],
              'warning': 'Available subset is not the full official test set when cases are missing. '
                         'Further tuning after viewing these scores consumes this hold-out; '
                         'a new independent set is needed for future final claims.'}
    save_summary(output, report)
    for record in cache['cases']:
        print(f'Evaluating {record["case"]} ({record["slices"]} slices)', flush=True)
        try:
            prediction, elapsed = _case_prediction(model, output, record, device, args.batch_size, .5)
            folder = output / record['cache_folder']
            reference = np.load(folder / 'reference_native.npy', allow_pickle=False) != 0
            if file_sha256(folder / 'reference_native.npy') != record['reference_sha256']:
                raise ValueError('Native expert reference changed since preparation')
            scores = overlap_metrics(prediction, reference)
            # Read validated native CT for fixed rule comparison and previews only.
            from lung_unet import load_cached_source
            hu = load_cached_source(record)
            baseline, _, baseline_ms = segment_volume(hu, record['spacing_zyx_mm'][1:])
            baseline_scores = overlap_metrics(baseline, reference)
            case_dir = output / record['case']
            case_dir.mkdir()
            np.save(case_dir / 'prediction_native.npy', prediction.astype(np.uint8) * 255)
            per_slice = [overlap_metrics(prediction[i], reference[i]) for i in range(len(reference))]
            nonempty = [i for i in range(len(reference)) if reference[i].any()]
            indices = {int(np.argmax(reference.sum(axis=(1, 2)))),
                       min(nonempty, key=lambda i: per_slice[i]['dice'])}
            empty_fp = [i for i in range(len(reference)) if not reference[i].any() and prediction[i].any()]
            if empty_fp:
                indices.add(max(empty_fp, key=lambda i: int(prediction[i].sum())))
            for index in sorted(indices):
                save_learned_preview(hu[index], reference[index], prediction[index], baseline[index],
                                     case_dir / f'comparison_slice_{index:03d}.png', record['case'], index,
                                     learned_title='Frozen 2D U-Net (green), held-out CPU/GPU')
            row = {'case': record['case'], 'split': 'held_out_test', 'slices': len(reference),
                   'learned_metrics_native': scores, 'baseline_metrics_native': baseline_scores,
                   'slice_quality': slice_quality_metrics(prediction, reference),
                   'per_slice_metrics': per_slice, 'preview_slices': sorted(indices),
                   'learned_execution': str(device), 'cached_input_to_native_mask_ms': elapsed,
                   'cpu_baseline_algorithm_ms': baseline_ms,
                   'timing_warning': 'Different scopes; no fair speedup, system FPS or FPGA claim'}
            write_json(case_dir / 'metrics.json', row)
            report['cases'].append(row)
            report.update(summarize(report['cases']))
            save_summary(output, report)
            print(f'  Dice {scores["dice"]:.6f} | IoU {scores["iou"]:.6f}', flush=True)
            del prediction, reference, hu, baseline
        except Exception as error:
            report['status'] = 'aborted_incomplete_evaluation'
            report['execution_failure'] = {'case': record['case'], 'error': str(error)}
            save_summary(output, report)
            raise  # Never quietly exclude an inference/metric failure.
    report.update(summarize(report['cases']))
    report['status'] = 'complete_available_subset' if cache['failures'] else 'complete_audited_test_set'
    report['reserved_candidates_evaluated'] = sum(r['case'] in protocol['reserved_available_candidates']
                                                 for r in report['cases'])
    report['checkpoint_unchanged'] = file_sha256(args.checkpoint) == protocol['checkpoint_sha256']
    if not report['checkpoint_unchanged']:
        report['status'] = 'invalid_checkpoint_changed'
    save_summary(output, report)
    if not report['cases'] or not report['checkpoint_unchanged']:
        raise ValueError('No valid test result or frozen checkpoint changed; inspect evaluation.json')
    print(f'Held-out results: {output / "evaluation.json"}\n'
          f'Coverage {len(report["cases"])}/{report["expected_test_cases"]} cases, '
          f'{report["evaluated_slices"]} slices, mean case Dice {report["dice"]["mean"]:.6f}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Original training/validation manifest, NOT a test manifest')
    parser.add_argument('--checkpoint', required=True, help='One already selected trusted best.pt; no model search')
    parser.add_argument('--audit', required=True, help='Download audit listing every test folder, including failures')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--outdir', required=True, help='New empty output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser


if __name__ == '__main__':
    run_heldout(build_parser().parse_args())
