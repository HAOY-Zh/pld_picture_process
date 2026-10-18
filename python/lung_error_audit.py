"""Read-only lung-label and internal-FN audit; no training or mask correction.

Outputs are descriptive development diagnostics, not a fresh blind evaluation.
2-D enclosed background is a geometric property, not a diagnosis/anatomical label.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy import ndimage as ndi

from lung_external import archive_members, lps_zyx, read_member
from lung_segmentation import fresh_output, write_json
from lung_unet import _load_source, file_sha256, mapped_npy, resize_stack
from lungseg.metrics import overlap_metrics


HU_EDGES = [-np.inf, -1000, -700, -400, -100, 200, 400, np.inf]


def binary(mask):
    mask = np.asarray(mask)
    if mask.size == 0 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Need nonempty binary 0/1/255 mask')
    return mask != 0


def enclosed_background(mask):
    """4-connected background components not touching any image border."""
    mask = binary(mask)
    if mask.ndim != 2:
        raise ValueError('Enclosed-background analysis is explicitly 2-D')
    labels, count = ndi.label(~mask, structure=ndi.generate_binary_structure(2, 1))
    exterior = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    keep[exterior] = False
    return keep[labels], labels


def error_maps(prediction, reference, spacing_yx=(1., 1.), interior_mm=2.):
    prediction, reference = binary(prediction), binary(reference)
    spacing = np.asarray(spacing_yx, dtype=float)
    if (prediction.shape != reference.shape or prediction.ndim != 2 or
            spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any() or
            not np.isfinite(interior_mm) or interior_mm <= 0):
        raise ValueError('Need same 2-D grid, positive physical spacing and interior distance')
    fn = reference & ~prediction
    enclosed, labels = enclosed_background(prediction)
    closed_fn = fn & enclosed
    # Pad explicitly, so an all-white reference still has a meaningful image edge.
    distance = ndi.distance_transform_edt(np.pad(reference, 1), sampling=spacing)[1:-1, 1:-1]
    deep_fn = fn & (distance >= interior_mm)
    maps = {'closed_fn': closed_fn, 'open_fn': fn & ~enclosed,
            'deep_fn': deep_fn, 'edge_fn': fn & ~deep_fn,
            'fp': prediction & ~reference, 'tp': prediction & reference}
    stats = {key + '_pixels': int(value.sum()) for key, value in maps.items()}
    stats['fn_pixels'] = int(fn.sum())
    stats['closed_fn_components'] = len(np.unique(labels[closed_fn]))
    stats['reference_pixels'] = int(reference.sum())
    reference_holes, _ = enclosed_background(reference)
    stats['reference_enclosed_background_pixels'] = int(reference_holes.sum())
    return stats, maps


def density_counts(hu, reference, prediction=None):
    """HU strata only; never interpret these bins as infection/normal labels."""
    hu, reference = np.asarray(hu), binary(reference)
    if hu.shape != reference.shape or not np.isfinite(hu).all():
        raise ValueError('HU/reference grid mismatch or nonfinite values')
    prediction = binary(prediction) if prediction is not None else reference
    if prediction.shape != reference.shape:
        raise ValueError('Prediction shape mismatch')
    values, missed = hu[reference], hu[reference & ~prediction]
    return np.histogram(values, HU_EDGES)[0], np.histogram(missed, HU_EDGES)[0]


def preview(hu, reference, prediction, path, title, spacing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats, maps = error_maps(prediction, reference, spacing)
    figure, axes = plt.subplots(1, 5, figsize=(19, 4.4))
    for axis, mask, color, label in zip(axes[:3], [None, reference, prediction],
            ['white', 'cyan', 'lime'], ['Original CT', 'Expert lung (cyan)', 'Unmodified model (green)']):
        axis.imshow(hu, cmap='gray', vmin=-1000, vmax=400)
        if mask is not None and mask.any() and not mask.all():
            axis.contour(mask, levels=[.5], colors=[color], linewidths=.65)
        axis.set_title(label)
    axes[3].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Actual model mask\nwhite = lung, black = non-lung')
    colors = np.zeros((*reference.shape, 3))
    colors[maps['tp']] = .6
    colors[maps['fp']] = [1, 0, 0]
    colors[maps['open_fn']] = [0, .4, 1]
    colors[maps['closed_fn']] = [1, 0, 1]
    axes[4].imshow(colors)
    axes[4].set_title('Magenta: enclosed FN; blue: other FN\nred: extra; gray: correct lung')
    for axis in axes:
        axis.axis('off')
    figure.suptitle(f'{title} | closed FN={stats["closed_fn_pixels"]}, '
                   f'other FN={stats["open_fn_pixels"]} | education only')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def target_preview(hu, native, small, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    restored = resize_stack(small[None], native.shape[0], 0)[0] >= .5
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(hu, cmap='gray', vmin=-1000, vmax=400)
    if native.any() and not native.all():
        axes[0].contour(native, levels=[.5], colors=['cyan'], linewidths=.65)
    axes[0].set_title('Original CT + expert lung')
    axes[1].imshow(native, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Native expert target')
    axes[2].imshow(small, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Actual 256 x 256 training target')
    errors = np.zeros((*native.shape, 3))
    errors[native & restored] = .6
    errors[native & ~restored] = [0, .4, 1]
    errors[restored & ~native] = [1, 0, 0]
    axes[3].imshow(errors)
    axes[3].set_title('Nearest resize roundtrip ONLY\nblue: lost; red: gained')
    for axis in axes:
        axis.axis('off')
    figure.suptitle(title + ' | label audit, NOT a model prediction')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def check_training_labels(reference, labels, size):
    reference, labels = binary(reference), binary(labels)
    expected = resize_stack(reference, size, 0) >= .5
    if not np.array_equal(expected, labels):
        raise ValueError('Training cache differs from nearest-resized native expert target')
    restored = resize_stack(labels, reference.shape[1], 0) >= .5
    return overlap_metrics(restored, reference)


def analyze_volume(hu, reference, prediction, spacing_yx, case, cohort, model, output,
                   forced=()):
    reference, prediction = binary(reference), binary(prediction)
    if hu.shape != reference.shape or reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError('Native HU/reference/prediction volume mismatch')
    rows, totals = [], defaultdict(int)
    for index in range(len(reference)):
        stats, _ = error_maps(prediction[index], reference[index], spacing_yx)
        row = dict(case=case, cohort=cohort, model=model, slice=index, **stats)
        rows.append(row)
        for key, value in stats.items():
            totals[key] += value
    counts, missed = density_counts(hu, reference, prediction)
    selected = {max(range(len(rows)), key=lambda i: rows[i]['closed_fn_pixels']),
                max(range(len(rows)), key=lambda i: rows[i]['deep_fn_pixels'])}
    selected.update(i for i in forced if 0 <= i < len(rows))
    folder = output / cohort / case / model
    folder.mkdir(parents=True)
    for index in sorted(selected):
        preview(hu[index], reference[index], prediction[index], folder/f'internal_errors_{index:03d}.png',
                f'{case} | {model} | slice {index}', spacing_yx)
    report = dict(case=case, cohort=cohort, model=model, slices=len(rows),
                  metrics=overlap_metrics(prediction, reference), totals=dict(totals),
                  reference_hu_bin_voxels=counts.tolist(), missed_hu_bin_voxels=missed.tolist(),
                  preview_slices=sorted(selected))
    write_json(folder/'diagnostics.json', report)
    return report, rows


def run(args):
    output = fresh_output(args.outdir)
    tracked = {}

    def track(path):
        path = Path(path).resolve()
        tracked[str(path)] = file_sha256(path)
        return path

    dataset_path, heldout_path, external_path = map(track, [args.dataset, args.heldout, args.external])
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    heldout = json.loads(heldout_path.read_text(encoding='utf-8'))
    external = json.loads(external_path.read_text(encoding='utf-8'))
    track(args.checkpoint)
    track(args.external_masks)
    report = {'status': 'running', 'purpose': 'Development error diagnosis; NOT a new blind test',
        'definitions': {'closed_fn': 'Reference-lung voxels in a 4-connected 2-D prediction-background component not touching the image border',
            'open_fn': 'Remaining false-negative voxels (border-connected prediction background)',
            'deep_fn': 'False negatives >=2 mm inside the native expert mask, using 2-D EDT and physical pixel spacing',
            'warning': 'Closed/open describe mask geometry, not normal tissue or disease. No mask has been repaired.',
            'hu_bins': ['<-1000', '[-1000,-700)', '[-700,-400)', '[-400,-100)', '[-100,200)', '[200,400)', '>=400'],
            'hu_warning': 'HU strata are NOT disease labels',
            'roundtrip': 'Nearest down/up-resize diagnostic, NOT model prediction or independent rasterizer validation'},
        'label_audit': [], 'prediction_audit': [], 'failures': []}
    rows = []
    write_json(output/'audit.json', report)
    total_dicom = len(dataset['cases']) + len(heldout['cases'])
    groups = [(dataset_path.parent, r, 'development') for r in dataset['cases']]
    groups += [(heldout_path.parent, r, 'heldout') for r in heldout['cases']]
    for number, (root, record, cohort) in enumerate(groups, 1):
        name, cache = record['case'], root / record['cache_folder']
        print(f'[{number}/{total_dicom}] Source/label audit {name}', flush=True)
        try:
            source_folder = Path(args.input_root)/name
            hu, original, spacing, files, patient, frame, _ = _load_source(name, source_folder)
            reference_path = track(cache/'reference_native.npy')
            with mapped_npy(reference_path) as array:
                reference = binary(array)
            if (not np.array_equal(original, reference) or
                    files != record['source_ct_files_sorted'] or
                    patient != record['patient_key'] or frame != record['frame_uid'] or
                    not np.allclose(spacing, record['spacing_zyx_mm'], rtol=0, atol=1e-6)):
                raise ValueError('Source expert raster/CT identity/physical geometry differs from cache')
            # Record source RTSTRUCT hashes as well; CT is re-read by strict _load_source.
            from lungseg.dicom_io import discover
            _, structures, _ = discover(source_folder)
            track(structures[0][0])
            reference_holes = sum(int(enclosed_background(mask)[0].sum()) for mask in reference)
            counts, _ = density_counts(hu, reference)
            label_row = dict(case=name, split=record['split'], source_raster_matches_native=True,
                             reference_hu_bin_voxels=counts.tolist(), native_reference_hole_pixels=reference_holes)
            if cohort == 'development':
                track(cache/'labels.npy')
                track(cache/'images.npy')
                with mapped_npy(cache/'labels.npy') as labels, mapped_npy(cache/'images.npy') as images:
                    roundtrip = check_training_labels(reference, labels, dataset['image_size'])
                    expected_images = ((resize_stack(np.clip(hu, -1000, 400), dataset['image_size'], 1)
                                        + 1000)/1400).astype(np.float16)
                    if not np.array_equal(expected_images, images):
                        raise ValueError('HU normalization/resize differs from saved training input')
                    del expected_images
                    restored = resize_stack(labels, reference.shape[1], 0) >= .5
                    lost_deep = 0
                    for j in range(len(reference)):
                        lost_deep += error_maps(restored[j], reference[j], spacing[1:])[0]['deep_fn_pixels']
                    label_row.update(training_label_matches_nearest=True, training_input_matches_source=True,
                                     resize_roundtrip=roundtrip, resize_roundtrip_deep_fn_pixels=lost_deep)
                    if name in {'LCTSC-Train-S2-001', 'LCTSC-Train-S3-001', 'LCTSC-Train-S2-006'}:
                        # Choose by reference high-HU content, not model performance.
                        index = int(np.argmax(((hu >= -300) & reference).sum(axis=(1, 2))))
                        target_preview(hu[index], reference[index], labels[index],
                                       output/f'training_target_{name}_{index:03d}.png', name+f' | slice {index}')
                    del restored
                if record['split'] == 'validation':
                    prediction_path = Path(args.validation)/name/'prediction_native.npy'
                    track(prediction_path)
                    with mapped_npy(prediction_path) as array:
                        result, slices = analyze_volume(hu, reference, array, spacing[1:], name,
                                                       'validation', 'fp32', output)
                    report['prediction_audit'].append(result)
                    rows.extend(slices)
            else:
                prediction_path = root/name/'prediction_native.npy'
                track(prediction_path)
                with mapped_npy(prediction_path) as array:
                    result, slices = analyze_volume(hu, reference, array, spacing[1:], name,
                                                   'heldout', 'fp32', output)
                report['prediction_audit'].append(result)
                rows.extend(slices)
            report['label_audit'].append(label_row)
            del hu, original, reference
        except (ValueError, KeyError, OSError) as error:
            report['failures'].append(dict(case=name, error=str(error)))
            write_json(output/'audit.json', report)
            raise
        write_json(output/'audit.json', report)
    with zipfile.ZipFile(args.external_masks) as archive:
        members = archive_members(archive)
        for record in external['cases']:
            if record.get('status') != 'ready_primary_hu':
                continue
            name, cache = record['case'], external_path.parent/record['cache_folder']
            print(f'External mask/error audit {name}', flush=True)
            mask, _ = read_member(archive, members[name])
            original, _ = lps_zyx(mask)
            if not np.isin(original, [0, 1, 2]).all():
                raise ValueError('External reference is not expert left/right lung labels')
            track(cache/'reference_native.npy')
            track(cache/'hu_native.npy')
            with mapped_npy(cache/'reference_native.npy') as r, mapped_npy(cache/'hu_native.npy') as hu:
                reference = binary(r)
                if not np.array_equal(reference, original != 0):
                    raise ValueError('External expert source differs from native cache')
                report['label_audit'].append(dict(case=name, split='external_test', source_mask_matches_native=True))
                for model, filename in [('fp32', 'float_prediction_native.npy'), ('int8', 'int8_prediction_native.npy')]:
                    path = Path(args.external_evaluation)/name/filename
                    track(path)
                    forced = [80] if name == 'coronacases_003' else [114] if name == 'coronacases_009' else []
                    with mapped_npy(path) as prediction:
                        result, slices = analyze_volume(hu, reference, prediction,
                            record['ct_geometry']['spacing_zyx'][1:], name, 'external', model, output, forced)
                    report['prediction_audit'].append(result)
                    rows.extend(slices)
            write_json(output/'audit.json', report)
    summary = {}
    for item in report['prediction_audit']:
        key = item['cohort']+'/'+item['model']
        target = summary.setdefault(key, {'cases': 0, 'slices': 0, 'totals': defaultdict(int),
            'reference_hu_bin_voxels': np.zeros(7, dtype=np.int64), 'missed_hu_bin_voxels': np.zeros(7, dtype=np.int64)})
        target['cases'] += 1
        target['slices'] += item['slices']
        for field, value in item['totals'].items():
            target['totals'][field] += value
        target['reference_hu_bin_voxels'] += item['reference_hu_bin_voxels']
        target['missed_hu_bin_voxels'] += item['missed_hu_bin_voxels']
    for target in summary.values():
        target['totals'] = dict(target['totals'])
        target['reference_hu_bin_voxels'] = target['reference_hu_bin_voxels'].tolist()
        target['missed_hu_bin_voxels'] = target['missed_hu_bin_voxels'].tolist()
    report['summary'] = summary
    for path, digest in tracked.items():
        if file_sha256(path) != digest:
            raise ValueError(f'Input changed during audit: {path}')
    report['unchanged_input_sha256'] = tracked
    report['status'] = 'complete_diagnostic_only'
    with (output/'slice_errors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/'audit.json', report)
    print(json.dumps(summary, indent=2), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', default='data/lung_learning_20261009/dataset.json')
    result.add_argument('--heldout', default='data/lung_training_20261011/heldout_2d_evaluation/test_dataset.json')
    result.add_argument('--validation', default='data/lung_training_20261011/early_stop_evaluation')
    result.add_argument('--external', default='data/lung_external_20261012/prepared/external_dataset.json')
    result.add_argument('--external-evaluation', default='data/lung_external_20261012/evaluation')
    result.add_argument('--external-masks', default='D:/CT3/Lung_Mask.zip')
    result.add_argument('--checkpoint', default='data/lung_training_20261011/early_stop_run/best.pt')
    result.add_argument('--input-root', default='D:/CT2/lctsc')
    result.add_argument('--outdir', required=True)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
