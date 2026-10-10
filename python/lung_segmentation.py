"""Offline LCTSC inspection, preparation and baseline evaluation.

Run from the project root. See docs/lung_segmentation.md.
No PYNQ import, FPGA programming, training or clinical diagnosis occurs here.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pydicom

from lungseg.baseline import BaselineConfig, segment_slice, segment_volume
from lungseg.dicom_io import discover, load_ct, rasterize_lungs, referenced_series
from lungseg.metrics import overlap_metrics
from lungseg.report import save_preview


def fresh_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output directory is not empty; choose a new directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def inspect_data(args):
    series, structures, skipped = discover(args.input)
    print(f'CT series: {len(series)} | RTSTRUCT files: {len(structures)} | Other files: {skipped}')
    for uid, entries in sorted(series.items()):
        first = entries[0][1]
        print(f'\nCT series UID: {uid}\n  slices: {len(entries)}'
              f' | size: {first.Rows} x {first.Columns}'
              f' | description: {getattr(first, "SeriesDescription", "(none)")}')
    for path, rt in structures:
        names = ', '.join(f'{r.ROINumber}:{r.ROIName}'
                          for r in getattr(rt, 'StructureSetROISequence', []))
        print(f'\nRTSTRUCT: {path}\n  referenced series: {sorted(referenced_series(rt))}'
              f'\n  ROIs: {names}')
    if not series:
        raise ValueError('No CT series found; point --input at the extracted DICOM directory')


def audit_data(args):
    """Classify downloaded cases without changing data or evaluating test cases.

    Header checks detect missing referenced images. They do not establish that
    every pixel file decodes or that a segmentation algorithm is accurate.
    """
    root = Path(args.input)
    if not root.is_dir():
        raise ValueError('Audit input must be a directory containing LCTSC case folders')
    out = Path(args.outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output directory is not empty; choose a new audit output directory')
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('LCTSC-'))
    if not folders:
        raise ValueError('No LCTSC case folders found')
    records = []
    for index, folder in enumerate(folders, 1):
        series, structures, skipped = discover(folder)
        reasons, missing = [], set()
        n_ct = sum(len(entries) for entries in series.values())
        if not n_ct:
            reasons.append('No CT image files found')
        if not structures:
            reasons.append('No RTSTRUCT found')
        if len(series) > 1 or len(structures) > 1:
            reasons.append('Multiple series/annotations need explicit selection')
        if len(series) == 1 and len(structures) == 1:
            uid, entries = next(iter(series.items()))
            rt = structures[0][1]
            refs = referenced_series(rt)
            if refs and uid not in refs:
                reasons.append('RTSTRUCT references a different CT series')
            actual = {str(ds.SOPInstanceUID) for _, ds in entries}
            if len(actual) != len(entries):
                reasons.append('Duplicate CT SOP Instance UID')
            # Include the series-level image list, not only slices bearing lung contours.
            expected = set()
            for frame in getattr(rt, 'ReferencedFrameOfReferenceSequence', []):
                for study in getattr(frame, 'RTReferencedStudySequence', []):
                    for ref_series in getattr(study, 'RTReferencedSeriesSequence', []):
                        if str(ref_series.SeriesInstanceUID) == uid:
                            expected.update(str(c.ReferencedSOPInstanceUID)
                                            for c in getattr(ref_series, 'ContourImageSequence', []))
            for roi in getattr(rt, 'ROIContourSequence', []):
                for contour in getattr(roi, 'ContourSequence', []):
                    expected.update(str(c.ReferencedSOPInstanceUID)
                                    for c in getattr(contour, 'ContourImageSequence', []))
            missing = expected - actual
            if missing:
                reasons.append(f'{len(missing)} referenced CT images are missing')
            if any(int(getattr(ds, 'NumberOfFrames', 1)) != 1 for _, ds in entries):
                reasons.append('Multiframe CT is unsupported by this version')
        split = 'train' if folder.name.startswith('LCTSC-Train-') else 'test'
        records.append({'case': folder.name, 'split': split,
                        'status': 'exclude_for_now' if reasons else 'available_for_preparation',
                        'ct_slices': n_ct, 'ct_series': len(series), 'rtstruct_files': len(structures),
                        'missing_referenced_slices': len(missing), 'reasons': reasons,
                        'input_path': str(folder), 'other_files': skipped})
        if index % 20 == 0:
            print(f'Checked {index}/{len(folders)} cases', flush=True)
    train = [r['case'] for r in records if r['split'] == 'train' and not r['reasons']]
    test = [r['case'] for r in records if r['split'] == 'test' and not r['reasons']]
    excluded = [r['case'] for r in records if r['reasons']]
    output = fresh_output(args.outdir)
    result = {'input_root': str(root), 'cases': records,
              'available_train_cases': train, 'held_out_test_cases': test,
              'exclude_for_now': excluded,
              'scope': 'Read-only header/reference audit, not pixel-decode or medical accuracy validation',
              'policy': 'Use training cases for development; preserve test cases for independent evaluation.'}
    write_json(output / 'audit.json', result)
    with (output / 'case_inventory.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['case', 'split', 'status', 'ct_slices', 'rtstruct_files',
                  'missing_referenced_slices', 'reasons']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in records:
            writer.writerow({**row, 'reasons': '; '.join(row['reasons'])})
    lines = ['# LCTSC 下载清点', '', f'原始路径：`{root}`', '',
             f'- 病例目录：{len(records)}', f'- 可进入准备步骤的训练病例：{len(train)}',
             f'- 留作独立评价的测试病例：{len(test)}', f'- 暂时排除：{len(excluded)}', '',
             '这里只检查文件头、序列关联及引用切片是否缺失；不保证像素解码、',
             '轮廓转换或算法准确率。原始文件未修改、未删除。', '',
             '## 暂时排除（补齐后可以重新清点）', '',
             '| 病例 | CT 切片数 | 标注数 | 原因 |', '|---|---:|---:|---|']
    lines.extend(f"| {r['case']} | {r['ct_slices']} | {r['rtstruct_files']} | {'; '.join(r['reasons'])} |"
                 for r in records if r['reasons'])
    lines.extend(['', '## 当前训练候选', '', *[f'- {name}' for name in train], '',
                  '## 独立测试候选：不参与调参', '', *[f'- {name}' for name in test], ''])
    (output / 'download_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Total {len(records)} | train available {len(train)} | '
          f'test held out {len(test)} | exclude for now {len(excluded)}')
    print(f'Inventory: {output / "case_inventory.csv"}\nReport: {output / "download_audit.md"}')


def prepare(args):
    series, structures, _ = discover(args.input)
    if args.rtstruct:
        path = Path(args.rtstruct)
        structures = [(path, pydicom.dcmread(path, stop_before_pixels=True))]
    if args.series:
        if args.series not in series:
            raise ValueError('Selected CT series UID not found; run inspect first')
        selected_uid = args.series
    elif len(series) == 1:
        selected_uid = next(iter(series))
    else:
        raise ValueError('Need exactly one CT series or explicit --series UID; run inspect first')
    matches = [(path, rt) for path, rt in structures
               if selected_uid in referenced_series(rt)]
    if args.rtstruct and not matches and not referenced_series(structures[0][1]):
        matches = structures  # explicit file still checked using ROI frame/SOP references
    if len(matches) != 1:
        raise ValueError('Need one matching RTSTRUCT; download annotation too or set --rtstruct PATH')
    print('Loading CT and checking geometry...')
    volume = load_ct(series[selected_uid])
    print('Converting expert left/right lung contours...')
    masks, roi_info = rasterize_lungs(volume, matches[0][1], args.left_roi, args.right_roi,
                                     args.allow_reference_overlap)
    if roi_info['overlap_check']['voxels']:
        print(f'Warning: preserving {roi_info["overlap_check"]["voxels"]} overlapping reference voxels; '
              'evaluation target is the left/right union.')
    reference = masks['left'] | masks['right']
    output = fresh_output(args.outdir)
    np.savez_compressed(output / 'case.npz', hu=volume.hu,
                        reference_left=masks['left'], reference_right=masks['right'],
                        spacing_zyx=volume.spacing, positions=volume.positions,
                        orientation=volume.orientation, sop_uids=np.array(volume.sop_uids),
                        series_uid=np.array(volume.series_uid), frame_uid=np.array(volume.frame_uid))
    notes = [
        'Only classic single-frame CT on a uniform, nonsheared grid is supported.',
        'HU stays float32; no resizing, resampling or signed-int16 conversion was applied.',
        'A slice without a contour is represented as empty: visually verify annotation completeness.',
        'Rasterization uses pixel-center even-odd fill with polygon boundary included; '
        'vendor rasterization may differ by edge pixels.',
        'LCTSC contour policy may exclude tumor/hilar regions: this is the dataset target, '
        'not a universal whole-lung definition.',
        'Education/engineering demonstration only; not for clinical diagnosis.'
    ]
    meta = {'source_collection': 'LCTSC (user supplied; verify actual provenance)',
            'dataset_url': 'https://www.cancerimagingarchive.net/collection/lctsc/',
            'shape_zyx': list(volume.hu.shape), 'spacing_zyx_mm': volume.spacing.tolist(),
            'series_uid': volume.series_uid, 'frame_uid': volume.frame_uid,
            'roi_info': roi_info, 'warnings': notes}
    write_json(output / 'metadata.json', meta)
    # Show the expert label independently of any automatic segmentation success.
    index = int(np.argmax(reference.sum(axis=(1, 2))))
    _, stages, _ = segment_slice(volume.hu[index], volume.spacing[1:])
    save_preview(volume.hu[index], reference[index], stages, output / 'alignment_preview.png', index)
    print(f'Prepared: {output / "case.npz"}\nAlignment preview: {output / "alignment_preview.png"}')
    print('First inspect the cyan expert contour alignment; baseline accuracy is not guaranteed.')


def load_case(path):
    with np.load(path, allow_pickle=False) as case:
        hu = case['hu']
        left, right = case['reference_left'], case['reference_right']
        spacing = case['spacing_zyx']
    if hu.ndim != 3 or not hu.size or not np.isfinite(hu).all():
        raise ValueError('Case needs a nonempty finite 3-D HU volume')
    if left.shape != hu.shape or right.shape != hu.shape:
        raise ValueError('Case annotation and HU shapes differ')
    if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
        raise ValueError('Prepared reference masks must be binary 0/1')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Case spacing must be positive (slice, row, column)')
    left, right = left != 0, right != 0
    if not left.any() or not right.any():
        raise ValueError('Need separate nonempty left/right reference masks')
    overlap = int(np.count_nonzero(left & right))
    if overlap:
        print(f'Warning: reference labels overlap at {overlap} voxels; using union, not left/right metrics.')
    return hu, left | right, spacing


def evaluate(args):
    config = BaselineConfig(args.threshold, args.min_area, args.max_components,
                            args.close_iterations)
    config.validate()
    hu, reference, spacing = load_case(args.case)
    if args.slice is not None and not 0 <= args.slice < len(hu):
        raise ValueError(f'Preview slice must be in 0..{len(hu) - 1}')
    # Fail before lengthy computation rather than overwrite existing results.
    out = Path(args.outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output directory is not empty; choose a new output directory')
    print('Running CPU baseline (NOT FPGA or neural network inference)...')
    prediction, diagnostics, elapsed = segment_volume(hu, spacing[1:], config)
    scores = overlap_metrics(prediction, reference)
    per_slice = []
    for index, info in enumerate(diagnostics):
        info['metrics'] = overlap_metrics(prediction[index], reference[index])
        per_slice.append(info)
    valid_dice = [row['metrics']['dice'] for row in per_slice if row['metrics']['dice'] is not None]
    voxel_ml = float(np.prod(spacing)) / 1000
    result = {'method': 'untrained 2D threshold/components/closing/fill-holes baseline',
              'reference_target': 'Union of expert left/right masks; original labels are not modified',
              'execution': 'CPU only; no hardware acceleration in this evaluation',
              'config': asdict(config), 'volume_metrics': scores,
              'mean_nonempty_slice_dice': float(np.mean(valid_dice)) if valid_dice else None,
              'both_empty_slices_excluded_from_mean': sum(row['metrics']['both_empty'] for row in per_slice),
              'cpu_algorithm_ms': elapsed,
              'mean_cpu_algorithm_ms_per_slice': elapsed / len(hu),
              'timing_excludes': 'DICOM I/O, rasterization, plotting, FPGA/DMA and UI',
              'prediction_volume_ml': float(prediction.sum()) * voxel_ml,
              'reference_volume_ml': float(reference.sum()) * voxel_ml,
              'slices': per_slice,
              'warning': 'No medical accuracy threshold is asserted. Inspect failures; '
                         'verify complete reference contours. Not for clinical diagnosis.'}
    output = fresh_output(args.outdir)
    np.savez_compressed(output / 'prediction.npz', prediction=prediction, spacing_zyx=spacing)
    write_json(output / 'metrics.json', result)
    index = args.slice if args.slice is not None else int(np.argmax(reference.sum(axis=(1, 2))))
    _, stages, _ = segment_slice(hu[index], spacing[1:], config)
    save_preview(hu[index], reference[index], stages, output / 'preview.png', index)
    # Optional standalone PL morphology test input. This is the selected pre-closing
    # candidate, not an expert mask and not a claim that lung segmentation ran in PL.
    np.save(output / 'candidate_mask.npy', stages['chosen'].astype(np.uint8) * 255)
    np.save(output / 'reference_mask.npy', reference[index].astype(np.uint8) * 255)
    write_json(output / 'exported_slice.json', {'slice_index': index,
               'shape': list(hu[index].shape), 'mask_encoding': 'uint8, background=0, foreground=255',
               'warning': 'Standalone PL test input only; current morphology bitstreams require 512x512.'})
    print(f'Volume Dice: {scores["dice"]:.4f} | IoU: {scores["iou"]:.4f}')
    print(f'CPU algorithm: {elapsed:.1f} ms / {len(hu)} slices (not system FPS)')
    print(f'Preview: {output / "preview.png"}\nMetrics: {output / "metrics.json"}')


def build_parser():
    parser = argparse.ArgumentParser(description='Offline lung segmentation development (education only)')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='Classify downloaded case folders and missing referenced images')
    audit.add_argument('--input', required=True)
    audit.add_argument('--outdir', required=True, help='New/empty audit output directory')
    audit.set_defaults(func=audit_data)
    inspect = commands.add_parser('inspect', help='List CT series, annotation files and ROI names')
    inspect.add_argument('--input', required=True, help='Extracted DICOM directory')
    inspect.set_defaults(func=inspect_data)
    prep = commands.add_parser('prepare', help='Align CT with expert contours and create case.npz')
    prep.add_argument('--input', required=True)
    prep.add_argument('--series', help='Explicit CT SeriesInstanceUID when input contains multiple series')
    prep.add_argument('--rtstruct', help='Explicit matching RTSTRUCT file')
    prep.add_argument('--left-roi', help='Exact ROI name or ROI number if automatic names do not match')
    prep.add_argument('--right-roi', help='Exact ROI name or ROI number')
    prep.add_argument('--allow-reference-overlap', action='store_true',
                      help='Only after visual inspection: preserve overlapping labels and evaluate their union')
    prep.add_argument('--outdir', required=True, help='New/empty directory; existing results are protected')
    prep.set_defaults(func=prepare)
    eval_parser = commands.add_parser('evaluate', help='Run baseline and compare with expert masks')
    eval_parser.add_argument('--case', required=True, help='Prepared case.npz')
    eval_parser.add_argument('--outdir', required=True)
    eval_parser.add_argument('--threshold', type=float, default=-400)
    eval_parser.add_argument('--min-area', type=float, default=25, help='Minimum candidate area, mm^2')
    eval_parser.add_argument('--max-components', type=int, default=2)
    eval_parser.add_argument('--close-iterations', type=int, default=1,
                             help='Number of complete 3x3 closing operations; 0 bypasses')
    eval_parser.add_argument('--slice', type=int, help='Zero-based preview slice; full volume still evaluated')
    eval_parser.set_defaults(func=evaluate)
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
