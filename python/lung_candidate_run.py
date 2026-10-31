"""Freeze a scan split, then run only development nodule-candidate experiments."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_candidate_data import (development_records, freeze_split, load_development_case,
                                  read_split, sha_file, write_json)
from lung_candidates import CandidateConfig, detect_candidates, log_kernels, point_metrics


def predict_lungs(hu, geometry, model, device, batch_size):
    """Frozen original FP32 model; exact historical float16 input roundtrip."""
    from lung_unet import resize_stack, torch_module
    from lung_body_gate import gate_slice
    torch = torch_module()
    raw, body = np.zeros(hu.shape, bool), np.zeros(hu.shape, bool)
    start = perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(hu), batch_size):
            end = min(len(hu), begin+batch_size)
            image = ((resize_stack(np.clip(hu[begin:end], -1000, 400), 256, 1)+1000)/1400)
            image = image.astype(np.float16).astype(np.float32)
            probability = torch.sigmoid(model(torch.from_numpy(image[:, None]).to(device)))
            probability = torch.nn.functional.interpolate(probability, size=hu.shape[1:],
                                                           mode='bilinear', align_corners=False)
            raw[begin:end] = (probability[:, 0] >= .5).cpu().numpy()
            for z in range(begin, end):
                body[z], _ = gate_slice(hu[z], raw[z], geometry['spacing_zyx_mm'][1:])
    return raw, body, (perf_counter()-start)*1000


def excluded_findings(path):
    groups = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            groups.setdefault(row['seriesuid'], []).append(dict(
                center_lps_xyz_mm=[float(row[k]) for k in ('coordX', 'coordY', 'coordZ')],
                diameter_mm=float(row['diameter_mm'])))
    return groups


def coverage(mask, nodules, geometry, offset=(0, 0, 0)):
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    basis, origin = np.asarray(geometry['basis_lps_zyx']), np.asarray(geometry['origin_lps_mm'])
    offset = np.asarray(offset)
    rows = []
    for n in nodules:
        p = np.linalg.solve(basis, np.asarray(n['center_lps_xyz_mm'])-origin)-offset
        nearest = np.rint(p).astype(int)
        center_kept = bool(mask[tuple(nearest)]) if ((nearest >= 0) & (nearest < mask.shape)).all() else False
        radius = n['diameter_mm']/2
        lo = np.maximum(0, np.floor(p-radius/spacing).astype(int))
        hi = np.minimum(mask.shape, np.ceil(p+radius/spacing).astype(int)+1)
        any_kept = False
        if (lo < hi).all():
            sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            grid = np.ogrid[sl]
            sphere = sum(((g-c)*s)**2 for g, c, s in zip(grid, p, spacing)) < radius**2
            any_kept = bool((mask[sl] & sphere).any())
        rows.append(dict(center_kept=center_kept, any_reference_sphere_kept=any_kept))
    return dict(reference_nodules=len(rows), centers_kept=sum(r['center_kept'] for r in rows),
                reference_spheres_intersecting=sum(r['any_reference_sphere_kept'] for r in rows), details=rows,
                sphere_role='reference matching neighborhood only, not expert segmentation')


def preview(hu, result, references, geometry, folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    indices = {int(np.rint(n['center_index_zyx'][0])) for n in references}
    if not indices:
        indices = {c['center_index_zyx'][0] for c in result['candidates'][:2]}
    if not indices:
        indices = {len(hu)//2}
    files = []
    for z in sorted(indices):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(hu[z], cmap='gray', vmin=-1350, vmax=150)
        shown = 0
        for c in result['candidates']:
            z0, z1, y0, y1, x0, x1 = c['bbox_zyx']
            if z0 <= z < z1:
                ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False,
                                      edgecolor='orange', linewidth=.7, alpha=.65))
                shown += 1
        for n in references:
            p = np.asarray(n['center_index_zyx'])
            if abs(p[0]-z)*spacing[0] <= n['diameter_mm']/2:
                ax.add_patch(Ellipse((p[2], p[1]), n['diameter_mm']/spacing[2],
                                     n['diameter_mm']/spacing[1], fill=False, edgecolor='cyan', linewidth=1.8))
                ax.plot(p[2], p[1], '+', color='cyan')
        ax.set_title(f'Development axial z={z}; orange=candidate bbox ({shown})\n'
                     'cyan=reference center/diameter; no diagnosis or precise nodule contour')
        ax.axis('off')
        path = folder/f'candidates_axial_{z:04d}.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        files.append(path.name)
    return files


def aggregate(report, output):
    rows = report['cases']
    nodules = sum(r['metrics']['reference_nodules'] for r in rows)
    detected = sum(r['metrics']['detected'] for r in rows)
    fp = sum(r['metrics']['false_positive_candidates'] for r in rows)
    report['summary'] = dict(completed_scans=len(rows), expected_scans=report['expected_scans'],
        reference_nodules=nodules, detected=detected, sensitivity=detected/nodules if nodules else None,
        false_positive_candidates=fp, fp_per_scan=fp/len(rows) if rows else None,
        heldout_scans_run=0, official_froc_cpm=False,
        candidate_total=sum(r['metrics']['candidates'] for r in rows))
    write_json(output/'report.json', report)
    with (output/'case_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['series_uid', 'reference_nodules', 'detected', 'sensitivity', 'fp_candidates',
                         'candidates', 'lung_ms', 'candidate_ms'])
        for row in rows:
            m = row['metrics']
            writer.writerow([row['series_uid'], m['reference_nodules'], m['detected'], m['sensitivity'],
                m['false_positive_candidates'], m['candidates'], row['lung_ms'], row['candidate_ms']])


def run_development(args):
    from lung_unet import make_model, select_device, torch_module
    split = read_split(args.split)
    records = development_records(split, args.cohort)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('Limit must be positive')
        records = records[:args.limit]
    output = args.outdir.resolve()
    if output.exists():
        raise ValueError('Output exists; preserve prior experiments')
    if not 1 <= args.batch_size <= 32:
        raise ValueError('Batch size must be 1..32')
    values = json.loads(args.params.read_text()) if args.params else {}
    if 'sigma_mm' in values:
        values['sigma_mm'] = tuple(values['sigma_mm'])
    cfg = CandidateConfig(**values).checked()
    structure = None
    if args.with_structure:
        from lung_candidate_structure import BALANCED_PILOT_CONFIG
        structure = BALANCED_PILOT_CONFIG
    sources = [args.split, args.checkpoint, args.excluded,
               Path(__file__), Path(__file__).with_name('lung_candidates.py'),
               Path(__file__).with_name('lung_candidate_data.py'),
               Path(__file__).with_name('lung_unet.py'), Path(__file__).with_name('lung_body_gate.py'),
               Path(__file__).with_name('lung_body_crop.py')]
    if structure is not None:
        sources.append(Path(__file__).with_name('lung_candidate_structure.py'))
    frozen = {str(p.resolve(strict=True)): sha_file(p) for p in sources}
    torch = torch_module()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint['image_size'] != 256 or checkpoint.get('context_slices', 1) != 1 or
            checkpoint.get('threshold', .5) != .5 or checkpoint['epoch'] != 12):
        raise ValueError('This development entry expects frozen original 2D epoch12/256/.5')
    model = make_model(checkpoint['base_channels'], 1).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    excluded = excluded_findings(args.excluded)
    output.mkdir(parents=True)
    protocol = dict(scope='development only; CPU candidate reference; GPU/CPU frozen FP32 lung predictor',
        frozen_sources=frozen, split_summary=split['summary'], selected_uids=[r['series_uid'] for r in records],
        cohort=args.cohort, limit=args.limit, parameters=asdict(cfg), lung_device=str(device),
        structure_parameters=asdict(structure) if structure else None,
        lung_model='frozen original FP32 epoch12; not INT8 or PL',
        annotations_not_algorithm_input=True, heldout_execution_allowed=False,
        identity_limit=split['split_unit'], clinical_use=False)
    write_json(output/'protocol.json', protocol)
    report = dict(status='RUNNING', expected_scans=len(records), cases=[], failures=[],
                  protocol_sha256=sha_file(output/'protocol.json'), scope=protocol['scope'])
    aggregate(report, output)
    start = perf_counter()
    try:
        for i, record in enumerate(records, 1):
            uid = record['series_uid']
            folder = output/uid
            folder.mkdir()
            print(f'[{i}/{len(records)}] {uid}: loading native HU', flush=True)
            hu, geometry = load_development_case(split, record)
            raw, body, lung_ms = predict_lungs(hu, geometry, model, device, args.batch_size)
            print(f'  lung prediction {lung_ms/1000:.2f}s; extracting candidates', flush=True)
            if structure is None:
                result, seed = detect_candidates(hu, body, geometry, cfg)
            else:
                from lung_candidate_structure import extract_suspicious_regions
                result, seed = extract_suspicious_regions(hu, body, geometry, cfg, structure)
                result['stage_centers_index_zyx']['structure'] = [c['center_index_zyx'] for c in result['candidates']]
            result.update(series_uid=uid, geometry=geometry, source_shape_zyx=list(hu.shape),
                input_raw_sha256=record['raw_sha256'], split='development',
                lung_backend=f'FROZEN_FP32_REFERENCE_{device}', lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                algorithm_source_sha256=frozen[str(Path(__file__).with_name('lung_candidates.py').resolve())])
            # Evaluation is deliberately after candidate generation; no labels enter detect_candidates.
            centers = [c['center_lps_xyz_mm'] for c in result['candidates']]
            metrics = point_metrics(centers, record['nodules'], excluded.get(uid, ()))
            stages = {}
            basis, origin = np.asarray(geometry['basis_lps_zyx']), np.asarray(geometry['origin_lps_mm'])
            for stage, points in result['stage_centers_index_zyx'].items():
                world = np.asarray(points).reshape(-1, 3) @ basis.T + origin
                stages[stage] = point_metrics(world, record['nodules'], excluded.get(uid, ()))
            offset = np.asarray(seed['bbox_zyx'])[::2]
            cover = dict(raw_lung=coverage(raw, record['nodules'], geometry),
                         body_lung=coverage(body, record['nodules'], geometry),
                         search=coverage(seed['search_mask'], record['nodules'], geometry),
                         coarse_seed=coverage(seed['seed_mask'], record['nodules'], geometry, offset))
            write_json(folder/'candidates.json', result)
            write_json(folder/'metrics.json', dict(metrics=metrics, stages=stages, coverage=cover))
            np.savez_compressed(folder/'masks_packed.npz', shape_zyx=np.array(hu.shape),
                lung_raw=np.packbits(raw), lung_body=np.packbits(body), search=np.packbits(seed['search_mask']),
                seed=np.packbits(seed['seed_mask']), seed_shape_zyx=np.array(seed['seed_mask'].shape),
                seed_bbox_zyx=np.array(seed['bbox_zyx']))
            kernels = []
            for sigma in cfg.sigma_mm:
                (gy, dy), (gx, dx) = log_kernels(sigma, geometry['spacing_zyx_mm'][1:], cfg.truncate)
                kernels.append(dict(sigma_mm=sigma, gaussian_y=gy.tolist(), derivative_y=dy.tolist(),
                                    gaussian_x=gx.tolist(), derivative_x=dx.tolist()))
            write_json(folder/'filter_contract.json', dict(kernels=kernels, boundary='reflect',
                computation='-sigma_mm**2 * (Gy_second/Gx + Gy/Gx_second); derivatives already per mm^2',
                candidate_owner='A: software and FPGA', roi_consumer='C: candidate centers/bbox',
                reference_dtype='float32 image/output; float64 coefficients; FPGA quantization not yet validated'))
            images = preview(hu, result, record['nodules'], geometry, folder)
            row = dict(series_uid=uid, metrics=metrics, stages=stages, coverage=cover, previews=images,
                       lung_ms=lung_ms, candidate_ms=result['timings_ms'].get('candidate_total', result['timings_ms']['total']))
            report['cases'].append(row)
            aggregate(report, output)
            print(f'  detected={metrics["detected"]}/{metrics["reference_nodules"]}; '
                  f'candidates={metrics["candidates"]}; FP={metrics["false_positive_candidates"]}; '
                  f'candidate_s={row["candidate_ms"]/1000:.2f}', flush=True)
            del hu, raw, body, result, seed
        for path, expected in frozen.items():
            if sha_file(path) != expected:
                raise ValueError('A frozen source changed during execution: '+path)
        report['status'] = 'COMPLETE'
    except Exception as error:
        report['status'] = 'FAIL'
        report['failures'].append(dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        report['elapsed_seconds'] = perf_counter()-start
        aggregate(report, output)
    print(json.dumps(report['summary'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    split = sub.add_parser('freeze-split')
    split.add_argument('--audit', type=Path, required=True)
    split.add_argument('--outdir', type=Path, required=True)
    split.add_argument('--seed', type=int, default=20261018)
    run = sub.add_parser('run-dev')
    run.add_argument('--split', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--excluded', type=Path, required=True)
    run.add_argument('--outdir', type=Path, required=True)
    run.add_argument('--cohort', choices=['pilot', 'check-development', 'all-development'], default='pilot')
    run.add_argument('--limit', type=int)
    run.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    run.add_argument('--batch-size', type=int, default=16)
    run.add_argument('--params', type=Path)
    run.add_argument('--with-structure', action='store_true', help='Use pilot-selected 3-D structure operating point')
    args = parser.parse_args()
    if args.command == 'freeze-split':
        print(json.dumps(freeze_split(args.audit, args.outdir, args.seed)['summary'], indent=2))
    else:
        run_development(args)


if __name__ == '__main__':
    main()
