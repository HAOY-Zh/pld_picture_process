"""Frozen-model development-only screening of conservative 3D island removal.

Filter inputs: binary prediction, physical spacing and fixed parameters ONLY.
Expert masks are read separately for evaluation, never to choose component IDs.
CPU geometry experiment; not clinical use, retraining or FPGA implementation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from lung_segmentation import fresh_output, write_json
from lung_unet import (_case_prediction, file_sha256, make_model, mapped_npy,
                       read_manifest, select_device, torch_module)

PRIMARY = dict(max_island_ml=5., min_distance_mm=20.)
GRID = [dict(max_island_ml=v, min_distance_mm=d) for v in [1., 3., 5.] for d in [10., 20., 30.]]


def binary_volume(value):
    a = np.asarray(value)
    if a.ndim != 3 or not a.size or not np.isin(a, [0, 1, 255]).all():
        raise ValueError('Need a nonempty 3D binary mask')
    return a != 0


def analyze(prediction, spacing, max_query_ml=5., minimum_main_ml=50., anchor_fraction=.02):
    """26-connected components and nearest anchor voxel-centre distances (mm)."""
    a = binary_volume(prediction)
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Need positive finite ZYX spacing')
    if not np.isfinite([max_query_ml, minimum_main_ml, anchor_fraction]).all() or min(max_query_ml, minimum_main_ml) <= 0 or not 0 < anchor_fraction <= 1:
        raise ValueError('Invalid geometry analysis parameters')
    start = perf_counter()
    labels, count = ndimage.label(a, structure=ndimage.generate_binary_structure(3, 3))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    sizes[0] = 0
    factor = float(np.prod(spacing) / 1000.)
    volumes = sizes * factor
    largest = float(volumes.max())
    reliable = largest >= minimum_main_ml
    # Multiple sufficiently large components are protected, not always exactly two.
    anchor_limit = max(10., largest * anchor_fraction)
    anchors = np.flatnonzero((volumes >= anchor_limit) & (sizes > 0)) if reliable else np.array([], int)
    lookup = np.zeros(count+1, bool); lookup[anchors] = True
    tree = None
    if len(anchors):
        anchor_mask = lookup[labels]
        surface = anchor_mask & ~ndimage.binary_erosion(anchor_mask, border_value=0)
        points = np.argwhere(surface).astype(float) * spacing
        tree = cKDTree(points)
        del anchor_mask, surface, points
    objects = ndimage.find_objects(labels)
    components = []
    for i, box in enumerate(objects, 1):
        if box is None:
            continue
        distance = 0. if lookup[i] else None
        if tree is not None and not lookup[i] and volumes[i] <= max_query_ml:
            local = np.argwhere(labels[box] == i)
            points = (local + [s.start for s in box]) * spacing
            distance = float(tree.query(points, workers=1)[0].min())
        components.append(dict(id=i, voxels=int(sizes[i]), volume_ml=float(volumes[i]),
            anchor=bool(lookup[i]), nearest_anchor_mm=distance,
            bbox_zyx=[[s.start, s.stop] for s in box]))
    info = dict(connectivity=26, components=components, anchors=anchors.tolist(),
        spacing_zyx_mm=spacing.tolist(), max_query_ml=max_query_ml,
        minimum_main_ml=minimum_main_ml, anchor_fraction=anchor_fraction,
        largest_component_ml=largest, reliable_anchor=reliable,
        cpu_analysis_ms=(perf_counter()-start)*1000,
        distance_definition='Minimum Euclidean distance between native voxel centres, spacing-scaled mm')
    return labels, info


def removed_ids(info, max_island_ml=5., min_distance_mm=20.):
    if not np.isfinite([max_island_ml, min_distance_mm]).all() or min(max_island_ml, min_distance_mm) < 0:
        raise ValueError('Nonnegative finite filtering parameters required')
    if max_island_ml > info['max_query_ml']:
        raise ValueError('Distance analysis does not cover requested component volume')
    if not info['reliable_anchor']:
        return []
    return [c['id'] for c in info['components'] if not c['anchor'] and
        c['volume_ml'] <= max_island_ml and c['nearest_anchor_mm'] is not None and
        c['nearest_anchor_mm'] > min_distance_mm]


def materialize(labels, ids):
    lookup = np.ones(int(labels.max())+1, bool)
    lookup[0] = False
    lookup[ids] = False
    return lookup[labels]


def clean_prediction(prediction, spacing, **parameters):
    labels, info = analyze(prediction, spacing, max_query_ml=max(5., parameters.get('max_island_ml', 5.)))
    ids = removed_ids(info, **parameters)
    return materialize(labels, ids), dict(info, removed_component_ids=ids)


def evaluation_tables(labels, reference):
    """Scoring only, AFTER geometry decisions. Never provided to removed_ids."""
    g = binary_volume(reference)
    if g.shape != labels.shape:
        raise ValueError('Reference/prediction grid mismatch')
    n = int(labels.max()) + 1
    counts, overlaps = [], []
    for z in range(len(labels)):
        counts.append(np.bincount(labels[z].ravel(), minlength=n))
        overlaps.append(np.bincount(labels[z][g[z]], minlength=n))
    counts, overlaps = np.array(counts), np.array(overlaps)
    counts[:, 0] = 0; overlaps[:, 0] = 0
    return counts, overlaps, g.sum(axis=(1, 2))


def score(tables, ids):
    counts, overlaps, reference_area = tables
    pp = counts.sum(axis=1) - counts[:, ids].sum(axis=1)
    intersections = overlaps.sum(axis=1) - overlaps[:, ids].sum(axis=1)
    tp = int(intersections.sum()); fp = int(pp.sum()) - tp; fn = int(reference_area.sum()) - tp
    denominator = 2*tp+fp+fn
    dice = np.divide(2.*intersections, reference_area+pp, out=np.zeros(len(pp), float), where=(reference_area+pp)>0)
    small = (reference_area > 0) & (reference_area <= 1000)
    empty = reference_area == 0
    removed_tp = int(overlaps[:, ids].sum())
    removed_all = int(counts[:, ids].sum())
    return dict(dice=2*tp/denominator if denominator else None, true_positive=tp,
        false_positive=fp, false_negative=fn, removed_true_lung_voxels=removed_tp,
        removed_false_positive_voxels=removed_all-removed_tp,
        removed_components=len(ids), empty_slices_with_prediction=int((pp[empty]>0).sum()),
        empty_false_positive_voxels=int(pp[empty].sum()), small_slices=int(small.sum()),
        mean_small_slice_dice=float(dice[small].mean()) if small.any() else None,
        lost_positive_slices=int(((reference_area>0)&(counts.sum(axis=1)>0)&(pp==0)).sum()))


def summarize(rows):
    if not rows:
        raise ValueError('No validation cases')
    result = dict(mean_case_dice=float(np.mean([r['dice'] for r in rows])),
        worst_case_dice=min(r['dice'] for r in rows), cases=len(rows))
    for key in ['true_positive', 'false_positive', 'false_negative', 'removed_true_lung_voxels',
                'removed_false_positive_voxels', 'removed_components', 'empty_slices_with_prediction',
                'empty_false_positive_voxels', 'small_slices', 'lost_positive_slices']:
        result[key] = sum(r[key] for r in rows)
    result['mean_small_slice_dice'] = (sum(r['small_slices'] * r['mean_small_slice_dice']
        for r in rows if r['small_slices']) / result['small_slices']) if result['small_slices'] else None
    return result


def comparison_preview(reference, original, cleaned, spacing, index, path, case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    views = [(reference, 'Expert'), (original, 'Original prediction'), (cleaned, 'Conservative cleanup')]
    y = reference.shape[1]//2
    for j, (mask, name) in enumerate(views):
        axes[0,j].imshow(mask[index], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0,j].set_title(f'{name} | axial {index}')
        axes[1,j].imshow(mask[::-1,y,:], cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                         aspect=spacing[0]/spacing[2])
        axes[1,j].set_title(f'{name} | coronal {y}')
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(case+' | native binary masks, no smoothing | education only')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run(args):
    root, manifest = read_manifest(args.dataset)
    records = [r for r in manifest['cases'] if r['split']=='validation']
    if not records or any('Test-' in r['case'] for r in records):
        raise ValueError('Development validation cases only')
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError('Positive batch/thread settings required')
    torch = torch_module(); torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint.get('context_slices',1)!=1 or checkpoint.get('kind')=='soft_level_v1' or
        checkpoint['dataset_fingerprint']!=manifest['fingerprint'] or checkpoint['hu_clip']!=[-1000,400] or
        checkpoint['image_size']!=manifest['image_size'] or checkpoint['threshold']!=.5):
        raise ValueError('Frozen original full-field 2D model required')
    protected = [Path(args.dataset), Path(args.checkpoint), Path(__file__)]
    for r in records:
        protected += [root/r['cache_folder']/name for name in ['images.npy','reference_native.npy']]
    hashes = {str(p.resolve()):file_sha256(p) for p in protected}
    output = fresh_output(args.outdir)
    protocol = dict(primary=PRIMARY, descriptive_grid=GRID, connectivity=26,
        anchors='Protect components >= max(10 mL, 2% of largest). If largest <50 mL, do nothing.',
        filter='Remove ONLY non-anchor component with volume <= limit AND minimum physical distance > limit.',
        primary_acceptance='All six development cases: no true lung voxel removed, no case Dice decrease, some FP removed.',
        selection='Primary 5 mL /20 mm fixed before evaluation; grid descriptive, not auto-picked by GT.',
        validation_cases=[r['case'] for r in records], train_cases=[r['case'] for r in manifest['cases'] if r['split']=='train'],
        input_sha256=hashes, warning='Repeated development validation, not new blind test. No Test/COVID accessed. CPU geometry, not PL.',
        model_epoch=checkpoint['epoch'], model_sha256=file_sha256(args.checkpoint))
    write_json(output/'protocol.json',protocol)
    report = dict(status='running',protocol=protocol,cases=[],original_model_retained=True)
    write_json(output/'evaluation.json',report)
    device = select_device(args.device)
    model = make_model(checkpoint['base_channels'],1).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    previous = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        for r in records:
            print('Frozen inference and 3D geometry: '+r['case'],flush=True)
            original, inference_ms = _case_prediction(model,root,r,device,args.batch_size)
            labels, info = analyze(original,r['spacing_zyx_mm'])
            # Geometry decisions are committed before this case's GT is read.
            decisions = [('original',[]), ('primary',removed_ids(info,**PRIMARY))]
            ranked = sorted(info['components'],key=lambda c:(-c['voxels'],c['id']))
            decisions += [('largest_two_control',[c['id'] for c in ranked[2:]])]
            decisions += [(f'grid_{c["max_island_ml"]:g}ml_{c["min_distance_mm"]:g}mm',removed_ids(info,**c)) for c in GRID]
            folder=output/r['case'];folder.mkdir()
            write_json(folder/'geometry_decisions.json',dict(info,removed_by_rule=dict(decisions),expert_used=False))
            with mapped_npy(root/r['cache_folder']/'reference_native.npy') as reference:
                tables=evaluation_tables(labels,reference)
                scores={name:score(tables,ids) for name,ids in decisions}
                cleaned=materialize(labels,decisions[1][1])
                control=materialize(labels,decisions[2][1])
                np.savez_compressed(folder/'predictions_native.npz',original=original,primary=cleaned,largest_two=control)
                # Pick maximal removed slice only for transparent visualization, never for rule choice.
                removed_area=(original&~cleaned).sum(axis=(1,2))
                index=int(removed_area.argmax()) if removed_area.any() else len(original)//2
                comparison_preview(reference!=0,original,cleaned,r['spacing_zyx_mm'],index,folder/'preview.png',r['case'])
                reference_before=hashes[str((root/r['cache_folder']/'reference_native.npy').resolve())]
            overlaps=tables[1].sum(axis=0)
            component_evaluation=[dict(c,expert_overlap_voxels=int(overlaps[c['id']])) for c in info['components']]
            row=dict(case=r['case'],split=r['split'],shape=r['native_shape'],spacing_zyx_mm=r['spacing_zyx_mm'],
                scores=scores,components=component_evaluation,inference_ms=inference_ms,cpu_analysis_ms=info['cpu_analysis_ms'],
                prediction_sha256=file_sha256(folder/'predictions_native.npz'),reference_sha256=reference_before,
                preview_axial_slice=index)
            report['cases'].append(row)
            write_json(output/'evaluation.json',report)
            print(f'  Dice {scores["original"]["dice"]:.6f} -> {scores["primary"]["dice"]:.6f}; '
                  f'removed FP {scores["primary"]["removed_false_positive_voxels"]}, true lung {scores["primary"]["removed_true_lung_voxels"]}',flush=True)
            del original,cleaned,control,labels,tables
        summaries={name:summarize([r['scores'][name] for r in report['cases']]) for name,_ in decisions}
        accept=(summaries['primary']['removed_true_lung_voxels']==0 and
                summaries['primary']['removed_false_positive_voxels']>0 and
                all(r['scores']['primary']['dice']>=r['scores']['original']['dice'] for r in report['cases']))
        for p,expected in hashes.items():
            if file_sha256(p)!=expected:
                raise ValueError('Protected original input changed: '+p)
        report.update(status='complete_development_screen',summaries=summaries,protected_inputs_unchanged=True,
            primary_accepted_for_optional_development_view=bool(accept),
            decision='offer_primary_as_optional_not_default' if accept else 'do_not_adopt_primary',
            warning='No independent Test/COVID validation performed; original checkpoint and predictions remain available.')
        write_json(output/'evaluation.json',report)
        print(json.dumps(dict(decision=report['decision'],original=summaries['original'],primary=summaries['primary']),indent=2),flush=True)
    except BaseException as e:
        report.update(status='aborted_incomplete',error=str(e));write_json(output/'evaluation.json',report);raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    p.add_argument('--outdir',required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    return p


if __name__=='__main__':
    run(parser().parse_args())
