"""Plot completed v1/v2 development results; no new neural inference."""
import json
import numpy as np
from lung_candidate_data import load_development_case, sha_file, write_json
from lung_nodule_cnn_hardneg import BASE, DEFAULT
from lung_nodule_cnn_experiment import verify, records_for


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    out = DEFAULT
    plan = verify(out)
    comparison = json.loads((out / 'comparison.json').read_text('utf-8'))
    if sha_file(out / 'model.pt') != comparison['model_sha256']:
        raise ValueError('Changed model')
    folder = out / 'effect_review'
    folder.mkdir(exist_ok=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in [('v1', 'CNN v1'), ('v2', 'CNN v2 hard backgrounds'),
                       ('svm_same_scans', 'Rule + SVM')]:
        curve = comparison[key]['curve']
        ax.plot(curve['fp_per_scan'], curve['sensitivity'], label=label)
    ax.set(xlim=(0, 8), ylim=(0, 1), xlabel='False positives per scan', ylabel='Sensitivity',
           title='Development calibration: same 19 scans / 19 reference nodules')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'froc.png', dpi=150)
    plt.close(fig)
    references = []
    for case in comparison['cases']:
        for index, (a, b) in enumerate(zip(case['v1']['reference_details'], case['v2']['reference_details'])):
            references.append(dict(uid=case['series_uid'], reference=a['reference'], index=index,
                                   v1=a['detected'], v2=b['detected']))
    selection = []
    for label, predicate in [('Recovered by v2', lambda r: r['v2'] and not r['v1']),
                              ('Lost by v2', lambda r: r['v1'] and not r['v2']),
                              ('Both detected', lambda r: r['v1'] and r['v2']),
                              ('Both missed', lambda r: not r['v1'] and not r['v2'])]:
        matches = [r for r in references if predicate(r)]
        if matches:
            selection.append((label, matches[0]))
    split, records = records_for(plan, 'calibration_scans')
    records = {r['series_uid']: r for r in records}
    thresholds = {key: next(p['threshold'] for p in comparison[key]['operating_points'] if p['fp_budget'] == 4.)
                  for key in ('v1', 'v2')}
    # Describe score/ranking misses versus missing candidate coverage separately.
    # Only completed predictions are read; these references never feed training.
    prior_review = json.loads((BASE / 'effect_review/summary.json').read_text('utf-8'))
    svm_hits = {(r['series_uid'], r['index']): r['svm'] for r in prior_review['per_reference']}
    errors = []
    for item in references:
        uid = item['uid']
        annotation = records[uid]['nodules'][item['index']]
        center = np.asarray(annotation['center_lps_xyz_mm'])
        detail = dict(item, diameter_mm=annotation['diameter_mm'],
                      svm_at4=svm_hits[(uid, item['index'])])
        for key, root in [('v1', BASE), ('v2', out)]:
            packet = json.loads((root / 'calibration_scans' / (uid + '.json')).read_text('utf-8'))
            world = np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3)
            scores = np.asarray([c['score'] for c in packet['candidates']])
            distance = np.linalg.norm(world - center, axis=1)
            hits = distance < annotation['diameter_mm'] / 2
            detail[key + '_candidate_covered'] = bool(hits.any())
            detail[key + '_best_matching_score'] = float(scores[hits].max()) if hits.any() else None
            detail[key + '_nearest_candidate_distance_mm'] = float(distance.min()) if len(distance) else None
            detail[key + '_miss_kind_at4'] = ('hit' if item[key] else
                'score_below_operating_threshold' if hits.any() else 'no_matching_decoded_candidate')
        errors.append(detail)
    bins = []
    for name, lower, upper in [('under_6mm', 0, 6), ('6_to_under_10mm', 6, 10), ('at_least_10mm', 10, float('inf'))]:
        group = [r for r in errors if lower <= r['diameter_mm'] < upper]
        bins.append(dict(size_group=name, references=len(group), v1_detected=sum(r['v1'] for r in group),
                         v2_detected=sum(r['v2'] for r in group), svm_detected=sum(r['svm_at4'] for r in group)))
    write_json(folder / 'error_analysis.json', dict(scope='Retrospective development calibration only',
        neural_forwards=0, per_reference=errors, size_groups=bins,
        gained_vs_svm_at4=sum(r['v2'] and not r['svm_at4'] for r in errors),
        lost_vs_svm_at4=sum(r['svm_at4'] and not r['v2'] for r in errors),
        v2_misses_at4=sum(not r['v2'] for r in errors),
        v2_missing_candidates=sum(not r['v2_candidate_covered'] for r in errors),
        heldout_scans_run=0))
    fig, axes = plt.subplots(len(selection), 3, figsize=(11, 3.6 * len(selection)), squeeze=False)
    for row, (category, item) in enumerate(selection):
        uid = item['uid']
        record = records[uid]
        hu, _ = load_development_case(split, record)
        spacing = np.asarray(record['spacing_zyx_mm'])
        center = np.asarray(item['reference']['center_index_zyx'])
        z, y, x = np.rint(center).astype(int)
        hy, hx = np.ceil(25 / spacing[1:]).astype(int)
        y0, y1 = max(0, y-hy), min(hu.shape[1], y+hy+1)
        x0, x1 = max(0, x-hx), min(hu.shape[2], x+hx+1)
        outputs = [[]]
        for key, root in [('v1', BASE), ('v2', out)]:
            packet = json.loads((root / 'calibration_scans' / (uid + '.json')).read_text('utf-8'))
            outputs.append([c for c in packet['candidates'] if thresholds[key] is not None and c['score'] >= thresholds[key]])
        for column, ax in enumerate(axes[row]):
            ax.imshow(hu[z, y0:y1, x0:x1], cmap='gray', vmin=-1000, vmax=400,
                      extent=(x0*spacing[2], x1*spacing[2], y1*spacing[1], y0*spacing[1]))
            ax.add_patch(Circle((center[2]*spacing[2], center[1]*spacing[1]),
                item['reference']['diameter_mm']/2, fill=False, edgecolor='cyan', linestyle='--'))
            ax.plot(center[2]*spacing[2], center[1]*spacing[1], '+', color='cyan')
            for candidate in outputs[column]:
                z0, z1, by0, by1, bx0, bx1 = candidate['bbox_zyx']
                if z0 <= z < z1 and by0 < y1 and by1 > y0 and bx0 < x1 and bx1 > x0:
                    ax.add_patch(Rectangle((bx0*spacing[2], by0*spacing[1]), (bx1-bx0)*spacing[2],
                        (by1-by0)*spacing[1], fill=False, edgecolor='orange'))
            status = '' if not column else (' HIT' if item['v1' if column == 1 else 'v2'] else ' MISS')
            ax.set_title(('Reference', 'CNN v1', 'CNN v2')[column] + status)
            ax.set_xlim(x0*spacing[2], x1*spacing[2])
            ax.set_ylim(y1*spacing[1], y0*spacing[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if not column:
                ax.set_ylabel(category + f"\n{item['reference']['diameter_mm']:.2f} mm | ...{uid[-8:]}", fontsize=9)
        del hu
    fig.suptitle('Each model at its development 4-FP/scan point\nCyan: reference center/diameter, not contour. Orange: predicted ROI. HIT/MISS uses 3D matching.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(folder / 'examples.png', dpi=140)
    plt.close(fig)
    write_json(folder / 'review.json', dict(comparison_sha256=sha_file(out / 'comparison.json'),
        source_sha256=sha_file(__file__), examples=[dict(category=c, **r) for c, r in selection],
        neural_forwards=0, development_cases_read=len(selection), heldout_scans_run=0))
    print('Saved development FROC and predicted ROI examples; no neural inference rerun')


if __name__ == '__main__':
    main()
