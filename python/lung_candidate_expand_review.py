"""Review frozen development effects using saved candidates/scores, without inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from lung_candidate_data import read_split, load_development_case, sha_file, write_json
from lung_candidate_rules_run import ROOT, SPLIT


def main():
    out = ROOT / 'data/lung_candidate_expand_20261018'
    report_path = out / 'evaluation/report.json'
    report = json.loads(report_path.read_text('utf-8'))
    audit = json.loads((out / 'evaluation/audit.json').read_text('utf-8'))
    if audit['status'] != 'PASS':
        raise ValueError('A successful frozen evaluation is required')
    review = out / 'effect_review'
    review.mkdir(exist_ok=False)
    cohort = report['cohorts']['fresh_development_12']
    profiles = cohort['fixed_profiles']
    models = ('old_svm', 'previous_paper', 'expanded')
    per_model = {name: {c['series_uid']: c for c in profiles[name]['4.0']['metrics']['cases']}
                 for name in models}
    references = []
    for uid, case in per_model['expanded'].items():
        for index, detail in enumerate(case['after']['reference_details']):
            item = dict(series_uid=uid, reference_index=index, reference=detail['reference'],
                        primary_detected=case['before']['reference_details'][index]['detected'])
            item['detected'] = {name: per_model[name][uid]['after']['reference_details'][index]['detected']
                                for name in models}
            references.append(item)
    gained = [r for r in references if r['detected']['expanded'] and not r['detected']['previous_paper']]
    lost = [r for r in references if not r['detected']['expanded'] and r['detected']['previous_paper']]
    primary_misses = [r for r in references if not r['primary_detected']]
    classifier_misses = [r for r in references if r['primary_detected'] and not r['detected']['expanded']]
    summary = dict(status='COMPLETE_EFFECT_REVIEW', report_sha256=sha_file(report_path),
                   audit_sha256=sha_file(out/'evaluation/audit.json'),
                   primary=cohort['primary_detectable'],
                   fixed_profiles={name: {key: dict(threshold=value['threshold'],
                       **value['metrics']['summary']['after']) for key, value in profiles[name].items()}
                       for name in models},
                   gained_vs_previous_at_frozen_4=len(gained), lost_vs_previous_at_frozen_4=len(lost),
                   primary_missed_references=primary_misses, classifier_missed_references=classifier_misses,
                   per_reference=references, heldout_scans_run=0,
                   interpretation='Development evidence only. Thresholds frozen before these 12 scans; '
                                  '4.0 is the training operating-point label, not guaranteed validation FP.',
                   image_policy='Select gained, primary-missed and classifier-missed examples by ordered '
                                'reference metadata. Crops centered on reference for retrospective review only; '
                                'no rerun of segmentation, candidate extraction, scoring or threshold tuning.')
    selected = [('Newly detected', r) for r in gained[:2]]
    selected += [('Primary-stage miss', primary_misses[0]), ('Classifier-stage miss', classifier_misses[0])]
    split = read_split(SPLIT)
    records = {r['series_uid']: r for r in split['cases']}
    with np.load(out/'evaluation/fresh_development_12_scores.npz', allow_pickle=False) as a:
        new_scores = {k: a[k] for k in a.files}
    with np.load(out/'evaluation/fresh_development_12_previous_paper_scores.npz', allow_pickle=False) as a:
        prior_scores = {k: a[k] for k in a.files}
    fig, axes = plt.subplots(len(selected), 4, figsize=(15, 3.8*len(selected)))
    for row, (reason, item) in enumerate(selected):
        uid = item['series_uid']
        hu, _ = load_development_case(split, records[uid])
        candidates = json.loads((out/'cases'/uid/'candidates.json').read_text('utf-8'))['candidates']
        center = np.array(item['reference']['center_index_zyx'])
        spacing = np.array(records[uid]['spacing_zyx_mm'])
        z, y, x = np.rint(center).astype(int)
        half = np.ceil(25 / spacing[1:]).astype(int)
        y0, y1 = max(0, y-half[0]), min(hu.shape[1], y+half[0]+1)
        x0, x1 = max(0, x-half[1]), min(hu.shape[2], x+half[1]+1)
        thresholds = [None, None, profiles['previous_paper']['4.0']['threshold'],
                      profiles['expanded']['4.0']['threshold']]
        scores = [None, None, prior_scores[uid], new_scores[uid]]
        titles = ['Reference point + diameter', 'Primary candidates', 'Previous SVM', 'Expanded training SVM']
        for col, ax in enumerate(axes[row]):
            ax.imshow(hu[z, y0:y1, x0:x1], cmap='gray', vmin=-1000, vmax=400,
                      extent=(x0*spacing[2], x1*spacing[2], y1*spacing[1], y0*spacing[1]))
            cx, cy = center[2]*spacing[2], center[1]*spacing[1]
            ax.add_patch(Circle((cx, cy), item['reference']['diameter_mm']/2,
                                fill=False, edgecolor='cyan', linewidth=1.2, linestyle='--'))
            ax.plot(cx, cy, '+', color='cyan', markersize=8)
            if col:
                for i, c in enumerate(candidates):
                    if scores[col] is not None and scores[col][i] < thresholds[col]:
                        continue
                    z0, z1, by0, by1, bx0, bx1 = c['bbox_zyx']
                    if z0 <= z < z1 and by0 < y1 and by1 > y0 and bx0 < x1 and bx1 > x0:
                        ax.add_patch(Rectangle((bx0*spacing[2], by0*spacing[1]),
                                              (bx1-bx0)*spacing[2], (by1-by0)*spacing[1],
                                              fill=False, edgecolor='orange', linewidth=1.2))
            status = '' if col == 0 else ('HIT' if (item['primary_detected'] if col == 1 else
                       item['detected']['previous_paper' if col == 2 else 'expanded']) else 'MISS')
            ax.set_title(titles[col] + (' | 3D '+status if status else ''), fontsize=10)
            ax.set_xlim(x0*spacing[2], x1*spacing[2]); ax.set_ylim(y1*spacing[1], y0*spacing[1])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{reason}\nUID ...{uid[-8:]} | ref {item['reference_index']}\n"
                              f"diameter {item['reference']['diameter_mm']:.2f} mm | z={z}", fontsize=9)
        del hu
    fig.suptitle('Fresh development: success and failure review at frozen 4-FP training profiles\n'
                 'Cyan: reference center/diameter (not a segmentation contour). Orange: predicted region bounds.\n'
                 'HIT/MISS uses 3D center matching; boxes show only this axial slice. Reference-centered crops are review only.',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(review/'reference_review.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, label in zip(models, ('Old 13-feature SVM', 'Previous paper-inspired SVM', 'Expanded 72-scan SVM')):
        curve = cohort['froc'][name]['curve']
        ax.plot(curve['fp_per_scan'], curve['sensitivity'], label=label)
    ax.axhline(cohort['primary_detectable']['sensitivity'], color='gray', linestyle='--', label='Primary-stage ceiling (20/23)')
    ax.set(xlim=(0, 8), ylim=(0, 1), xlabel='False positives per scan', ylabel='Reference nodule sensitivity',
           title='Same fresh 12 development scans / 23 reference nodules')
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(review/'fresh_12_froc.png', dpi=150); plt.close(fig)
    summary['pictured_references'] = [dict(reason=reason, **item) for reason, item in selected]
    write_json(review/'effect_summary.json', summary)
    print(json.dumps(dict(gained=len(gained), lost=len(lost), primary_misses=len(primary_misses),
                          classifier_misses=len(classifier_misses), output=str(review))))


if __name__ == '__main__':
    main()
