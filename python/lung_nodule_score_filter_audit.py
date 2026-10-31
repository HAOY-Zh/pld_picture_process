"""Quantify threshold filtering separately from fixed cross-slice rescoring.

Read completed calibration outputs only; no CT pixels or network inference.
"""
from pathlib import Path
import numpy as np
from lung_nodule_expanded_evaluation import (OUT, verify, calibration_records, verified_done,
    load_packet, read, save, sha_file, EXCLUDED, excluded_findings)
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts


def detected_set(result):
    return {(case['series_uid'], i) for case in result['cases']
            for i, detail in enumerate(case['metric']['reference_details']) if detail['detected']}


def brief(result):
    return {k:v for k,v in result.items() if k != 'cases'}


def run(out=OUT):
    plan, manifest = verify(out)
    identity = sha_file(out/'plan.json')
    frozen = read(out/'calibration_freeze.json')
    report = read(out/'evaluation.json')
    report_sha = sha_file(out/'evaluation.json')
    if report_sha != frozen['evaluation_sha256']:
        raise ValueError('Changed frozen evaluation')
    excluded = excluded_findings(EXCLUDED)
    rows = {'raw':[], 'cross':[]}
    for r in calibration_records(manifest):
        uid = r['series_uid']
        folder = out/'cases'/uid
        verified_done(folder, identity)
        packet = load_packet(folder/'expanded247.json.gz')
        xyz = np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3)
        with np.load(folder/'expanded247.npz', allow_pickle=False) as a:
            for mode in rows:
                rows[mode].append(dict(uid=uid, world=xyz, record=r,
                    excluded=excluded.get(uid, []), scores=a[mode+'_scores'].copy()))

    def checked(mode, threshold):
        result = summarize_fixed(rows[mode], threshold)
        independent = independent_counts(rows[mode], threshold)
        for key, value in independent.items():
            if result[key] != value:
                raise AssertionError('Independent count mismatch')
        return result

    baseline = checked('raw', 0.)
    main = report['results']['expanded247_cross']['profiles']['4.0']
    cross_check = checked('cross', main['threshold'])
    for key in ('detected', 'reference_nodules', 'false_positive_candidates'):
        if cross_check[key] != main[key]:
            raise AssertionError('Frozen main result mismatch')
    same_numeric_raw = checked('raw', main['threshold'])
    primary = []
    for profile in ('1.0', '4.0', '8.0'):
        a = report['results']['expanded247_raw']['profiles'][profile]
        b = report['results']['expanded247_cross']['profiles'][profile]
        for mode, value in [('raw', a), ('cross', b)]:
            counted = independent_counts(rows[mode], value['threshold'])
            if any(counted[k] != value[k] for k in counted):
                raise AssertionError('Frozen profile count mismatch')
        ah, bh = detected_set(a), detected_set(b)
        primary.append(dict(fp_budget=float(profile), raw=brief(a), cross=brief(b),
            gained=sorted(bh-ah), lost=sorted(ah-bh), net_detected=len(bh)-len(ah)))
    all_points = []
    for a, b in zip(report['results']['expanded247_raw']['froc']['operating_points'],
                    report['results']['expanded247_cross']['froc']['operating_points']):
        all_points.append(dict(fp_budget=a['fp_budget'], raw_detected=a['detected'],
            cross_detected=b['detected'], raw_actual_fp_per_scan=a['actual_fp_per_scan'],
            cross_actual_fp_per_scan=b['actual_fp_per_scan']))
    initial_hits, final_hits = detected_set(baseline), detected_set(main)
    audit = dict(status='PASS', cohort='calibration47_not_test', model='expanded247',
        evaluation_sha256=report_sha, source_sha256=sha_file(Path(__file__)),
        baseline_definition='All stored post-lung-gate/post-NMS candidates from CNN decode floor .02; not unscreened CT voxels',
        baseline=brief(baseline), final_main=brief(main),
        threshold_effect=dict(false_positives_removed=baseline['false_positive_candidates']-main['false_positive_candidates'],
            fraction_false_positives_removed=1-main['false_positive_candidates']/baseline['false_positive_candidates'],
            references_lost=sorted(initial_hits-final_hits), references_lost_count=len(initial_hits-final_hits)),
        cross_scoring_at_same_numeric_cut=dict(raw=brief(same_numeric_raw), cross=brief(main),
            false_positives_removed=same_numeric_raw['false_positive_candidates']-main['false_positive_candidates'],
            fraction_false_positives_removed=1-main['false_positive_candidates']/same_numeric_raw['false_positive_candidates'],
            gained=sorted(detected_set(main)-detected_set(same_numeric_raw)),
            lost=sorted(detected_set(same_numeric_raw)-detected_set(main)),
            caveat='Scores were transformed; equal numeric thresholds do not represent equal operating points'),
        cross_scoring_at_equal_fp_budgets=primary, all_froc_points=all_points,
        ct_pixels_loaded=0, network_inference_runs=0, new56_accessed=0)
    save(out/'score_filter_effect_audit.json', audit)
    lines = ['# 阈值过滤与跨层评分的独立影响', '',
        '依据已保存的47例开发校准输出统计，无CT读取、无重新推理、无阈值调整。', '',
        '|阶段|检出/参考结节|误报候选总数|平均误报/整卷CT|', '|---|---:|---:|---:|']
    for label, value in [('宽松候选池（CNN解码下限0.02，已经过肺范围及三维合并）', baseline),
                         ('仅原始CNN评分，数值阈值同为0.68318', same_numeric_raw),
                         ('最终折中档（跨层评分及冻结阈值）', main)]:
        lines.append(f"|{label}|{value['detected']}/{value['reference_nodules']}|{value['false_positive_candidates']}|{value['actual_fp_per_scan']:.2f}|")
    lines += ['', f"阈值从宽松候选池中排除{audit['threshold_effect']['false_positives_removed']}个参考标准下的误报候选，"
        f"同时损失{len(initial_hits-final_hits)}个本可命中的参考结节。这不等于跨层评分算法单独带来的收益。", '',
        f"固定相同数值阈值时，跨层修正把误报从{same_numeric_raw['false_positive_candidates']}降至"
        f"{main['false_positive_candidates']}，检出的参考结节集合新增"
        f"{len(detected_set(main)-detected_set(same_numeric_raw))}个、丢失"
        f"{len(detected_set(same_numeric_raw)-detected_set(main))}个。"
        '若允许两个评分模式各自校准至4误报/CT，二者均检出50/56个；因此不能仅据相同数值阈值的对照宣称提升检出率。', '',
        '|平均误报预算/CT|原始CNN检出|加入跨层评分检出|新增检出|丢失检出|', '|---:|---:|---:|---:|---:|']
    for p in primary:
        lines.append(f"|{p['fp_budget']:.0f}|{p['raw']['detected']}/56|{p['cross']['detected']}/56|{len(p['gained'])}|{len(p['lost'])}|")
    lines += ['', '跨层评分对各档位的作用不一致，应结合完整FROC及新增/丢失的具体结节评价。'
        '例如0.25及0.5误报预算下，原始CNN分别检出39及41个，跨层版分别为38及40个。'
        '直接降低阈值会同时引入更多误报。以上均为开发校准统计，新56例测试仍未运行。', '']
    (out/'score_filter_effect.md').write_text('\n'.join(lines), encoding='utf-8')
    if sha_file(out/'evaluation.json') != report_sha:
        raise AssertionError('Frozen evaluation changed during audit')
    print('BASELINE', brief(baseline))
    print('FINAL', brief(main))
    print('FILTER', audit['threshold_effect'])
    print('SAME_NUMERIC_RAW', brief(same_numeric_raw))
    print('EQUAL_FP', [{k:v for k,v in p.items() if k not in ('raw','cross')} for p in primary])
    print('ALL_POINTS', all_points)


if __name__ == '__main__':
    run()
