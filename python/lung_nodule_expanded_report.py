"""Audit complete paired calibration artifacts and describe the full result."""
from pathlib import Path
import numpy as np
from lung_nodule_expanded_evaluation import (OUT, LABELS, verify, calibration_records,
    verified_done, load_packet, read, save, sha_file, rerank_scores)


def report(out=OUT):
    plan, manifest = verify(out)
    evaluation = read(out/'evaluation.json')
    frozen = read(out/'calibration_freeze.json')
    identity = sha_file(out/'plan.json')
    if (evaluation['status'] != 'COMPLETE' or evaluation['scans'] != 47 or
            frozen['evaluation_sha256'] != sha_file(out/'evaluation.json')):
        raise ValueError('Need complete frozen calibration47')
    ranker = read(Path(plan['ranker']))
    reused = 0
    for r in calibration_records(manifest):
        folder = out/'cases'/r['series_uid']
        verified_done(folder, identity)
        for name in LABELS:
            packet = load_packet(folder/(name+'.json.gz'))
            cs = packet['candidates']
            coordinates = np.array([c['center_index_zyx'] for c in cs]).reshape(-1, 3)
            world = np.array([c['center_lps_xyz_mm'] for c in cs]).reshape(-1, 3)
            expected = coordinates@np.array(r['basis_lps_zyx']).T+np.array(r['origin_lps_mm'])
            np.testing.assert_allclose(world, expected, atol=1e-6, rtol=0)
            q = np.rint(coordinates).astype(int)
            if not ((q >= 0) & (q < r['shape_zyx'])).all():
                raise ValueError('Candidate outside native decode gate')
            with np.load(folder/(name+'.npz'), allow_pickle=False) as a:
                np.testing.assert_array_equal(a['raw_scores'], [c['score'] for c in cs])
                np.testing.assert_allclose(a['cross_scores'], rerank_scores(a['raw_scores'], a['features'], ranker),
                                           atol=1e-14, rtol=0)
            if name == 'old53' and packet['baseline_output_reused']:
                original = read(Path(plan['reused_baseline'][r['series_uid']]))
                if original['candidates'] != cs:
                    raise ValueError('Cached original candidates changed')
                with np.load(folder/(name+'.npz'), allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['features'], original['features'])
                reused += 1
    if reused != 19:
        raise ValueError('Expected 19 exact cached baselines')
    results = evaluation['results']
    case_maps = {n:{c['series_uid']:c['metric'] for c in results[n+'_cross']['profiles']['4.0']['cases']} for n in LABELS}
    changes, strata = [], {}
    subgroups = {n:{} for n in LABELS}
    for r in calibration_records(manifest):
        uid = r['series_uid']
        group = 'original19' if uid in plan['reused_baseline'] else 'added28'
        for n in LABELS:
            m = case_maps[n][uid]
            s = subgroups[n].setdefault(group, dict(scans=0, references=0, detected=0, false_positives=0))
            s['scans'] += 1
            s['references'] += m['reference_nodules']
            s['detected'] += m['detected']
            s['false_positives'] += m['false_positive_candidates']
        for a, b in zip(case_maps['old53'][uid]['reference_details'],case_maps['expanded247'][uid]['reference_details']):
            if a['reference'] != b['reference']:
                raise ValueError('Reference order mismatch')
            d = b['reference']['diameter_mm']
            size = 'under6' if d < 6 else '6to10' if d < 10 else 'over10'
            s = strata.setdefault(size, dict(references=0, old_detected=0, new_detected=0))
            s['references'] += 1
            s['old_detected'] += int(a['detected'])
            s['new_detected'] += int(b['detected'])
            if a['detected'] != b['detected']:
                changes.append(dict(series_uid=uid, diameter_mm=d, old_detected=a['detected'],
                    new_detected=b['detected'], center_lps_xyz_mm=b['reference']['center_lps_xyz_mm']))
    audit = dict(status='PASS', scans=47, reference_nodules=56, model_coordinates_and_scores_checked=True,
        exact_cached_baselines_reused=reused, fixed_counts_and_stable_froc_audit='PASS',
        evaluation_sha256=sha_file(out/'evaluation.json'), protocol_sha256=identity,
        additional_audit_source_sha256=sha_file(Path(__file__)), sealed_test_scans_accessed=0)
    if (out/'worker_identity.json').exists():
        worker = read(out/'worker_identity.json')
        if worker['plan_sha256'] != identity or worker['source_sha256'] != sha_file(Path(__file__).with_name('lung_nodule_expanded_eval_worker.py')):
            raise ValueError('Companion worker source/protocol identity changed')
        audit['parallel_worker'] = worker
    save(out/'audit.json', audit)
    save(out/'paired_main_analysis.json', dict(profile='cross calibrated <=4 FP/CT on all47',
        groups=subgroups, sizes=strata, changed_references=changes,
        gained=sum(c['new_detected'] for c in changes), lost=sum(c['old_detected'] for c in changes),
        interpretation='subgroups use same all47 threshold, never separately recalibrated; development only'))
    print('AUDIT PASS; all47 geometry and raw/cross scores verified; original19 bitwise reused')
    print('MAIN', {n:{k:v for k,v in results[n+'_cross']['profiles']['4.0'].items() if k!='cases'} for n in LABELS})
    print('PAIRED', 'gained', sum(c['new_detected'] for c in changes), 'lost', sum(c['old_detected'] for c in changes))
    print('SIZES', strata)
    old = results['old53_cross']['profiles']
    new = results['expanded247_cross']['profiles']
    lines = ['# 肺结节扩充训练：47例开发校准结果', '',
        '2026-10-20。47例、56个参考结节全部评价完成；包括16例无该参考标准收录结节的CT。'
        '这是开发校准结果，新56例测试尚未运行，不能当作新测试成绩。', '',
        '两个网络均为13,157参数；旧版由53例训练，新版在247例上额外续训30轮。'
        '两者使用相同预测肺范围、物理网格、输入批次、解码、三维合并和冻结跨层评分。', '',
        '## 相同误报预算下的新旧对照', '',
        '|档位|旧版检出/参考|旧版检出率|旧版平均误报/CT|新版检出/参考|新版检出率|新版平均误报/CT|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for profile, title in [('1.0', '低误报，预算1/CT'), ('4.0', '折中，预算4/CT'), ('8.0', '高召回，预算8/CT')]:
        a, b = old[profile], new[profile]
        lines.append(f"|{title}|{a['detected']}/{a['reference_nodules']}|{a['sensitivity']:.1%}|{a['actual_fp_per_scan']:.2f}|"
                     f"{b['detected']}/{b['reference_nodules']}|{b['sensitivity']:.1%}|{b['actual_fp_per_scan']:.2f}|")
    delta = new['4.0']['detected']-old['4.0']['detected']
    direction = '增加' if delta >= 0 else '减少'
    lines += ['', f"折中档检出数{direction}{abs(delta)}个；逐结节匹配显示新版新增检出"
              f"{sum(c['new_detected'] for c in changes)}个，同时丢失旧版已检出的{sum(c['old_detected'] for c in changes)}个。",
        '', '每个模型的阈值均只依据完整47例校准，在其原始候选曲线上选择不超过相应误报预算的运行点；'
        '同一参考结节的重复命中不计额外检出，参考忽略标注的命中不计误报，无Top-K截断。'
        '“误报/CT”按全部47卷计算，包括无参考结节病例。', '',
        '## 原始CNN评分对照', '',
        '|评分|旧版折中档检出率|旧版误报/CT|新版折中档检出率|新版误报/CT|',
        '|---|---:|---:|---:|---:|']
    for mode, title in [('raw', '原始CNN'), ('cross', '相同冻结跨层评分')]:
        a = results['old53_'+mode]['profiles']['4.0']
        b = results['expanded247_'+mode]['profiles']['4.0']
        lines.append(f"|{title}|{a['sensitivity']:.1%}|{a['actual_fp_per_scan']:.2f}|{b['sensitivity']:.1%}|{b['actual_fp_per_scan']:.2f}|")
    lines += ['', '## 折中档细分', '', '|结节直径|参考结节|旧版检出|新版检出|', '|---|---:|---:|---:|']
    for key, title in [('under6', '<6 mm'), ('6to10', '6至<10 mm'), ('over10', '≥10 mm')]:
        s = strata.get(key, dict(references=0, old_detected=0, new_detected=0))
        lines.append(f"|{title}|{s['references']}|{s['old_detected']}|{s['new_detected']}|")
    lines += ['', '|校准来源|模型|检出/参考|平均误报/CT|', '|---|---|---:|---:|']
    for group, title in [('original19', '原19例'), ('added28', '新增28例')]:
        for name, label in [('old53', '旧版'), ('expanded247', '新版')]:
            s = subgroups[name][group]
            lines.append(f"|{title}|{label}|{s['detected']}/{s['references']}|{s['false_positives']/s['scans']:.2f}|")
    lines += ['', '上面的细分沿用完整47例选定的同一个折中阈值，没有对子组单独重新选阈值。', '',
        '## 核验及使用范围', '',
        '- 全47例模型/输入身份、物理坐标、原始与跨层评分核验通过；原19例旧模型输出及特征精确复用。',
        '- 固定点的检出和误报计数另行核对，FROC采用独立ROC整数计数复核。',
        '- 为缩短等待增加反向病例工作进程，复用同一冻结推理函数，通过完整目录原子提交避免覆盖；源码身份另存。',
        '- 新版同时包含数据扩充、额外训练及新增背景采样教师的影响，不能全部单独归因于病例数量。',
        '- 没有重新训练跨层评分器，没有修改LDCT或U-Net；网页默认模型尚未替换。',
        '- 这些结果不能和旧45例测试结果直接计算提升，也不是官方LUNA16十折成绩或FPGA测速。', '',
        '下一步使用已冻结阈值评价新56例，才能进一步判断泛化效果；不要基于新测试结果重新挑运行阈值。', '',
        '原始记录：[完整统计](../data/lung_nodule_expanded_calibration_20261020_v1/evaluation.json)、'
        '[核验](../data/lung_nodule_expanded_calibration_20261020_v1/audit.json)、'
        '[逐结节变化](../data/lung_nodule_expanded_calibration_20261020_v1/paired_main_analysis.json)、'
        '[冻结运行阈值](../data/lung_nodule_expanded_calibration_20261020_v1/calibration_freeze.json)。', '']
    doc = out.parents[1]/'docs/lung_nodule_expanded_calibration_20261020.md'
    doc.write_text('\n'.join(lines), encoding='utf-8')
    training_doc = doc.with_name('lung_nodule_expanded_training_20261020.md')
    text = training_doc.read_text(encoding='utf-8')
    text = text.replace('18:30启动47例开发校准的新旧模型评价，尚待全部完成统计；新56例测试继续封存。',
        '47例开发校准的新旧模型评价及独立核验已完成，'
        '[检出率和误报结果](lung_nodule_expanded_calibration_20261020.md)已补充；新56例测试继续封存。')
    training_doc.write_text(text, encoding='utf-8')
    print('DOCUMENT', doc)


if __name__ == '__main__':
    report()
