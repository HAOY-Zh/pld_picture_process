"""Publish local launcher only after all three physical full-case comparisons pass.

Does not touch the FPGA, stop a process, or enable an unmeasured transport.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'data/CT_Studio_ExactSpeed_20261027_04'
OUT = ROOT / 'data/exact_speed_board_acceptance_20261027'


def main():
    complete = json.loads((OUT / 'full/complete_ldct_segmentation_nodule.json').read_text(encoding='utf8'))
    assert complete['status'] == 'PHYSICAL_FULL_CASE_EXACT_PASS'
    assert not complete['cleanup_changed'] and not complete['labels_used']
    reports = complete['results']
    assert set(reports) == {'ldct', 'segmentation', 'nodule'}
    for name, report in reports.items():
        assert report['all_outputs_exact']
        assert 0 < report['optimized']['task_ms'] < report['baseline']['task_ms'], name
    assert reports['ldct']['optimized']['slices_exact'] == 351
    assert reports['segmentation']['optimized']['slices_exact'] == 351
    assert reports['segmentation']['optimized']['all_native_masks_exact']
    assert reports['nodule']['optimized']['full_heads_exact'] == 225
    assert reports['nodule']['optimized']['ranked_candidates_exact'] == 5753
    assert reports['nodule']['optimized']['selected_candidates_exact'] == 9
    assert reports['ldct']['optimized']['ldct_transport'] == 'original_v1_raw'
    for name in ('ct_workspace_web.py', 'start_ct_studio_demo.py'):
        text = (APP / 'python' / name).read_text(encoding='utf8')
        assert 'from ct_dual_io_client import DualSegStudioClient as SharedSegStudioClient' in text
        assert "'exact_host_raw_r3'" in text
    launcher = ROOT / '启动最新版CT工作台.cmd'
    prior = launcher.read_bytes()
    backup = OUT / 'previous_latest_launcher.cmd'
    if backup.exists():
        assert backup.read_bytes() == prior or b'CT_Studio_ExactSpeed_20261027_04' in prior
    else:
        backup.write_bytes(prior)
    updated = prior.replace(b'CT_Studio_DualExact_20261026_02', b'CT_Studio_ExactSpeed_20261027_04')
    assert updated.count(b'CT_Studio_ExactSpeed_20261027_04') == 2
    launcher.write_bytes(updated)
    acceptance = dict(status='PHYSICAL_FULL_CASE_EXACT_ACCEPTED',
        app=str(APP), host_speed_profile='exact_host_raw_r3', default_ldct_transport='original_v1_raw',
        service_bundle_sha256='ee06e8fddd97ba4a316b68dd4781affe4c362d41cb1616c156eb01299d1d1d3e',
        cleanup='legacy', browser_drawing_included=False,
        report_sha256=hashlib.sha256((OUT / 'full/complete_ldct_segmentation_nodule.json').read_bytes()).hexdigest(),
        tested_case_native_slices=351, expert_accuracy_re_evaluated=False, results=reports)
    (OUT / 'accepted.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding='utf8')
    candidate = json.loads((APP / 'exact_speed_candidate.json').read_text(encoding='utf8'))
    candidate.update(status='PHYSICAL_FULL_CASE_EXACT_ACCEPTED', production_not_modified=False,
        default_ldct_transport='original_v1_raw', acceptance_report=str(OUT / 'accepted.json'))
    (APP / 'exact_speed_candidate.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 三项功能实板部署与同输入验收（2026-10-27）', '',
        '同一例 351 层 CT，旧、新实现使用同一板端服务、同一模型/参数/bit 和时钟。各做一次完整病例，另对通信做 3 轮交替抽样对照。三维清理固定为 legacy。', '',
        '| 功能 | 原版整例 | 新版整例 | 耗时减少 |', '|---|---:|---:|---:|']
    labels = dict(ldct='LDCT 增强', segmentation='肺分割与重建', nodule='结节候选检测')
    for name, result in reports.items():
        lines.append('| %s | %.3f s | %.3f s | %.2f%% |' % (labels[name], result['baseline']['task_ms']/1000,
            result['optimized']['task_ms']/1000, result['latency_reduction_percent']))
    lines.extend(['', '完整任务计时包含电脑准备、串行板端请求、结果保存与相应后处理；不包含首次读取 DICOM、启动自检、首次 bit 切换预热和浏览器绘制。预热切换的独立耗时本轮未保留，实际首次运行还要另计。它是当前病例的实测，不是多个病例或多轮整例的统计结论。', '',
        '数值核对：LDCT 351 张输出哈希一致；分割 351 张完整输入/输出头与 raw/body/cleaned 三维掩膜逐元素一致，体积 3629.099121 mL；结节 225 个完整检测平面输入/输出、5753 个完整排序候选和最终 9 个候选一致。没有新专家标注评价，不把一致性当作新 Dice 或检出率。', '',
        'LDCT 压缩通信试验被拒绝：相同 10 层、3 轮交替测试，原始传输平均 518.47 ms，压缩后 704.27 ms，慢 35.84%。实际网页继续原始传输，采用电脑后台保存。分割与结节采用 V3。', '',
        '板端目录 `/home/xilinx/ct_test/ct_studio_exact_speed_20261027_04`，服务 `ct-studio-exact-speed-r4.service`，启动完整硬件自检约 146.9 秒通过；没有跳过真实硬件自检。', '',
        '电脑端目录 `data/CT_Studio_ExactSpeed_20261027_04`，主机标记 `exact_host_raw_r3`，默认 legacy 清理。`启动最新版CT工作台.cmd` 指向此目录；旧应用、旧服务和启动器备份保留。', '',
        '没有更改 RTL、模型、量化、剪枝、阈值、采样坐标或评分；21 项数值资产逐文件一致。不能声称本轮 FPGA 资源或功耗降低，也不能声称三项均达到 10%。', '',
        '完整记录：`data/exact_speed_board_acceptance_20261027/accepted.json`、`full/*_result.json` 与 `pilot/*_result.json`。网页监听与连接、开机服务配置另见同目录 `web_deployment.json`。', ''])
    (ROOT / 'docs/三项功能实板部署验收_20261027.md').write_text('\n'.join(lines), encoding='utf8')
    for name, result in reports.items():
        print(name, round(result['baseline']['task_ms']/1000, 3), '->',
              round(result['optimized']['task_ms']/1000, 3), 's;',
              round(result['latency_reduction_percent'], 2), '%; ALL_EXACT')
    print('Latest launcher points to physically accepted raw-LDCT/V3-segmentation host.')


if __name__ == '__main__':
    main()
