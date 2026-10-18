# Standalone moderate-candidate adaptation; prior frozen runner remains unchanged.
"""Create a readable report/paired plot only after the complete frozen test audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_prune_fulltest import REBUILT
from lung_segmentation import write_json
from lung_unet import file_sha256,mapped_npy


def run(root):
    root=Path(root).resolve()
    if (root/'full_test_report.md').exists():raise FileExistsError('Preserve generated full-test report')
    report=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    audit=json.loads((root/'returned_mask_audit.json').read_text(encoding='utf-8'))
    if audit['status']!='verified_all_27_frozen_test_cases_saved_masks' or audit['evaluation_sha256']!=file_sha256(root/'evaluation.json'):
        raise ValueError('Need independently verified full test')
    lines=['# 冻结剪枝模型全量测试结果','',
        '实际完成全部27个可用病例、5286张切片。以下主要比较使用与FPGA相同的整数计算规范，在电脑上执行，不能作为实板速度或临床可靠性证明。','',
        '两模型均使用原HU预处理、256输入、sigmoid后原生双线性恢复、0.5阈值和相同CT-only身体约束/三维清理。',
        '候选固定为 moderate；没有训练、重新校准、改阈值或根据测试分数再选模型。',
        '新候选FP32和定点均对全部5286层重新推理。原模型预测复用上一轮已独立审核的原生掩膜；逐文件校验哈希，重新执行后处理并核对全部重叠指标。',
        '原模型二维边界指标在确认输入、专家标注及最终掩膜完全相同后复用，新候选边界重新计算。','',
        '本轮结构在查看上一轮测试结果后提出，这些病例属于重复确认，不能视作新的独立盲测。',
        '本轮与上一轮balanced还存在初始化、训练预算及随机种子差异；候选间的效果差异不能单独归因于保留dec2通道。','',
        '| 数据集 | 病例/切片 | 原模型最终Dice | 剪枝最终Dice | 变化（百分点） | 最差单病例变化（百分点） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    cohort_names={'lctsc_heldout':'LCTSC留出子集','covid_external':'COVID原HU外部集'}
    for cohort,data in report['cohorts'].items():
        s=data['summaries'];p=data['pairs']['fixed_final']
        lines.append('| {} | {}/{} | {:.6f} | {:.6f} | {:+.3f} | {:+.3f} |'.format(cohort_names[cohort],data['cases'],data['slices'],
            s['original_fixed_final']['mean_case_dice'],s['pruned_fixed_final']['mean_case_dice'],
            100*p['mean_case_dice_delta'],100*p['worst_case_dice_delta']))
    lines+=['','平均Dice先按病例计算再平均；两个数据集分开汇报。用户接受的1个百分点指数据集均值绝对差0.01，不能把某个病例的退化隐藏在均值中。','',
        '| 数据集 | 原模型小肺Dice | 剪枝小肺Dice | 原/剪枝空肺假阳性像素 | 超过1个百分点下降的病例 |','| --- | ---: | ---: | ---: | ---: |']
    for cohort,data in report['cohorts'].items():
        a=data['summaries']['original_fixed_final']['slice_quality'];b=data['summaries']['pruned_fixed_final']['slice_quality']
        lines.append('| {} | {:.6f} | {:.6f} | {} / {} | {} |'.format(cohort_names[cohort],
            a['groups']['small']['mean_slice_dice'],b['groups']['small']['mean_slice_dice'],a['empty_false_positive_pixels'],b['empty_false_positive_pixels'],
            data['pairs']['fixed_final']['cases_dropped_more_than_one_percentage_point']))
    lines+=['','## 原始输出与FP32对照','',
        '| 数据集 | 路径 | 原模型Dice | 剪枝Dice | 变化（百分点） |','| --- | --- | ---: | ---: | ---: |']
    for cohort,data in report['cohorts'].items():
        for kind,stage in [('fp32','raw'),('fixed','raw'),('fp32','final')]:
            a=data['summaries']['original_'+kind+'_'+stage]['mean_case_dice'];b=data['summaries']['pruned_'+kind+'_'+stage]['mean_case_dice']
            lines.append('| {} | {} {} | {:.6f} | {:.6f} | {:+.3f} |'.format(cohort_names[cohort],kind,stage,a,b,100*(b-a)))
    lines+=['','## 全部病例：定点最终结果','',
        '| 病例 | 切片 | 原模型Dice | 剪枝Dice | 变化（百分点） | 新增/恢复零重叠阳性层 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['cases']:
        a=row['scores']['original_fixed_final']['metrics']['dice'];b=row['scores']['pruned_fixed_final']['metrics']['dice']
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:+.3f} | {} / {} |'.format(row['case'],row['slices'],a,b,100*(b-a),
            row['paired_slice_audit']['newly_zero_overlap_slices'],row['paired_slice_audit']['recovered_overlap_slices']))
    lines+=['','新增/恢复按与专家区域是否存在真阳性重叠计算。非空但全是假阳性的预测，不计为成功检出。',
        '逐切片详情在各病例的 fixed_final_slice_metrics.csv；全部原尺寸预测保存在该病例四份 *_native.npz。','',
        '## 覆盖范围与解释','',
        '- 24个LCTSC官方测试目录中，17个CT/专家标注可用；7个仍只有RTSTRUCT、没有CT，不能算作已完成24例官方测试。',
        '- COVID全部20份数据已有审计；10份原HU纳入，另10份已窗宽化UINT8按原协议排除，不根据分数剔除。',
        '- 旧缓存已清理，本轮从原始CT/RTSTRUCT/官方肺标注ZIP重建；所有27例输入与原生专家掩膜均与历史SHA256逐字节一致。',
        '- 原模型与候选的FP32及硬件整数参考均重新推理。缓存、检查点、代码、阈值、后处理参数均冻结并核验。',
        '- 这是对已选定候选的一次确认。原模型此前看过这批测试结果，不能称作全新盲测；本轮结束后也没有再据此调参。',
        '- 专家标签未修改、未重采样。外部NIfTI缺少PatientID/SOP，精确输入不重复不能证明患者身份完全独立。',
        '- FPGA时延与完整实板病例推理仍需板端验收；电脑数值评价不代替它们。','',
        '独立审计：27例、5286层、324组原始/身体约束/最终掩膜的TP/FP/FN/Dice重算通过。','']
    lines+=['## 逐病例风险与边界复核','',
        '| 病例 | 原/剪枝空肺FP像素 | 原/剪枝零重叠阳性层 | 原/剪枝二维平均边界距离mm |',
        '| --- | ---: | ---: | ---: |']
    for row in report['cases']:
        a=row['scores']['original_fixed_final'];b=row['scores']['pruned_fixed_final']
        lines.append('| {} | {} / {} | {} / {} | {:.3f} / {:.3f} |'.format(row['case'],
            a['slice_quality']['empty_false_positive_pixels'],b['slice_quality']['empty_false_positive_pixels'],
            row['missed']['original_fixed']['positive_slices_with_zero_true_positive'],
            row['missed']['pruned_fixed']['positive_slices_with_zero_true_positive'],
            row['boundaries']['original_fixed']['mean_symmetric_boundary_mm'],
            row['boundaries']['pruned_fixed']['mean_symmetric_boundary_mm']))
    lines+=['','边界距离为所有有标注且预测非空切片上的原生二维对称边界距离；完全空预测另计，不能被距离均值掩盖。',
        '此表不能当作三维表面距离或临床结论。','']
    severe=[r for r in report['cases'] if r['scores']['pruned_fixed_final']['metrics']['dice']-r['scores']['original_fixed_final']['metrics']['dice']<-.01]
    lines+=['## 采用判断','']
    if severe:
        lines+=['本轮存在单病例Dice下降超过1个百分点，当前候选继续保留为性能实验，**不建议直接替换原模型**。',
            '平均值是否满足用户允许范围与困难病例是否退化是两项不同判断；个别明显退化仍需保留并解释。','',
            '超过1个百分点下降的病例：'+ '、'.join(r['case'] for r in severe)+'。','']
    else:
        lines+=['本轮没有单病例Dice下降超过1个百分点。仍需结合小肺/空肺及边界变化、实板提速作采用判断；此次不自动替换模型。','']
    (root/'full_test_report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['cases'];names=[r['case'] for r in rows];old=np.array([r['scores']['original_fixed_final']['metrics']['dice'] for r in rows])
    new=np.array([r['scores']['pruned_fixed_final']['metrics']['dice'] for r in rows]);delta=100*(new-old);y=np.arange(len(rows))
    fig,(left,right)=plt.subplots(1,2,figsize=(14,13),sharey=True,gridspec_kw={'width_ratios':[1.4,1]})
    left.plot(old,y,'o',label='Original',color='#697785');left.plot(new,y,'o',label='Pruned',color='#008a82')
    for z in y:left.plot([old[z],new[z]],[z,z],color='#b2bcc1',zorder=0)
    left.set_yticks(y,names);left.invert_yaxis();left.set_xlabel('Native patient-volume Dice');left.legend();left.grid(axis='x',alpha=.25)
    right.barh(y,delta,color=['#cf5146' if d<0 else '#008a82' for d in delta]);right.axvline(0,color='black',linewidth=.8)
    right.axvline(-1,color='#9b5a00',linestyle='--',label='-1 pp reference')
    right.set_xlabel('Pruned - original (percentage points)');right.legend();right.grid(axis='x',alpha=.25)
    for ax in [left,right]:ax.axhline(16.5,color='#8a8a8a',linestyle=':',linewidth=.8)
    fig.suptitle('All 27 frozen cases: exact Q31 reference, identical native postprocessing\nWorkstation accuracy comparison; no board-speed claim')
    fig.tight_layout();fig.savefig(root/'paired_fixed_dice.png',dpi=150);plt.close(fig)
    # Post-hoc visualization: three largest final Dice losses,
    # with the candidate's highest FP+FN slice in each; never choose an inference input.
    cache=json.loads((REBUILT/'dataset.json').read_text(encoding='utf-8'));lookup={r['case']:r for r in cache['cases']}
    worst=sorted(rows,key=lambda r:r['scores']['pruned_fixed_final']['metrics']['dice']-r['scores']['original_fixed_final']['metrics']['dice'])[:3]
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for number,row in enumerate(worst):
        record=lookup[row['case']];folder=REBUILT/record['cache_folder']
        with np.load(root/row['case']/'original_fixed_native.npz') as a:old_mask=a['final']
        with np.load(root/row['case']/'pruned_fixed_native.npz') as a:new_mask=a['final']
        with mapped_npy(folder/'reference_native.npy') as truth,mapped_npy(folder/'hu_native.npy') as hu:
            z=int(np.argmax((new_mask!=(truth!=0)).sum((1,2))))
            for col,(predicted,title) in enumerate([(None,'Expert'),(old_mask,'Original'),(new_mask,'Pruned')]):
                ax=axes[number,col];ax.imshow(hu[z],cmap='gray',vmin=-1000,vmax=400)
                if truth[z].any():ax.contour(truth[z]!=0,levels=[.5],colors=['lime'],linewidths=.7)
                if predicted is not None and predicted[z].any():ax.contour(predicted[z],levels=[.5],colors=['orange'],linewidths=.7)
                ax.set_title(row['case']+' z='+str(z)+' | '+title);ax.axis('off')
    fig.suptitle('Three largest Dice losses, highest-error slice per case\nPost-hoc review: green=expert, orange=prediction; not training or case selection')
    fig.tight_layout();fig.savefig(root/'worst_case_native_review.png',dpi=140);plt.close(fig)
    files=[root/'full_test_report.md',root/'paired_fixed_dice.png',root/'worst_case_native_review.png']
    write_json(root/'presentation_verification.json',dict(status='generated_audited_full_test_report',
        evaluation_sha256=file_sha256(root/'evaluation.json'),mask_audit_sha256=file_sha256(root/'returned_mask_audit.json'),
        outputs_sha256={p.name:file_sha256(p) for p in files},source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)}))
    print('Readable full-test report and paired/failure figures:',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data/lung_prune_moderate_fulltest_20261017')
    run(p.parse_args().root)
