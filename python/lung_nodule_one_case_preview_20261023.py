"""Render one existing113-test prediction; never run a neural network."""
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from lung_candidate_data import sha_file
from lung_candidate_run import excluded_findings
from lung_candidate_rules_run import EXCLUDED
from lung_nodule_sealed113_evaluation import OUT as TEST, load_test_volume
from luna16_multibundle import read_manifest

UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.288701997968615460794642979503'
OUT = TEST/'preview_20261023'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    plan = read(TEST/'plan.json'); report = read(TEST/'evaluation.json'); audit = read(TEST/'audit.json')
    if sha_file(TEST/'evaluation.json') != audit['evaluation_sha256'] or audit['status'] != 'PASS':
        raise ValueError('Final evaluation identity changed')
    case = TEST/'cases'/UID; done = read(case/'complete.json')
    if done['plan_sha256'] != sha_file(TEST/'plan.json'):
        raise ValueError('Wrong completed case protocol')
    for name, digest in done['artifacts_sha256'].items():
        if Path(name).name != name or sha_file(case/name) != digest:
            raise ValueError('Saved prediction changed')
    with gzip.open(case/'prediction.json.gz', 'rt', encoding='utf-8') as f:
        packet = json.load(f)
    manifest = read_manifest(Path(plan['manifest']))
    record = next(r for r in manifest['cases'] if r['series_uid'] == UID)
    if packet['model_sha256'] != plan['model_sha256'] or packet['input_raw_sha256'] != record['raw_sha256']:
        raise ValueError('Wrong model/input')
    hu, geometry = load_test_volume(manifest, UID, plan)
    threshold = plan['thresholds']['cross']['4.0']
    with np.load(case/'scores.npz', allow_pickle=False) as archive:
        scores = archive['cross_scores'].copy()
    references = record['nodules']; excluded = excluded_findings(EXCLUDED).get(UID, [])
    shown = []
    for i in np.flatnonzero(scores >= threshold):
        c = dict(packet['candidates'][i]); world = np.asarray(c['center_lps_xyz_mm'])
        hits = [j for j, r in enumerate(references)
                if np.linalg.norm(world-r['center_lps_xyz_mm']) < r['diameter_mm']/2]
        ignored = any(np.linalg.norm(world-r['center_lps_xyz_mm']) < (r['diameter_mm'] if r['diameter_mm'] >= 0 else 10.)/2
                      for r in excluded)
        c.update(cross_score=float(scores[i]), reference_indices=hits,
                 evaluation_class='matched' if hits else 'excluded' if ignored else 'false_positive')
        shown.append(c)
    metric = next(r['metric'] for r in report['results']['cross']['4.0']['cases'] if r['series_uid'] == UID)
    counts = {kind:sum(c['evaluation_class']==kind for c in shown) for kind in ('matched','excluded','false_positive')}
    detected = len({j for c in shown for j in c['reference_indices']})
    if (detected != metric['detected'] or counts['false_positive'] != metric['false_positive_candidates']
            or counts['excluded'] != metric['ignored_excluded_candidates'] or len(shown) != metric['candidates']):
        raise ValueError('Preview disagrees with formal per-case metrics')
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    colors = dict(matched='#ffb347', false_positive='#ff5e6c', excluded='#9da6b2')
    cyan = '#31e3df'; spacing = np.asarray(geometry['spacing_zyx_mm'])
    fig = plt.figure(figsize=(12,15), facecolor='#0b1520')
    grid = fig.add_gridspec(3,2,left=.035,right=.975,bottom=.065,top=.845,hspace=.20,wspace=.08)
    fig.text(.04,.975,'当前2.5D CNN · 单病例效果',color='white',fontsize=22,weight='bold',va='top')
    fig.text(.04,.942,f'166张切片  |  参考结节 {detected}/{len(references)} 检出  |  计分误报 {counts["false_positive"]} 个',
             color='white',fontsize=13,va='top')
    fig.text(.04,.916,f'固定阈值 {threshold:.6f}  |  模型保留 {len(shown)} 个候选，其中 {counts["excluded"]} 个在排除标注内、不计误报',
             color='#becbd8',fontsize=10,va='top')
    for x,text,color in ((.04,'青：参考中心/直径',cyan),(.285,'橙：匹配参考的候选',colors['matched']),
                         (.545,'红：计分误报',colors['false_positive']),(.75,'灰：排除标注内候选',colors['excluded'])):
        fig.text(x,.890,text,color=color,fontsize=10,va='top')

    def panel(position,z,title,center=None):
        ax = fig.add_subplot(grid[position]); ax.set_facecolor('#081018')
        ax.imshow(hu[z],cmap='gray',vmin=-1350,vmax=150,interpolation='nearest')
        for c in shown:
            z0,z1,y0,y1,x0,x1 = c['bbox_zyx']
            if z0 <= z < z1:
                color=colors[c['evaluation_class']]
                ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1.5))
                if center is not None:
                    ax.plot(c['center_index_zyx'][2],c['center_index_zyx'][1],'+',color=color,markersize=9)
        for j,r in enumerate(references):
            rz,ry,rx = r['center_index_zyx']; radius=r['diameter_mm']/2
            dz=(z-rz)*spacing[0]
            if abs(dz) < radius:
                cross_radius=np.sqrt(radius**2-dz**2)
                ax.add_patch(Ellipse((rx,ry),2*cross_radius/spacing[2],2*cross_radius/spacing[1],
                                     fill=False,edgecolor=cyan,linewidth=1.7))
                ax.plot(rx,ry,'+',color=cyan,markersize=10)
                if center is None: ax.text(rx+7,ry-6,f'R{j+1}',color=cyan,fontsize=9,weight='bold')
        if center is not None:
            cy,cx=center; dy=16/spacing[1]; dx=16/spacing[2]
            ax.set_xlim(max(-.5,cx-dx),min(hu.shape[2]-.5,cx+dx))
            ax.set_ylim(min(hu.shape[1]-.5,cy+dy),max(-.5,cy-dy))
        ax.set_title(title,color='white',fontsize=11,pad=9); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#304356')
        return ax

    slices=sorted({int(np.rint(r['center_index_zyx'][0])) for r in references})
    for col,z in enumerate(slices[:2]):
        panel((0,col),z,f'整张CT及候选叠加 · 第{z+1}层（索引{z}）')
    for j,r in enumerate(references):
        z,y,x=r['center_index_zyx']; z=int(np.rint(z))
        hit=any(j in c['reference_indices'] for c in shown)
        panel((1+j//2,j%2),z,f'参考结节 R{j+1} · {r["diameter_mm"]:.1f} mm · {"检出" if hit else "漏检"}',(y,x))
    fp=max((c for c in shown if c['evaluation_class']=='false_positive'),key=lambda c:c['cross_score'])
    z,y,x=fp['center_index_zyx']; z=int(np.clip(np.rint(z),0,len(hu)-1))
    panel((2,1),z,f'误报示例 · 分数 {fp["cross_score"]:.3f} · 第{z+1}层',(y,x))
    fig.text(.04,.042,'参考圈由LUNA16中心和直径绘制，不是专家逐像素轮廓；矩形是模型候选范围。',color='#becbd8',fontsize=10)
    fig.text(.04,.022,'本图只展示同一病例的部分切片与局部；统计来自完整病例。候选类别使用标注事后核对，不参与推理。',color='#becbd8',fontsize=9)
    OUT.mkdir(exist_ok=True)
    image=OUT/'one_case_overview.png'; fig.savefig(image,dpi=160,facecolor=fig.get_facecolor()); plt.close(fig)
    metadata=dict(series_uid=UID,selection='Median historical wall-time among reference-positive113-test cases; no best-score selection',
        model_sha256=plan['model_sha256'],threshold=threshold,metric=metric,selected_candidates=shown,
        note='Rendering saved outputs only; excluded findings are unscored, not confirmed benign or confirmed nodules.',
        prediction_sha256=sha_file(case/'prediction.json.gz'),scores_sha256=sha_file(case/'scores.npz'),
        image_sha256=sha_file(image),render_script_sha256=sha_file(Path(__file__)),model_inference_repeated=False)
    (OUT/'one_case.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(image=str(image.resolve()),counts=counts,references=len(references),detected=detected,total_candidates=len(shown)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
