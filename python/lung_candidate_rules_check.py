"""Apply a fixed experimental ranker to six preselected NEW development scans.

No model/threshold updates are made here. Original frozen U-Net predicts lungs;
LDCT and board deployment are not modified. The reserved 45 scans are refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from lung_candidate_data import read_split,load_development_case,sha_file,write_json
from lung_candidate_run import predict_lungs,excluded_findings
from lung_candidate_structure import select_structural_candidates,BALANCED_PILOT_CONFIG
from lung_candidate_refined import extract_refined_regions
from lung_candidates import point_metrics

ROOT=Path(__file__).resolve().parents[1]


def compare_preview(hu,old,new,references,geometry,folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle,Ellipse
    spacing=np.asarray(geometry['spacing_zyx_mm'])
    indices=sorted({int(np.rint(n['center_index_zyx'][0])) for n in references}) or [len(hu)//2]
    paths=[]
    for z in indices:
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        for ax,candidates,title in zip(axes,[old,new],['Original v3','Experimental refinement']):
            ax.imshow(hu[z],cmap='gray',vmin=-1350,vmax=150);shown=0
            for c in candidates:
                z0,z1,y0,y1,x0,x1=c['bbox_zyx']
                if z0<=z<z1:
                    ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='orange',linewidth=.7));shown+=1
            for n in references:
                p=n['center_index_zyx'];radius=n['diameter_mm']/2
                if abs(p[0]-z)*spacing[0]<=radius:
                    ax.add_patch(Ellipse((p[2],p[1]),2*radius/spacing[2],2*radius/spacing[1],fill=False,edgecolor='cyan',linewidth=1.5))
                    ax.plot(p[2],p[1],'+',color='cyan')
            ax.set_title(f'{title}: z={z}, {shown} boxes');ax.axis('off')
        fig.suptitle('All retained boxes on this slice; cyan = reference center/diameter')
        path=folder/f'comparison_z{z:04d}.png';fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig);paths.append(path.name)
    return paths


def run(model_path,plan_path,out):
    from lung_unet import torch_module,make_model,select_device
    if out.exists():raise ValueError('Use a fresh output; do not tune after reading new checks')
    split_path=ROOT/'data/lung_candidate_split_20261018/split.json'
    checkpoint_path=ROOT/'data/lung_training_20261011/early_stop_run/best.pt'
    excluded_path=ROOT/'data/lung_candidate_planning_20261017/annotations_excluded.csv'
    split=read_split(split_path);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    records={r['series_uid']:r for r in split['cases']};selected=plan['selected_uids']
    if (len(selected)!=6 or len(set(selected))!=6 or set(selected)&set(plan['prior_24_excluded']) or
        any(records[uid]['split']!='development' for uid in selected)):raise ValueError('Invalid new development cohort')
    rank_model=json.loads(model_path.read_text(encoding='utf-8'))
    frozen={str(p.resolve()):sha_file(p) for p in [model_path,plan_path,split_path,checkpoint_path,excluded_path,
        Path(__file__),ROOT/'python/lung_candidate_refined.py',ROOT/'python/lung_candidate_rules.py',
        ROOT/'python/lung_candidate_feature_rank.py',ROOT/'python/lung_candidates.py',
        ROOT/'python/lung_candidate_structure.py',ROOT/'python/lung_candidate_run.py',ROOT/'python/lung_unet.py']}
    torch=torch_module();torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=select_device('cuda');cp=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if cp['image_size']!=256 or cp.get('context_slices',1)!=1 or cp['epoch']!=12 or cp.get('threshold',.5)!=.5:
        raise ValueError('Need original frozen lung predictor')
    lung_model=make_model(cp['base_channels'],1).to(device).eval();lung_model.load_state_dict(cp['model'])
    ignored=excluded_findings(excluded_path);out.mkdir(parents=True)
    report=dict(status='RUNNING',scope='six additional development scans, fixed rank model and threshold; not final heldout',
        selected_uids=selected,frozen_sources=frozen,heldout_scans_run=0,cases=[],
        lung_backend='frozen original FP32 CUDA; not INT8/PL',new_neural_network=False)
    write_json(out/'report.json',report)
    for i,uid in enumerate(selected,1):
        print(f'[{i}/6] new development check {uid}',flush=True);r=records[uid]
        hu,geometry=load_development_case(split,r)
        raw,body,lung_ms=predict_lungs(hu,geometry,lung_model,device,16);del raw
        start=perf_counter();result,debug=extract_refined_regions(hu,body,geometry,rank_model)
        candidate_ms=(perf_counter()-start)*1000
        baseline=select_structural_candidates(debug['coarse_candidates'][:debug['original_coarse_count']],
            debug['hessian_features'][:debug['original_coarse_count']],geometry,BALANCED_PILOT_CONFIG)
        del debug
        result.update(series_uid=uid,input_raw_sha256=r['raw_sha256'],
            lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),lung_backend=report['lung_backend'])
        # All references are consumed only after candidate generation.
        before=point_metrics([c['center_lps_xyz_mm'] for c in baseline],r['nodules'],ignored.get(uid,[]))
        after=point_metrics([c['center_lps_xyz_mm'] for c in result['candidates']],r['nodules'],ignored.get(uid,[]))
        folder=out/uid;folder.mkdir();write_json(folder/'candidates.json',result)
        write_json(folder/'metrics.json',dict(original_v3=before,refined=after))
        np.savez_compressed(folder/'lung_mask_packed.npz',shape_zyx=np.array(hu.shape),lung_body=np.packbits(body))
        images=compare_preview(hu,baseline,result['candidates'],r['nodules'],geometry,folder)
        paired=list(zip(before['reference_details'],after['reference_details']))
        report['cases'].append(dict(series_uid=uid,original_v3=before,refined=after,previews=images,
            lost_original_detections=sum(a['detected'] and not b['detected'] for a,b in paired),
            gained_detections=sum(not a['detected'] and b['detected'] for a,b in paired),
            lung_ms=lung_ms,candidate_ms=candidate_ms,
            timing='full refined CPU candidate path, includes complementary filter; baseline reused measurements so no baseline speedup claim'))
        write_json(out/'report.json',report)
        print('  original/refined detected',before['detected'],after['detected'],'/',after['reference_nodules'],
              'FP',before['false_positive_candidates'],after['false_positive_candidates'],flush=True)
        del hu,body,result,baseline
    for path,expected in frozen.items():
        if sha_file(path)!=expected:raise ValueError('Model, source or fixed plan changed during check')
    report['summary']={name:dict(reference_nodules=sum(r[name]['reference_nodules'] for r in report['cases']),
        detected=sum(r[name]['detected'] for r in report['cases']),
        false_positive_candidates=sum(r[name]['false_positive_candidates'] for r in report['cases']),
        fp_per_scan=sum(r[name]['false_positive_candidates'] for r in report['cases'])/6) for name in ['original_v3','refined']}
    report['lost_original_detections']=sum(r['lost_original_detections'] for r in report['cases'])
    report['gained_detections']=sum(r['gained_detections'] for r in report['cases'])
    report['status']='COMPLETE';write_json(out/'report.json',report)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();run(args.model,args.plan,args.outdir)
