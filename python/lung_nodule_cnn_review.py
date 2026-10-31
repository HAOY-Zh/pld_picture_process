"""Summarize a completed first CNN calibration without rerunning inference."""
from pathlib import Path
import json
import numpy as np
from lung_candidate_data import write_json,sha_file,load_development_case
from lung_nodule_cnn_experiment import verify,records_for,EXPAND
from lung_candidate_expand_train import load_rows
from lung_candidate_rules_run import ROOT,EXCLUDED
from lung_candidate_run import excluded_findings
from lung_candidates import point_metrics


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    out=ROOT/'data/lung_nodule_cnn_20261019_v1';plan=verify(out)
    report=json.loads((out/'calibration/report.json').read_text('utf-8'))
    freeze=json.loads((out/'calibration/freeze.json').read_text('utf-8'))
    if sha_file(out/'model.pt')!=freeze['model_sha256']:raise ValueError('Changed model')
    if sha_file(out/'calibration/report.json')!=freeze['report_sha256']:raise ValueError('Changed calibration')
    split,records=records_for(plan,'calibration_scans');ignored=excluded_findings(EXCLUDED)
    folder=out/'effect_review';folder.mkdir(exist_ok=False)
    old_rows=load_rows(EXPAND,plan['calibration_scans']);old_by={r['uid']:r for r in old_rows}
    with np.load(EXPAND/'training/selected_oof_scores.npz',allow_pickle=False) as a:old_scores={r['series_uid']:a[r['series_uid']] for r in records}
    thresholds={}
    for name in ('cnn','svm_same_scans'):
        thresholds[name]=next(p['threshold'] for p in report[name]['operating_points'] if p['fp_budget']==4.)
    per_reference=[];timings={};schedule=[];cases=[];packets={};cnn_rows=[];cnn_scores={}
    for r in records:
        uid=r['series_uid'];packet=json.loads((out/'calibration_scans'/(uid+'.json')).read_text('utf-8'))
        if packet['model_sha256']!=freeze['model_sha256'] or packet['input_raw_sha256']!=r['raw_sha256']:raise ValueError('Packet identity mismatch')
        centers=np.array([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1,3)
        world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
        np.testing.assert_allclose(world,centers@np.array(r['basis_lps_zyx']).T+np.array(r['origin_lps_mm']),atol=1e-6,rtol=0)
        cnn_rows.append(dict(uid=uid,record=r,excluded=ignored.get(uid,[]),world=world))
        cnn_scores[uid]=np.array([c['score'] for c in packet['candidates']])
        packets[uid]=packet
        for k,v in packet['timings_ms'].items():timings.setdefault(k,[]).append(v)
        schedule.append(packet['schedule'])
        cnn=[c for c in packet['candidates'] if thresholds['cnn'] is not None and c['score']>=thresholds['cnn']]
        old=[c for c,s in zip(old_by[uid]['packet']['candidates'],old_scores[uid]) if thresholds['svm_same_scans'] is not None and s>=thresholds['svm_same_scans']]
        new_metric=point_metrics([c['center_lps_xyz_mm'] for c in cnn],r['nodules'],ignored.get(uid,[]))
        old_metric=point_metrics([c['center_lps_xyz_mm'] for c in old],r['nodules'],ignored.get(uid,[]))
        cases.append(dict(series_uid=uid,cnn=new_metric,svm=old_metric))
        for i,(n,o) in enumerate(zip(new_metric['reference_details'],old_metric['reference_details'])):
            per_reference.append(dict(series_uid=uid,index=i,reference=n['reference'],cnn=n['detected'],svm=o['detected']))
    from lung_candidate_lowfp_audit import check_froc
    check_froc(cnn_rows,cnn_scores,report['cnn']);check_froc(old_rows,old_scores,report['svm_same_scans'])
    import torch
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    if set(checkpoint['training_scans'])!=set(plan['fit_scans']) or set(plan['fit_scans'])&set(plan['calibration_scans']):raise ValueError('Training/calibration identity mismatch')
    if checkpoint['plan_sha256']!=sha_file(out/'plan.json'):raise ValueError('Checkpoint plan mismatch')
    if not all(torch.isfinite(v).all() for v in checkpoint['state'].values()):raise ValueError('Nonfinite weights')
    write_json(folder/'audit.json',dict(status='PASS',calibration_scans_checked=len(records),
        fit_scans=len(checkpoint['training_scans']),model_and_plan_identity_verified=True,
        independent_froc_verified=True,LPS_geometry_verified=True,
        protected_files_unchanged=len(plan['protected_sha256']),heldout_scans_run=0,
        review12_run=False,neural_forwards_rerun=0))
    selected=[]
    for category,predicate in [('CNN missed; SVM detected',lambda r:not r['cnn'] and r['svm']),
                               ('CNN detected; SVM missed',lambda r:r['cnn'] and not r['svm']),
                               ('Both detected',lambda r:r['cnn'] and r['svm'])]:
        matches=[r for r in per_reference if predicate(r)]
        if matches:selected.append((category,matches[0]))
    if not selected:selected=[('Neither detected',per_reference[0])]
    fig,axes=plt.subplots(len(selected),3,figsize=(11,3.7*len(selected)),squeeze=False)
    by_record={r['series_uid']:r for r in records}
    for row,(category,item) in enumerate(selected):
        uid=item['series_uid'];r=by_record[uid];hu,_=load_development_case(split,r)
        spacing=np.array(r['spacing_zyx_mm']);center=np.array(item['reference']['center_index_zyx']);z,y,x=np.rint(center).astype(int)
        hy,hx=np.ceil(25/spacing[1:]).astype(int);y0,y1=max(0,y-hy),min(hu.shape[1],y+hy+1);x0,x1=max(0,x-hx),min(hu.shape[2],x+hx+1)
        filtered=[[],[c for c,s in zip(old_by[uid]['packet']['candidates'],old_scores[uid]) if thresholds['svm_same_scans'] is not None and s>=thresholds['svm_same_scans']],
                     [c for c in packets[uid]['candidates'] if thresholds['cnn'] is not None and c['score']>=thresholds['cnn']]]
        for col,ax in enumerate(axes[row]):
            ax.imshow(hu[z,y0:y1,x0:x1],cmap='gray',vmin=-1000,vmax=400,
                      extent=(x0*spacing[2],x1*spacing[2],y1*spacing[1],y0*spacing[1]))
            ax.add_patch(Circle((center[2]*spacing[2],center[1]*spacing[1]),item['reference']['diameter_mm']/2,fill=False,edgecolor='cyan',linestyle='--'))
            ax.plot(center[2]*spacing[2],center[1]*spacing[1],'+',color='cyan')
            for c in filtered[col]:
                z0,z1,by0,by1,bx0,bx1=c['bbox_zyx']
                if z0<=z<z1 and by0<y1 and by1>y0 and bx0<x1 and bx1>x0:
                    ax.add_patch(Rectangle((bx0*spacing[2],by0*spacing[1]),(bx1-bx0)*spacing[2],(by1-by0)*spacing[1],fill=False,edgecolor='orange'))
            status='' if col==0 else ('HIT' if item['svm' if col==1 else 'cnn'] else 'MISS')
            ax.set_title(('Reference','Existing SVM','New CNN')[col]+' '+status,fontsize=10)
            ax.set_xlim(x0*spacing[2],x1*spacing[2]);ax.set_ylim(y1*spacing[1],y0*spacing[1]);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(category+'\n...'+uid[-8:]+f" | {item['reference']['diameter_mm']:.2f} mm",fontsize=9)
        del hu
    fig.suptitle('Internal calibration examples at each model\'s 4-FP/scan curve point\nCyan = reference center/diameter, not contour; orange = predicted box. 3D hit/miss; axial boxes only.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(folder/'examples.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,label in [('cnn','New 2.5D CNN'),('svm_same_scans','Existing rule + SVM')]:
        c=report[name]['curve'];ax.plot(c['fp_per_scan'],c['sensitivity'],label=label)
    ax.set(xlim=(0,8),ylim=(0,1),xlabel='False positives per scan',ylabel='Sensitivity',title='Same internal calibration 19 scans / 19 nodules')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(folder/'froc.png',dpi=140);plt.close(fig)
    summary=dict(status='COMPLETE_INTERNAL_CALIBRATION_ONLY',scans=len(records),reference_nodules=len(per_reference),
        cpm={name:report[name]['cpm_interpolated'] for name in ('cnn','svm_same_scans')},
        operating_points={name:report[name]['operating_points'] for name in ('cnn','svm_same_scans')},
        mean_timings_ms={k:float(np.mean(v)) for k,v in timings.items()},
        schedule=dict(dense_tiles=sum(s['dense_grid_tiles'] for s in schedule),lung_tiles=sum(s['lung_tiles'] for s in schedule)),
        gained_vs_svm=sum(r['cnn'] and not r['svm'] for r in per_reference),lost_vs_svm=sum(r['svm'] and not r['cnn'] for r in per_reference),
        cases=cases,per_reference=per_reference,model_sha256=freeze['model_sha256'],
        calibration_sha256=sha_file(out/'calibration/report.json'),heldout_scans_run=0,review12_run=False,
        notes='Curve operating points are calibration estimates. Timings use cached FP32 lung masks and CUDA CNN; not PL or web end-to-end. No default promotion.')
    write_json(folder/'summary.json',summary)
    print(json.dumps(dict(cpm=summary['cpm'],mean_timings_ms=summary['mean_timings_ms'],gained=summary['gained_vs_svm'],lost=summary['lost_vs_svm'])))


if __name__=='__main__':main()
