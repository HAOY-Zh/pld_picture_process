"""Independent completed-v3 provenance/score review and plots; no new training."""
import json
import numpy as np
from lung_candidate_data import sha_file, load_development_case
from lung_candidate_run import excluded_findings
from lung_candidate_rules_run import EXCLUDED
from lung_candidate_lowfp_audit import check_froc
from lung_candidates import point_metrics
from lung_nodule_cnn_balanced import OUT, V2, VARIANTS, read, save, verify
from lung_nodule_cnn_experiment import records_for


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    out=OUT
    plan=verify(out)
    comparison=read(out/'comparison.json')
    folder=out/'effect_review'
    folder.mkdir(exist_ok=False)
    cat=read(out/'training_catalogue.json')
    if set(i['series_uid'] for i in cat['items'])!=set(plan['fit_scans']):
        raise ValueError('Training catalogue cohort mismatch')
    for path,digest in cat['cache_sha256'].items():
        if sha_file(path)!=digest:
            raise ValueError('Changed training input')
    for v in VARIANTS:
        f=read(out/v/'model_freeze.json')
        if (sha_file(out/'training_catalogue.json')!=f['catalogue_sha256'] or
                sha_file(out/v/'model.pt')!=f['model_sha256'] or
                f['plan_sha256']!=sha_file(out/'plan.json')):
            raise ValueError('Changed training provenance')
        checkpoint=torch.load(out/v/'model.pt',map_location='cpu',weights_only=True)
        if set(checkpoint['training_scans'])!=set(plan['fit_scans']) or checkpoint['variant']!=v:
            raise ValueError('Wrong checkpoint cohort/variant')
    batch_counts=[read(out/v/'model_freeze.json')['training_batches'] for v in VARIANTS]
    if len(set(batch_counts))!=1:
        raise ValueError('Unequal extra optimization steps')
    split,records=records_for(plan,'calibration_scans')
    ignored=excluded_findings(EXCLUDED)
    roots={'v2':V2,**{v:out/v for v in VARIANTS}}
    curves={'v2':comparison['v2'],**comparison['variants']}
    thresholds={v:next(p['threshold'] for p in curve['operating_points'] if p['fp_budget']==4.) for v,curve in curves.items()}
    packets={v:{} for v in roots}
    rows={v:[] for v in roots}
    scores={v:{} for v in roots}
    references=[]
    cases={v:{c['series_uid']:c['metric'] for c in comparison['cases'][v]} for v in VARIANTS}
    v2cases={c['series_uid']:c['v2'] for c in read(V2/'comparison.json')['cases']}
    by_record={r['series_uid']:r for r in records}
    for r in records:
        uid=r['series_uid']
        for v,root in roots.items():
            packet=read(root/'calibration_scans'/(uid+'.json'))
            packets[v][uid]=packet
            world=np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
            rows[v].append(dict(uid=uid,record=r,excluded=ignored.get(uid,[]),world=world))
            scores[v][uid]=np.asarray([c['score'] for c in packet['candidates']])
        for j,n in enumerate(r['nodules']):
            item=dict(series_uid=uid,index=j,reference=n,diameter_mm=n['diameter_mm'])
            for v in roots:
                metric=v2cases[uid] if v=='v2' else cases[v][uid]
                world=rows[v][-1]['world']
                distance=np.linalg.norm(world-n['center_lps_xyz_mm'],axis=1)
                hit=distance<n['diameter_mm']/2
                matching=np.flatnonzero(hit)
                best=int(matching[np.argmax(scores[v][uid][matching])]) if len(matching) else None
                best_candidate=packets[v][uid]['candidates'][best] if best is not None else None
                item[v]=dict(detected_at4=metric['reference_details'][j]['detected'],
                    matching_candidate_present=bool(hit.any()),
                    best_matching_score=float(scores[v][uid][hit].max()) if hit.any() else None,
                    best_matching_center_distance_mm=float(distance[best]) if best is not None else None,
                    estimated_diameter_mm=best_candidate['estimated_scale_diameter_mm'] if best_candidate else None,
                    nearest_center_distance_mm=float(distance.min()) if len(distance) else None)
            references.append(item)
    for v in roots:
        check_froc(rows[v],scores[v],curves[v])
    strata=[]
    for label,lo,hi in [('under6',0,6),('6to10',6,10),('over10',10,float('inf'))]:
        group=[r for r in references if lo<=r['diameter_mm']<hi]
        strata.append(dict(size_group=label,references=len(group),
            detected_at4={v:sum(r[v]['detected_at4'] for r in group) for v in roots}))
    localization={}
    for v in roots:
        detected=[r for r in references if r[v]['detected_at4']]
        localization[v]=dict(detected_references=len(detected),
            median_center_distance_mm=float(np.median([r[v]['best_matching_center_distance_mm'] for r in detected])) if detected else None,
            median_diameter_absolute_error_mm=float(np.median([abs(r[v]['estimated_diameter_mm']-r['diameter_mm']) for r in detected])) if detected else None,
            scope='Only detected references, groups differ across models; center/diameter annotations, not voxel contour accuracy')
    save(folder/'error_analysis.json',dict(per_reference=references,size_groups=strata,
        localization_on_detected=localization,
        thresholds_at4=thresholds,scope='Already viewed development19; not independent test',neural_forwards=0,heldout_scans_run=0))
    fig,ax=plt.subplots(figsize=(8,5))
    for v,label in [('v2','CNN v2'),('continued_control','Same 30 extra epochs, control'),
                     ('balanced_orientation','Balanced + orientation'),('svm','Rule + SVM')]:
        curve=comparison['svm_same_scans'] if v=='svm' else curves[v]
        ax.plot(curve['curve']['fp_per_scan'],curve['curve']['sensitivity'],label=label)
    ax.set(xlim=(0,8),ylim=(0,1),xlabel='False positives per scan',ylabel='Sensitivity',
           title='Same development19: matched-step CNN training comparison')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'froc.png',dpi=150)
    plt.close(fig)
    treatment='balanced_orientation'
    selection=[]
    for label,predicate in [
        ('Balanced recovers versus v2',lambda r:r[treatment]['detected_at4'] and not r['v2']['detected_at4']),
        ('Balanced loses versus v2',lambda r:r['v2']['detected_at4'] and not r[treatment]['detected_at4']),
        ('Control detects; balanced misses',lambda r:r['continued_control']['detected_at4'] and not r[treatment]['detected_at4']),
        ('Both new variants miss',lambda r:not r['continued_control']['detected_at4'] and not r[treatment]['detected_at4'])]:
        chosen=[r for r in references if predicate(r)]
        if chosen:
            selection.append((label,chosen[0]))
    if not selection:
        selection=[('Example',references[0])]
    fig,axes=plt.subplots(len(selection),3,figsize=(11,3.6*len(selection)),squeeze=False)
    for row,(category,item) in enumerate(selection):
        uid=item['series_uid']
        r=by_record[uid]
        hu,_=load_development_case(split,r)
        spacing=np.asarray(r['spacing_zyx_mm'])
        center=np.asarray(item['reference']['center_index_zyx'])
        z,y,x=np.rint(center).astype(int)
        hy,hx=np.ceil(25/spacing[1:]).astype(int)
        y0,y1=max(0,y-hy),min(hu.shape[1],y+hy+1)
        x0,x1=max(0,x-hx),min(hu.shape[2],x+hx+1)
        for col,v in enumerate(roots):
            ax=axes[row,col]
            ax.imshow(hu[z,y0:y1,x0:x1],cmap='gray',vmin=-1000,vmax=400,
                      extent=(x0*spacing[2],x1*spacing[2],y1*spacing[1],y0*spacing[1]))
            ax.add_patch(Circle((center[2]*spacing[2],center[1]*spacing[1]),item['diameter_mm']/2,
                                fill=False,edgecolor='cyan',linestyle='--'))
            ax.plot(center[2]*spacing[2],center[1]*spacing[1],'+',color='cyan')
            for c in packets[v][uid]['candidates']:
                if thresholds[v] is None or c['score']<thresholds[v]:
                    continue
                z0,z1,by0,by1,bx0,bx1=c['bbox_zyx']
                if z0<=z<z1 and by0<y1 and by1>y0 and bx0<x1 and bx1>x0:
                    ax.add_patch(Rectangle((bx0*spacing[2],by0*spacing[1]),(bx1-bx0)*spacing[2],
                        (by1-by0)*spacing[1],fill=False,edgecolor='orange'))
            ax.set_title(('CNN v2','Extra-step control','Balanced + orientation')[col]+(' HIT' if item[v]['detected_at4'] else ' MISS'))
            ax.set_xlim(x0*spacing[2],x1*spacing[2])
            ax.set_ylim(y1*spacing[1],y0*spacing[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if col==0:
                ax.set_ylabel(category+f"\n{item['diameter_mm']:.2f} mm | ...{uid[-8:]}",fontsize=8)
        del hu
    fig.suptitle('Each model at its own development 4-FP/scan threshold\nCyan: reference center/diameter, not contour. Orange: predicted ROI. HIT/MISS uses 3D matching.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(folder/'examples.png',dpi=140)
    plt.close(fig)
    save(folder/'audit.json',dict(status='PASS',independent_froc_verified=list(roots),
        matched_extra_training_batches=batch_counts[0],parameters_per_variant=13157,
        protected_files_verified=len(plan['protected_sha256']),
        training_catalogue_sha256=sha_file(out/'training_catalogue.json'),
        comparison_sha256=sha_file(out/'comparison.json'),source_sha256=sha_file(__file__),
        fit_scans=53,calibration_scans=19,heldout_scans_run=0,neural_forwards_rerun=0,
        review12_run=False,default_promoted=False,fpga_validated=False))
    print('REVIEW COMPLETE',json.dumps(dict(size_groups=strata,matched_batches=batch_counts),ensure_ascii=False))


if __name__=='__main__':
    main()
