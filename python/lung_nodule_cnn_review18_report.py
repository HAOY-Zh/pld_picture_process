"""Review completed frozen-threshold extra cases; plots without neural reruns."""
import numpy as np
from lung_candidate_data import sha_file, load_development_case
from lung_candidate_rules_run import EXCLUDED
from lung_candidate_run import excluded_findings
from lung_nodule_cnn_experiment import verify, records_for
from lung_nodule_cnn_balanced import read, save
from lung_nodule_cnn_review18 import OUT, COHORTS, independent_counts


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    out = OUT
    plan = verify(out)
    report = read(out/'report.json')
    if report['plan_sha256'] != sha_file(out/'plan.json') or report['thresholds'] != plan['thresholds']:
        raise ValueError('Changed frozen plan or thresholds')
    folder = out/'effect_review'
    folder.mkdir(exist_ok=False)
    ignored = excluded_findings(EXCLUDED)
    models = plan['models']
    all_records = {}
    split = None
    audited = 0
    for cohort in COHORTS:
        split,records = records_for(plan,cohort)
        all_records.update({r['series_uid']:r for r in records})
        for name,info in models.items():
            if sha_file(info['path']) != info['model_sha256'] or sha_file(info['threshold_source']) != info['threshold_source_sha256']:
                raise ValueError('Changed model or calibration source')
            rows = []
            for r in records:
                packet = read(out/cohort/r['series_uid']/(name+'.json'))
                if (packet['input_raw_sha256'] != r['raw_sha256'] or
                        packet['model_sha256'] != info['model_sha256'] or
                        packet['plan_sha256'] != sha_file(out/'plan.json')):
                    raise ValueError('Packet identity mismatch')
                index = np.asarray([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1,3)
                world = np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
                np.testing.assert_allclose(world,index@np.asarray(r['basis_lps_zyx']).T+r['origin_lps_mm'],atol=1e-6,rtol=0)
                rows.append(dict(world=world,scores=np.asarray([c['score'] for c in packet['candidates']]),
                                 record=r,excluded=ignored.get(r['series_uid'],[])))
                audited += 1
            for profile,threshold in plan['thresholds'][name].items():
                direct = independent_counts(rows,threshold)
                expected = report['cohorts'][cohort][name]['profiles'][profile]
                if any(direct[k] != expected[k] for k in direct):
                    raise ValueError('Fixed-threshold direct counts mismatch')
    refs = report['per_reference']
    strata = []
    changes = {}
    fp_distribution = {}
    for cohort in COHORTS:
        selected = [r for r in refs if r['cohort'] == cohort]
        changes[cohort] = dict(
            gained=sum(r['v3_balanced']['detected_at_fixed4'] and not r['v2']['detected_at_fixed4'] for r in selected),
            lost=sum(r['v2']['detected_at_fixed4'] and not r['v3_balanced']['detected_at_fixed4'] for r in selected),
            v3_missing_candidates=sum(not r['v3_balanced']['candidate_present'] for r in selected))
        fp_distribution[cohort] = {}
        for name in models:
            values = [c['metric']['false_positive_candidates'] for c in report['cohorts'][cohort][name]['profiles']['4.0']['cases']]
            fp_distribution[cohort][name] = dict(minimum=int(min(values)),median=float(np.median(values)),
                                                  maximum=int(max(values)),mean=float(np.mean(values)))
        for label,lo,hi in [('under6',0,6),('6to10',6,10),('over10',10,float('inf'))]:
            group = [r for r in selected if lo <= r['reference']['diameter_mm'] < hi]
            strata.append(dict(cohort=cohort,size_group=label,references=len(group),
                detected_at_fixed4={name:sum(r[name]['detected_at_fixed4'] for r in group) for name in models}))
    save(folder/'summary.json',dict(changes_vs_v2=changes,size_groups=strata,
        false_positive_distribution_at_fixed4=fp_distribution,
        interpretation='Pre-existing development19 thresholds, nominal4 is not guaranteed actual4 FP/scan',
        model_sha256={name:info['model_sha256'] for name,info in models.items()},heldout_scans_run=0))
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for ax,cohort in zip(axes,COHORTS):
        for name,label in [('v2','CNN v2'),('v3_balanced','CNN v3 balanced')]:
            profiles = report['cohorts'][cohort][name]['profiles']
            points = [profiles[p] for p in ('1.0','4.0','8.0')]
            ax.plot([p['actual_fp_per_scan'] for p in points],[p['sensitivity'] for p in points],'-o',label=label)
            for p,tag in zip(points,('cal-1','cal-4','cal-8')):
                ax.annotate(tag,(p['actual_fp_per_scan'],p['sensitivity']),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(ylim=(0,1.04),xlabel='Actual false positives per scan',ylabel='Sensitivity',
               title='Review12 / 23 nodules' if cohort=='review_12' else 'Reused6 / 5 nodules')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Frozen development19 threshold profiles on extra cases\nThree fixed settings per model; no threshold tuning on these cohorts',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(folder/'fixed_profiles.png',dpi=150)
    plt.close(fig)
    selection = []
    categories = [
        ('V3 recovers versus V2',lambda r:r['cohort']=='review_12' and r['v3_balanced']['detected_at_fixed4'] and not r['v2']['detected_at_fixed4']),
        ('V3 loses versus V2',lambda r:r['cohort']=='review_12' and r['v2']['detected_at_fixed4'] and not r['v3_balanced']['detected_at_fixed4']),
        ('Both miss on review12',lambda r:r['cohort']=='review_12' and not r['v2']['detected_at_fixed4'] and not r['v3_balanced']['detected_at_fixed4']),
        ('Missed on reused6',lambda r:r['cohort']=='reused_6' and not r['v3_balanced']['detected_at_fixed4'])]
    for label,predicate in categories:
        options = [r for r in refs if predicate(r)]
        if options:
            selection.append((label,options[0]))
    if not selection:
        selection = [('Review example',refs[0])]
    fig,axes = plt.subplots(len(selection),3,figsize=(11,3.6*len(selection)),squeeze=False)
    for row,(category,item) in enumerate(selection):
        uid,cohort = item['series_uid'],item['cohort']
        record = all_records[uid]
        hu,_ = load_development_case(split,record)
        spacing = np.asarray(record['spacing_zyx_mm'])
        center = np.asarray(item['reference']['center_index_zyx'])
        z,y,x = np.rint(center).astype(int)
        hy,hx = np.ceil(25/spacing[1:]).astype(int)
        y0,y1 = max(0,y-hy),min(hu.shape[1],y+hy+1)
        x0,x1 = max(0,x-hx),min(hu.shape[2],x+hx+1)
        outputs = [[]]
        for name in models:
            packet = read(out/cohort/uid/(name+'.json'))
            threshold = plan['thresholds'][name]['4.0']
            outputs.append([c for c in packet['candidates'] if threshold is not None and c['score'] >= threshold])
        for column,ax in enumerate(axes[row]):
            ax.imshow(hu[z,y0:y1,x0:x1],cmap='gray',vmin=-1000,vmax=400,
                      extent=(x0*spacing[2],x1*spacing[2],y1*spacing[1],y0*spacing[1]))
            ax.add_patch(Circle((center[2]*spacing[2],center[1]*spacing[1]),item['reference']['diameter_mm']/2,
                                fill=False,edgecolor='cyan',linestyle='--'))
            ax.plot(center[2]*spacing[2],center[1]*spacing[1],'+',color='cyan')
            for c in outputs[column]:
                z0,z1,by0,by1,bx0,bx1 = c['bbox_zyx']
                if z0 <= z < z1 and by0 < y1 and by1 > y0 and bx0 < x1 and bx1 > x0:
                    ax.add_patch(Rectangle((bx0*spacing[2],by0*spacing[1]),(bx1-bx0)*spacing[2],
                        (by1-by0)*spacing[1],fill=False,edgecolor='orange'))
            status = '' if column==0 else (' HIT' if item['v2' if column==1 else 'v3_balanced']['detected_at_fixed4'] else ' MISS')
            ax.set_title(('Reference','CNN v2','CNN v3')[column]+status)
            ax.set_xlim(x0*spacing[2],x1*spacing[2])
            ax.set_ylim(y1*spacing[1],y0*spacing[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if column==0:
                ax.set_ylabel(category+f"\n{item['reference']['diameter_mm']:.2f} mm | ...{uid[-8:]}",fontsize=9)
        del hu
    fig.suptitle('Extra-case predictions at each model\'s frozen nominal4 threshold\nCyan: reference center/diameter, not contour. Orange: predicted ROI. HIT/MISS uses 3D matching.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(folder/'examples.png',dpi=140)
    plt.close(fig)
    # Whole-slice example of a reference-free scan with many false positives.
    # Selection is retrospective illustration only, never detector input.
    negative_cases = [c for c in report['cohorts']['review_12']['v3_balanced']['profiles']['4.0']['cases']
                      if c['metric']['reference_nodules']==0]
    if negative_cases:
        worst = max(negative_cases,key=lambda c:c['metric']['false_positive_candidates'])
        uid = worst['series_uid']
        record = all_records[uid]
        hu,_ = load_development_case(split,record)
        outputs = {}
        for name in models:
            packet = read(out/'review_12'/uid/(name+'.json'))
            threshold = plan['thresholds'][name]['4.0']
            outputs[name] = [c for c in packet['candidates'] if threshold is not None and c['score']>=threshold]
        density = np.zeros(len(hu),int)
        for c in outputs['v3_balanced']:
            z0,z1 = c['bbox_zyx'][:2]
            density[z0:z1] += 1
        z = int(density.argmax()) if density.any() else len(hu)//2
        fig,axes = plt.subplots(1,3,figsize=(13,5))
        for column,name in enumerate([None,*models]):
            ax = axes[column]
            ax.imshow(hu[z],cmap='gray',vmin=-1000,vmax=400)
            if name is not None:
                for c in outputs[name]:
                    z0,z1,y0,y1,x0,x1 = c['bbox_zyx']
                    if z0<=z<z1:
                        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='orange',linewidth=1))
                count = next(c['metric']['false_positive_candidates'] for c in report['cohorts']['review_12'][name]['profiles']['4.0']['cases'] if c['series_uid']==uid)
                ax.set_title(('CNN v2' if name=='v2' else 'CNN v3')+f' | whole-scan FP={count}')
            else:
                ax.set_title(f'CT slice z={z}')
            ax.axis('off')
        fig.suptitle('Reference-standard-negative scan: selected ROI boxes on one axial slice\nFP totals refer to the whole scan; excluded findings are ignored by the metric, not automatically healthy.',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.89))
        fig.savefig(folder/'false_positive_example.png',dpi=140)
        plt.close(fig)
        save(folder/'false_positive_example.json',dict(series_uid=uid,native_z=z,selection='Highest v3 fixed4 FP count among review12 cases without reference-standard nodules',
            figure='Only one slice shown; metric FP count covers whole scan',neural_forwards=0))
        del hu
    save(folder/'audit.json',dict(status='PASS',packets_verified=audited,
        protected_files_verified=len(plan['protected_sha256']),report_sha256=sha_file(out/'report.json'),
        review_source_sha256=sha_file(__file__),independent_fixed_threshold_counts_verified=True,
        calibration_source_hashes_verified=True,model_hashes_verified=True,
        training_performed=False,recalibration_performed=False,neural_forwards_rerun=0,heldout_scans_run=0))
    print('REVIEW PLOTS/AUDIT COMPLETE',changes,flush=True)


if __name__=='__main__':
    main()
