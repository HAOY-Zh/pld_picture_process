"""Frozen review12 + reused6 evaluation, no training or recalibration.

Compare v2 and selected v3 using each model's pre-existing development19
thresholds. The old6 cohort is reported separately. Sealed45 has no entrypoint.
"""
import argparse
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_candidate_data import sha_file, load_development_case
from lung_candidate_rules_run import ROOT, EXCLUDED
from lung_candidate_run import excluded_findings
from lung_candidates import checked_inputs, point_metrics
import lung_nodule_cnn_experiment as first
from lung_nodule_cnn import (config_from_dict, model, search_mask, tiles,
                             sample_patch, decode, consolidate)
from lung_nodule_cnn_hardneg import DEFAULT as V2, setup_torch
from lung_nodule_cnn_balanced import OUT as V3, read, save

OUT = ROOT/'data/lung_nodule_cnn_review18_20261019'
MODELS = {'v2':V2, 'v3_balanced':V3/'balanced_orientation'}
COHORTS = ('review_12','reused_6')


def eligible_uids(plan, cohort):
    if cohort not in COHORTS:
        raise ValueError('Only declared review12/reused6; no heldout or training entrypoint')
    selected = list(plan[cohort])
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate review case')
    if set(selected) & set(plan['fit_scans'] + plan['calibration_scans']):
        raise ValueError('Review overlaps training/calibration')
    if set(plan['review_12']) & set(plan['reused_6']):
        raise ValueError('Review cohorts overlap')
    return selected


def summarize_fixed(rows, threshold):
    cases = []
    detected = refs = false_positive = candidates = 0
    for row in rows:
        score = np.asarray(row['scores'])
        world = np.asarray(row['world']).reshape(-1,3)
        keep = score >= threshold if threshold is not None else np.zeros(len(score),bool)
        metric = point_metrics(world[keep],row['record']['nodules'],row['excluded'])
        cases.append(dict(series_uid=row['uid'],metric=metric))
        detected += metric['detected']
        refs += metric['reference_nodules']
        false_positive += metric['false_positive_candidates']
        candidates += int(keep.sum())
    return dict(threshold=threshold,scans=len(rows),reference_nodules=refs,detected=detected,
                sensitivity=detected/refs if refs else None,false_positive_candidates=false_positive,
                actual_fp_per_scan=false_positive/len(rows),selected_candidates=candidates,cases=cases)


def independent_counts(rows,threshold):
    detected = refs = false_positive = 0
    for row in rows:
        xyz = np.asarray(row['world']).reshape(-1,3)
        scores = np.asarray(row['scores'])
        keep = scores >= threshold if threshold is not None else np.zeros(len(scores),bool)
        xyz = xyz[keep]
        nodules = row['record']['nodules']
        refs += len(nodules)
        if nodules:
            centers = np.asarray([n['center_lps_xyz_mm'] for n in nodules])
            radius = np.asarray([n['diameter_mm']/2 for n in nodules])
            matches = np.linalg.norm(xyz[:,None,:]-centers[None,:,:],axis=2) < radius[None,:]
            detected += int(matches.any(axis=0).sum())
            reference_hit = matches.any(axis=1)
        else:
            reference_hit = np.zeros(len(xyz),bool)
        ignore = np.zeros(len(xyz),bool)
        for finding in row['excluded']:
            diameter = finding['diameter_mm'] if finding['diameter_mm'] >= 0 else 10.
            ignore |= np.linalg.norm(xyz-finding['center_lps_xyz_mm'],axis=1) < diameter/2
        false_positive += int((~reference_hit & ~ignore).sum())
    return dict(detected=detected,reference_nodules=refs,false_positive_candidates=false_positive)


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve first review; resume existing frozen plan')
    parent = first.verify(V3)
    protected = dict(parent['protected_sha256'])
    thresholds, models = {}, {}
    for name, root in MODELS.items():
        f = read(root/'model_freeze.json')
        if sha_file(root/'model.pt') != f['model_sha256']:
            raise ValueError('Changed frozen model')
        import torch
        ckpt = torch.load(root/'model.pt',map_location='cpu',weights_only=True)
        if set(ckpt['training_scans']) != set(parent['fit_scans']) or ckpt['config'] != parent['config']:
            raise ValueError('Model training/config mismatch')
        source_plan = V3/'plan.json' if name == 'v3_balanced' else V2/'plan.json'
        if ckpt['plan_sha256'] != sha_file(source_plan):
            raise ValueError('Checkpoint does not match its frozen source plan')
        source = root/'calibration.json' if name == 'v3_balanced' else root/'calibration/report.json'
        calibration = read(source)
        if name == 'v3_balanced' and calibration['model_sha256'] != f['model_sha256']:
            raise ValueError('Wrong calibration identity')
        if name == 'v2':
            fixed = read(root/'calibration/freeze.json')
            if (fixed['model_sha256'] != f['model_sha256'] or fixed['report_sha256'] != sha_file(source)):
                raise ValueError('Changed v2 thresholds')
            protected[str(root/'calibration/freeze.json')] = sha_file(root/'calibration/freeze.json')
        thresholds[name] = {str(p['fp_budget']):p['threshold'] for p in calibration['cnn']['operating_points']
                            if p['fp_budget'] in (1.,4.,8.)}
        models[name] = dict(path=str(root/'model.pt'),model_sha256=f['model_sha256'],
                            source_plan_sha256=ckpt['plan_sha256'],threshold_source=str(source),
                            threshold_source_sha256=sha_file(source))
        for path in (root/'model.pt',root/'model_freeze.json',source):
            protected[str(path)] = sha_file(path)
    for path in (V3/'plan.json',V3/'plan.sha256',Path(__file__).resolve(),
                 ROOT/'python/tests/test_lung_nodule_cnn_review18.py'):
        protected[str(path)] = sha_file(path)
    plan = {k:parent[k] for k in ('fit_scans','calibration_scans','review_12','reused_6','config')}
    for cohort in COHORTS:
        eligible_uids(plan,cohort)
        first.records_for(plan,cohort)
    plan.update(version='frozen_cnn_review18_v1',models=models,thresholds=thresholds,seed=2026101904,
        primary_profile='4.0',additional_profiles=['1.0','8.0'],
        threshold_policy='Copy pre-existing development19 thresholds; no threshold fitting on review cases',
        scope='review12 previously used by classical/SVM development; reused6 old development; neither is a new clinical external test',
        lung_mask_source='cached original FP32 U-Net predictions, not deployed pruned INT8',
        shared_preprocessing=True,protected_sha256=protected,
        training_performed=False,recalibration_performed=False,heldout_scans_run=0,fpga_validated=False)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in (Path(__file__).resolve(),ROOT/'python/tests/test_lung_nodule_cnn_review18.py'):
        (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN: 12 + 6 review cases, v2/v3, fixed pre-existing 1/4/8 profiles; no sealed45',flush=True)


def infer(out):
    plan = first.verify(out)
    torch = setup_torch(plan['seed'])
    cfg = config_from_dict(plan['config'])
    from threadpoolctl import threadpool_limits
    ignored = excluded_findings(EXCLUDED)
    nets = {}
    for name, info in plan['models'].items():
        ckpt = torch.load(info['path'],map_location='cpu',weights_only=True)
        if (sha_file(info['path']) != info['model_sha256'] or
                set(ckpt['training_scans']) != set(plan['fit_scans'])):
            raise ValueError('Changed model/training cohort')
        net = model(cfg).cuda().eval()
        net.load_state_dict(ckpt['state'])
        nets[name] = net
    identity = sha_file(out/'plan.json')
    with threadpool_limits(limits=1), torch.inference_mode():
        for cohort in COHORTS:
            eligible_uids(plan,cohort)
            split,records = first.records_for(plan,cohort)
            for i,r in enumerate(records,1):
                uid = r['series_uid']
                folder = out/cohort/uid
                folder.mkdir(parents=True,exist_ok=True)
                pending = []
                for name,info in plan['models'].items():
                    dest = folder/(name+'.json')
                    if dest.exists():
                        packet = read(dest)
                        if (packet['model_sha256'] != info['model_sha256'] or packet['plan_sha256'] != identity or
                                packet['input_raw_sha256'] != r['raw_sha256']):
                            raise ValueError('Changed completed review case')
                    else:
                        pending.append(name)
                if not pending:
                    continue
                tick = perf_counter()
                hu,geometry = load_development_case(split,r)
                body = first.load_mask(r,split)
                checked_inputs(hu,body,geometry)
                spacing = np.asarray(geometry['spacing_zyx_mm'])
                search = search_mask(body,spacing,cfg)
                jobs,schedule = tiles(search,spacing,cfg)
                result = {n:[[],[],[]] for n in pending}
                for begin in range(0,len(jobs),48):
                    group = jobs[begin:begin+48]
                    image = np.stack([sample_patch(hu,spacing,j,cfg) for j in group])
                    tensor = torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                    for name in pending:
                        prediction = nets[name](tensor).cpu().numpy()
                        for store,values in zip(result[name],decode(prediction,group,geometry,cfg,search)):
                            store.append(values)
                metrics = {}
                for name in pending:
                    values = [np.concatenate(a) for a in result[name]] if jobs else [[],[],[]]
                    candidates = consolidate(*values,geometry,cfg)
                    for c in candidates:
                        box = np.asarray(c['bbox_zyx']).reshape(3,2)
                        box[:,0] = np.maximum(box[:,0],0)
                        box[:,1] = np.minimum(box[:,1],hu.shape)
                        c['bbox_zyx'] = box.ravel().tolist()
                    packet = dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_guided_cnn_'+name,
                        backend='FP32_CUDA_REFERENCE_NOT_FPGA',status='success',series_uid=uid,cohort=cohort,
                        input_raw_sha256=r['raw_sha256'],model_sha256=plan['models'][name]['model_sha256'],
                        plan_sha256=identity,lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                        source_shape_zyx=list(hu.shape),geometry=geometry,parameters=plan['config'],schedule=schedule,
                        candidates=candidates,shared_pair_seconds=perf_counter()-tick,
                        scope='Fixed-threshold review; shared pair wall time, not per-model/PL/web latency')
                    save(folder/(name+'.json'),packet)
                    t = plan['thresholds'][name]['4.0']
                    selected = [c['center_lps_xyz_mm'] for c in candidates if t is not None and c['score'] >= t]
                    m = point_metrics(selected,r['nodules'],ignored.get(uid,[]))
                    metrics[name] = dict(detected=m['detected'],references=m['reference_nodules'],
                                         false_positives=m['false_positive_candidates'])
                print('REVIEW',cohort,i,'/',len(records),'tiles',len(jobs),'fixed4',metrics,
                      'shared_seconds',round(perf_counter()-tick,2),flush=True)
                save(out/'progress.json',dict(cohort=cohort,completed_in_cohort=i,cohort_total=len(records),
                    series_uid=uid,last_case_fixed4=metrics,heldout_scans_run=0))
                del hu,body,search,result
                gc.collect()


def summarize(out):
    plan = first.verify(out)
    if (out/'report.json').exists():
        raise FileExistsError('Preserve first fixed-threshold review report')
    ignored = excluded_findings(EXCLUDED)
    reports = {}
    references = []
    for cohort in COHORTS:
        _,records = first.records_for(plan,cohort)
        reports[cohort] = {}
        rows_by = {}
        for name,info in plan['models'].items():
            rows = []
            for r in records:
                uid = r['series_uid']
                packet = read(out/cohort/uid/(name+'.json'))
                if (packet['model_sha256'] != info['model_sha256'] or packet['input_raw_sha256'] != r['raw_sha256'] or
                        packet['plan_sha256'] != sha_file(out/'plan.json')):
                    raise ValueError('Review identity mismatch')
                world = np.asarray([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
                index = np.asarray([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1,3)
                np.testing.assert_allclose(world,index@np.asarray(r['basis_lps_zyx']).T+r['origin_lps_mm'],atol=1e-6,rtol=0)
                rows.append(dict(uid=uid,record=r,excluded=ignored.get(uid,[]),world=world,
                                 scores=np.asarray([c['score'] for c in packet['candidates']]),packet=packet))
            profiles = {}
            for profile,threshold in plan['thresholds'][name].items():
                profile_report = summarize_fixed(rows,threshold)
                direct = independent_counts(rows,threshold)
                for key,value in direct.items():
                    if profile_report[key] != value:
                        raise AssertionError('Independent fixed-threshold count mismatch')
                profiles[profile] = profile_report
            coverage = summarize_fixed(rows,None)
            covered = sum(point_metrics(row['world'],row['record']['nodules'],row['excluded'])['detected'] for row in rows)
            reports[cohort][name] = dict(profiles=profiles,candidate_coverage=dict(
                detected=covered,reference_nodules=coverage['reference_nodules'],decode_floor=plan['config']['decode_floor']),
                interpretation='Nominal profiles use original development19 thresholds; actual FP/scan is measured on this cohort')
            rows_by[name] = rows
        for index,r in enumerate(records):
            uid = r['series_uid']
            for field in ('input_raw_sha256','lung_mask_sha256','geometry','parameters','schedule','source_shape_zyx'):
                if rows_by['v2'][index]['packet'][field] != rows_by['v3_balanced'][index]['packet'][field]:
                    raise AssertionError('Unmatched review inputs')
            for j,n in enumerate(r['nodules']):
                item = dict(cohort=cohort,series_uid=uid,index=j,reference=n)
                for name in plan['models']:
                    row = rows_by[name][index]
                    distance = np.linalg.norm(row['world']-n['center_lps_xyz_mm'],axis=1)
                    matches = distance < n['diameter_mm']/2
                    hits = np.flatnonzero(matches)
                    best = int(hits[np.argmax(row['scores'][hits])]) if len(hits) else None
                    profile = reports[cohort][name]['profiles']['4.0']['cases'][index]['metric']['reference_details'][j]
                    item[name] = dict(detected_at_fixed4=profile['detected'],candidate_present=bool(matches.any()),
                        best_matching_score=float(row['scores'][best]) if best is not None else None,
                        best_matching_distance_mm=float(distance[best]) if best is not None else None,
                        nearest_distance_mm=float(distance.min()) if len(distance) else None)
                references.append(item)
        print('COHORT_RESULT',cohort,{n:{'detected':v['profiles']['4.0']['detected'],
            'references':v['profiles']['4.0']['reference_nodules'],'actual_fp_per_scan':v['profiles']['4.0']['actual_fp_per_scan']}
            for n,v in reports[cohort].items()},flush=True)
    save(out/'report.json',dict(status='COMPLETE_FIXED_THRESHOLD_REVIEW',cohorts=reports,
        per_reference=references,thresholds=plan['thresholds'],primary_profile='4.0',
        plan_sha256=sha_file(out/'plan.json'),protected_files_verified=len(plan['protected_sha256']),
        independent_fixed_threshold_counts_verified=True,geometry_and_shared_inputs_verified=True,
        heldout_scans_run=0,training_performed=False,recalibration_performed=False,fpga_validated=False,
        interpretation='12/6 separate development checks, old algorithms have seen these cohorts. No threshold optimization or selection on these results.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    args = parser.parse_args()
    out = args.outdir.resolve()
    if not out.exists():
        freeze(out)
    stage = 'verify'
    try:
        first.verify(out)
        if not (out/'report.json').exists():
            stage = 'infer'
            save(out/'pipeline_state.json',dict(status='RUNNING',stage=stage))
            infer(out)
            stage = 'summarize'
            save(out/'pipeline_state.json',dict(status='RUNNING',stage=stage))
            summarize(out)
        first.verify(out)
        save(out/'pipeline_state.json',dict(status='COMPLETE_FIXED_THRESHOLD_REVIEW',heldout_scans_run=0))
    except Exception as error:
        save(out/'pipeline_state.json',dict(status='FAILED',stage=stage,error=repr(error)))
        raise


if __name__=='__main__':
    main()
