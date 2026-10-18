"""Export an offline, lazy-loaded browser for ALL existing validation cases.

Reuses frozen predictions from the accepted cleanup screen. No new inference,
training, threshold selection, Test/COVID access or unknown-case import.
"""
import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pydicom

from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256, load_cached_source, mapped_npy, read_manifest
from lung_volume_viewer import geometry, html_export, mesh, packed, payload
from lungseg.metrics import overlap_metrics

CASE_ID = re.compile(r'LCTSC-Train-S\d-\d{3}')


def checked_inputs(dataset, screen, checkpoint):
    dataset, screen, checkpoint = [Path(p).resolve(strict=True) for p in [dataset,screen,checkpoint]]
    root, manifest = read_manifest(dataset)
    report = json.loads(screen.read_text(encoding='utf-8'))
    if (report.get('status')!='complete_development_screen' or
        not report.get('protected_inputs_unchanged') or
        not report.get('primary_accepted_for_optional_development_view')):
        raise ValueError('Need a completed, accepted frozen cleanup screen')
    records = [r for r in manifest['cases'] if r['split']=='validation']
    names = [r['case'] for r in records]
    rows = report['cases']
    if (not records or len(rows)!=len(records) or len({r['case'] for r in rows})!=len(rows) or
        set(names)!=set(report['protocol']['validation_cases']) or
        set(names)!={r['case'] for r in rows} or
        any(r['split']!='validation' or not CASE_ID.fullmatch(r['case']) for r in rows)):
        raise ValueError('Screen must contain exactly ALL unchanged development validation cases')
    original_hashes = report['protocol']['input_sha256']
    if (original_hashes.get(str(dataset))!=file_sha256(dataset) or
        original_hashes.get(str(checkpoint))!=file_sha256(checkpoint) or
        report['protocol']['model_sha256']!=file_sha256(checkpoint)):
        raise ValueError('Dataset/checkpoint does not match frozen screen')
    for path, expected in original_hashes.items():
        if file_sha256(path)!=expected:
            raise ValueError('Original screen input changed: '+path)
    tracked = dict(original_hashes)
    tracked.update({str(p):file_sha256(p) for p in [screen,Path(__file__).resolve(),
        Path(__file__).with_name('lung_volume_viewer.py'),
        Path(__file__).with_name('lungseg')/'volume_viewer.html',
        Path(__file__).with_name('lungseg')/'case_browser.html']})
    for r in records:
        if not CASE_ID.fullmatch(r['case']):
            raise ValueError('Invalid case identifier')
        row = next(row for row in rows if row['case']==r['case'])
        artifact = (screen.parent/r['case']/'predictions_native.npz').resolve(strict=True)
        if artifact.parent.parent!=screen.parent or file_sha256(artifact)!=row['prediction_sha256']:
            raise ValueError('Prediction artifact path/hash mismatch')
        reference = (root/r['cache_folder']/'reference_native.npy').resolve(strict=True)
        if file_sha256(reference)!=row['reference_sha256']:
            raise ValueError('Expert reference changed since screen')
        tracked[str(artifact)] = row['prediction_sha256']
        for path in r['source_ct_files_sorted']:
            p=Path(path).resolve(strict=True);tracked[str(p)]=file_sha256(p)
    return root,manifest,report,records,tracked


def checked_body_screen(path, records, model_hash, tracked):
    if path is None:
        return None
    path=Path(path).resolve(strict=True);report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('status')!='complete_body_gate_development_screen' or not report.get('accepted_for_optional_view') or
        not report.get('protected_inputs_unchanged') or report['protocol']['model_sha256']!=model_hash or
        {r['case'] for r in report['cases']}!={r['case'] for r in records} or len(report['cases'])!=len(records)):
        raise ValueError('Accepted complete matching body-gate screen required')
    for p,h in report['protocol']['input_sha256'].items():
        if file_sha256(p)!=h:raise ValueError('Body gate input changed: '+p)
        if p in tracked and tracked[p]!=h:raise ValueError('Conflicting body-gate source identity')
        tracked[p]=h
    tracked[str(path)]=file_sha256(path)
    for row in report['cases']:
        folder=(path.parent/row['case']).resolve(strict=True)
        if folder.parent!=path.parent or row['split']!='validation' or row['expert_outside_envelope_voxels']!=0:
            raise ValueError('Invalid body gate case/coverage')
        for name,key in [('predictions_native.npz','prediction_sha256'),('body_geometry.json','body_geometry_sha256')]:
            file=folder/name
            if file_sha256(file)!=row[key]:raise ValueError('Body gate artifact changed')
            tracked[str(file)]=row[key]
    return dict(path=path,report=report)


def case_payload(root, record, row, screen, report, body_screen=None):
    """Read native CT/GT and reuse immutable screened masks; do not call model."""
    hu = load_cached_source(record)
    with mapped_npy(root/record['cache_folder']/'reference_native.npy') as ref:
        expert = ref!=0
    with np.load(Path(screen).parent/record['case']/'predictions_native.npz') as archive:
        original,cleaned=archive['original'],archive['primary']
    if (list(hu.shape)!=record['native_shape'] or expert.shape!=hu.shape or
        original.shape!=hu.shape or cleaned.shape!=hu.shape or
        not np.isin(original,[0,1]).all() or not np.isin(cleaned,[0,1]).all()):
        raise ValueError('Native grids or mask values differ')
    original,cleaned=original!=0,cleaned!=0
    if (cleaned&~original).any() or (expert&original&~cleaned).any():
        raise ValueError('Cleanup subset/zero true-removal safeguard failed')
    scores={}
    for name,mask in [('original',original),('primary',cleaned)]:
        scores[name]=overlap_metrics(mask,expert)
        for key in ['dice','true_positive','false_positive','false_negative']:
            if scores[name][key]!=row['scores'][name][key]:
                raise ValueError('Screen scores do not match native masks')
    if not expert.any():
        raise ValueError('This development-case browser expects a nonempty whole-case expert lung mask')
    header=pydicom.dcmread(record['source_ct_files_sorted'][0],stop_before_pixels=True)
    geom=geometry(header.ImagePositionPatient,header.ImageOrientationPatient,record['spacing_zyx_mm'])
    factor=float(np.prod(geom['spacing_zyx_mm'])/1000)
    expert_ml=int(expert.sum())*factor;original_ml=int(original.sum())*factor;cleaned_ml=int(cleaned.sum())*factor
    # Display choice uses prediction alone, not GT, and does not change inference.
    initial=int(original.sum(axis=(1,2)).argmax()) if original.any() else len(original)//2
    start=perf_counter()
    surfaces={name:mesh(a,geom) for name,a in [('expert',expert),('prediction',original)]}
    surfaces['cleaned']=surfaces['prediction'] if np.array_equal(original,cleaned) else mesh(cleaned,geom)
    reconstruction_ms=(perf_counter()-start)*1000
    metadata=dict(case=record['case'],split='validation',initial_slice=initial,
        model='Retained Small 2D U-Net, original epoch '+str(report['protocol']['model_epoch']),
        checkpoint_sha256=report['protocol']['model_sha256'],metrics=scores['original'],
        expert_volume_ml=expert_ml,prediction_volume_ml=original_ml,
        volume_relative_difference=(original_ml-expert_ml)/expert_ml,
        computer_inference_ms=row['inference_ms'],native_reconstruction_ms=reconstruction_ms,
        inference_device='recorded computer run',full_resolution_mesh=True,smoothing=False,
        hole_filling=False,component_removal=False,mesh_decimation=False,
        prediction_source='Saved frozen cleanup-screen masks; no new inference during export/browsing',
        disclaimer='公开数据；仅用于工程教学与算法演示，不作为临床诊断依据。预计算电脑结果，不是板端性能。')
    metadata['cleanup']=dict(metrics=scores['primary'],prediction_volume_ml=cleaned_ml,
        volume_relative_difference=(cleaned_ml-expert_ml)/expert_ml,
        parameters=report['protocol']['primary'],cpu_analysis_ms=row['cpu_analysis_ms'],
        removed_components=row['scores']['primary']['removed_components'],removed_voxels=int((original&~cleaned).sum()),
        warning='Optional conservative 3D postprocess; development validation only, not PL or new blind test')
    data=payload(hu,expert,original,geom,{k:v for k,v in surfaces.items() if k!='cleaned'},metadata)
    data['cleaned']=packed(cleaned,True)
    data['meshes']['cleaned']={k:packed(v) for k,v in surfaces['cleaned'].items()}
    if body_screen:
        from lung_body_crop import checked_plan
        context=body_screen['report'];body_row=next(row for row in context['cases'] if row['case']==record['case'])
        folder=body_screen['path'].parent/record['case']
        with np.load(folder/'predictions_native.npz') as archive:
            if not np.array_equal(archive['original'],original):raise ValueError('Body gate original prediction mismatch')
            variants={name:archive[name] for name in ['body','body_cleaned']}
        body_geometry=json.loads((folder/'body_geometry.json').read_text(encoding='utf-8'))
        if len(body_geometry['plans'])!=len(hu):raise ValueError('Body geometry slice count differs')
        for plan in body_geometry['plans']:checked_plan(plan,hu.shape[1:])
        for z,plan in enumerate(body_geometry['plans']):
            y0,y1,x0,x1=plan['box_y0_y1_x0_x1'];expected=np.zeros(hu.shape[1:],bool)
            expected[y0:y1,x0:x1]=original[z,y0:y1,x0:x1]
            if not np.array_equal(expected,variants['body'][z]):raise ValueError('Body mask does not match native envelope')
        if (variants['body_cleaned']&~variants['body']).any():raise ValueError('Combined mask expanded beyond 2D body gate')
        data['body_geometry']=body_geometry
        data['variants']={};metadata['variants']={}
        for name,mask in variants.items():
            if mask.shape!=original.shape or not np.isin(mask,[0,1]).all() or (mask&~original).any() or (expert&original&~mask).any():
                raise ValueError('Body-gated mask grid/subset/zero true-removal mismatch')
            metrics=overlap_metrics(mask,expert)
            for key in ['dice','true_positive','false_positive','false_negative']:
                if metrics[key]!=body_row['scores'][name][key]:raise ValueError('Body gate metrics mismatch')
            surface=mesh(mask,geom);data['variants'][name]=packed(mask,True)
            data['meshes'][name]={k:packed(v) for k,v in surface.items()}
            volume=int(mask.sum())*factor
            metadata['variants'][name]=dict(metrics=metrics,prediction_volume_ml=volume,
                volume_relative_difference=(volume-expert_ml)/expert_ml,removed_voxels=int((original&~mask).sum()),
                body_gate_cpu_ms=body_row['body_gate_cpu_ms'],approximate_body_envelope=True,
                body_parameters=context['protocol']['parameters'],warning='Development-only output constraint; no retraining or PL.')
    return data


def portal_export(catalog, path):
    entries=catalog['cases']
    if (not entries or len({e['case'] for e in entries})!=len(entries) or
        catalog['default_case'] not in {e['case'] for e in entries}):
        raise ValueError('Nonempty unique cases and an existing default case required')
    for e in entries:
        if not CASE_ID.fullmatch(e['case']) or e['url']!=f'cases/{e["case"]}/viewer.html':
            raise ValueError('Only controlled relative local case URLs allowed')
    text=(Path(__file__).with_name('lungseg')/'case_browser.html').read_text(encoding='utf-8')
    encoded=json.dumps(catalog,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    encoded=encoded.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    Path(path).write_text(text.replace('__CATALOG__',encoded),encoding='utf-8')


def build(args):
    root,manifest,screen,records,tracked=checked_inputs(args.dataset,args.screen,args.checkpoint)
    body_screen=checked_body_screen(getattr(args,'body_screen',None),records,screen['protocol']['model_sha256'],tracked)
    if args.default_case not in {r['case'] for r in records}:
        raise ValueError('Default case must be one of all validation cases')
    output=fresh_output(args.outdir)
    result=dict(status='running',input_sha256=tracked,cases=[],inference_rerun=False,unknown_import_supported=False)
    write_json(output/'report.json',result)
    try:
        for index,r in enumerate(records,1):
            print(f'[{index}/{len(records)}] Reading native CT and exporting {r["case"]}...',flush=True)
            row=next(row for row in screen['cases'] if row['case']==r['case'])
            data=case_payload(root,r,row,args.screen,screen,body_screen)
            folder=output/'cases'/r['case'];folder.mkdir(parents=True)
            html_export(data,folder/'viewer.html')
            metadata=data['metadata'];write_json(folder/'case.json',dict(shape=data['shape'],geometry=data['geometry'],metadata=metadata))
            entry=dict(case=r['case'],url=f'cases/{r["case"]}/viewer.html',shape=data['shape'],
                initial_slice=metadata['initial_slice'],original_dice=metadata['metrics']['dice'],
                cleaned_dice=metadata['cleanup']['metrics']['dice'],removed_voxels=metadata['cleanup']['removed_voxels'],
                bytes=(folder/'viewer.html').stat().st_size,html_sha256=file_sha256(folder/'viewer.html'),
                case_metadata_sha256=file_sha256(folder/'case.json'))
            if body_screen:
                entry.update(body_dice=metadata['variants']['body']['metrics']['dice'],
                    body_cleaned_dice=metadata['variants']['body_cleaned']['metrics']['dice'])
            result['cases'].append(entry);write_json(output/'report.json',result)
            print(f'  Completed {entry["bytes"]/1024**2:.1f} MiB; original masks retained',flush=True)
            del data,metadata
        for path,expected in tracked.items():
            if file_sha256(path)!=expected:
                raise ValueError('Protected input changed during export: '+path)
        catalog=dict(schema=1,default_case=args.default_case,cases=result['cases'],
            warning='Six development validation cases, precomputed computer results. No live inference or unknown import.')
        write_json(output/'catalog.json',catalog);portal_export(catalog,output/'index.html')
        result.update(status='complete_offline_validation_case_browser',protected_inputs_unchanged=True,
            dataset_fingerprint=manifest['fingerprint'],checkpoint_sha256=screen['protocol']['model_sha256'],
            catalog_sha256=file_sha256(output/'catalog.json'),index_sha256=file_sha256(output/'index.html'),
            default_case=args.default_case,default_variant='original',
            body_gate_enabled=bool(body_screen),
            warning='Only existing validation results; no Test/COVID, training, PL inference or clinical validation.')
        write_json(output/'report.json',result)
        print('Open: '+str(output/'index.html'),flush=True)
    except BaseException as e:
        result.update(status='aborted_incomplete',error=str(e));write_json(output/'report.json',result);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--screen',default='data/lung_island_cleanup_20261014/screen/evaluation.json')
    p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    p.add_argument('--body-screen',help='Accepted matching 2D body-envelope output-gate evaluation.json')
    p.add_argument('--default-case',default='LCTSC-Train-S3-008')
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':build(parser().parse_args())
