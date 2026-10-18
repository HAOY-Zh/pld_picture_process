"""Frozen CPU INT8 predictions + CT-only body gate + unchanged 3D cleanup.

Development validation only. No training/calibration/Test access or FPGA claim.
Saved prediction masks are checked against original metrics and fixed live probes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_body_gate import gate_volume, save_preview
from lung_case_browser import checked_inputs
from lung_island_cleanup import PRIMARY, clean_prediction
from lung_quantize import restore_quantized
from lung_segmentation import fresh_output, write_json
from lung_unet import (aggregate_slice_quality, file_sha256, load_cached_source,
                       mapped_npy, slice_quality_metrics, torch_module)
from lungseg.metrics import overlap_metrics


def final_masks(hu, raw, spacing):
    body, plans, elapsed = gate_volume(hu, raw, spacing[1:])
    final, geometry = clean_prediction(body, spacing, **PRIMARY)
    return body, final, plans, elapsed, geometry


def read_body_masks(path, expected_hash):
    if file_sha256(path)!=expected_hash:raise ValueError('Accepted FP32 body artifact hash mismatch')
    with np.load(path) as a:
        raw,final=a['original'],a['body_cleaned']
    if (raw.shape!=final.shape or raw.ndim!=3 or not np.isin(raw,[0,1]).all() or
        not np.isin(final,[0,1]).all() or (final.astype(bool)&~raw.astype(bool)).any()):
        raise ValueError('Invalid accepted native FP32 subset masks')
    return raw!=0,final!=0


def score_mask(mask, reference):
    return dict(metrics=overlap_metrics(mask, reference),
                slice_quality=slice_quality_metrics(mask, reference))


def removed_counts(raw, final, reference):
    raw, final, reference = [np.asarray(a) != 0 for a in [raw, final, reference]]
    if raw.shape != final.shape or reference.shape != raw.shape or (final & ~raw).any():
        raise ValueError('Final postprocess must be a native subset, not expand labels')
    removed = raw & ~final
    return dict(true_positive=int((removed & reference).sum()),
                false_positive=int((removed & ~reference).sum()))


def live_probe(model, root, record, expected, batch=4):
    """Predetermined indices, not chosen by GT/score. No observer updates."""
    torch = torch_module()
    indices = sorted({0, record['slices']//2, record['slices']-1} |
                     ({186} if record['case']=='LCTSC-Train-S3-002' else set()))
    with mapped_npy(root/record['cache_folder']/'images.npy') as images:
        for start in range(0, len(indices), batch):
            selected = indices[start:start+batch]
            inputs = torch.from_numpy(np.array(images[selected], dtype=np.float32))[:, None]
            with torch.inference_mode():
                probabilities = torch.nn.functional.interpolate(torch.sigmoid(model(inputs)),
                    size=tuple(record['native_shape'][1:]), mode='bilinear', align_corners=False)
            actual = probabilities[:, 0].numpy() >= .5
            if not np.array_equal(actual, expected[selected] != 0):
                raise ValueError('Frozen INT8 live probe does not reproduce saved masks')
    return indices


def summarize_rows(rows, stages):
    return {stage: dict(mean_case_dice=float(np.mean([r['scores'][stage]['metrics']['dice'] for r in rows])),
        true_positive=sum(r['scores'][stage]['metrics']['true_positive'] for r in rows),
        false_positive=sum(r['scores'][stage]['metrics']['false_positive'] for r in rows),
        false_negative=sum(r['scores'][stage]['metrics']['false_negative'] for r in rows),
        slice_quality=aggregate_slice_quality([r['scores'][stage]['slice_quality'] for r in rows]))
        for stage in stages}


def acceptance(rows, summaries, mean_drop=.005, small_drop=.03):
    delta = summaries['int8_final']['mean_case_dice']-summaries['fp32_final']['mean_case_dice']
    small = summaries['int8_final']['slice_quality']['groups']['small']['mean_slice_dice']
    baseline = summaries['fp32_final']['slice_quality']['groups']['small']['mean_slice_dice']
    small_delta = None if small is None or baseline is None else small-baseline
    return dict(accepted_for_hardware_design=bool(rows and delta >= -mean_drop and
        (small_delta is None or small_delta >= -small_drop) and
        all(r['removed']['true_positive']==0 and r['expert_outside_envelope_voxels']==0 for r in rows)),
        final_mean_dice_delta=delta, final_small_slice_dice_delta=small_delta,
        warning='CPU quantization development screen only; not fixed-point whole-network or PL equivalence')


def run(args):
    root, manifest, old, records, tracked = checked_inputs(args.dataset, args.old_screen, args.checkpoint)
    # Old presentation files are not part of this numerical experiment.
    tracked = {p:h for p,h in tracked.items() if Path(p).suffix not in ['.py', '.html']}
    body_path = Path(args.body_screen).resolve(strict=True)
    body_report = json.loads(body_path.read_text(encoding='utf-8'))
    quant_dir = Path(args.quantization).resolve(strict=True)
    quant_report = json.loads((quant_dir/'evaluation.json').read_text(encoding='utf-8'))
    quant_protocol = json.loads((quant_dir/'protocol.json').read_text(encoding='utf-8'))
    names = [r['case'] for r in records]
    model_hash = file_sha256(args.checkpoint)
    if (body_report.get('status')!='complete_body_gate_development_screen' or
        not body_report.get('accepted_for_optional_view') or
        not body_report.get('protected_inputs_unchanged') or
        body_report['protocol']['model_sha256']!=model_hash or
        quant_report.get('status')!='complete_validation_ptq' or
        not quant_report.get('source_checkpoint_unchanged') or
        quant_report['checkpoint_sha256']!=model_hash or
        quant_protocol['checkpoint_sha256']!=model_hash or
        quant_protocol['dataset_sha256']!=file_sha256(args.dataset) or
        quant_protocol['dataset_fingerprint']!=manifest['fingerprint'] or quant_protocol['test_used'] or
        quant_protocol['image_size']!=manifest['image_size'] or quant_protocol['threshold']!=.5 or
        quant_protocol['calibration_cases']!=[r['case'] for r in manifest['cases'] if r['split']=='train'] or
        quant_protocol['validation_cases']!=names or
        any(len(report['cases'])!=len(names) or {r['case'] for r in report['cases']}!=set(names)
            for report in [body_report, quant_report])):
        raise ValueError('Need exact accepted body screen and frozen matching INT8 validation')
    for p,h in body_report['protocol']['input_sha256'].items():
        if file_sha256(p)!=h: raise ValueError('Body experiment source changed: '+p)
        tracked[p]=h
    for p in [body_path, quant_dir/'evaluation.json', quant_dir/'protocol.json',
              quant_dir/'int8_model.pt', quant_dir/'int8_core_parameters.npz', quant_dir/'core_manifest.json',
              Path(__file__).resolve(), Path(__file__).with_name('lung_body_gate.py'),
              Path(__file__).with_name('lung_quantize.py')]:
        tracked[str(p.resolve())]=file_sha256(p)
    for r in records:
        for p in [quant_dir/r['case']/'int8_prediction_native.npy',
                  quant_dir/r['case']/'float_prediction_native.npy',
                  body_path.parent/r['case']/'predictions_native.npz']:
            tracked[str(p.resolve())]=file_sha256(p)
    model, saved = restore_quantized(quant_dir/'int8_model.pt')
    if (saved['source_checkpoint_sha256']!=model_hash or saved['dataset_fingerprint']!=manifest['fingerprint'] or
        saved['image_size']!=manifest['image_size'] or saved['threshold']!=.5 or saved['context_slices']!=1):
        raise ValueError('INT8 model identity mismatch')
    torch_module().set_num_threads(args.threads)
    output=fresh_output(args.outdir)
    protocol=dict(input_sha256=tracked, validation_cases=names, threshold=.5,
        parameters=body_report['protocol']['parameters'], island_parameters=PRIMARY,
        engineering_screen=dict(max_mean_dice_drop=.005,max_small_slice_dice_drop=.03,
            no_removed_int8_true_positive=True),
        retraining=False, recalibration=False, test_used=False,
        prediction_source='Previously validated frozen CPU INT8 native masks; fixed live probes per case',
        order='Frozen 2D INT8 -> sigmoid/native bilinear threshold on CPU -> CT-only body gate -> optional 3D cleanup')
    write_json(output/'protocol.json', protocol)
    report=dict(status='running',protocol=protocol,cases=[])
    stages=['fp32_raw','fp32_final','int8_raw','int8_body','int8_final']
    try:
        for r in records:
            name=r['case']; print('Final INT8 pipeline: '+name,flush=True)
            hu=load_cached_source(r)
            body_row=next(row for row in body_report['cases'] if row['case']==name)
            fp32,fp32_final=read_body_masks(body_path.parent/name/'predictions_native.npz',body_row['prediction_sha256'])
            with mapped_npy(quant_dir/name/'int8_prediction_native.npy') as a:
                raw=np.array(a!=0)
            with mapped_npy(quant_dir/name/'float_prediction_native.npy') as a:
                if not np.array_equal(a!=0,fp32):raise ValueError('Frozen FP32 prediction differs')
            if raw.shape!=hu.shape or fp32.shape!=hu.shape: raise ValueError('Native shape mismatch')
            probes=live_probe(model,root,r,raw)
            body,final,plans,elapsed,geo=final_masks(hu,raw,r['spacing_zyx_mm'])
            folder=output/name;folder.mkdir()
            # CT-only decisions precede reading expert mask.
            write_json(folder/'body_geometry.json',dict(plans=plans,expert_used=False))
            write_json(folder/'island_geometry.json',geo)
            np.savez_compressed(folder/'predictions_native.npz',int8_raw=raw,int8_body=body,int8_final=final)
            with mapped_npy(root/r['cache_folder']/'reference_native.npy') as reference:
                scores={stage:score_mask(a,reference) for stage,a in
                    zip(stages,[fp32,fp32_final,raw,body,final])}
                previous=next(row for row in quant_report['cases'] if row['case']==name)
                for metric in ['dice','true_positive','false_positive','false_negative']:
                    if scores['fp32_final']['metrics'][metric]!=body_row['scores']['body_cleaned'][metric]:
                        raise ValueError('Accepted FP32 final screen scores mismatch')
                for stage,key in [('fp32_raw','fp32'),('int8_raw','int8')]:
                    if scores[stage]['metrics']!=previous[key]:raise ValueError('Frozen native scores mismatch')
                removed=removed_counts(raw,final,reference)
                uncovered=sum(int(np.count_nonzero(reference[z])-np.count_nonzero(reference[z,
                    plan['box_y0_y1_x0_x1'][0]:plan['box_y0_y1_x0_x1'][1],
                    plan['box_y0_y1_x0_x1'][2]:plan['box_y0_y1_x0_x1'][3]])) for z,plan in enumerate(plans))
                index=186 if name=='LCTSC-Train-S3-002' else int((raw&~final).sum((1,2)).argmax())
                save_preview(hu[index],reference[index]!=0,raw[index],body[index],final[index],
                    plans[index],folder/'preview.png',name+' | frozen CPU INT8',index)
                inspected=dict(index=index,expert=int(np.count_nonzero(reference[index])),
                    raw=int(raw[index].sum()),body=int(body[index].sum()),final=int(final[index].sum()))
            row=dict(case=name,split='validation',scores=scores,removed=removed,
                expert_outside_envelope_voxels=uncovered,live_probe_indices=probes,
                body_cpu_ms=elapsed,island_cpu_ms=geo['cpu_analysis_ms'],inspected_slice=inspected,
                final_dice_delta=scores['int8_final']['metrics']['dice']-scores['fp32_final']['metrics']['dice'],
                prediction_sha256=file_sha256(folder/'predictions_native.npz'))
            report['cases'].append(row);write_json(output/'evaluation.json',report)
            print(f'  FP32 final {scores["fp32_final"]["metrics"]["dice"]:.6f}; INT8 final {scores["int8_final"]["metrics"]["dice"]:.6f}; removed TP {removed["true_positive"]}',flush=True)
            del hu,fp32,fp32_final,raw,body,final
        summaries=summarize_rows(report['cases'],stages)
        for p,h in tracked.items():
            if file_sha256(p)!=h:raise ValueError('Protected input changed: '+p)
        report.update(status='complete_cpu_int8_final_pipeline',summaries=summaries,
            protected_inputs_unchanged=True,**acceptance(report['cases'],summaries))
        write_json(output/'evaluation.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['cases','protocol','summaries']},indent=2),flush=True)
    except BaseException as e:
        report.update(status='aborted_incomplete',error=str(e));write_json(output/'evaluation.json',report);raise
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--old-screen',default='data/lung_island_cleanup_20261014/screen/evaluation.json')
    p.add_argument('--body-screen',default='data/lung_body_gate_20261014/screen/evaluation.json')
    p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    p.add_argument('--quantization',default='data/lung_training_20261011/int8_ptq_validation_v2')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
