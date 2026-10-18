"""Verify actual FP32 logits and CPU/PL baseline identities after board return."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from verify_lung_prepared_board import verify as verify_int8
from lung_unet import file_sha256
from lung_segmentation import write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_threeway_combo import validate


def stats(record,count):
    a=np.asarray(record['samples_ms'],dtype=np.float64)
    if a.shape!=(count,) or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('Invalid samples')
    for k,v in [('mean_ms',np.mean(a)),('median_ms',np.median(a)),('min_ms',np.min(a)),('max_ms',np.max(a))]:
        if not math.isfinite(record[k]) or abs(record[k]-v)>1e-6:raise ValueError('Invalid statistics: '+k)


def verify(root,package,stem):
    root=Path(root).resolve();package=Path(package).resolve();out=root/(stem+'_threeway_verification.json')
    if out.exists():raise FileExistsError('Preserve previous acceptance')
    manifest=validate(package)
    def path(s):return root/(stem+s)
    def read(s):return json.loads(path(s+'.json').read_text(encoding='utf-8'))
    # Re-run immutable INT8 verifier with fresh destination, not a manually
    # edited acceptance. This validates actual head, all commands and clocks.
    if path('_verification.json').exists():raise FileExistsError('Use this verifier first, do not overwrite existing INT8 proof')
    fp=read('_fp32');summary=read('_threeway');r=read('')
    if fp['status']!='passed_fp32_cpu_phase' or summary['status']!='passed_same_board_fp32_int8_fpga':raise ValueError('Not threeway PASS')
    if any(x['model_label']!=manifest['model_label'] or x['checkpoint_sha256']!=manifest['checkpoint_sha256'] for x in [fp,summary]):raise ValueError('Different FP32 model identity')
    if fp['package_sha256']!=file_sha256(package/'fp32_manifest.json') or fp['threads']!=2 or abs(fp['cpu_mhz']-650)>.05:raise ValueError('FP32 identity/CPU differs')
    if (fp['case'],fp['slice_index'],fp['runs'],fp['warmup'])!=(manifest['case'],manifest['slice_index'],r['runs'],r['warmup']):raise ValueError('Different CT/repeats')
    if (fp['atol'],fp['rtol'])!=(manifest['atol'],manifest['rtol']):raise ValueError('Changed tolerance')
    stats(fp,r['runs'])
    if len(fp['clock_reads'])!=r['runs'] or any(abs(c[k]-650)>.05 for c in fp['clock_reads'] for k in ['before_mhz','after_mhz']):raise ValueError('FP32 clocks changed')
    if set(fp['node_max_abs_error'])!=set(manifest['node_names']) or any(not math.isfinite(v) or v<0 for v in fp['node_max_abs_error'].values()):raise ValueError('Incomplete FP32 trace check')
    for n,h in fp['build']['sources_sha256'].items():
        if file_sha256(package/n)!=h:raise ValueError('Different native FP32 source')
    if '-O3' not in fp['build']['command'] or '-mfpu=neon' not in fp['build']['command'] or '-fopenmp' not in fp['build']['command']:raise ValueError('Not optimized native ARM build')
    if fp['head_sha256']!=file_sha256(path('_fp32_head.npz')):raise ValueError('Changed actual FP32 head')
    with np.load(path('_fp32_head.npz'),allow_pickle=False) as a:head=a['head_logits']
    with np.load(package/'fp32_reference.npz',allow_pickle=False) as a:ref=a['head']
    if head.dtype!=np.float32 or head.shape!=ref.shape:raise ValueError('Wrong FP32 head format')
    np.testing.assert_allclose(head,ref,atol=manifest['atol'],rtol=manifest['rtol'])
    if abs(fp['final_head_max_abs_error']-float(np.max(np.abs(head-ref))))>1e-12:raise ValueError('Wrong float error')
    for suffix,key in [('_fp32.json','fp32_report_sha256'),('.json','int8_report_sha256'),('_clock.json','clock_sha256'),('_selection.json','selection_sha256')]:
        if summary[key]!=file_sha256(path(suffix)):raise ValueError('Changed report cross-reference')
    times=[fp['mean_ms'],r['cpu_two_threads']['mean_ms'],r['optimized_pl']['mean_ms']]
    for k,v in zip(['fp32_ms','cpu_int8_ms','fpga_int8_ms'],times):
        if abs(summary[k]-v)>1e-9:raise ValueError('Summary time differs')
    for k,a,b in [('cpu_fp32_over_cpu_int8',0,1),('cpu_int8_over_fpga_int8',1,2),('cpu_fp32_over_fpga_int8',0,2)]:
        if abs(summary[k]-times[a]/times[b])>1e-9:raise ValueError('Speedup arithmetic differs')
    accepted=verify_int8(root,package,stem)
    result=dict(status='verified_returned_threeway_actual_fp32_and_int8_heads',fp32_ms=times[0],cpu_int8_ms=times[1],fpga_int8_ms=times[2],
        ratios={k:summary[k] for k in ['cpu_fp32_over_cpu_int8','cpu_int8_over_fpga_int8','cpu_fp32_over_fpga_int8']},
        actual_fp32_head_max_abs_error=float(np.max(np.abs(head-ref))),int8_head_bit_exact=accepted['actual_head_codes_bit_exact'],
        returned_files_sha256={path(s).name:file_sha256(path(s)) for s in ['.json','_fp32.json','_fp32_head.npz','_threeway.json','_head.npz','_clock.json','_selection.json','_diagnostic.json','_cpu.json']},
        warning='Original FP32 and frozen INT8 are different numerical models; one repeated validation slice, not dataset accuracy or whole-system throughput')
    write_json(out,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default='data/lung_threeway_20261016');ap.add_argument('--package',default='data/lung_threeway_20261016/lung_threeway');ap.add_argument('--stem',default='lung_threeway_result')
    a=ap.parse_args();verify(a.root,a.package,a.stem)
