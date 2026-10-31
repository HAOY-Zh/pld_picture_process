"""Aggregate all paired cases, independently count metrics and seal receipts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
from nodule_plan import sha
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts


def report(folder,require_complete=False):
    folder=Path(folder).resolve();protocol=json.loads((folder/'protocol.json').read_text())
    rows=[];old=[];files={};tiles=planes=values=0
    for uid in protocol['uids']:
        path=folder/(uid+'.json')
        if not path.exists():continue
        a=json.loads(path.read_text())
        if (a['status']!='exact_full_volume_regression_pass' or a['old_sha256']!=a['new_sha256']
                or a['row']!=a['old_row'] or a['row']['uid']!=uid):raise ValueError('Changed/mismatched case '+uid)
        rows.append(a['row']);old.append(a['old_row']);files[path.name]=sha(path)
        tiles+=a['tiles'];planes+=a['planes'];values+=a['required_head_values']
    if require_complete and len(rows)!=len(protocol['uids']):raise ValueError('Cases still incomplete')
    before=summarize_fixed(old,protocol['threshold']);after=summarize_fixed(rows,protocol['threshold'])
    if before!=after:raise ValueError('Quality changed')
    count=independent_counts(rows,protocol['threshold'])
    if any(count[k]!=after[k] for k in count):raise ValueError('Independent counts disagree')
    # Existing FP32 outputs, same scans, same fixed threshold. No rerun or tuning.
    fp32=[]
    import numpy as np
    import gzip
    for row in rows:
        d=ROOT/'data/lung_nodule_five_bundle_20261021_v1/evaluation_cases'/row['uid']
        complete=json.loads((d/'complete.json').read_text())
        for name in ['expanded.json.gz','expanded.npz']:
            if sha(d/name)!=complete['artifacts_sha256'][name]:raise ValueError('Changed historical FP32 output')
        with gzip.open(d/'expanded.json.gz','rt') as f:p=json.load(f)
        with np.load(d/'expanded.npz') as z:scores=z['cross_scores'].tolist()
        fp32.append(dict(row,world=[c['center_lps_xyz_mm'] for c in p['candidates']],scores=scores))
    result=dict(status='complete' if len(rows)==len(protocol['uids']) else 'in_progress',completed=len(rows),expected=len(protocol['uids']),
        baseline_int8=before,canvas_int8=after,historical_fp32_same_threshold=summarize_fixed(fp32,protocol['threshold']),
        sensitivity_delta=0.,fp_per_scan_delta=0.,all_required_values_and_candidates_exact=True,
        tiles=tiles,detector_planes=planes,required_head_values=values,
        scope=protocol['cohort'],threshold_refitted=False,board_timing_measured=False,
        protocol_sha256=sha(folder/'protocol.json'),case_files_sha256=files,
        note='Any FP32-versus-current-INT8 difference predates this spatial reuse optimization. This is calibration regression, not independent test performance.')
    path=folder/('final_report.json' if require_complete else 'aggregate_progress.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if require_complete:(folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','completed','tiles','detector_planes','required_head_values']},indent=2))
    for k in ['baseline_int8','canvas_int8','historical_fp32_same_threshold']:
        print(k, {n:result[k][n] for n in ['detected','reference_nodules','sensitivity','actual_fp_per_scan']})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();report(a.out,a.require_complete)
