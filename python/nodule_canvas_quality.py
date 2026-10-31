"""Paired deployed-INT8 regression on the frozen 75 calibration CTs.

Not training, threshold tuning, blind testing, or CUDA performance measurement.
Check EVERY selected tile's decoder-required cells and full-volume candidates.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
from nodule_plan import load_package,sha
from nodule_canvas import extract_tile_heads,valid_dependency_proof
from nodule_fast_input import SharedPlaneSampler
from nodule_int8_cuda_reference import IntegerNodule
from lung_nodule_cnn import config_from_dict,search_mask,tiles,decode,consolidate
from luna16_multibundle import read_manifest,records_for,load_development_volume
from lung_candidate_run import predict_lungs,excluded_findings
from lung_candidate_rules_run import EXCLUDED
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_continuity import rerank_scores
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts

THRESHOLD=0.6418673229668569


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def run(out,start_index=0,end_index=75):
    import torch
    out=Path(out).resolve();out.mkdir(exist_ok=True)
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board'
    _,_,model,params=load_package(parent);cfg=config_from_dict(model['detector_config'])
    data=read_manifest(ROOT/'data/luna16_expansion_20261021_v2/manifest.json')
    records=records_for(data,'calibration');ranker=json.loads((parent/'nodule_ranker.json').read_text())
    protocol=dict(parent_sha256=sha(parent/'nodule_manifest.json'),uids=[r['series_uid'] for r in records],
        threshold=THRESHOLD,threshold_refitted=False,cohort='calibration_75_not_blind_test',
        dependency_proof=valid_dependency_proof(model),sources_sha256={str(p):sha(p) for p in
            [Path(__file__),ROOT/'python/nodule_int8_cuda_reference.py',ROOT/'pynq/nodule_canvas.py',ROOT/'pynq/nodule_fast_input.py']})
    pp=out/'protocol.json'
    if pp.exists():
        prior=json.loads(pp.read_text())
        # Keep the initial receipt. Driver execution fixes may resume saved
        # exact cases; inference engine sources/protocol stay frozen, and every
        # reconstructed lung input must match the historical SHA256 exactly.
        previous=prior.pop('sources_sha256');current=dict(protocol);current.pop('sources_sha256')
        if prior!=current or any(previous[k]!=v for k,v in protocol['sources_sha256'].items() if k!=str(Path(__file__))):
            raise ValueError('Protocol/numerical sources changed')
        if previous[str(Path(__file__))]!=sha(__file__):
            save(out/('driver_execution_'+str(start_index)+'.json'),dict(reason='Execution fixes: undefined-sensitivity display, disjoint ranges, historical lung batch16/TF32-off settings; numerical nodule engine and protocol remain frozen',
                initial_sha256=previous[str(Path(__file__))],current_sha256=sha(__file__)))
    else:save(pp,protocol)
    net=IntegerNodule(model,params);excluded=excluded_findings(EXCLUDED);rows=[];oldrows=[];lungnet=None
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        for index,r in enumerate(records):
            if not start_index<=index<end_index:continue
            uid=r['series_uid'];dest=out/(uid+'.json')
            if dest.exists():
                packet=json.loads(dest.read_text());rows.append(packet['row']);oldrows.append(packet['old_row']);continue
            tick=time.perf_counter();hu,g=load_development_volume(data,uid,'calibration')
            maskfile=ROOT/'data/lung_nodule_context_20261020_v1/calibration_masks'/(uid+'.npz')
            if maskfile.exists():
                with np.load(maskfile) as z:lung=np.unpackbits(z['body'],count=int(np.prod(z['shape_zyx']))).reshape(z['shape_zyx']).astype(bool)
            else:
                if lungnet is None:
                    from lung_unet import make_model
                    cp=torch.load(ROOT/'data/lung_training_20261011/early_stop_run/best.pt',map_location='cpu',weights_only=True)
                    lungnet=make_model(cp['base_channels'],cp.get('context_slices',1)).cuda().eval();lungnet.load_state_dict(cp['model'])
                torch.backends.cudnn.benchmark=False
                torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
                _,lung,_=predict_lungs(hu,g,lungnet,'cuda',16)
            oldpath=ROOT/'data/lung_nodule_five_bundle_20261021_v1/evaluation_cases'/uid/'expanded.json.gz'
            with gzip.open(oldpath,'rt',encoding='utf-8') as f:oldpacket=json.load(f)
            masksha=hashlib.sha256(lung.tobytes()).hexdigest()
            if masksha!=oldpacket['lung_mask_sha256']:raise ValueError('Mask differs from frozen FP32 evaluation')
            search=search_mask(lung,g['spacing_zyx_mm'],cfg);jobs,schedule=tiles(search,g['spacing_zyx_mm'],cfg)
            groups=defaultdict(list)
            for j in jobs:groups[j[0]].append(j)
            parts=[[],[],[]];oldparts=[[],[],[]];cells=0;oldhash=hashlib.sha256();newhash=hashlib.sha256()
            for zi,group in enumerate(groups.values()):
                sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],group,cfg)
                old=np.concatenate([net(np.stack([sampler.patch(j) for j in group[b:b+16]])) for b in range(0,len(group),16)])
                h,w=sampler.context.shape[1:]
                if h<=1024 and w<=1024:
                    new=extract_tile_heads(net(sampler.context[None])[0],group,sampler.y0,sampler.x0,cfg)
                else:raise ValueError('Unsupported canvas; no silent truncation')
                a=old[:,:,15:49,15:49];b=new[:,:,15:49,15:49]
                np.testing.assert_array_equal(a,b);cells+=a.size
                oldhash.update(a.tobytes());newhash.update(b.tobytes())
                before=decode(old,group,g,cfg,search);after=decode(new,group,g,cfg,search)
                for x,y in zip(before,after):np.testing.assert_array_equal(x,y)
                for bank,decoded in [(parts,after),(oldparts,before)]:
                    for destbank,value in zip(bank,decoded):destbank.append(value)
                if zi%100==0:print('Case {}/75, plane {}/{}'.format(index+1,zi+1,len(groups)),flush=True)
            oldc=consolidate(*(np.concatenate(a) for a in oldparts),g,cfg) if jobs else []
            candidates=consolidate(*(np.concatenate(a) for a in parts),g,cfg) if jobs else []
            if candidates!=oldc:raise AssertionError('Whole-volume NMS differs')
            for bank in [candidates,oldc]:
                for c in bank:
                    box=np.asarray(c['bbox_zyx']).reshape(3,2);box[:,0]=np.maximum(0,box[:,0]);box[:,1]=np.minimum(hu.shape,box[:,1]);c['bbox_zyx']=box.ravel().tolist()
            # Two independent calls also check deterministic downstream scoring.
            features,_=evidence_for_packet(hu,g['spacing_zyx_mm'],candidates)
            oldfeatures,_=evidence_for_packet(hu,g['spacing_zyx_mm'],oldc)
            scores=rerank_scores(np.asarray([c['score'] for c in candidates]),features,ranker)
            oldscores=rerank_scores(np.asarray([c['score'] for c in oldc]),oldfeatures,ranker)
            np.testing.assert_array_equal(scores,oldscores)
            base=dict(uid=uid,record=r,excluded=excluded.get(uid,[]))
            row=dict(base,world=[c['center_lps_xyz_mm'] for c in candidates],scores=scores.tolist())
            oldrow=dict(base,world=[c['center_lps_xyz_mm'] for c in oldc],scores=oldscores.tolist())
            packet=dict(status='exact_full_volume_regression_pass',row=row,old_row=oldrow,tiles=len(jobs),planes=len(groups),
                required_head_values=cells,old_sha256=oldhash.hexdigest(),new_sha256=newhash.hexdigest(),
                predicted_mask_sha256=masksha,elapsed_sec=time.perf_counter()-tick)
            save(out/(uid+'.json'),packet);rows.append(row);oldrows.append(oldrow)
            oldmetric=summarize_fixed(oldrows,THRESHOLD);newmetric=summarize_fixed(rows,THRESHOLD)
            if oldmetric!=newmetric:raise AssertionError('Detection quality changed')
            counts=independent_counts(rows,THRESHOLD)
            if any(counts[k]!=newmetric[k] for k in counts):raise AssertionError('Metric count discrepancy')
            report=dict(status='complete' if len(rows)==len(records) else 'in_progress',completed=len(rows),expected=len(records),
                baseline_int8=oldmetric,canvas_int8=newmetric,all_required_values_and_candidates_exact=True,
                precision_change=0,elapsed_sec=time.perf_counter()-start,scope=protocol['cohort'],
                fp32_published_numbers_not_substituted=True,board_timing_measured=False)
            save(out/('report.json' if start_index==0 else 'report_worker_'+str(start_index)+'.json'),report)
            sensitivity='N/A' if newmetric['sensitivity'] is None else '{:.2%}'.format(newmetric['sensitivity'])
            print('Completed {}/75: sensitivity {}, FP/CT {:.3f}, EXACT; case {:.1f}s'.format(len(rows),sensitivity,newmetric['actual_fp_per_scan'],packet['elapsed_sec']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int,default=75)
    a=p.parse_args();run(a.out,a.start_index,a.end_index)
