"""Exercise the portable environment against saved real-board outputs.

No new model/threshold is fitted. CT data and validation results stay outside
the redistributable package. Board inference is optional and never retried.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--app',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('postprocess','nodule'),required=True);args=p.parse_args()
    app=args.app.resolve();root=args.workspace.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    sys.path.insert(0,str(app/'python'))
    import numpy as np
    import torch
    from lung_live_data import read_ct,input_codes,native_masks
    from lung_island_cleanup import clean_prediction,PRIMARY
    torch.set_num_threads(2)
    case=root/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/08ea2953f1664d608f11b863f07fe68e'
    info=json.loads((case/'input.json').read_text(encoding='utf8'))
    hu,geom,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if details['hu_sha256']!=info['hu_sha256']:raise ValueError('Input CT identity changed')
    package=app/'data/ct_studio_integration_20261025'
    model=json.loads((package/'model.json').read_text(encoding='utf8'))
    baseline=json.loads((root/'data/dual_pipeline_optimization_20261026/seg_optimized/result.json').read_text())
    volume=json.loads((case/(baseline['job_id']+'_volume.json')).read_text(encoding='utf8'))
    began=time.perf_counter()
    if args.mode=='postprocess':
        with np.load(case/(baseline['job_id']+'_masks.npz'),allow_pickle=False)as z:
            masks={k:z[k]for k in z.files}
        body=np.zeros_like(masks['body'])
        for z in range(len(hu)):
            rid=volume['slice_receipts'][str(z)]
            with np.load(case/rid/'actual_head.npz',allow_pickle=False)as saved:
                np.testing.assert_array_equal(input_codes(hu[z],model),saved['input_codes'])
                raw,body[z],plan=native_masks(saved['head_codes'],hu[z],geom,model)
                np.testing.assert_array_equal(raw,masks['raw'][z])
                np.testing.assert_array_equal(body[z],masks['body'][z])
            if z%50==0:print('PORTABLE_EXACT_NATIVE_POSTPROCESS',z+1,'/',len(hu),flush=True)
        cleaned,cleanup=clean_prediction(body,geom['spacing_zyx_mm'],**PRIMARY)
        np.testing.assert_array_equal(cleaned,masks['cleaned'])
        actual_volume=float(cleaned.sum()*np.prod(geom['spacing_zyx_mm'])/1000)
        assert actual_volume==volume['volume_ml']
        result=dict(status='PORTABLE_351_INPUTS_AND_ALL_NATIVE_3D_MASKS_EXACT_PASS',slices=len(hu),volume_ml=actual_volume,
            elapsed_ms=(time.perf_counter()-began)*1000)
    else:
        from start_ct_studio_demo import configuration
        from nodule_detector import NoduleCaseProvider
        from nodule_canvas_contract import digest
        client,_=configuration(package/'host_config.json');client.health()
        provider=NoduleCaseProvider(package,client)
        oldjob=json.loads((root/'data/dual_pipeline_optimization_20261026/nodule_full_case/result.json').read_text())['job_id']
        old=case/'nodule'/oldjob;packet=json.loads((old/'prediction.json').read_text(encoding='utf8'))
        context={k:packet[k]for k in ('case_id','analysis_source_id','context_id','source_hu_sha256','geometry_sha256','segmentation_signature')}
        context.update(shape=list(hu.shape),geometry=geom,whole_case_ready=True)
        with np.load(case/(baseline['job_id']+'_masks.npz'),allow_pickle=False)as z:lung=z['body']
        # Use the same mask stage selected by the production provider. Its
        # context/provenance records the source, rather than using annotations.
        provenance=packet['detector_provenance']
        print('PORTABLE_NODULE_MODEL',provenance.get('pipeline'),flush=True)
        def progress(stage,done,total):
            if stage!='inference'or done%400<30:print('PORTABLE_NODULE',stage,done,'/',total,flush=True)
        candidates,new_provenance=provider.run(hu,lung,geom,context,out,progress,lambda:False)
        assert candidates==packet['candidates'],'Candidate output changed'
        with gzip.open(old/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:expected=json.load(f)
        with gzip.open(out/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:actual=json.load(f)
        assert expected==actual,'Full candidate scores changed'
        original=[json.loads(l)for l in(old/'canvas_receipts.jsonl').open()]
        current=[json.loads(l)for l in(out/'canvas_receipts.jsonl').open()]
        assert len(original)==len(current)
        for a,b in zip(original,current):
            assert a['jobs']==b['jobs']
            assert a['receipt']['input_sha256']==b['receipt']['input_sha256']
            assert a['receipt']['output_sha256']==b['receipt']['output_sha256']
        result=dict(status='PORTABLE_ACTUAL_BOARD_ALL_CANVAS_HEADS_AND_CANDIDATES_EXACT_PASS',
            canvases=len(current),raw_candidates=len(actual),selected_candidates=len(candidates),
            elapsed_ms=(time.perf_counter()-began)*1000,detector_provenance=new_provenance)
    result.update(runtime_executable=sys.executable,numpy_version=np.__version__,torch_version=torch.__version__)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k!='detector_provenance'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
