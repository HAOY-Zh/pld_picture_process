"""Physical same-input old/new comparisons; all hardware calls are serial.

Uses existing frozen real-case records for equality, never reference annotations.
The complete-task timings include actual PC preparation/save/postprocessing but
exclude initial file loading, startup selftests and browser drawing. Warm overlay
cost is recorded separately and never charged to only one comparison arm.
"""
import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
OLD=ROOT/'data/CT_Studio_DualExact_20261026_02'
PACKAGE=NEW/'data/ct_studio_integration_20261025'
OUT=ROOT/'data/exact_speed_board_acceptance_20261027'
sys.path[:0]=[str(NEW/'python')]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def stats(values):
    values=np.asarray(values,float)
    return dict(mean_ms=float(values.mean()),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples_ms=values.tolist())


def summarize(functions):
    folder=OUT/'full'
    reports={name:json.loads((folder/(name+'_result.json')).read_text(encoding='utf8'))for name in functions.split(',')}
    assert all(r['all_outputs_exact']for r in reports.values())
    target=folder/('complete_'+functions.replace(',','_')+'.json')
    write(target,dict(status='PHYSICAL_FULL_CASE_EXACT_PASS',results=reports,
        browser_drawing_included=False,labels_used=False,cleanup_changed=False))
    print('COMPLETE_REPORT_SAVED_FROM_VERIFIED_COMPONENT_RESULTS',target,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('pilot','full','summarize'))
    parser.add_argument('--functions',default='ldct,segmentation,nodule');args=parser.parse_args()
    if args.phase=='summarize':
        summarize(args.functions);return  # No board connection or computation.
    selected=set(args.functions.split(','))
    from start_ct_studio_demo import configuration
    from ct_shared_seg_client import SharedSegStudioClient
    from ct_dual_io_client import DualSegStudioClient
    from lung_live_data import read_ct
    from nodule_canvas_contract import digest
    from lung_unet import torch_module
    torch_module().set_num_threads(2)
    client,manifest=configuration(PACKAGE/'host_config.json');client.health()
    legacy=SharedSegStudioClient(client.url,client.token,manifest);legacy.health()
    # Physical pilot rejected LDCT compression: CPU codec cost exceeded LAN savings.
    # Keep the new exact host pipeline, but use the verified original LDCT route.
    ldct_client=DualSegStudioClient(client.url,client.token,manifest);ldct_client.health()
    state=json.loads((ROOT/'data/current_effect_audit_20261026/state.json').read_text(encoding='utf8'))
    results=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'
    reference_case=results/state['case_id'];base=results/state['series_id']
    info=json.loads((base/'input.json').read_text(encoding='utf8'))
    original,geometry,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if digest(original.tobytes())!=info['hu_sha256']or geometry!=state['geometry']:raise ValueError('Original case changed')
    ldct_reference=sorted(state['ldct']['results'],key=lambda row:row['index'])
    enhanced=np.empty_like(original)
    for row in ldct_reference:
        with np.load(base/'ldct'/(row['result_id']+'.npz'),allow_pickle=False)as z:enhanced[row['index']]=z['hu']
    assert digest(enhanced.tobytes())==state['analysis_source']['hu_sha256']
    reference_volume=json.loads((reference_case/(state['volume_job_id']+'_volume.json')).read_text(encoding='utf8'))
    with np.load(reference_case/(state['volume_job_id']+'_masks.npz'),allow_pickle=False)as z:
        expected_masks={k:z[k]for k in z.files}
    folder=OUT/args.phase;folder.mkdir(exist_ok=True)
    write(folder/'protocol.json',dict(source_hu_sha256=info['hu_sha256'],enhanced_hu_sha256=state['analysis_source']['hu_sha256'],
        slices=len(original),service_bundle_sha256=manifest['service_bundle_sha256'],cleanup='legacy',
        scope='Same case/input, physical PL calls; full tasks include PC save/postprocessing, exclude initial CT read and browser drawing.',expert_labels_used=False))
    if args.phase=='pilot':
        indices=list(range(35,len(original),35))
        output={}
        for function in ['ldct','segmentation']:
            if function not in selected:continue
            rows=[]
            for rep in range(3):
                for name,c in ([('baseline',legacy),('optimized',client)]if rep%2==0 else [('optimized',client),('baseline',legacy)]):
                    if function=='ldct':c.denoise(original[175],'cnn')
                    else:
                        rid=reference_volume['slice_receipts']['175']
                        with np.load(reference_case/rid/'actual_head.npz',allow_pickle=False)as z:c.infer(z['input_codes'])
                    times=[]
                    for index in indices:
                        if function=='ldct':
                            actual,receipt=c.denoise(original[index],'cnn');expected=ldct_reference[index]['output_sha256']
                            assert digest(actual.tobytes())==expected
                        else:
                            rid=reference_volume['slice_receipts'][str(index)]
                            with np.load(reference_case/rid/'actual_head.npz',allow_pickle=False)as z:
                                actual,receipt=c.infer(z['input_codes']);np.testing.assert_array_equal(actual,z['head_codes'])
                        times.append(receipt['roundtrip_ms'])
                    row=dict(repetition=rep,mode=name,**stats(times));rows.append(row)
                    print('PHYSICAL_PILOT',function,rep,name,round(row['mean_ms'],3),'ms, ALL_EXACT',flush=True)
                    write(folder/(function+'_progress.json'),rows)
            a=np.mean([r['mean_ms']for r in rows if r['mode']=='baseline']);b=np.mean([r['mean_ms']for r in rows if r['mode']=='optimized'])
            output[function]=dict(status='PHYSICAL_PAIRED_EXACT_PASS',rows=rows,baseline_mean_ms=a,optimized_mean_ms=b,latency_reduction_percent=100*(1-b/a))
            write(folder/(function+'_result.json'),output[function])
        return
    old_ldct=load('baseline_ldct_actual',OLD/'python/ldct_workspace.py')
    new_ldct=load('new_ldct_actual',NEW/'python/ldct_workspace.py')
    old_lung=load('baseline_lung_actual',OLD/'python/lung_live_web.py')
    import lung_live_web as new_lung
    old_nodule=load('baseline_nodule_actual',OLD/'python/nodule_detector.py')
    import nodule_detector as new_nodule
    model=json.loads((PACKAGE/'model.json').read_text(encoding='utf8'))
    all_results={}

    def make_controller(cls,c,hu,name,analysis_source):
        out=folder/name;out.mkdir(exist_ok=False)
        controller=cls(c,model,out);case_id=uuid.uuid4().hex
        controller.hu=hu;controller.geom=geometry;controller.details=copy.deepcopy(details)
        descriptor=dict(source_id=analysis_source,label='Physical speed comparison',case_id=case_id,
            hu_sha256=digest(hu.tobytes()))
        controller.details.update(analysis_source=descriptor,hu_sha256=descriptor['hu_sha256'])
        controller.case_folder=out/case_id;controller.case_folder.mkdir();write(controller.case_folder/'input.json',controller.details)
        controller.raw=np.zeros(hu.shape,bool);controller.body=np.zeros(hu.shape,bool)
        controller.known=np.zeros(len(hu),bool);controller.status.update(case_id=case_id,phase='idle',board_connected=True)
        base_update=controller._update;last=[-50];started=[time.perf_counter()]
        def progress(**fields):
            base_update(**fields);done=fields.get('completed',-1)
            if done-last[0]>=50 or done==len(hu):
                print('PHYSICAL_FULL',name,done,'/',len(hu),round(time.perf_counter()-started[0],1),'s',flush=True);last[0]=done
                write(out/'progress.json',dict(completed=done,total=len(hu),elapsed_s=time.perf_counter()-started[0]))
        controller._update=progress
        return controller

    if 'ldct'in selected:
        timings={};receipts={}
        for name,mixin,c in [('baseline',old_ldct.LDCTWorkspaceMixin,legacy),('optimized',new_ldct.LDCTWorkspaceMixin,ldct_client)]:
            c.denoise(original[175],'cnn')  # Warmup outside measured task, same overlay state.
            cls=type('PhysicalLDCT_'+name,(mixin,new_lung.Controller),{})
            h=make_controller(cls,c,original,'ldct_'+name,'original')
            h.series_id=h.status['case_id'];h.original_hu=original;h.original_details=copy.deepcopy(h.details)
            h.source_choices={'original':h.details['analysis_source']};h.analysis_source_id='original'
            h.ldct_identity=manifest['ldct'];h.ldct_groups={'measured':{}};h.ldct_running=True
            began=time.perf_counter();h._enhance(list(range(len(original))),'FPGA_LDCT','cnn','measured');elapsed=(time.perf_counter()-began)*1000
            if h.status['phase']!='done':raise RuntimeError(h.status)
            assert len(h.ldct_groups['measured'])==len(original)
            records=[h.ldct_groups['measured'][z]['metadata']for z in range(len(original))]
            for index,row in enumerate(records):
                assert row['input_sha256']==ldct_reference[index]['input_sha256']
                assert row['output_sha256']==ldct_reference[index]['output_sha256']
            timings[name]=elapsed;receipts[name]=dict(task_ms=elapsed,ldct_transport='original_v1_raw',
                roundtrip_mean_ms=float(np.mean([r['hardware_receipt']['roundtrip_ms']for r in records])),
                module_mean_ms=float(np.mean([r['processing_ms']for r in records])),slices_exact=len(records))
            write(folder/('ldct_'+name+'_result.json'),receipts[name]);print('FULL_LDCT_EXACT',name,round(elapsed/1000,3),'s',flush=True)
            del h
        all_results['ldct']=dict(baseline=receipts['baseline'],optimized=receipts['optimized'],
            latency_reduction_percent=100*(1-timings['optimized']/timings['baseline']),all_outputs_exact=True)
        write(folder/'ldct_result.json',all_results['ldct'])
    if 'segmentation'in selected:
        measurements={}
        for name,cls,c in [('baseline',old_lung.Controller,legacy),('optimized',new_lung.Controller,client)]:
            rid=reference_volume['slice_receipts']['175']
            with np.load(reference_case/rid/'actual_head.npz',allow_pickle=False)as z:c.infer(z['input_codes'])
            h=make_controller(cls,c,enhanced,'segmentation_'+name,state['analysis_source_id'])
            began=time.perf_counter();h._run(dict(mode='volume',index=0,_indices=list(range(len(enhanced)))))
            elapsed=(time.perf_counter()-began)*1000
            assert h.status['phase']=='done'
            meta=json.loads((h.case_folder/(h.status['volume_job_id']+'_volume.json')).read_text(encoding='utf8'))
            records=[]
            for index,rid in h.receipts.items():
                expected=reference_case/reference_volume['slice_receipts'][str(index)]/'actual_head.npz'
                with np.load(h.case_folder/rid/'actual_head.npz',allow_pickle=False)as new,np.load(expected,allow_pickle=False)as old:
                    for key in ('input_codes','head_codes'):np.testing.assert_array_equal(new[key],old[key])
                records.append(json.loads((h.case_folder/rid/'receipt.json').read_text(encoding='utf8')))
            for key in ('raw','body','cleaned'):np.testing.assert_array_equal(getattr(h,key),expected_masks[key])
            assert meta['volume_ml']==reference_volume['volume_ml']
            measurements[name]=dict(task_ms=elapsed,slices_exact=len(records),all_native_masks_exact=True,
                volume_ml=meta['volume_ml'],roundtrip_mean_ms=float(np.mean([r['roundtrip_ms']for r in records])),
                network_mean_ms=float(np.mean([r['stats']['host_total_ms']for r in records])),
                pl_active_mean_ms=float(np.mean([r['stats']['pl_command_cycles']/100000 for r in records])))
            write(folder/('segmentation_'+name+'_result.json'),measurements[name]);print('FULL_SEGMENTATION_EXACT',name,round(elapsed/1000,3),'s',flush=True)
            del h
        all_results['segmentation']=dict(**measurements,all_outputs_exact=True,
            latency_reduction_percent=100*(1-measurements['optimized']['task_ms']/measurements['baseline']['task_ms']))
        write(folder/'segmentation_result.json',all_results['segmentation'])
    if 'nodule'in selected:
        reference=reference_case/'nodule'/state['nodule']['job']['job_id']
        expected=json.loads((reference/'prediction.json').read_text(encoding='utf8'))
        with gzip.open(reference/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:ranked_expected=json.load(f)
        old_records=[json.loads(l)for l in(reference/'canvas_receipts.jsonl').open()]
        measurements={}
        for name,cls in [('baseline',old_nodule.NoduleCaseProvider),('optimized',new_nodule.NoduleCaseProvider)]:
            provider=cls(PACKAGE,client);dest=folder/('nodule_'+name);dest.mkdir(exist_ok=False)
            printed=[-400]
            def progress(stage,done,total):
                if done-printed[0]>=400 or(stage=='inference'and done==total):
                    print('PHYSICAL_FULL_NODULE',name,stage,done,'/',total,flush=True);printed[0]=done
                    write(dest/'progress.json',dict(stage=stage,completed=done,total=total))
            began=time.perf_counter();selected,provenance=provider.run(enhanced,expected_masks['body'],geometry,
                state['nodule']['context'],dest,progress,lambda:False);elapsed=(time.perf_counter()-began)*1000
            assert selected==expected['candidates']
            with gzip.open(dest/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:assert json.load(f)==ranked_expected
            actual_records=[json.loads(l)for l in(dest/'canvas_receipts.jsonl').open()]
            assert len(actual_records)==len(old_records)
            for a,b in zip(actual_records,old_records):
                assert a['jobs']==b['jobs']
                for key in ('input_sha256','output_sha256','output_shape'):assert a['receipt'][key]==b['receipt'][key]
            measurements[name]=dict(task_ms=elapsed,full_heads_exact=len(actual_records),
                ranked_candidates_exact=len(ranked_expected),selected_candidates_exact=len(selected),timing=provenance['timing'])
            write(dest/'result.json',measurements[name]);print('FULL_NODULE_EXACT',name,round(elapsed/1000,3),'s',flush=True)
        all_results['nodule']=dict(**measurements,all_outputs_exact=True,
            latency_reduction_percent=100*(1-measurements['optimized']['task_ms']/measurements['baseline']['task_ms']))
        write(folder/'nodule_result.json',all_results['nodule'])
    summarize(args.functions)


if __name__=='__main__':main()
