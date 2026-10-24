"""Audit completed real-case runs, preserving all original outputs and receipts."""
import argparse,hashlib,itertools,json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def compare(before,after,out,roi_before=None,roi_after=None):
    a,b=read(before/'prediction.json'),read(after/'prediction.json')
    ja,jb=read(before/'job.json'),read(after/'job.json')
    assert ja['status']==jb['status']=='COMPLETE'
    for key in ('case_id','analysis_source_id','context_id','source_hu_sha256','geometry_sha256','segmentation_signature','candidate_count','candidates'):
        assert a[key]==b[key],'Changed detection result: '+key
    for key in ('shared_nodule','detector_config','sampling_decode_code_sha256','scoring_mode','lung_mask_sha256','lung_mask_stage',
                'raw_decode_floor','additional_score_filter','ranker_applied','candidate_order','schedule'):
        assert a['detector_provenance'][key]==b['detector_provenance'][key],'Changed numerical pipeline: '+key
    count=0;switched=[0.,0.];requests=[set(),set()]
    with (before/'patch_receipts.jsonl').open(encoding='utf8')as fa,(after/'patch_receipts.jsonl').open(encoding='utf8')as fb:
        for left,right in itertools.zip_longest(fa,fb):
            assert left is not None and right is not None,'Different patch count'
            x,y=json.loads(left),json.loads(right)
            assert x['index']==y['index']==count and x['job']==y['job'],'Changed patch order or sampling job'
            for key in ('input_sha256','output_sha256','shared_nodule','input_shape','output_shape'):
                assert x['receipt'][key]==y['receipt'][key],'Changed patch '+str(count)+': '+key
            for side,row in enumerate((x,y)):
                r=row['receipt'];s=r['stats']
                assert r['state']==dict(busy=False,faulted=False,dma_may_access=False)
                assert s['command_count']==len(s['commands'])==15 and s['buffer_allocations_per_frame']==0
                assert s['overlay_reloaded']is False
                assert abs(s['actual_pl_mhz_before']-100)<.5 and abs(s['actual_pl_mhz_after']-100)<.5
                switched[side]+=s['overlay_switch_ms']
                requests[side].add(r.get('batch_id',r['request_id']))
            count+=1
    assert count==ja['total']==jb['total']==ja['completed']==jb['completed']
    result=dict(status='PASS_REAL_CASE_SPEED_AND_EXACT_PARITY',case_id=a['case_id'],analysis_source_id=a['analysis_source_id'],
        source_hu_sha256=a['source_hu_sha256'],geometry_sha256=a['geometry_sha256'],patch_count=count,
        all_patch_inputs_bit_exact=True,all_patch_heads_bit_exact=True,all_candidates_exact=True,
        candidate_count=a['candidate_count'],candidate_sha256=fingerprint(a['candidates']),
        model_weights_and_bits_unchanged=True,scoring_and_sampling_unchanged=True,
        before_job=ja['job_id'],after_job=jb['job_id'],before_seconds=ja['elapsed_ms']/1000,after_seconds=jb['elapsed_ms']/1000,
        speedup=ja['elapsed_ms']/jb['elapsed_ms'],reduction_percent=100*(1-jb['elapsed_ms']/ja['elapsed_ms']),
        before_requests=len(requests[0]),after_requests=len(requests[1]),before_switch_ms=switched[0],after_switch_ms=switched[1],
        before_timing=a['detector_provenance']['timing'],after_timing=b['detector_provenance']['timing'],
        detection_quality_accepted=False,scope='Same single real case; actual FPGA execution in both runs, no timing replay or cached neural outputs.')
    if roi_before is not None:
        ra,rb=read(roi_before),read(roi_after)
        for key in ('valid_count','hu_statistics','z_bounds','roi_yx','processed_indices','source_hu_sha256','shape','geometry',
                    'use_hu','hu_range','use_lung_mask','processed_slices','complete','status'):
            assert ra[key]==rb[key],'Changed ROI: '+key
        assert ra['backend']==rb['backend']=='FPGA_ROI'
        assert ra['candidate_reference']['candidate_id']==rb['candidate_reference']['candidate_id']
        assert ra['candidate_reference']['original_bbox_zyx']==rb['candidate_reference']['original_bbox_zyx']
        result.update(auto_roi_exact=True,roi_slices=rb['processed_slices'],roi_statistics=rb['hu_statistics'])
    with out.open('x',encoding='utf8')as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k]for k in ('status','patch_count','candidate_count','before_seconds','after_seconds','speedup','reduction_percent')},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--roi-before',type=Path);p.add_argument('--roi-after',type=Path)
    args=p.parse_args();compare(args.before,args.after,args.out,args.roi_before,args.roi_after)
