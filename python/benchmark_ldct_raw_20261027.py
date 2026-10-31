"""Physical LDCT candidate AB/BA comparison; no deployment, no replay.

Requires the independently staged matching raw candidate owner to be ready.
Original whole-case outputs are checked against immutable saved actual results.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_RawLDCT_20261027_05'
sys.path.insert(0,str(APP/'python'))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    from start_ct_studio_demo import configuration,read_web_info
    from ct_dual_io_client import DualSegStudioClient
    from lung_live_data import read_ct
    from lung_live_web import Controller
    from ldct_workspace import LDCTWorkspaceMixin
    from lung_unet import torch_module
    torch_module().set_num_threads(2)
    web=read_web_info('http://127.0.0.1:8099/')
    if web is not None and (web[2]['phase']in('loading','computing','processing','connecting','reconstructing')or web[2].get('ldct',{}).get('running')):
        raise RuntimeError('Current web task active; no benchmark')
    package=APP/'data/ct_studio_integration_20261025';client,manifest=configuration(package/'host_config.json');client.health()
    baseline=DualSegStudioClient(client.url,client.token,manifest);baseline.health()
    health=client.call('/v1/shared/platform',timeout=10)
    if not health.get('ready')or health.get('busy')or health.get('error'):raise RuntimeError('Owner not healthy/idle')
    state=json.loads((ROOT/'data/current_effect_audit_20261026/state.json').read_text(encoding='utf8'))
    saved=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'/state['series_id']
    source=json.loads((saved/'input.json').read_text(encoding='utf8'))
    hu,geom,details=read_ct(source['source'],confirm_hu=True,series_uid=source.get('series_uid',''),allow_inconsistent_frame_uids=source.get('frame_uid_exception_applied')is True)
    if digest(hu)!=source['hu_sha256']:raise ValueError('Reference CT changed')
    reference={r['index']:r for r in state['ldct']['results']}
    output=ROOT/'data/threshold_ldct_iteration_20261027'/('physical_full'if args.full else'physical_pilot')
    output.mkdir(exist_ok=False);rows=[]
    write(output/'protocol.json',dict(service_bundle_sha256=manifest['service_bundle_sha256'],input_sha256=digest(hu),
        transport_only=True,clock_model_bit_unchanged=True,all_inference_serial=True,expert_accuracy_re_evaluated=False))
    if not args.full:
        for rep in range(3):
            for name,c in ([('baseline',baseline),('candidate',client)]if rep%2==0 else[('candidate',client),('baseline',baseline)]):
                # Same overlay warmup excluded from both arms; never retry POSTs.
                c.denoise(hu[175],'cnn');samples=[];module=[]
                for index in range(35,len(hu),35):
                    actual,receipt=c.denoise(hu[index],'cnn')
                    if digest(actual)!=reference[index]['output_sha256']:raise AssertionError('Actual HU differs')
                    samples.append(receipt['roundtrip_ms']);module.append(receipt['stats']['total_ms'])
                row=dict(repetition=rep,mode=name,roundtrip_mean_ms=float(np.mean(samples)),module_mean_ms=float(np.mean(module)),samples_ms=samples,all_outputs_exact=True)
                rows.append(row);write(output/'progress.json',rows);print(name,rep,round(row['roundtrip_mean_ms'],3),'ms, ALL_EXACT',flush=True)
        a=np.mean([r['roundtrip_mean_ms']for r in rows if r['mode']=='baseline']);b=np.mean([r['roundtrip_mean_ms']for r in rows if r['mode']=='candidate'])
        write(output/'result.json',dict(status='PHYSICAL_PAIRED_OUTPUTS_EXACT',baseline_mean_ms=a,candidate_mean_ms=b,latency_reduction_percent=100*(1-b/a),rows=rows))
        return
    class Task(LDCTWorkspaceMixin,Controller):pass
    timings={}
    for name,c in [('baseline',baseline),('candidate',client)]:
        c.denoise(hu[175],'cnn');out=output/name;out.mkdir()
        h=Task(c,json.loads((package/'model.json').read_text()),out)
        h.hu=h.original_hu=hu;h.geom=geom;h.details=copy.deepcopy(details);h.original_details=copy.deepcopy(details)
        key=uuid.uuid4().hex;h.series_id=key;h.status.update(case_id=key,phase='idle',board_connected=True)
        h.case_folder=out/key;h.case_folder.mkdir();h.details['analysis_source']=dict(source_id='original',case_id=key,hu_sha256=digest(hu))
        h.source_choices={'original':h.details['analysis_source']};h.ldct_identity=manifest['ldct'];h.ldct_groups={'measured':{}};h.ldct_running=True
        original_update=h._update
        def progress(**values):
            original_update(**values)
            if values.get('completed',-1)%50==0 or values.get('completed')==len(hu):
                print(name,values.get('completed'),flush=True)
        h._update=progress
        began=time.perf_counter();h._enhance(list(range(len(hu))),'FPGA_LDCT','cnn','measured');elapsed=(time.perf_counter()-began)*1000
        if h.status['phase']!='done':raise RuntimeError(h.status)
        records=[h.ldct_groups['measured'][i]['metadata']for i in range(len(hu))]
        for i,row in enumerate(records):
            if row['output_sha256']!=reference[i]['output_sha256']or row['input_sha256']!=reference[i]['input_sha256']:raise AssertionError('Complete case differs')
        timings[name]=dict(task_ms=elapsed,slices_exact=len(records),roundtrip_mean_ms=float(np.mean([r['hardware_receipt']['roundtrip_ms']for r in records])))
        write(output/(name+'_result.json'),timings[name]);print(name,round(elapsed/1000,3),'s ALL_EXACT',flush=True)
        del h
    write(output/'result.json',dict(status='PHYSICAL_FULL_CASE_EXACT',**timings,
        latency_reduction_percent=100*(1-timings['candidate']['task_ms']/timings['baseline']['task_ms']),
        browser_drawing_and_first_overlay_switch_included=False))


if __name__=='__main__':main()
