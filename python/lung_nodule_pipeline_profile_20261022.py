"""Diagnostic timings on two already-exposed CTs; no model/threshold changes.

Wrap existing functions without changing their outputs. This is an instrumented
single-run CPU/CUDA profile, not a new blind test or an FPGA benchmark.
"""
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from lung_candidate_rules_run import ROOT
from lung_candidate_data import sha_file
from lung_nodule_cnn_balanced import read, save
from lung_nodule_sealed113_evaluation import OUT as TEST, verify, load_test_volume
from lung_nodule_heldout_eval import load_packet
from lung_nodule_cnn import model, config_from_dict
from lung_nodule_cnn_hardneg import setup_torch
from lung_candidate_run import predict_lungs
from lung_nodule_continuity import rerank_scores
import lung_nodule_expanded_evaluation as engine
import lung_body_gate
import lung_unet

OUT = ROOT/'data/lung_nodule_pipeline_profile_20261022_v1'


def run():
    if OUT.exists():
        raise FileExistsError('Preserve the completed diagnostic; do not repeat automatically')
    plan, manifest, records = verify(TEST)
    ordered = sorted(records, key=lambda r: read(TEST/'cases'/r['series_uid']/'complete.json')['seconds'])
    selected = [ordered[len(ordered)//2], ordered[3*len(ordered)//4]]
    OUT.mkdir()
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        test_plan_sha256=sha_file(TEST/'plan.json'), model_sha256=plan['model_sha256'],
        source_sha256=sha_file(Path(__file__)), selected_uids=[r['series_uid'] for r in selected],
        selection='50th and75th percentile of completed case wall time; no reference/accuracy selection',
        purpose='instrument already-exposed cases only; no training, tuning or additional accuracy evaluation',
        setup_and_model_warmup_excluded=True, disk_cache_state_uncontrolled=True,
        forward_measurement='CUDA events around model forward; device-only nested within wall times',
        fpga_validated=False)
    save(OUT/'plan.json', protocol)
    torch = setup_torch(plan['seed'])
    cfg = config_from_dict(plan['config'])
    cp = torch.load(plan['model'], map_location='cpu', weights_only=True)
    net = model(cfg).cuda().eval(); net.load_state_dict(cp['state']); del cp
    cp = torch.load(plan['lung_model'], map_location='cpu', weights_only=True)
    lung = lung_unet.make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model']); del cp
    ranker = read(Path(plan['ranker']))
    info = {}
    with torch.inference_mode():
        for name, module, shape in [('lung', lung, (1,1,256,256)), ('nodule', net, (1,5,128,128))]:
            macs = [0]
            def counter(layer, inputs, output):
                macs[0] += output.numel()*(layer.in_channels//layer.groups)*int(np.prod(layer.kernel_size))
            hooks = [m.register_forward_hook(counter) for m in module.modules() if isinstance(m,torch.nn.Conv2d)]
            module(torch.zeros(shape, device='cuda')); torch.cuda.synchronize()
            for h in hooks: h.remove()
            info[name] = dict(parameters=sum(p.numel() for p in module.parameters()),
                              input_shape=list(shape), conv_macs_per_input=macs[0])
        for _ in range(3):
            lung(torch.zeros((16,1,256,256),device='cuda'))
            net(torch.zeros((48,5,128,128),device='cuda'))
        torch.cuda.synchronize()
    timing = {}
    def wrap(owner, name, key):
        original = getattr(owner, name)
        def measured(*args, **kwargs):
            start = perf_counter()
            result = original(*args, **kwargs)
            timing[key] = timing.get(key, 0.)+perf_counter()-start
            return result
        setattr(owner, name, measured)
    wrap(lung_body_gate, 'gate_slice', 'lung_body_envelope_cpu_seconds')
    wrap(lung_unet, 'resize_stack', 'lung_resize_cpu_seconds')
    for name in ('search_mask','tiles','sample_patch_reuse','decode','consolidate','evidence_for_packet'):
        wrap(engine, name, name+'_seconds')
    events = []
    def track(module, name):
        original = module.forward
        def measured(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); result = original(*args, **kwargs); end.record()
            events.append((name, start, end))
            return result
        module.forward = measured
    track(lung, 'lung'); track(net, 'nodule')
    results = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(selected, 1):
            timing.clear(); events.clear(); uid = r['series_uid']
            save(OUT/'progress.json',dict(status='RUNNING',completed=len(results),expected=2,current_uid=uid))
            start = perf_counter(); hu, geometry = load_test_volume(manifest, uid, plan)
            read_seconds = perf_counter()-start
            start = perf_counter(); raw, body, _ = predict_lungs(hu, geometry, lung, 'cuda', 16); del raw
            lung_seconds = perf_counter()-start
            start = perf_counter(); output, schedule = engine.paired_inference(hu, body, geometry, {'expanded':net}, cfg, torch)
            nodule_seconds = perf_counter()-start
            candidates, features, evidence = output['expanded']
            start = perf_counter()
            scores = np.array([c['score'] for c in candidates]); cross = rerank_scores(scores, features, ranker)
            scoring_seconds = perf_counter()-start
            torch.cuda.synchronize()
            gpu = {name:sum(s.elapsed_time(e)/1000 for n,s,e in events if n==name) for name in ('lung','nodule')}
            previous = load_packet(TEST/'cases'/uid/'prediction.json.gz')
            with np.load(TEST/'cases'/uid/'scores.npz',allow_pickle=False) as a:
                scores_match = len(scores)==len(a['raw_scores']) and np.allclose(scores,a['raw_scores'],atol=1e-6,rtol=1e-5) and np.allclose(cross,a['cross_scores'],atol=1e-6,rtol=1e-5)
            centers_match = len(candidates)==len(previous['candidates']) and np.allclose(
                np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3),
                np.array([c['center_lps_xyz_mm'] for c in previous['candidates']]).reshape(-1,3),atol=1e-6,rtol=0)
            body_match = hashlib.sha256(body.tobytes()).hexdigest()==previous['lung_mask_sha256']
            if not (scores_match and centers_match and body_match):
                raise ValueError('Diagnostic output differs from the frozen test prediction')
            result = dict(series_uid=uid,shape_zyx=list(hu.shape),schedule=schedule,
                historical_complete_seconds=read(TEST/'cases'/uid/'complete.json')['seconds'],
                archive_read_hash_seconds=read_seconds,lung_total_seconds=lung_seconds,
                nodule_total_seconds=nodule_seconds,scoring_seconds=scoring_seconds,
                stage_wall_seconds=dict(timing),cuda_forward_seconds=gpu,
                gpu_forward_is_nested_not_additional=True,outputs_match_frozen_test=True,
                conv_macs_lung=info['lung']['conv_macs_per_input']*len(hu),
                conv_macs_nodule=info['nodule']['conv_macs_per_input']*schedule['lung_tiles'],
                excludes='model load/warmup, result serialization, web transport/rendering, mesh generation')
            results.append(result); save(OUT/(uid+'.json'),result)
            print('PROFILE',i,'/2',result,flush=True)
            del hu,body,output,candidates,features,evidence,previous; gc.collect()
    save(OUT/'results.json',dict(status='COMPLETE',models=info,cases=results,protocol=protocol))
    save(OUT/'progress.json',dict(status='COMPLETE',completed=2,expected=2))
    print('PROFILE COMPLETE',info,flush=True)


if __name__ == '__main__':
    run()
