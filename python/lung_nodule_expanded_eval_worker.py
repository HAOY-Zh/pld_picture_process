"""Reverse-order companion for untouched calibration CTs, using frozen inference.

Commit a complete case directory by atomic rename ONLY if the main evaluator
has not created that directory. Never overwrite main-process partial results.
"""
import gc
import hashlib
import os
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import lung_nodule_expanded_evaluation as e


def commit_case(staging, destination):
    # On Windows, directory rename cannot replace an existing directory.
    # The main evaluator mkdir precedes its completion check, so it either
    # owns this case already or sees our whole immutable completed case.
    try:
        os.rename(staging, destination)
    except FileExistsError:
        return False
    return True


def run():
    out = e.OUT
    plan, manifest = e.verify(out)
    identity = e.sha_file(out/'plan.json')
    torch = e.setup_torch(plan['seed'])
    from lung_unet import make_model
    cfg = e.config_from_dict(plan['config'])
    nets = {}
    for name, info in plan['models'].items():
        cp = torch.load(info['path'], map_location='cpu', weights_only=True)
        if e.config_from_dict(cp['config']) != cfg:
            raise ValueError('Wrong model config')
        nets[name] = e.model(cfg).cuda().eval()
        nets[name].load_state_dict(cp['state'])
    cp = torch.load(e.LUNG, map_location='cpu', weights_only=True)
    lung = make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model'])
    ranker = e.read(e.Path(plan['ranker']))
    work = out/'worker_staging'
    work.mkdir(exist_ok=True)
    e.save(out/'worker_identity.json', dict(source_sha256=e.sha_file(e.Path(__file__)), plan_sha256=identity,
        policy='reverse untouched cases, identical paired_inference, atomic non-overwriting case directory commit'))
    with threadpool_limits(limits=1), torch.inference_mode():
        for r in reversed(e.calibration_records(manifest)):
            uid = r['series_uid']
            destination = out/'cases'/uid
            if destination.exists() or uid in plan['reused_baseline']:
                continue
            staging = work/uid
            staging.mkdir(exist_ok=True)
            if (staging/'complete.json').exists():
                e.verified_done(staging, identity)
                commit_case(staging, destination)
                continue
            tick = perf_counter()
            print('WORKER LOAD', uid, flush=True)
            hu, geometry = e.load_development_volume(manifest, uid, 'calibration')
            raw, body, _ = e.predict_lungs(hu, geometry, lung, 'cuda', 16)
            del raw
            body_sha = hashlib.sha256(body.tobytes()).hexdigest()
            result, schedule = e.paired_inference(hu, body, geometry, nets, cfg, torch)
            names = []
            for name in e.LABELS:
                candidates, features, evidence = result[name]
                scores = np.asarray([c['score'] for c in candidates])
                cross = e.rerank_scores(scores, features, ranker)
                packet = dict(series_uid=uid, plan_sha256=identity, model_sha256=plan['models'][name]['sha256'],
                    input_raw_sha256=r['raw_sha256'], lung_mask_sha256=body_sha,
                    source_shape_zyx=list(hu.shape), geometry=geometry, candidates=candidates,
                    evidence=evidence, baseline_output_reused=False, schedule=schedule,
                    backend='FP32_CUDA_NOT_FPGA', annotations_used_by_inference=False)
                a, b = name+'.json.gz', name+'.npz'
                e.save_packet(staging/a, packet)
                e.save_npz(staging/b, raw_scores=scores, cross_scores=cross, features=features)
                names.extend((a, b))
            e.save(staging/'complete.json', dict(series_uid=uid, plan_sha256=identity,
                artifacts_sha256={n:e.sha_file(staging/n) for n in names}, shared_seconds=perf_counter()-tick,
                worker_source_sha256=e.sha_file(e.Path(__file__))))
            committed = commit_case(staging, destination)
            print('WORKER DONE', uid, 'committed', committed, 'seconds', round(perf_counter()-tick, 2), flush=True)
            del hu, body, result, candidates, features, evidence, packet
            gc.collect()
    print('WORKER COMPLETE; main process will audit and summarize all47', flush=True)


if __name__ == '__main__':
    run()
