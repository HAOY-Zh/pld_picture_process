"""Reverse small-CT companion. No training; publish complete cases without overwrite.

Pause before every GPU batch and expensive stage while the main evaluator owns
any predeclared timing CT. The main process computes those timing CTs itself.
"""
import ctypes
import gc
import hashlib
import os
from pathlib import Path
import time
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

import lung_nodule_context_comparison as e
from lung_nodule_expanded_eval_worker import commit_case


def running(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def active_timing(out, variant, plan):
    return any((out/variant/'cases'/u).exists() and
               not (out/variant/'cases'/u/'complete.json').exists() for u in plan['timing_uids'])


def pause(out, variant, plan, main_pid):
    while active_timing(out, variant, plan):
        if not running(main_pid):
            raise RuntimeError('Main evaluator stopped; companion stops too')
        time.sleep(1)


def wait_after_main_stage(out, variant, plan, main_pid):
    if not running(main_pid):
        raise RuntimeError('Main task stopped')
    pause(out, variant, plan, main_pid)


def evaluate(out, variant, main_pid):
    plan, manifest = e.verify(out)
    folder = out/variant
    frozen = e.read(folder/'model_freeze.json')
    if frozen['model_sha256'] != e.sha_file(folder/'model.pt') or frozen['optimizer_updates'] != 29070:
        raise ValueError('Wrong completed variant model')
    identity = e.sha_file(out/'plan.json')
    torch = e.setup_torch(plan['seed'])
    cfg, context = e.config_from_dict(plan['config']), tuple(plan['variants'][variant])
    cp = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if tuple(cp['input_context_mm']) != context or set(cp['training_scans']) != set(plan['training_uids']):
        raise ValueError('Context/training mismatch')
    net = e.context_model(cfg, context).cuda().eval()
    net.load_state_dict(cp['state'])
    from lung_unet import make_model
    import lung_nodule_cnn_experiment as first
    cp = torch.load(e.LUNG, map_location='cpu', weights_only=True)
    lung = make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model'])
    split, ranker = e.read_split(e.SPLIT), e.read(e.RANKER)
    work = out/'worker_staging'/variant
    work.mkdir(parents=True, exist_ok=True)
    masks = out/'calibration_masks'
    masks.mkdir(exist_ok=True)
    (folder/'cases').mkdir(exist_ok=True)
    worker_sha = e.sha_file(Path(__file__))

    class PausingNet:
        def __call__(self, x):
            wait_after_main_stage(out, variant, plan, main_pid)
            return net(x)

    completed = 0
    with threadpool_limits(limits=1), torch.inference_mode():
        for r in reversed(e.calibration_records(manifest)):
            uid = r['series_uid']
            destination = folder/'cases'/uid
            if uid in plan['timing_uids'] or r['shape_zyx'][0] > 400 or destination.exists():
                continue
            wait_after_main_stage(out, variant, plan, main_pid)
            # Recheck after a timing pause; main may have reached this case.
            if destination.exists():
                continue
            staging = work/uid
            staging.mkdir(exist_ok=True)
            tick = perf_counter()
            baseline_dir = e.CAL/'cases'/uid
            e.verified_done(baseline_dir, e.sha_file(e.CAL/'plan.json'))
            old = e.load_packet(baseline_dir/'expanded247.json.gz')
            hu, geometry = e.load_development_volume(manifest, uid, 'calibration')
            load_seconds = perf_counter()-tick
            if old['input_raw_sha256'] != r['raw_sha256'] or old['geometry'] != geometry:
                raise ValueError('Wrong baseline pixels/geometry')
            wait_after_main_stage(out, variant, plan, main_pid)
            mask_path = masks/(uid+'.npz')
            if mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as a:
                    body = np.unpackbits(a['body'], count=int(np.prod(hu.shape))).reshape(hu.shape).astype(bool)
            else:
                if r['origin_cohort'] == 'original135':
                    body = first.load_mask(r, split)
                else:
                    raw, body, _ = e.predict_lungs(hu, geometry, lung, 'cuda', 16)
                    del raw
                # Unique temporary filename prevents collisions with main's writer.
                mask_temp = staging/'mask_publish.npz'
                e.save_npz(mask_temp, body=np.packbits(body), shape_zyx=np.asarray(hu.shape))
                try:
                    os.rename(mask_temp, mask_path)
                except FileExistsError:
                    pass
            mask_sha = hashlib.sha256(body.tobytes()).hexdigest()
            if mask_sha != old['lung_mask_sha256']:
                raise ValueError('Companion lung mask differs from baseline')
            spacing = np.asarray(geometry['spacing_zyx_mm'])
            wait_after_main_stage(out, variant, plan, main_pid)
            stage_tick = perf_counter()
            search = e.search_mask(body, spacing, cfg)
            jobs, schedule = e.tiles(search, spacing, cfg)
            grid_seconds = perf_counter()-stage_tick
            if schedule != old['schedule']:
                raise ValueError('Different search schedule')
            values, seconds = e.forward_jobs(hu, geometry, search, jobs, PausingNet(), cfg, context, torch)
            wait_after_main_stage(out, variant, plan, main_pid)
            stage_tick = perf_counter()
            candidates = e.consolidate(*values, geometry, cfg) if jobs else []
            for c in candidates:
                box = np.asarray(c['bbox_zyx']).reshape(3, 2)
                box[:, 0] = np.maximum(box[:, 0], 0)
                box[:, 1] = np.minimum(box[:, 1], hu.shape)
                c['bbox_zyx'] = box.ravel().tolist()
            seconds['consolidate'] = perf_counter()-stage_tick
            wait_after_main_stage(out, variant, plan, main_pid)
            stage_tick = perf_counter()
            features, evidence = e.evidence_for_packet(hu, spacing, candidates)
            raw_scores = np.asarray([c['score'] for c in candidates])
            cross_scores = e.rerank_scores(raw_scores, features, ranker)
            seconds['evidence_and_rerank'] = perf_counter()-stage_tick
            packet = dict(series_uid=uid, variant=variant, plan_sha256=identity,
                model_sha256=frozen['model_sha256'], input_raw_sha256=r['raw_sha256'],
                lung_mask_sha256=mask_sha, source_shape_zyx=list(hu.shape), geometry=geometry,
                candidates=candidates, evidence=evidence, schedule=schedule, input_context_mm=context,
                annotations_used_by_inference=False, backend='FP32_CUDA_NOT_FPGA',
                seconds=seconds, load_seconds=load_seconds, grid_seconds=grid_seconds,
                timing_probe=None, companion_source_sha256=worker_sha)
            e.save_packet(staging/'prediction.json.gz', packet)
            e.save_npz(staging/'scores.npz', raw_scores=raw_scores, cross_scores=cross_scores, features=features)
            e.save(staging/'complete.json', dict(plan_sha256=identity, series_uid=uid,
                artifacts_sha256={n:e.sha_file(staging/n) for n in ('prediction.json.gz', 'scores.npz')},
                seconds=perf_counter()-tick, worker_source_sha256=worker_sha))
            committed = commit_case(staging, destination)
            completed += int(committed)
            e.save(out/'worker_progress.json', dict(variant=variant, committed=completed, last_uid=uid,
                sealed_test_scans_accessed=0))
            print('WORKER', variant, 'committed', committed, 'seconds', round(perf_counter()-tick, 2), flush=True)
            del hu, body, search, values, candidates, features, evidence, old, packet
            gc.collect()
    del net, lung
    gc.collect()
    torch.cuda.empty_cache()


def main():
    out = e.OUT
    import json
    launch = json.loads((out/'launch.json').read_text('utf-8-sig'))
    main_pid = launch['pid']
    identity_path = out/'worker_identity.json'
    source_sha = e.sha_file(Path(__file__))
    identity = dict(source_sha256=source_sha, plan_sha256=e.sha_file(out/'plan.json'),
        extra_sources_sha256={str(p):e.sha_file(p) for p in
            (e.ROOT/'python/lung_nodule_expanded_eval_worker.py',
             e.ROOT/'python/tests/test_lung_nodule_context_worker.py')},
        scope='calibration only; depth <=400; skip all timing CTs; pause before GPU batches/stages during main timing CT',
        inference='identical frozen forward_jobs, decode, consolidate, evidence, rerank',
        publish='atomic directory rename, never overwrite main cases', sealed_test_scans_accessed=0)
    if identity_path.exists() and e.read(identity_path) != identity:
        raise ValueError('Changed companion identity')
    e.save(identity_path, identity)
    for variant in e.VARIANTS:
        while not (out/variant/'model_freeze.json').exists():
            if not running(main_pid):
                print('MAIN STOPPED; worker exits', flush=True)
                return
            time.sleep(3)
        if not (out/variant/'evaluation.json').exists():
            evaluate(out, variant, main_pid)
    print('WORKER ALL VARIANTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
