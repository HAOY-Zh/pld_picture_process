"""Standalone timing only, after all training/evaluation CUDA processes exit.

No accuracy evaluation, NMS, reference matching or threshold tuning. Four CTs
were selected by depth before training. Full selected-job pipelines replace the
unreliable 192-spaced-job extrapolation. Network timing uses sustained warmup,
CUDA events and balanced execution orders, with equal-architecture control.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'data/lung_nodule_context_20261020_v1'
NAMES = ('baseline5', 'near_far5', 'context7', 'context9')


def alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    h = kernel.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(h))


def control_ratio(resident):
    """Same architecture must have comparable sustained forward latency."""
    import statistics
    before, after = resident['baseline5'], resident['near_far5']
    if not before or not after or min(before+after) <= 0:
        raise ValueError('Invalid positive latency samples')
    ratio = statistics.median(after)/statistics.median(before)
    return dict(ratio=ratio, acceptable=.75 <= ratio <= 1.25,
                rule='same five-input architecture median ratio in [0.75,1.25]')


def warm(torch, nets, tensors, seconds=3.):
    tick = time.perf_counter()
    iterations = 0
    while time.perf_counter()-tick < seconds:
        for name in nets:
            for _ in range(8):
                nets[name](tensors[name])
        torch.cuda.synchronize()
        iterations += 1
    return iterations


def run(names=NAMES):
    import gc
    import hashlib
    import numpy as np
    from threadpoolctl import threadpool_limits
    import lung_nodule_context_comparison as e
    plan, manifest = e.verify(OUT)
    folder = OUT/'standalone_timing'
    folder.mkdir(exist_ok=True)
    main_pid = json.loads((OUT/'launch.json').read_text('utf-8-sig'))['pid']
    worker_pid = json.loads((OUT/'worker_supervisor_launch.json').read_text('utf-8-sig'))['pid']
    if alive(main_pid) or alive(worker_pid):
        raise RuntimeError('Training/evaluation processes still active; refuse timing')
    cfg = e.config_from_dict(plan['config'])
    records = {r['series_uid']:r for r in e.calibration_records(manifest)}
    active_variants = tuple(n for n in names if n != 'baseline5')
    contexts = {'baseline5':e.BASE_CONTEXT, **{n:e.VARIANTS[n] for n in active_variants}}
    model_paths = {'baseline5':e.TRAIN/'model.pt', **{n:OUT/n/'model.pt' for n in active_variants}}
    for name in active_variants:
        if e.read(OUT/name/'result_brief.json')['status'] != 'AUDITED_COMPLETE':
            raise ValueError('Model accuracy audit incomplete')
    hashes = {n:e.sha_file(p) for n, p in model_paths.items()}
    protocol = dict(plan_sha256=e.sha_file(OUT/'plan.json'), model_sha256=hashes,
        source_sha256=e.sha_file(Path(__file__)), timing_uids=plan['timing_uids'], active_models=list(names),
        tests_sha256=e.sha_file(ROOT/'python/tests/test_lung_nodule_context_timing.py'),
        input='original HU and the exact cached lung prediction; all lung grid jobs',
        pipeline='sampling + H2D/FP32 network/D2H + decode; no NMS, ranker, lung network, disk read or network transport',
        resident=f'batch48, 3-second sustained warmup, CUDA events, {2*len(names)} balanced-order rounds of 40 forwards',
        whole_ct_order='rotating order across four predeclared CTs', control_rule='same-architecture ratio 0.75 to 1.25',
        fpga_validated=False, accuracy_evaluation_repeated=False, sealed_test_scans_accessed=0)
    protocol_path = folder/'plan.json'
    if protocol_path.exists():
        if e.read(protocol_path) != protocol:
            raise ValueError('Changed standalone timing protocol')
    else:
        e.save(protocol_path, protocol)
    if (folder/'results.json').exists():
        print('Timing already complete; preserving it', flush=True)
        return
    torch = e.setup_torch(plan['seed'])
    nets = {}
    for name, path in model_paths.items():
        cp = torch.load(path, map_location='cpu', weights_only=True)
        if name == 'baseline5' and hashes[name] != plan['baseline_sha256']:
            raise ValueError('Changed baseline model')
        if name != 'baseline5' and hashes[name] != e.read(OUT/name/'model_freeze.json')['model_sha256']:
            raise ValueError('Changed context model')
        net = e.context_model(cfg, contexts[name]).cuda().float().eval()
        net.load_state_dict(cp['state'])
        if any(p.dtype != torch.float32 or not p.is_cuda for p in net.parameters()):
            raise ValueError('Non-FP32/CUDA parameter')
        nets[name] = net
    results = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for case_index, uid in enumerate(plan['timing_uids']):
            e.save(folder/'progress.json', dict(status='RUNNING', completed=len(results), expected=4,
                current_ct=case_index+1, stage='load'))
            case_path = folder/(uid+'.json')
            if case_path.exists():
                result = e.read(case_path)
                if result['protocol_sha256'] != e.sha_file(protocol_path):
                    raise ValueError('Wrong completed timing case')
                results.append(result)
                continue
            record = records[uid]
            hu, geometry = e.load_development_volume(manifest, uid, 'calibration')
            spacing = np.asarray(geometry['spacing_zyx_mm'])
            with np.load(OUT/'calibration_masks'/(uid+'.npz'), allow_pickle=False) as a:
                body = np.unpackbits(a['body'], count=int(np.prod(hu.shape))).reshape(hu.shape).astype(bool)
            previous = e.load_packet(e.CAL/'cases'/uid/'expanded247.json.gz')
            mask_sha = hashlib.sha256(body.tobytes()).hexdigest()
            if previous['lung_mask_sha256'] != mask_sha or previous['input_raw_sha256'] != record['raw_sha256']:
                raise ValueError('Timing input/mask mismatch')
            search = e.search_mask(body, spacing, cfg)
            jobs, schedule = e.tiles(search, spacing, cfg)
            if schedule != previous['schedule'] or len(jobs) < 48:
                raise ValueError('Wrong timing grid or insufficient batch')
            cached = e.FloatPlaneCache(hu)
            array = np.stack([e.sample_context(cached, spacing, j, cfg, e.ALL_CONTEXT) for j in jobs[:48]])
            tensors = {n:torch.from_numpy(array[:, [e.ALL_CONTEXT.index(c) for c in context]].copy()).cuda().float()
                       .clamp(-1000, 400).add_(1000).div_(1400) for n, context in contexts.items()}
            del cached, array
            warm(torch, nets, tensors)
            resident = {n:[] for n in names}
            for repeat in range(2*len(names)):
                shift = (repeat+case_index) % len(names)
                order = names[shift:]+names[:shift]
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(40):
                        nets[name](tensors[name])
                    end.record()
                    end.synchronize()
                    resident[name].append(start.elapsed_time(end)/1000/40/48)
            control = control_ratio(resident)
            pipelines = {}
            shift = case_index % len(names)
            order = names[shift:]+names[:shift]
            for name in order:
                e.save(folder/'progress.json', dict(status='RUNNING', completed=len(results), expected=4,
                    current_ct=case_index+1, stage='full_ct_pipeline', active_model=name))
                warm(torch, nets, tensors)
                tick = time.perf_counter()
                values, stages = e.forward_jobs(hu, geometry, search, jobs, nets[name], cfg, contexts[name], torch)
                elapsed = time.perf_counter()-tick
                pipelines[name] = dict(wall_seconds=elapsed, stages=stages)
                del values
                print('TIMING CT', case_index+1, '/4', name, 'jobs', len(jobs), 'seconds', round(elapsed, 3), flush=True)
            result = dict(series_uid=uid, protocol_sha256=e.sha_file(protocol_path),
                jobs=len(jobs), source_shape_zyx=list(hu.shape), pipelines=pipelines,
                resident_seconds_per_patch=resident, same_architecture_control=control)
            e.save(case_path, result)
            results.append(result)
            e.save(folder/'progress.json', dict(completed=len(results), expected=4))
            del hu, body, search, previous, tensors
            gc.collect()
            torch.cuda.empty_cache()
    summary = {}
    for name in active_variants:
        differences = [r['pipelines'][name]['wall_seconds']-r['pipelines']['baseline5']['wall_seconds'] for r in results]
        summary[name] = dict(full_ct_pipeline_extra_seconds=differences,
            mean_full_ct_pipeline_extra_seconds=float(np.mean(differences)),
            full_ct_pipeline_ratio=sum(r['pipelines'][name]['wall_seconds'] for r in results)/
                                   sum(r['pipelines']['baseline5']['wall_seconds'] for r in results),
            median_resident_forward_ratio=float(np.median([np.median(r['resident_seconds_per_patch'][name])/
                                  np.median(r['resident_seconds_per_patch']['baseline5']) for r in results])))
    report = dict(status='COMPLETE', protocol_sha256=e.sha_file(protocol_path), cases=results,
        summary=summary, same_architecture_controls_pass=all(r['same_architecture_control']['acceptable'] for r in results),
        backend=torch.cuda.get_device_name(), precision='FP32, TF32 off, no AMP',
        scope=protocol['pipeline'], accuracy_evaluation_repeated=False,
        fpga_validated=False, sealed_test_scans_accessed=0)
    e.save(folder/'results.json', report)
    e.save(folder/'progress.json', dict(status='COMPLETE', completed=4, expected=4))
    print('STANDALONE TIMING COMPLETE', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=NAMES[1:], default=list(NAMES[1:]))
    args = parser.parse_args()
    if 'near_far5' not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error('Include near_far5 control and no duplicate variants')
    names = ('baseline5', *args.variants)
    if args.wait:
        main_pid = json.loads((OUT/'launch.json').read_text('utf-8-sig'))['pid']
        worker_pid = json.loads((OUT/'worker_supervisor_launch.json').read_text('utf-8-sig'))['pid']
        for name in args.variants:
            while not (OUT/name/'result_brief.json').exists():
                if not alive(main_pid) and not (OUT/name/'evaluation.json').exists():
                    raise RuntimeError('Main task stopped before '+name)
                time.sleep(3)
        while alive(main_pid) or alive(worker_pid):
            time.sleep(3)
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--variants', *args.variants], cwd=str(ROOT),
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        run(names)


if __name__ == '__main__':
    main()
