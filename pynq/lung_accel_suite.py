"""Original/pruned FP32, INT8 ARM and fully optimized INT8 PL, same board.

Only run after the existing healthy service releases PL. No automatic reset,
clock rollback or recovery after a failed hardware operation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import sys
import types
from datetime import datetime, timezone

_owner_module=sys.modules.setdefault('_lung_accel_suite_owners',types.ModuleType('_lung_accel_suite_owners'))
if not hasattr(_owner_module,'retained'):_owner_module.retained=[]
_RETAINED_LOCKS=_owner_module.retained


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def relative_parts(name):
    """Manifest paths are portable; also read existing Windows-generated keys."""
    if not isinstance(name,str) or not name or '\x00' in name:
        raise ValueError('Invalid suite relative path')
    relative=PurePosixPath(name.replace('\\','/'))
    if (relative.is_absolute() or PureWindowsPath(name).drive or
            not relative.parts or '..' in relative.parts):
        raise ValueError('Invalid suite relative path: '+name)
    return relative.parts


def validate(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'suite_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='lung_all_acceleration_suite_v1':raise ValueError('Wrong suite')
    for name,h in manifest['files_sha256'].items():
        p=folder.joinpath(*relative_parts(name)).resolve()
        if folder not in p.parents or not p.is_file() or sha(p)!=h:
            raise ValueError('Changed suite file: '+name)
    if manifest['models']!=['original','pruned']:raise ValueError('Incomplete model comparison')
    for label in manifest['models']:
        p=folder/label
        fp=json.loads((p/'fp32_manifest.json').read_text(encoding='utf-8'))
        clock=json.loads((p/'clock_manifest.json').read_text(encoding='utf-8'))
        if (fp['model_label']!=label or fp['schema']!='lung_fp32_threeway_v2' or
                clock['routed_design']['bit_sha256']!=manifest['bit_sha256'] or
                sha(p/'lung_network.bit')!=manifest['bit_sha256']):
            raise ValueError('Different model label or hardware identity')
    return manifest


def release_web(service):
    """Call in the notebook that owns the healthy, idle platform_service."""
    if not service.gate.acquire(False):raise RuntimeError('A web request is active; wait for completion')
    try:
        if service.failed or service.hardware.fault:
            raise RuntimeError('Faulted hardware owner retained; no release or reprogramming')
        service.hardware.release_idle()
        service.server.shutdown();service.server.server_close();service.thread.join(timeout=2)
        service.hardware.lock_file.close()
    finally:service.gate.release()
    print('Healthy web service released PL. Restart this notebook kernel before running the suite.',flush=True)


def run(args):
    folder=Path(__file__).resolve().parent
    manifest=validate(folder)
    if platform.machine().lower() not in ('armv7','armv7l'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=5:raise ValueError('Invalid repeat counts')
    if _RETAINED_LOCKS:raise RuntimeError('Prior incomplete suite owner retained; no automatic retry')
    # The same lock used by the web service, held across both model phases.
    import fcntl
    owner=open('/tmp/lung_live_z2.lock','a')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        owner.close();raise RuntimeError('Another service owns PL; release the healthy current owner first')
    _RETAINED_LOCKS.append(owner)
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=folder/'results'/timestamp;output.mkdir(parents=True,exist_ok=False)
    results={}
    try:
        for label in manifest['models']:
            path=folder/label
            prior=list(sys.path);sys.path.insert(0,str(path))
            try:
                spec=importlib.util.spec_from_file_location('_lung_combo_'+label,path/'lung_threeway_combo.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                phase=argparse.Namespace(report=str(output/(label+'.json')),runs=args.runs,
                    warmup=args.warmup,old_runs=0,timeout=args.timeout)
                print('\nMODEL:',label,flush=True)
                results[label]=module.run(phase)
            finally:sys.path[:]=prior
        old,new=results['original'],results['pruned']
        gain={key:1-new[key]/old[key] for key in ['fp32_ms','cpu_int8_ms','fpga_int8_ms']}
        record=dict(status='completed_six_way_board_measurement_pending_independent_return_audit',
            models=results,pruning_latency_reduction=gain,bit_sha256=manifest['bit_sha256'],
            suite_manifest_sha256=sha(folder/'suite_manifest.json'),
            scope='Same validation slice; resident complete networks. Setup/bit loading, preprocessing, postprocessing and web round trip excluded.',
            files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
        write_new(output/'suite_result.json',record)
        print('\nALL SIX MEASUREMENTS COMPLETE',flush=True)
        print('model        ARM FP32 ms   ARM INT8 ms   FPGA INT8 ms',flush=True)
        for label,r in results.items():
            print('{:12s} {:11.3f} {:13.3f} {:14.3f}'.format(label,r['fp32_ms'],r['cpu_int8_ms'],r['fpga_int8_ms']),flush=True)
        print('Pruning PL latency reduction: {:.2f}%'.format(100*gain['fpga_int8_ms']),flush=True)
        print('RETURN THIS RESULTS FOLDER:',str(output),flush=True)
        _RETAINED_LOCKS.remove(owner);owner.close()
        return record
    except BaseException as exc:
        write_new(output/'suite_failure.json',dict(status='stopped_incomplete',error_type=type(exc).__name__,error=str(exc),
            completed_models=list(results),hardware_owner_retained=True))
        print('Suite stopped. Keep this kernel and partial evidence; no automatic hardware recovery.',flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=int,default=20);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--timeout',type=float,default=30.)
    run(p.parse_args())
