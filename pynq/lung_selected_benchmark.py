"""Pin a frozen benchmark to THIS package even in a notebook with cached imports.

Never clear driver modules or their retained-CMA owners. Only the two file-rooted
entrypoints are loaded afresh, from verified explicit paths. All cached helper
code must match this package. Verify the requested bit BEFORE calling Overlay.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)


def validate_selection(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'selection_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='lung_pinned_entrypoint_v1': raise ValueError('Wrong selection manifest')
    for name,digest in manifest['files_sha256'].items():
        p=(folder/name).resolve()
        if p.parent!=folder or sha(p)!=digest: raise ValueError('Changed selected-package file: '+name)
    if set(manifest['files_sha256'])!={'lung_selected_benchmark.py','fast_manifest.json','clock_manifest.json',
                                      'lung_network.bit','lung_network.hwh'}:
        raise ValueError('Incomplete selection manifest')
    fast=json.loads((folder/'fast_manifest.json').read_text(encoding='utf-8'))
    for name,digest in fast['files_sha256'].items():
        p=(folder/name).resolve()
        if p.parent!=folder or sha(p)!=digest: raise ValueError('Changed runtime/fixture: '+name)
    clocks=json.loads((folder/'clock_manifest.json').read_text(encoding='utf-8'))
    for name,digest in clocks['files_sha256'].items():
        p=(folder/name).resolve()
        if p.parent!=folder or sha(p)!=digest: raise ValueError('Changed clock entrypoint')
    return manifest,fast


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,str(Path(path).resolve()))
    if spec is None or spec.loader is None: raise ImportError('Cannot load selected benchmark file')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def selected_entrypoints(folder,fast):
    """Preserve ALL existing module owners, including poisoned hardware drivers."""
    folder=Path(folder).resolve(); origins=[]
    rooted={'lung_fast_benchmark','lung_clock_benchmark'}
    for filename,digest in fast['files_sha256'].items():
        if not filename.endswith('.py') or filename[:-3] in rooted: continue
        name=filename[:-3]; module=sys.modules.get(name)
        if module is not None:
            source=getattr(module,'__file__',None)
            if source is None or sha(source)!=digest:
                raise RuntimeError('Different cached helper version: '+name+'; stop and use a fresh kernel only after safe hardware recovery')
            # These are unchanged injected-folder drivers/operators. Do NOT
            # delete them: their module globals may own timed-out CMA buffers.
            origins.append(dict(module=name,path=str(Path(source).resolve()),same_source_sha256=True))
    prior_path=list(sys.path)
    sys.path.insert(0,str(folder))
    try:
        bench=load_file('_lung_selected_benchmark',folder/'lung_fast_benchmark.py')
        # Restore ONLY this entrypoint alias. patch.dict(sys.modules) restores
        # the entire mapping, which could discard newly imported driver modules
        # owning retained CMA after a hardware timeout.
        missing=object(); previous=sys.modules.get('lung_fast_benchmark',missing)
        sys.modules['lung_fast_benchmark']=bench
        try:
            clocks=load_file('_lung_selected_clock',folder/'lung_clock_benchmark.py')
            if clocks.benchmark is not bench: raise AssertionError('Clock entry bound to wrong benchmark')
            if Path(bench.run.__globals__['__file__']).resolve()!=folder/'lung_fast_benchmark.py':
                raise AssertionError('Benchmark still has the wrong package root')
            yield bench,clocks,origins
        finally:
            if previous is missing: sys.modules.pop('lung_fast_benchmark',None)
            else: sys.modules['lung_fast_benchmark']=previous
    finally:
        sys.path[:]=prior_path


def overlay_guard(constructor,folder,expected,loads):
    folder=Path(folder).resolve(); bit=folder/'lung_network.bit';hwh=bit.with_suffix('.hwh')
    def guarded(*args,**kwargs):
        if loads: raise RuntimeError('Only one selected bit download allowed')
        for name,owners in [('lung_fast','_UNSAFE'),('lung_continuous','_UNSAFE_RUNS')]:
            module=sys.modules.get(name)
            if module is not None and getattr(module,owners,[]):
                raise RuntimeError('Retained uncertain hardware owners: '+name+'; shared board recovery required, no bit download')
        requested=args[0] if args else kwargs.get('bitfile_name')
        if requested is None or Path(requested).resolve()!=bit:
            raise RuntimeError('Refusing wrong bit path BEFORE download: '+str(requested))
        if kwargs.get('download',True) is not True or kwargs.get('partial',False):
            raise RuntimeError('Need a full selected overlay download')
        if sha(bit)!=expected['lung_network.bit'] or sha(hwh)!=expected['lung_network.hwh']:
            raise ValueError('Selected bit/HWH changed before download')
        print('VERIFIED bit path: '+str(bit),flush=True)
        print('VERIFIED bit SHA256: '+expected['lung_network.bit'],flush=True)
        overlay=constructor(*args,**kwargs)
        actual=Path(overlay.bitfile_name).resolve()
        if actual!=bit: raise RuntimeError('Overlay returned a different bit path; no inference started')
        loads.append(dict(requested_bit=str(bit),overlay_bitfile_name=str(actual),
                          bit_sha256=sha(bit),hwh_sha256=sha(hwh)))
        return overlay
    return guarded


def run(args):
    folder=Path(__file__).resolve().parent
    manifest,fast=validate_selection(folder)
    path=Path(args.report).resolve();selection_path=path.with_name(path.stem+'_selection.json')
    if any(p.exists() for p in [selection_path,path,path.with_name(path.stem+'_clock.json'),
            path.with_name(path.stem+'_cpu.json'),path.with_name(path.stem+'_diagnostic.json'),path.with_name(path.stem+'_head.npz')]):
        raise FileExistsError('Use fresh report names, preserving the suspect and partial evidence')
    import pynq
    loads=[];evidence=dict(selected_folder=str(folder),manifest_sha256=sha(folder/'selection_manifest.json'),
        expected_bit_sha256=manifest['files_sha256']['lung_network.bit'],
        expected_fast_manifest_sha256=manifest['files_sha256']['fast_manifest.json'])
    try:
        with selected_entrypoints(folder,fast) as (bench,clocks,origins):
            evidence.update(benchmark_file=str(Path(bench.__file__).resolve()),
                            clock_entry_file=str(Path(clocks.__file__).resolve()),cached_identical_helpers=origins)
            print('PINNED benchmark: '+evidence['benchmark_file'],flush=True)
            guarded=overlay_guard(pynq.Overlay,folder,manifest['files_sha256'],loads)
            with patch.object(pynq,'Overlay',guarded):
                summary=clocks.run(args)
        result=json.loads(path.read_text(encoding='utf-8'))
        if (len(loads)!=1 or result['bit_sha256']!=evidence['expected_bit_sha256']
                or result['package_sha256']!=evidence['expected_fast_manifest_sha256']):
            raise AssertionError('Result identity differs from selected bit/package; no speedup accepted')
        record=dict(status='passed_pinned_real_board_benchmark',**evidence,loads=loads,
            benchmark_sha256=sha(path),clock_evidence_sha256=sha(path.with_name(path.stem+'_clock.json')),
            optimized_pl_mean_ms=summary['optimized_pl_mean_ms'],cpu_over_pl=summary['cpu_over_pl'])
        write_new(selection_path,record)
        print('Selected bit/package identity: PASS. Evidence: '+str(selection_path),flush=True)
        return record
    except BaseException as exc:
        if not selection_path.exists():
            write_new(selection_path,dict(status='failed_pinned_benchmark',**evidence,loads=loads,
                                        error_type=type(exc).__name__,error=str(exc)))
        # Deliberately no module-cache clearing, subprocess exit, buffer free,
        # reset, reload, or clock rollback as a response to uncertain AXI state.
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_word_pinned_result.json')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--old-runs',type=int,choices=[0],default=0)
    parser.add_argument('--timeout',type=float,default=30.)
    run(parser.parse_args())
