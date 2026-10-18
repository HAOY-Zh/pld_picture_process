"""Run the frozen prepared-runtime benchmark at its routed 100 MHz, not HWH divisors.

The old benchmark and all old artifacts stay byte-identical. A scoped Overlay
factory sets ONLY FCLK0 after download, while both known engines are idle. A
scoped runtime subclass reads clocks before/after each complete forward. No
PLL/CPU/DDR setters, no cancellation/reset, no frequency above routed timing.
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

from lung_axil import LungAxil
from lung_tensor import TensorAxil
from lung_model_layers_test import sha
import lung_fast_benchmark as benchmark

TARGET_MHZ=100.0


def state(clocks):
    values=dict(fclk0_mhz=float(clocks.fclk0_mhz),cpu_mhz=float(clocks.cpu_mhz))
    if any(not math.isfinite(v) or v<=0 for v in values.values()):
        raise RuntimeError('Invalid clock readback')
    return values


def require_target(values,cpu_mhz):
    if abs(values['fclk0_mhz']-TARGET_MHZ)>0.001:
        raise RuntimeError('FCLK0 is not the routed 100 MHz; stop, do not trust cycle/second conversion')
    if abs(values['cpu_mhz']-cpu_mhz)>0.05:
        raise RuntimeError('CPU frequency changed; comparison invalid')


def clock_overlay(overlay,clocks,cpu_mhz,trace,timeout):
    # Read the actual hardware identity/status, not merely the metadata names.
    conv=LungAxil(overlay.lung_conv8_axil_0.mmio,timeout=timeout)
    tensor=TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=timeout)
    conv._idle();tensor._idle()
    before=state(clocks);trace.append(dict(stage='overlay_loaded_before_clock',**before))
    if abs(before['cpu_mhz']-cpu_mhz)>0.05: raise RuntimeError('CPU changed before clock calibration')
    # PYNQ chooses divisors from the CURRENT source PLL, not the Vivado source
    # PLL assumption. Setter runs BEFORE any allocations or compute STARTs.
    if abs(before['fclk0_mhz']-TARGET_MHZ)>0.001: clocks.fclk0_mhz=TARGET_MHZ
    after=state(clocks);trace.append(dict(stage='overlay_loaded_after_clock',**after))
    require_target(after,cpu_mhz)
    conv._idle();tensor._idle()
    print('Clock correction after bit download: {:.3f} -> {:.3f} MHz; CPU unchanged {:.3f} MHz'.format(
        before['fclk0_mhz'],after['fclk0_mhz'],after['cpu_mhz']),flush=True)
    return overlay


def checked_network(base,clocks,cpu_mhz,trace):
    class ClockCheckedUNet(base):
        def predict_codes(self,*args,**kwargs):
            before=state(clocks);trace.append(dict(stage='before_forward',**before))
            require_target(before,cpu_mhz)
            output,row=super().predict_codes(*args,**kwargs)
            after=state(clocks);trace.append(dict(stage='after_forward',**after))
            require_target(after,cpu_mhz)
            row=dict(row,clock_before=before,clock_after=after,
                     engine_active_ms_actual_clock=row['pl_command_cycles']/(after['fclk0_mhz']*1000))
            return output,row
    return ClockCheckedUNet


def validate_addon(folder):
    folder=Path(folder).resolve()
    _,_,_,_=benchmark.validate_package(folder)
    path=folder/'clock_manifest.json'; manifest=json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('schema')!='lung_routed_clock_benchmark_v1'
            or manifest['target_clock_mhz']!=TARGET_MHZ
            or manifest['original_fast_manifest_sha256']!=sha(folder/'fast_manifest.json')):
        raise ValueError('Wrong clock addon or changed original package')
    for name,digest in manifest['files_sha256'].items():
        p=(folder/name).resolve()
        if p.parent!=folder or sha(p)!=digest: raise ValueError('Changed clock addon: '+name)
    proof=manifest['routed_design']
    if (proof['clock_mhz']!=TARGET_MHZ or proof['wns_ns']<0 or proof['whs_ns']<0
            or proof['bit_sha256']!=sha(folder/'lung_network.bit')
            or proof['hwh_sha256']!=sha(folder/'lung_network.hwh')):
        raise ValueError('Need exact routed 100MHz bit/HWH proof')
    return manifest


def run(args):
    folder=Path(__file__).resolve().parent
    manifest=validate_addon(folder)
    report=Path(args.report).resolve(); clock_path=report.with_name(report.stem+'_clock.json')
    if clock_path.exists(): raise FileExistsError('Preserve earlier clock evidence; use a fresh report name')
    # The frozen runner checks its other four output paths before touching PL.
    import pynq
    clocks=pynq.Clocks; initial=state(clocks); trace=[dict(stage='before_cpu_baseline',**initial)]
    actual_overlay=pynq.Overlay; base=benchmark.FastUNet; loads=[]
    def factory(*a,**k):
        if loads: raise RuntimeError('Only one bit download allowed in this benchmark')
        overlay=actual_overlay(*a,**k);loads.append(overlay)
        return clock_overlay(overlay,clocks,initial['cpu_mhz'],trace,args.timeout)
    evidence=dict(addon_sha256=sha(folder/'clock_manifest.json'),
        target_clock_mhz=TARGET_MHZ,source_clock_pll_changed=False,cpu_or_ddr_settings_written=False,
        original_fast_manifest_sha256=manifest['original_fast_manifest_sha256'],
        routed_design=manifest['routed_design'])
    try:
        # Only construction is intercepted, and only inside this synchronous
        # call. Restore both original Python bindings on success OR exception.
        with patch.object(pynq,'Overlay',factory), patch.object(benchmark,'FastUNet',
                checked_network(base,clocks,initial['cpu_mhz'],trace)):
            result=benchmark.run(args)
        final=state(clocks);trace.append(dict(stage='benchmark_complete',**final));require_target(final,initial['cpu_mhz'])
        if len(loads)!=1: raise AssertionError('Benchmark did not perform exactly one controlled download')
        summary=dict(status='passed_real_board_100mhz_clock_verified_benchmark',**evidence,
            benchmark_sha256=sha(report),clock_trace=trace,clock_reads_inside_forward_wall_time=True,
            optimized_pl_mean_ms=result['optimized_pl']['mean_ms'],
            cpu_two_thread_mean_ms=result['cpu_two_threads']['mean_ms'],
            cpu_over_pl=result['speedup_cpu2_over_fast_pl'])
        benchmark.write_new(clock_path,summary)
        print('Verified FCLK0 100 MHz before/after every forward. Clock evidence: '+str(clock_path),flush=True)
        return summary
    except BaseException as exc:
        benchmark.write_new(clock_path,dict(status='failed_clock_checked_benchmark',**evidence,
                                          error_type=type(exc).__name__,error=str(exc),clock_trace=trace))
        # No automatic clock rollback/reset/free: an error may leave AXI active.
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_fast_100mhz_result.json')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--old-runs',type=int,choices=[0],default=0)
    parser.add_argument('--timeout',type=float,default=30.)
    run(parser.parse_args())
