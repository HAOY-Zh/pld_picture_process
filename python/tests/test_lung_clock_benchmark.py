import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_clock_benchmark as clock
from test_lung_axil import MMIO


class FakeClocks:
    def __init__(self,frequency=62.5,wrong=False,cpu_change=False):
        self.frequency=frequency;self.cpu_mhz=650.;self.writes=[];self.wrong=wrong;self.cpu_change=cpu_change
    @property
    def fclk0_mhz(self): return self.frequency
    @fclk0_mhz.setter
    def fclk0_mhz(self,value):
        self.writes.append(value)
        if not self.wrong: self.frequency=value
        if self.cpu_change: self.cpu_mhz=500.


def overlay():
    tensor=MMIO();tensor.reg[0]=0x54454e53
    return SimpleNamespace(lung_conv8_axil_0=SimpleNamespace(mmio=MMIO()),
                           lung_tensor_axil_0=SimpleNamespace(mmio=tensor))


class ClockTests(unittest.TestCase):
    def test_correct_idle_engines_and_unchanged_cpu(self):
        clocks=FakeClocks();trace=[];ol=overlay()
        self.assertIs(clock.clock_overlay(ol,clocks,650.,trace,1.),ol)
        self.assertEqual(clocks.writes,[100.]);self.assertEqual(trace[-1]['fclk0_mhz'],100.)
        self.assertEqual(trace[0]['fclk0_mhz'],62.5)
        self.assertEqual(ol.lung_conv8_axil_0.mmio.writes,[])  # No accelerator START/reset/clear.
        self.assertEqual(ol.lung_tensor_axil_0.mmio.writes,[])

    def test_already_100_no_write_and_busy_or_bad_id_never_sets_clock(self):
        clocks=FakeClocks(100.);clock.clock_overlay(overlay(),clocks,650.,[],1.)
        self.assertEqual(clocks.writes,[])
        for engine,status in [('conv',1),('conv',32),('tensor',1),('identity',0)]:
            ol=overlay();clocks=FakeClocks()
            if engine=='identity': ol.lung_conv8_axil_0.mmio.reg[0]=0
            else: getattr(ol,'lung_'+('conv8' if engine=='conv' else 'tensor')+'_axil_0').mmio.reg[8]=status
            with self.assertRaises(RuntimeError): clock.clock_overlay(ol,clocks,650.,[],1.)
            self.assertEqual(clocks.writes,[])

    def test_bad_readback_or_cpu_change_rejected_no_rollback(self):
        for c in [FakeClocks(wrong=True),FakeClocks(cpu_change=True)]:
            with self.assertRaises(RuntimeError): clock.clock_overlay(overlay(),c,650.,[],1.)
            self.assertEqual(c.writes,[100.])
        for value in [0.,float('nan'),float('inf')]:
            with self.assertRaises(RuntimeError): clock.state(FakeClocks(value))

    def test_forward_returns_actual_output_and_uses_observed_frequency(self):
        clocks=FakeClocks(100.);trace=[];calls=[];actual=object()
        class Base:
            def predict_codes(self,image,observer=None):
                calls.append((image,observer));return actual,dict(pl_command_cycles=123456)
        net=clock.checked_network(Base,clocks,650.,trace)()
        output,row=net.predict_codes('input',observer='observer')
        self.assertIs(output,actual);self.assertEqual(calls,[('input','observer')])
        self.assertEqual(row['engine_active_ms_actual_clock'],1.23456)
        self.assertEqual(len(trace),2)
        clocks.frequency=62.5
        with self.assertRaises(RuntimeError): net.predict_codes('input')
        self.assertEqual(len(calls),1)  # Reject BEFORE calling any operator.

    def test_clock_change_during_forward_rejected(self):
        clocks=FakeClocks(100.)
        class Base:
            def predict_codes(self,*a,**k):
                clocks.frequency=62.5;return 'output',dict(pl_command_cycles=10)
        with self.assertRaises(RuntimeError): clock.checked_network(Base,clocks,650.,[])().predict_codes('input')

    def test_scoped_overlay_hook_runs_after_download_and_restores_bindings(self):
        for fail in [False,True]:
            with tempfile.TemporaryDirectory() as temp:
                report=Path(temp)/'result.json';args=SimpleNamespace(report=str(report),timeout=1.)
                clocks=FakeClocks();events=[]
                def real_overlay(*a,**k):
                    self.assertEqual(clocks.frequency,62.5);events.append('download');return overlay()
                pynq=SimpleNamespace(Clocks=clocks,Overlay=real_overlay)
                class Base:
                    def predict_codes(self,*a,**k): return 'actual',dict(pl_command_cycles=100)
                def benchmark(args):
                    pynq.Overlay('bit')
                    self.assertEqual(clocks.frequency,100.)
                    out,row=clock.benchmark.FastUNet().predict_codes('input')
                    self.assertEqual(out,'actual')
                    if fail: raise TimeoutError('Simulated hardware timeout, NO recovery')
                    report.write_text('{}')
                    return dict(optimized_pl=dict(mean_ms=20.),cpu_two_threads=dict(mean_ms=10.),speedup_cpu2_over_fast_pl=.5)
                addon=dict(original_fast_manifest_sha256='original',routed_design=dict(clock_mhz=100))
                with patch.dict(sys.modules,{'pynq':pynq}),patch.object(clock,'validate_addon',return_value=addon), \
                        patch.object(clock,'sha',return_value='digest'),patch.object(clock.benchmark,'FastUNet',Base), \
                        patch.object(clock.benchmark,'run',side_effect=benchmark):
                    if fail:
                        with self.assertRaises(TimeoutError): clock.run(args)
                    else: self.assertEqual(clock.run(args)['cpu_over_pl'],.5)
                    self.assertIs(pynq.Overlay,real_overlay);self.assertIs(clock.benchmark.FastUNet,Base)
                result=json.loads(report.with_name('result_clock.json').read_text())
                self.assertEqual(result['status'].startswith('failed'),fail)
                self.assertEqual(clocks.writes,[100.]);self.assertEqual(events,['download'])


if __name__=='__main__': unittest.main()
