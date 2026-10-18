"""Resource ownership, partial-start faults, input contracts and tile framing."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'pynq'), str(ROOT/'python')]
from ldct_processor import LDCTProcessor, METHODS
from ctdenoise.pl_reference import fixed_filter
from ctdenoise.filters import unpack_hu


class CMA(np.ndarray):
    def __new__(cls, address):
        obj = np.zeros(4096, np.uint32).view(cls)
        obj.device_address = address
        return obj

    def __array_finalize__(self, obj):
        self.device_address = getattr(obj, 'device_address', 0)

    def freebuffer(self):
        raise AssertionError('Borrowed buffer must never be freed')


class GPIO:
    def __init__(self):
        self.cfg = self.status = 0
        self.writes = []

    def write(self, offset, value):
        self.writes.append((offset, value))
        if value & 0x80000000:
            raise AssertionError('Processor must never reset hardware')
        self.cfg = value

    def read(self, offset):
        return self.status


class Channel:
    def __init__(self, dma, receive):
        self.dma, self.receive = dma, receive
        self._first_transfer = True
        self._active_buffer = None

    def transfer(self, buffer):
        dma = self.dma
        dma.transfer_attempts += 1
        self._first_transfer = False
        self._active_buffer = buffer
        if dma.observer:
            assert dma.observer()['dma_may_access'], 'Flag must precede BOTH transfers'
        if self.receive:
            dma.rx = 0
            dma.destination = buffer
            if dma.failure == 'receive':
                raise RuntimeError('receive partial start')
            return
        dma.tx = 0
        if dma.failure == 'send':
            raise RuntimeError('send partial start')
        if dma.failure == 'timeout':
            return
        if dma.failure == 'dma_error':
            dma.tx = 0x10
            return
        cfg = dma.gpio.cfg
        mode, w, h = cfg & 15, (cfg >> 4) & 127, (cfg >> 11) & 127
        a = unpack_hu(buffer).reshape(h, w)
        q = a if mode == 6 else fixed_filter(a, mode)  # zero CNN is an independent identity fixture
        if cfg & (1 << 26):
            left, right, top, bottom = [(cfg >> s) & 3 for s in (18,20,22,24)]
            q = q[top:h-bottom, left:w-right]
        assert q.size == dma.destination.size
        dma.destination[:] = q.view(np.uint16).astype(np.uint32).ravel()
        dma.tx = dma.rx = 2
        if dma.failure == 'pl_error':
            dma.gpio.status = 2

    def wait(self):
        self.dma.waits += 1
        self._active_buffer = None


class DMA:
    def __init__(self, gpio):
        self.gpio, self.mmio = gpio, self
        self.tx = self.rx = 2
        self.failure = None
        self.observer = None
        self.transfer_attempts = self.waits = self.reset_writes = 0
        self.extra_reads = {0:1,0x30:1,0x28:0,0x58:0}
        self.recvchannel, self.sendchannel = Channel(self, True), Channel(self, False)

    def read(self, offset):
        return self.tx if offset == 4 else self.rx if offset == 0x34 else self.extra_reads[offset]

    def write(self, offset, value):
        self.reset_writes += 1
        raise AssertionError('Module must not reset DMA')


class ProcessorContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)/'model'
        self.model.mkdir()
        np.savez(self.model/'weights_integer.npz', w0=np.zeros(1, np.int8))
        sha = hashlib.sha256((self.model/'weights_integer.npz').read_bytes()).hexdigest()
        (self.model/'quantization.json').write_text(json.dumps(dict(checkpoint_sha256='a'*64, weights_sha256=sha)))

    def processor(self, clock=lambda: 50., timeout=.002):
        gpio = GPIO()
        dma = DMA(gpio)
        obj = LDCTProcessor(dma=dma, gpio=gpio, tx_buffer=CMA(0x10000), rx_buffer=CMA(0x100000),
            model_dir=self.model, read_clock_mhz=clock, expected_clock_mhz=50., timeout_s=timeout)
        dma.observer = obj.get_state
        return obj, dma, gpio

    def test_initialization_is_inert_and_results_are_independent(self):
        obj, dma, gpio = self.processor()
        self.assertEqual(dma.transfer_attempts, 0)
        self.assertEqual(gpio.writes, [])
        first = np.arange(512*512).astype(np.int16).reshape(512,512)
        r1 = obj.process_slice(first)
        self.assertEqual(r1['tiles'], 81)
        preserved = r1['hu'].copy()
        r2 = obj.process_slice(np.zeros((67,69), np.int16))
        self.assertEqual(r2['tiles'], 4)
        np.testing.assert_array_equal(first, preserved)
        np.testing.assert_array_equal(r1['hu'], preserved)
        self.assertFalse(np.shares_memory(r1['hu'], obj.rx_buffer))
        obj.assert_idle()
        self.assertEqual(obj.get_state(), dict(busy=False, faulted=False, dma_may_access=False, error=None))

    def test_declared_methods_and_degenerate_edges(self):
        rng = np.random.default_rng(71)
        for method, mode in METHODS.items():
            obj, _, _ = self.processor()
            image = rng.integers(-32768,32768,(9,11),dtype=np.int16)
            result = obj.process_slice(image, method)
            expected = image if mode == 6 else fixed_filter(image, mode)
            np.testing.assert_array_equal(result['hu'], expected)
        for shape in [(1,1), (1,69), (67,1), (512,511)]:
            obj, _, _ = self.processor()
            image = rng.integers(-32768,32768,shape,dtype=np.int16)
            np.testing.assert_array_equal(obj.process_slice(image)['hu'], image)

    def test_invalid_input_and_frequency_are_rejected_before_dma(self):
        cases = [np.zeros((0,3),np.int16), np.zeros((513,2),np.int16), np.zeros(2,np.int16),
                 np.zeros((2,2),np.float64), np.zeros((2,2),np.uint16),
                 np.array([[np.nan]],np.float32), np.array([[np.inf]],np.float32),
                 np.array([[32767.6]],np.float32), np.array([[-32768.6]],np.float32)]
        for image in cases:
            obj, dma, _ = self.processor()
            with self.assertRaises(ValueError):
                obj.process_slice(image)
            self.assertEqual(dma.transfer_attempts, 0)
            self.assertFalse(obj.faulted)
        obj, dma, _ = self.processor()
        with self.assertRaises(ValueError):
            obj.process_slice(np.zeros((1,1),np.int16), 'unknown')
        self.assertEqual(dma.transfer_attempts, 0)
        for bad in (100., 49.49, float('nan')):
            obj, dma, gpio = self.processor(clock=lambda: bad)
            with self.assertRaises(RuntimeError):
                obj.process_slice(np.zeros((2,3),np.int16))
            self.assertEqual(dma.transfer_attempts, 0)
            self.assertEqual(gpio.writes, [])

    def test_float32_ties_to_even_without_display_transform(self):
        obj, _, _ = self.processor()
        a = np.array([[-32768., -2.5, -1.5, -.5, .5, 1.5, 2.5, 32767.4]], np.float32)
        result = obj.process_slice(a)
        np.testing.assert_array_equal(result['hu'], np.rint(a).astype(np.int16))
        self.assertEqual(result['hu'].dtype, np.int16)

    def test_partial_starts_timeouts_and_hardware_errors_are_latched(self):
        for failure in ('receive', 'send', 'timeout', 'dma_error', 'pl_error'):
            obj, dma, _ = self.processor()
            dma.failure = failure
            with self.assertRaises((RuntimeError,TimeoutError)):
                obj.process_slice(np.zeros((2,3),np.int16))
            state = obj.get_state()
            self.assertFalse(state['busy'])
            self.assertTrue(state['faulted'])
            self.assertTrue(state['dma_may_access'])
            self.assertIsNotNone(state['error'])
            self.assertEqual(dma.reset_writes, 0)
            with self.assertRaises(RuntimeError):
                obj.assert_idle()
            attempts = dma.transfer_attempts
            with self.assertRaises(RuntimeError):
                obj.process_slice(np.zeros((2,3),np.int16))
            self.assertEqual(dma.transfer_attempts, attempts)

    def test_assert_idle_checks_real_status_without_reset(self):
        for problem in ('active', 'error', 'protocol'):
            obj, dma, gpio = self.processor()
            if problem == 'active': dma.rx = 0; dma.recvchannel._first_transfer = False
            if problem == 'error': dma.tx = 0x12
            if problem == 'protocol': gpio.status = 1
            with self.assertRaises(RuntimeError):
                obj.assert_idle()
            self.assertTrue(obj.faulted)
            self.assertTrue(obj.dma_may_access)
            self.assertEqual(dma.reset_writes, 0)

    def test_pristine_running_dma_without_idle_bit_is_verified_not_assumed(self):
        obj, dma, _ = self.processor()
        dma.tx = dma.rx = 0
        obj.assert_idle()
        image = np.zeros((2,3),np.int16)
        np.testing.assert_array_equal(obj.process_slice(image)['hu'],image)
        dma.tx = dma.rx = 0
        dma.recvchannel._first_transfer = dma.sendchannel._first_transfer = True
        with self.assertRaises(RuntimeError):
            obj.assert_idle()  # Pristine exception is forbidden after submission.
        for bad in ('buffer','length','stopped'):
            obj, dma, _ = self.processor()
            dma.tx = dma.rx = 0
            if bad == 'buffer': dma.recvchannel._active_buffer = np.empty(1)
            if bad == 'length': dma.extra_reads[0x58] = 4
            if bad == 'stopped': dma.extra_reads[0x30] = 0
            with self.assertRaises(RuntimeError): obj.assert_idle()

    def test_clock_change_after_frame_is_not_ignored(self):
        values = iter([50., 100.])
        obj, dma, _ = self.processor(clock=lambda: next(values))
        with self.assertRaises(RuntimeError):
            obj.process_slice(np.zeros((2,3),np.int16))
        self.assertTrue(obj.faulted)
        self.assertTrue(obj.dma_may_access)


if __name__ == '__main__':
    unittest.main()
