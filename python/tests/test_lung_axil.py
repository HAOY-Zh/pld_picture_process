from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_axil as a
import lung_axil_smoke as smoke
import numpy as np


class MMIO:
    def __init__(self):
        self.reg = {a.ID:0x4C554E47, a.VERSION:0x10000, a.STATUS:0, a.WEIGHT_BEATS:0, a.COMPLETED:0}
        self.writes = []; self.packets = []; self.flags = []; self.ticks = 0
        self.task = None; self.compute_error = False; self.reject_start = False; self.hang = False

    def read(self, offset):
        if offset == a.STATUS and self.ticks and not self.hang:
            self.ticks -= 1
            if self.ticks == 0:
                if self.task == 'compute':
                    self.reg[a.STATUS] = a.DONE | a.LOADED | (a.ERROR if self.compute_error else 0)
                    self.reg[a.COMPLETED] += 1
                    self.reg[a.CYCLES] = 1234
                else:
                    self.reg[a.STATUS] = a.LOADED if self.flags[-1] & 16 else a.LOADING
        return self.reg.get(offset, 0)

    def write(self, offset, value):
        self.writes.append((offset, value))
        self.reg[offset] = value
        if offset != a.CTRL:
            return
        if value == 2:
            self.reg[a.STATUS] &= ~(a.DONE | a.ERROR | a.CONTROL_ERROR)
        elif value == 1:
            if self.reject_start:
                self.reg[a.STATUS] = a.LOADED | a.CONTROL_ERROR
                return
            h, w = self.reg[a.SHAPE] & 1023, self.reg[a.SHAPE] >> 16
            geom = self.reg[a.GEOMETRY]
            delta = 2 if geom & 256 and not geom & 512 else 0
            self.reg[a.WRITTEN] = (h-delta)*(w-delta)
            self.reg[a.STATUS] = a.BUSY | a.LOADED
            self.task = 'compute'; self.ticks = 2
        elif value & 4:
            self.packets.append(sum(self.reg.get(a.WEIGHTS+4*i, 0) << (32*i) for i in range(16)))
            self.flags.append(value)
            self.reg[a.WEIGHT_BEATS] = (self.reg[a.WEIGHT_BEATS]+1) & 0xFFFFFFFF
            self.reg[a.STATUS] = a.BUSY
            self.task = 'weights'; self.ticks = 2


class Buffer:
    def __init__(self, address, size, name, events):
        self.physical_address, self.nbytes, self.name, self.events = address, size, name, events

    def flush(self): self.events.append(self.name+' flush')
    def invalidate(self): self.events.append(self.name+' invalidate')


def load(driver, packets=None, keep=1):
    driver.load_weights([1] if packets is None else packets, [-1]+[0]*7, [1 << 30]*8, [31]*8, 128, 128, keep, True)


class DriverTests(unittest.TestCase):
    def test_board_smoke_layout_packet_order_and_rne_reference(self):
        q = np.arange(11*3*5, dtype=np.uint8).reshape(11, 3, 5)
        words = smoke.feature_words(q, 128).view(np.uint8).reshape(3, 5, 2, 8)
        for c in range(11): self.assertTrue(np.array_equal(words[:, :, c//8, c%8], q[c]))
        self.assertTrue(np.all(words[:, :, 1, 3:] == 128))
        weights = np.zeros((9, 11, 3, 3), np.int8); weights[8, 10, 2, 1] = -3
        packets = smoke.weight_packets(weights, 1)
        self.assertEqual(len(packets), 18)
        self.assertEqual(packets[(2*3+1)*2+1], 253 << 16)
        values = np.array([[[127, 125, 129, 131]]], np.uint8)
        got = smoke.reference(values, np.ones((1, 1, 1, 1), np.int8), [0], 128, 128, padding=0)
        self.assertEqual(got.tolist(), [[[128, 126, 128, 130]]])

    def test_abi_and_polling_settings(self):
        mmio = MMIO(); mmio.reg[a.ID] = 0
        with self.assertRaises(RuntimeError): a.LungAxil(mmio)
        mmio = MMIO(); mmio.reg[a.VERSION] = 2
        with self.assertRaises(RuntimeError): a.LungAxil(mmio)
        for timeout, interval in [(0, 0), (-1, 0), (float('nan'), 0), (float('inf'), 0), (1, -1)]:
            with self.assertRaises(ValueError): a.LungAxil(MMIO(), timeout, interval)

    def test_geometry_byte_address_and_tail_output(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        driver.configure(0x1000, 0x2000, 5, 7, 11, 13, output_block=1, kernel=3, padding=0)
        self.assertEqual(mmio.reg[a.INPUT], 0x1000)
        self.assertEqual(mmio.reg[a.SHAPE], 5 | 7 << 16)
        self.assertEqual(mmio.reg[a.GEOMETRY], 11 | 256 | 2 << 16 | 1 << 24)
        self.assertEqual(driver._configuration['words'], 15)
        self.assertEqual(driver._configuration['packet_count'], 18)
        self.assertEqual(driver._configuration['keep'], 31)
        driver.configure(0xFFFFFFF8, 0, 1, 1, 1, 1, kernel=1, padding=0)

    def test_bad_configuration_does_not_write_mmio(self):
        good = dict(input_address=0x1000, output_address=0x2000, height=5, width=7,
                    channels=3, output_channels=1, kernel=3, padding=1)
        for key, value in [('height', 0), ('width', 513), ('channels', 97), ('kernel', 2),
                           ('output_channels', 97), ('output_block', 1), ('input_address', 3),
                           ('output_address', 0x1000), ('input_address', 0xFFFFFFF8), ('height', 1.2)]:
            mmio = MMIO(); driver = a.LungAxil(mmio); kwargs = {**good, key:value}
            with self.assertRaises(ValueError): driver.configure(**kwargs)
            self.assertEqual(mmio.writes, [])

    def test_weight_packets_and_counter_wrap(self):
        mmio = MMIO(); mmio.reg[a.WEIGHT_BEATS] = 0xFFFFFFFE; driver = a.LungAxil(mmio)
        packets = [0, (1 << 511) | 0x87654321, 2**512-1]
        load(driver, packets, keep=0x42)
        self.assertEqual(mmio.packets, packets)
        self.assertEqual(mmio.flags, [12, 4, 20])
        self.assertEqual(mmio.reg[a.BIASES], 0xFFFFFFFF)
        self.assertEqual(mmio.reg[a.QUANT], 0x01808042)
        self.assertEqual(mmio.reg[a.WEIGHT_BEATS], 1)
        before = len(mmio.writes)
        with self.assertRaises(ValueError):
            driver.load_weights([2**512], [0]*8, [1]*8, [0]*8, 0, 0, 1)
        with self.assertRaises(ValueError):
            driver.load_weights([1], [0]*8, [1]*8, [63]*8, 0, 0, 1)
        self.assertEqual(len(mmio.writes), before)

    def test_start_wait_reuse_and_frame_counters(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        with self.assertRaises(RuntimeError): driver.wait()
        with self.assertRaises(RuntimeError): driver.start()

        driver.configure(0x1000, 0x2000, 3, 5, 1, 1, kernel=1, padding=0); load(driver)
        for i in range(2):
            driver.start()
            with self.assertRaises(RuntimeError): driver.start()
            result = driver.wait()
            self.assertEqual(result['written_words'], 15)
            self.assertEqual(result['completed'], i+1)
        driver.configure(0x1000, 0x2000, 3, 5, 9, 1, kernel=1, padding=0)
        with self.assertRaises(ValueError): driver.start()

    def test_uncertain_start_write_does_not_allow_buffer_reuse(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        driver.configure(0x1000, 0x2000, 1, 1, 1, 1, kernel=1, padding=0); load(driver)
        with patch.object(mmio, 'write', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): driver.start()
        self.assertTrue(driver._inflight)
        self.assertTrue(driver._poisoned)
        with self.assertRaises(RuntimeError): driver.start()

    def test_busy_errors_and_timeout_require_recovery(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        driver.configure(0x1000, 0x2000, 1, 1, 1, 1, kernel=1, padding=0); load(driver)
        mmio.compute_error = True; driver.start()
        with self.assertRaises(RuntimeError): driver.wait()
        mmio.compute_error = False; mmio.reject_start = True; driver.start()
        with self.assertRaises(RuntimeError): driver.wait()
        mmio.reject_start = False; mmio.hang = True; driver.start()
        with patch.object(a.time, 'monotonic', side_effect=[0, 10]):
            with self.assertRaises(TimeoutError): driver.wait()
        with self.assertRaises(RuntimeError): driver.start()

    def test_run_checks_actual_buffers_and_syncs_in_order(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        driver.configure(0x1000, 0x2000, 1, 1, 1, 1, kernel=1, padding=0); load(driver)
        events = []; src = Buffer(0x1000, 8, 'input', events); dst = Buffer(0x2000, 8, 'output', events)
        self.assertEqual(driver.run(src, dst)['written_words'], 1)
        self.assertEqual(events, ['input flush', 'output flush', 'output invalidate'])
        events.clear(); dst.nbytes = 7
        with self.assertRaises(ValueError): driver.run(src, dst)
        self.assertEqual(events, [])

    def test_incomplete_or_busy_weight_blocks_are_not_reconfigured(self):
        mmio = MMIO(); driver = a.LungAxil(mmio)
        for status in [a.BUSY, a.LOADING]:
            mmio.reg[a.STATUS] = status
            with self.assertRaises(RuntimeError): driver.configure(0x1000, 0x2000, 1, 1, 1, 1, kernel=1, padding=0)
            with self.assertRaises(RuntimeError): load(driver)
        self.assertEqual(mmio.writes, [])


if __name__ == '__main__': unittest.main()
