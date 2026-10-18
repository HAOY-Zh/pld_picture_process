"""Borrowed-resource, HU-to-HU interface for the verified LDCT tile overlay.

Import/initialization never allocates CMA, downloads an Overlay, starts DMA,
changes clocks, or executes a CPU CNN. The caller owns its lock and recovery.
"""
import json
import math
import sys
import threading
from pathlib import Path
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct_denoise_tile_pl_test import run_frame_fast

INTERFACE_VERSION = 'ldct_slice_v1'
METHODS = {'bypass': 0, 'gaussian5': 1, 'bilateral5': 2, 'joint7_guide5': 3,
           'joint7_guide3': 4, 'guided3': 5, 'cnn': 6, 'bilateral7': 7}
CLOCK_TOLERANCE_MHZ = .5
MAX_DIMENSION = 512


def interface_contract(clock_mhz):
    """Single source for the platform-facing manifest and the Python wrapper."""
    return dict(interface_version=INTERFACE_VERSION, input='raw HU numpy int16 or float32, 2D',
        output='independent numpy int16 HU, unchanged shape/coordinates',
        rounding='np.rint ties-to-even; reject nonfinite or rounded int16 overflow',
        dimensions=dict(min_height=1, min_width=1, max_height=MAX_DIMENSION, max_width=MAX_DIMENSION),
        methods=METHODS, ip_names=dict(dma='axi_dma_0', gpio='axi_gpio_0'),
        buffers=dict(count=2, elements_each=4096, dtype='uint32', ownership='borrowed CMA; caller allocates/releases'),
        clock=dict(domain='FCLK0 / clk_fpga_0', expected_mhz=float(clock_mhz), tolerance_mhz=CLOCK_TOLERANCE_MHZ,
                   owner='unified service; processor only reads/checks before and after every frame'),
        tile=dict(max_input=[64,64], cnn_halo=3, cnn_output_core=[58,58], cnn_compact=True,
                  other_methods_output_core=[48,48], boundary='replicate per layer; internal halo cropped'),
        gpio=dict(config_offset=0, status_offset=8, mode_bits=[0,3], width_bits=[4,10], height_bits=[11,17],
                  trim_left_bits=[18,19], trim_right_bits=[20,21], trim_top_bits=[22,23], trim_bottom_bits=[24,25],
                  compact_enable_bit=26, reset_bit=31,
                  status_bits={'0':'bad TKEEP','1':'bad TLAST','2':'invalid mode','3':'invalid dimensions/crop','4':'divide zero'}),
        axis=dict(word_bits=32, hu_bits=[0,15], upper_bits='zero', byte_order='little', tkeep=15,
                  tlast='last returned word per tile; compact CNN returns core only'),
        timing_scope='pack+cache+DMA+PL+poll+stitch',
        fault_policy='latch faults; no retry/reset/free/CPU fallback; caller must recover and reconstruct processor',
        dma_idle_validation='DMASR Idle/Halted; before any submission only, also allow RS=1, LENGTH=0, PYNQ _first_transfer=True and _active_buffer=None',
        dependencies=dict(python='3.10.4 verified',numpy='NumPy required; tested version recorded in board result',pynq='3.1.1 verified; pristine-state check uses its channel bookkeeping'))


class LDCTProcessor:
    def __init__(self, *, dma, gpio, tx_buffer, rx_buffer, model_dir, read_clock_mhz,
                 expected_clock_mhz, timeout_s=10.):
        expected_clock_mhz = float(expected_clock_mhz)
        timeout_s = float(timeout_s)
        if not math.isfinite(expected_clock_mhz) or expected_clock_mhz not in (50., 100.):
            raise ValueError('Expected clock must match a verified 50/100 MHz release')
        if not math.isfinite(timeout_s) or timeout_s <= 0 or not callable(read_clock_mhz):
            raise ValueError('Positive timeout and clock reader required')
        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required')
        addresses = []
        for buffer in (tx_buffer, rx_buffer):
            if (not isinstance(buffer, np.ndarray) or buffer.dtype != np.dtype('uint32') or
                    buffer.ndim != 1 or buffer.size < 4096 or not buffer.flags.c_contiguous or not buffer.flags.writeable):
                raise ValueError('Writable contiguous 1D uint32 CMA buffers of >=4096 elements required')
            address = getattr(buffer, 'device_address', getattr(buffer, 'physical_address', None))
            if not isinstance(address, (int, np.integer)) or address < 0 or address % 4:
                raise ValueError('CMA buffer device/physical address required')
            addresses.append(int(address))
        if np.shares_memory(tx_buffer, rx_buffer) or max(addresses) < min(addresses[0]+tx_buffer.nbytes, addresses[1]+rx_buffer.nbytes):
            raise ValueError('TX/RX CMA buffers must not overlap')
        model_dir = Path(model_dir)
        spec = json.loads((model_dir/'quantization.json').read_text(encoding='utf-8'))
        self.model_id = spec['checkpoint_sha256']
        self.weights_sha256 = spec['weights_sha256']
        # Hash verification is light; no training stack or CPU inference import.
        import hashlib
        if hashlib.sha256((model_dir/'weights_integer.npz').read_bytes()).hexdigest() != self.weights_sha256:
            raise ValueError('Model weight identity mismatch')
        manifest_path = model_dir.parent/'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if (not manifest.get('cnn_compact_protocol') or
                    float(manifest['clock_mhz']) != expected_clock_mhz or manifest['model'] != spec):
                raise ValueError('Model, compact protocol and declared release clock must match')
        self.dma, self.gpio = dma, gpio
        self.tx_buffer, self.rx_buffer = tx_buffer, rx_buffer
        self.read_clock_mhz = read_clock_mhz
        self.expected_clock_mhz, self.timeout_s = expected_clock_mhz, timeout_s
        self.busy = self.faulted = self.dma_may_access = False
        self.error = None
        self._ever_submitted = False
        self._lock = threading.Lock()

    def get_state(self):
        return dict(busy=self.busy, faulted=self.faulted, dma_may_access=self.dma_may_access, error=self.error)

    def _check_hardware_idle(self):
        tx, rx = int(self.dma.mmio.read(4)), int(self.dma.mmio.read(0x34))
        if (tx | rx) & 0x70:
            raise RuntimeError('DMA error tx=0x%x rx=0x%x' % (tx, rx))
        for status, channel, control, length in [(tx,self.dma.sendchannel,0,0x28),
                                                 (rx,self.dma.recvchannel,0x30,0x58)]:
            if status & 3:  # Idle or Halted: no pending memory accesses.
                continue
            # PYNQ 3.1.1 starts RS before the first BTT command; DMASR.Idle is
            # then 0 even though no transfer was submitted. Accept only the
            # jointly verified pristine state, never this exception after ANY
            # possible transfer in this processor. No reset or DMA priming.
            pristine = (not self._ever_submitted and
                getattr(channel,'_first_transfer',False) is True and
                getattr(channel,'_active_buffer',False) is None and
                int(self.dma.mmio.read(control)) & 1 and int(self.dma.mmio.read(length)) == 0)
            if not pristine:
                raise RuntimeError('DMA channel neither idle/halted nor verified pristine')
        status = int(self.gpio.read(8))
        if status:
            raise RuntimeError('PL protocol status 0x%x' % status)

    def assert_idle(self):
        if self.busy or self.faulted or self.dma_may_access:
            raise RuntimeError('LDCT resources are not safely idle: '+str(self.get_state()))
        try:
            self._check_hardware_idle()
        except Exception as error:
            self.faulted = self.dma_may_access = True
            self.error = str(error)
            raise

    def _clock(self):
        value = float(self.read_clock_mhz())
        if not math.isfinite(value) or abs(value-self.expected_clock_mhz) > CLOCK_TOLERANCE_MHZ:
            raise RuntimeError('PL clock %.6g MHz does not match release %.6g +/- %.1f MHz' %
                               (value, self.expected_clock_mhz, CLOCK_TOLERANCE_MHZ))
        return value

    def before_dma(self):
        # Called BEFORE receive.transfer(), not merely before send.transfer().
        # Keep this conservative flag until the ENTIRE frame and protocol pass.
        self.dma_may_access = True
        self._ever_submitted = True

    @staticmethod
    def _validate_input(hu):
        if (not isinstance(hu, np.ndarray) or hu.dtype not in (np.dtype('int16'), np.dtype('float32')) or
                hu.ndim != 2 or not hu.size or max(hu.shape) > MAX_DIMENSION):
            raise ValueError('Nonempty 2D int16/float32 HU array; each dimension 1..512 required')
        if not np.isfinite(hu).all():
            raise ValueError('Finite HU required')
        if hu.dtype == np.float32:
            rounded = np.rint(hu)
            if rounded.min() < -32768 or rounded.max() > 32767:
                raise ValueError('Rounded HU exceeds int16 range')

    def process_slice(self, hu, method='cnn'):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent LDCT use is not supported; service must serialize requests')
        started = False
        try:
            if not isinstance(method, str) or method not in METHODS:
                raise ValueError('Unknown method; supported: '+', '.join(METHODS))
            self._validate_input(hu)
            self.assert_idle()
            before = self._clock()  # Reject wrong clock before GPIO/DMA mutation.
            self.busy = True
            started = True
            mode = METHODS[method]
            start = perf_counter()  # Includes HU conversion and output allocation.
            output, _, tiles = run_frame_fast(hu, mode, 58 if mode == 6 else 48,
                self.dma, self.gpio, self.tx_buffer, self.rx_buffer, self.timeout_s,
                compact=mode == 6, lifecycle=self)
            elapsed_ms = (perf_counter()-start)*1000
            after = self._clock()
            self._check_hardware_idle()
            self.dma_may_access = False
            return dict(hu=output, total_ms=elapsed_ms, timing_scope='pack+cache+DMA+PL+poll+stitch',
                        actual_pl_mhz_before=before, actual_pl_mhz_after=after, tiles=tiles,
                        method=method, model_id=self.model_id, interface_version=INTERFACE_VERSION)
        except Exception as error:
            if started:
                self.faulted = True
                self.error = str(error)
            raise
        finally:
            self.busy = False
            self._lock.release()
