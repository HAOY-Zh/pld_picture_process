"""ABI 1.0 MMIO driver for ONE convolution block, not a full U-Net driver.

Inject pynq.MMIO on the board; imports also work with a PC test double.
Timeout requires shared hardware/AXI recovery. CLEAR does not cancel AXI.
"""
import math
import operator
import time

ID, CTRL, STATUS, VERSION = 0x00, 0x04, 0x08, 0x0C
INPUT, OUTPUT, SHAPE, GEOMETRY, QUANT = 0x10, 0x14, 0x18, 0x1C, 0x20
WRITTEN, HITS, MISSES, CYCLES, COMPLETED, WEIGHT_BEATS = 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38
WEIGHTS, BIASES, MULTIPLIERS, SHIFTS = 0x100, 0x140, 0x160, 0x180
BUSY, DONE, ERROR, LOADED, READY, LOADING, CONTROL_ERROR = 1, 2, 4, 8, 16, 32, 256


def bounded(value, minimum, maximum, name):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(name + ' must be an integer') from exc
    if not minimum <= value <= maximum:
        raise ValueError(name + ' outside supported range')
    return value


class LungAxil:
    def __init__(self, mmio, timeout=5.0, poll_interval=0.0):
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval < 0:
            raise ValueError('Invalid polling settings')
        self.mmio, self.timeout, self.poll_interval = mmio, float(timeout), float(poll_interval)
        self._poisoned = self._inflight = False
        self._configuration = self._resident = None
        if mmio.read(ID) != 0x4C554E47 or mmio.read(VERSION) != 0x00010000:
            raise RuntimeError('Wrong accelerator ID/ABI; do not use an old bit/hwh')

    def _healthy(self):
        if self._poisoned:
            raise RuntimeError('Timed out: shared system recovery required before reuse')

    def _idle(self):
        self._healthy()
        status = self.mmio.read(STATUS)
        if self._inflight or status & (BUSY | LOADING):
            raise RuntimeError('Busy, incomplete weight block, or previous result not collected')
        return status

    def _poll(self, predicate):
        self._healthy()
        deadline = time.monotonic() + self.timeout
        while True:
            status = self.mmio.read(STATUS)
            if predicate(status):
                return status
            if time.monotonic() >= deadline:
                self._poisoned = True
                raise TimeoutError('Accelerator timeout; discard output and recover shared AXI system')
            if self.poll_interval:
                time.sleep(self.poll_interval)

    def configure(self, input_address, output_address, height, width, channels,
                  output_channels, output_block=0, kernel=3, padding=1):
        h = bounded(height, 1, 512, 'height'); w = bounded(width, 1, 512, 'width')
        cin = bounded(channels, 1, 96, 'channels'); cout = bounded(output_channels, 1, 96, 'output_channels')
        kernel = bounded(kernel, 1, 3, 'kernel'); padding = bounded(padding, 0, 1, 'padding')
        if kernel not in (1, 3) or (kernel == 1 and padding) or (kernel == 3 and not padding and min(h, w) < 3):
            raise ValueError('Unsupported convolution geometry')
        blocks = (cout + 7)//8; block = bounded(output_block, 0, blocks-1, 'output_block')
        src = bounded(input_address, 0, 2**32-1, 'input_address'); dst = bounded(output_address, 0, 2**32-1, 'output_address')
        oh, ow = h+2*padding-kernel+1, w+2*padding-kernel+1
        input_bytes, output_bytes = h*w*((cin+7)//8)*8, oh*ow*blocks*8
        if (src | dst) & 7 or src+input_bytes > 2**32 or dst+output_bytes > 2**32:
            raise ValueError('Buffers must be 8-byte aligned and inside 32-bit physical space')
        if src < dst+output_bytes and dst < src+input_bytes:
            raise ValueError('Input/output complete allocations must not overlap')
        self._idle()
        for register, value in [(INPUT, src), (OUTPUT, dst), (SHAPE, h | w << 16),
                (GEOMETRY, cin | int(kernel == 3) << 8 | padding << 9 | blocks << 16 | block << 24)]:
            self.mmio.write(register, value)
        self._configuration = dict(input_address=src, output_address=dst, input_bytes=input_bytes, output_bytes=output_bytes,
            words=oh*ow, packet_count=kernel*kernel*((cin+7)//8), keep=(1 << min(8, cout-block*8))-1)

    def load_weights(self, weights, biases, multipliers, shifts, input_zero, output_zero, output_keep, relu=True):
        packets = [bounded(v, 0, 2**512-1, 'weight packet') for v in weights]
        if not 1 <= len(packets) <= 128:
            raise ValueError('Weight block must contain 1..128 packets')
        bias = [bounded(v, -2**31, 2**31-1, 'bias') & 0xFFFFFFFF for v in biases]
        mult = [bounded(v, 0, 2**31-1, 'multiplier') for v in multipliers]
        shift = [bounded(v, 0, 62, 'shift') for v in shifts]
        if any(len(v) != 8 for v in [bias, mult, shift]):
            raise ValueError('Exactly eight lane parameters required')
        iz = bounded(input_zero, 0, 255, 'input_zero'); oz = bounded(output_zero, 0, 255, 'output_zero')
        keep = bounded(output_keep, 1, 255, 'output_keep'); relu = bounded(relu, 0, 1, 'relu')
        self._idle(); self._resident = None
        self.mmio.write(CTRL, 2); self.mmio.write(QUANT, keep | iz << 8 | oz << 16 | relu << 24)
        for registers, values in [(BIASES, bias), (MULTIPLIERS, mult), (SHIFTS, shift)]:
            for lane, value in enumerate(values):
                self.mmio.write(registers+4*lane, value)
        initial_count = self.mmio.read(WEIGHT_BEATS)
        for index, packet in enumerate(packets):
            for word in range(16):
                self.mmio.write(WEIGHTS+4*word, packet >> (word*32) & 0xFFFFFFFF)
            self.mmio.write(CTRL, 4 | int(index == 0)*8 | int(index == len(packets)-1)*16)
            status = self._poll(lambda s:not s & BUSY)
            if status & CONTROL_ERROR:
                raise RuntimeError('Weight push rejected; recover incomplete blocks explicitly')
        if not status & LOADED or status & LOADING or ((self.mmio.read(WEIGHT_BEATS)-initial_count) & 0xFFFFFFFF) != len(packets):
            raise RuntimeError('Incomplete weight load')
        self._resident = dict(packet_count=len(packets), keep=keep)

    def start(self):
        status = self._idle()
        if self._configuration is None or self._resident is None:
            raise RuntimeError('Configure and load weights before starting')
        if self._resident['packet_count'] != self._configuration['packet_count'] or self._resident['keep'] != self._configuration['keep']:
            raise ValueError('Resident weights do not match configured kernel/input/output block')
        if not status & LOADED:
            raise RuntimeError('Hardware weights are no longer loaded')
        self._inflight = True
        try:
            self.mmio.write(CTRL, 1)
        except BaseException:
            # A failed/interrupted host write does not prove PL did not start.
            self._poisoned = True
            raise

    def wait(self):
        self._healthy()
        if not self._inflight:
            raise RuntimeError('No command started')
        status = self._poll(lambda s:bool(s & (DONE | CONTROL_ERROR)) and not s & BUSY)
        self._inflight = False
        if status & (ERROR | CONTROL_ERROR):
            raise RuntimeError('Accelerator command failed; discard the complete output')
        statistics = {name:self.mmio.read(register) for name, register in [
            ('written_words', WRITTEN), ('cache_hits', HITS), ('cache_misses', MISSES), ('cycles', CYCLES), ('completed', COMPLETED)]}
        if statistics['written_words'] != self._configuration['words']:
            raise RuntimeError('Unexpected output word count; discard output')
        return statistics

    @staticmethod
    def _buffer_address(buffer):
        return int(buffer.physical_address if hasattr(buffer, 'physical_address') else buffer.device_address)

    def run(self, input_buffer, output_buffer):
        self._healthy(); config = self._configuration
        if config is None:
            raise RuntimeError('Configure buffers first')
        for buffer, address, size in [(input_buffer, config['input_address'], config['input_bytes']),
                                     (output_buffer, config['output_address'], config['output_bytes'])]:
            if self._buffer_address(buffer) != address or buffer.nbytes < size:
                raise ValueError('Actual buffer address/size differs from configured allocation')
            if not callable(getattr(buffer, 'flush', None)) or not callable(getattr(buffer, 'invalidate', None)):
                raise ValueError('Use PYNQ device buffers with cache synchronization')
        self._idle(); input_buffer.flush(); output_buffer.flush()
        self.start(); result = self.wait(); output_buffer.invalidate()
        return result
