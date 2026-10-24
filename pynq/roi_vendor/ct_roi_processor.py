"""ROI1 borrowed-resource adapter. No Overlay/clock/CMA allocation ownership.

B supplies initialized resources and serializes global module access. On DMA
fault/timeout this instance becomes faulted: B must recover the manager.
No reset, no buffer free, no automatic retry, and no CPU fallback here.
"""
from __future__ import annotations
from pathlib import Path
import sys
import threading
from time import perf_counter,sleep
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ct_roi_reference import validate_slice,configuration,pack_input,make_result


class RoiHardwareError(RuntimeError):
    pass


class CtRoiProcessor:
    def __init__(self,*,dma,geometry_gpio,bounds_gpio,hu_status_gpio,count_identity_gpio,
                 input_buffer,output_buffer,clock_hz,timeout_seconds=5.,wait_dma=None):
        if not np.isfinite(clock_hz) or clock_hz<=0 or not np.isfinite(timeout_seconds) or timeout_seconds<=0:
            raise ValueError('Provide actual manager-reported clock Hz and positive timeout')
        for buffer in (input_buffer,output_buffer):
            if buffer.dtype!=np.uint32 or buffer.ndim!=1 or not buffer.flags.c_contiguous:
                raise ValueError('Borrowed DMA buffers must be contiguous 1D uint32')
        if np.shares_memory(input_buffer,output_buffer):
            raise ValueError('Input/output DMA buffers must not alias')
        self.dma=dma; self.geometry=geometry_gpio; self.bounds=bounds_gpio
        self.hu_status=hu_status_gpio; self.count_identity=count_identity_gpio
        self.input_buffer=input_buffer; self.output_buffer=output_buffer
        self.clock_hz=float(clock_hz); self.timeout=float(timeout_seconds)
        self.wait_dma=wait_dma; self.faulted=False; self.lock=threading.Lock()
        if int(self.count_identity.channel2.read())!=0x524f4931:
            raise RoiHardwareError('ROI1 identity mismatch; do not bypass bit/HWH manager checks')

    def _wait(self):
        if self.wait_dma is not None:
            # Injected manager wait must honor timeout and expose DMA errors.
            self.wait_dma(self.dma,self.timeout)
            return
        deadline=perf_counter()+self.timeout
        while True:
            mmio=getattr(self.dma,'mmio',None)
            if mmio is not None:
                if (int(mmio.read(0x04))|int(mmio.read(0x34)))&0x70:
                    raise RoiHardwareError('AXI DMA internal/slave/decode error')
            if self.dma.sendchannel.idle and self.dma.recvchannel.idle:
                # PYNQ wait performs receive-buffer cache invalidation.
                self.dma.sendchannel.wait(); self.dma.recvchannel.wait()
                return
            if perf_counter()>=deadline:
                raise TimeoutError('ROI DMA timeout: manager recovery required; buffers remain borrowed')
            sleep(.0002)

    def process_slice(self,hu_slice_i16,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
        start=perf_counter()
        # All user-input checks happen before configuring/starting DMA.
        hu,roi,thresholds,use_hu,lung=validate_slice(hu_slice_i16,roi_yx,hu_range,use_hu,lung_mask)
        n=hu.size
        if n>self.input_buffer.size or n>self.output_buffer.size:
            raise ValueError('Borrowed buffers are too small for the native slice')
        params=configuration(hu,roi,thresholds,use_hu,lung)
        packed=pack_input(hu,lung)
        if not self.lock.acquire(blocking=False):
            raise RoiHardwareError('Concurrent calls to the same ROI processor are not allowed')
        try:
            if self.faulted:
                raise RoiHardwareError('ROI processor is faulted; ask manager to recover and rebind')
            status=int(self.hu_status.channel2.read())
            if status&1:
                raise RoiHardwareError('ROI is already busy; manager serialization required')
            self.geometry.channel1.write(params['geometry'],0xffffffff)
            self.geometry.channel2.write(params['modes'],0xffffffff)
            self.bounds.channel1.write(params['bounds_x'],0xffffffff)
            self.bounds.channel2.write(params['bounds_y'],0xffffffff)
            self.hu_status.channel1.write(params['thresholds'],0xffffffff)
            self.input_buffer[:n]=packed
            self.output_buffer[:n]=np.uint32(0xdeadbeef)
            prepared=perf_counter()
            try:
                self.dma.recvchannel.transfer(self.output_buffer,nbytes=n*4)
                self.dma.sendchannel.transfer(self.input_buffer,nbytes=n*4)
                self._wait()
                finished=perf_counter()
                status=int(self.hu_status.channel2.read())
                if status!=0x01000002:
                    raise RoiHardwareError(f'ROI status must be ABI1/done/no-errors, got {status:#010x}')
                hardware_count=int(self.count_identity.channel1.read())
                words=np.array(self.output_buffer[:n],copy=True)
                if (words>1).any():
                    raise RoiHardwareError('ROI output is not ABI1 0/1 words')
                mask=words.astype(np.uint8).reshape(hu.shape)
                if int(mask.sum(dtype=np.int64))!=hardware_count:
                    raise RoiHardwareError('ROI hardware count differs from returned mask sum')
            except Exception:
                self.faulted=True
                raise
            result=make_result(hu,roi,thresholds,use_hu,lung,mask,'FPGA',
                {'prepare_seconds':prepared-start,'dma_compute_seconds':finished-prepared,
                 'cpu_reference_seconds':None,'postprocess_seconds':None,'total_seconds':None},self.clock_hz)
            end=perf_counter()
            result['timing'].update(postprocess_seconds=end-finished,total_seconds=end-start)
            result['hardware_status']=status
            result['clock_source']='REPORTED_BY_RESOURCE_MANAGER'
            return result
        finally:
            self.lock.release()
