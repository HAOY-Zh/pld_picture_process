"""Borrowed ROI1 DMA plus RST1 GPIOs. Never loads Overlay, allocates CMA or retries."""
from time import perf_counter
import numpy as np
from ct_roi_processor import CtRoiProcessor, RoiHardwareError
from ct_roi_reference import validate_slice,configuration,pack_input,MODULE_VERSION,ALGORITHM_ID,PARAMS_VERSION
from roi_statistics_contract import read_statistics,IDENTITY,STATISTICS_ABI


class RoiStatisticsProcessor(CtRoiProcessor):
    def __init__(self,*,statistics_gpio,**resources):
        super().__init__(**resources)
        self.statistics_gpio=statistics_gpio
        if self._word('identity')!=IDENTITY:raise RoiHardwareError('Wrong RST1 bit/HWH; preserve owner')

    def _word(self,name):
        mapping={'sum_lo':('roi_stats_sum',1),'sum_hi':('roi_stats_sum',2),
            'sumsq_lo':('roi_stats_sumsq',1),'sumsq_hi':('roi_stats_sumsq',2),
            'minmax':('roi_stats_meta',1),'valid':('roi_stats_meta',2),'identity':('roi_stats_identity',1)}
        gpio,channel=mapping[name]
        return int(getattr(self.statistics_gpio[gpio],'channel'+str(channel)).read())

    def process_slice(self,hu_slice_i16,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
        start=perf_counter()
        hu,roi,limits,use_hu,lung=validate_slice(hu_slice_i16,roi_yx,hu_range,use_hu,lung_mask)
        n=hu.size
        if n>self.input_buffer.size or n>self.output_buffer.size:raise ValueError('Borrowed ROI buffers too small')
        params=configuration(hu,roi,limits,use_hu,lung);packed=pack_input(hu,lung)
        if not self.lock.acquire(False):raise RoiHardwareError('Concurrent RST1 calls')
        try:
            if self.faulted:raise RoiHardwareError('Faulted RST1 driver retained')
            if int(self.hu_status.channel2.read())&1:raise RoiHardwareError('ROI is busy')
            self.geometry.channel1.write(params['geometry'],0xffffffff)
            self.geometry.channel2.write(params['modes'],0xffffffff)
            self.bounds.channel1.write(params['bounds_x'],0xffffffff)
            self.bounds.channel2.write(params['bounds_y'],0xffffffff)
            self.hu_status.channel1.write(params['thresholds'],0xffffffff)
            self.input_buffer[:n]=packed;self.output_buffer[:n]=np.uint32(0xdeadbeef)
            prepared=perf_counter()
            try:
                self.dma.recvchannel.transfer(self.output_buffer,nbytes=n*4)
                self.dma.sendchannel.transfer(self.input_buffer,nbytes=n*4)
                self._wait();finished=perf_counter()
                status=int(self.hu_status.channel2.read());count=int(self.count_identity.channel1.read())
                words=np.array(self.output_buffer[:n],copy=True)
                if (words>1).any():raise RoiHardwareError('Invalid ROI mask words')
                mask=words.astype(np.uint8).reshape(hu.shape)
                if int(mask.sum(dtype=np.int64))!=count:raise RoiHardwareError('ROI mask/hardware count mismatch')
                stats=read_statistics(self._word,count,status)
                if int(self.hu_status.channel2.read())!=status or int(self.count_identity.channel1.read())!=count:
                    raise RoiHardwareError('Frame changed during statistics read')
            except BaseException:
                self.faulted=True;raise
            end=perf_counter()
            return dict(filtered_mask=mask,valid_count=count,backend='FPGA',status='SUCCESS',
                module_version=MODULE_VERSION,algorithm_id=ALGORITHM_ID,params_version=PARAMS_VERSION,
                shape_yx=list(hu.shape),roi_yx=list(roi),use_hu=use_hu,hu_range=list(limits) if use_hu else None,
                use_lung_mask=lung is not None,clock_hz=self.clock_hz,hu_statistics=stats,
                statistics_abi=STATISTICS_ABI,hardware_status=status,clock_source='REPORTED_BY_RESOURCE_MANAGER',
                timing=dict(prepare_seconds=prepared-start,dma_compute_seconds=finished-prepared,
                    postprocess_seconds=end-finished,total_seconds=end-start,cpu_reference_seconds=None))
        finally:self.lock.release()
