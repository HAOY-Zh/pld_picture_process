"""One hardware owner: unchanged LDCT/basic/segmentation plus ROI1."""
import base64
import json
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

import ct_platform_service as base
import ct_platform_fast_service as fast
import ct_studio_service as studio
from ct_roi_processor import CtRoiProcessor
from ct_roi_reference import CpuRoiProcessor
from roi_contract import BACKEND, TRANSPORT, decode, digest, validate_assets
from roi_statistics_contract import STATISTICS_ABI, GPIO_NAMES, validate_statistics

GUARD=np.uint32(0xA5C3917E)


def validate_package(folder):
    manifest=studio.validate_package(folder)
    validate_assets(folder,manifest['roi'])
    return manifest


class Hardware(studio.FastHardware):
    def _dma_status(self):
        if self.mode=='roi' and self.dma is not None:
            for offset in (0x04,0x34):
                status=int(self.dma.mmio.read(offset))
                if status&0x770:raise RuntimeError('ROI DMA error')
                if getattr(self,'_roi_transfer_started',False) and not status&2:
                    raise RuntimeError('ROI DMA not confirmed idle')
            return
        return super()._dma_status()

    def release_idle(self):
        if getattr(self,'roi_processor',None) is not None and self.roi_processor.faulted:
            raise RuntimeError('Faulted ROI retained; no release/reload')
        super().release_idle()
        self.roi_processor=None

    def switch(self,mode):
        if mode!='roi':return super().switch(mode)
        if self.fault:raise RuntimeError('Faulted owner retained: '+self.fault)
        if self.mode=='roi':self._dma_status();return 0.
        started=time.perf_counter()
        try:
            validate_package(self.folder)  # Check new assets before giving up old resources.
            self.release_idle();row=self.manifest['roi']
            self.overlay=self.pynq.Overlay(str(self.folder/row['bit']))
            self.clocks.fclk0_mhz=row['clock_mhz']
            self._check_clock(float(self.clocks.fclk0_mhz),100.,.5)
            self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
            self.dma=self.overlay.axi_dma_0
            for _ in range(2):self.buffers.append(self.pynq.allocate(shape=(base.PIXELS+16,),dtype=np.uint32))
            self.mode='roi';self._roi_transfer_started=False
            self._dma_status()
            for gpio,directions in ((self.overlay.roi_geometry,('out','out')),
                                    (self.overlay.roi_bounds,('out','out')),
                                    (self.overlay.roi_hu_status,('out','in')),
                                    (self.overlay.roi_count_identity,('in','in'))):
                gpio.channel1.setdirection(directions[0]);gpio.channel2.setdirection(directions[1])
            resources=dict(dma=self.dma,geometry_gpio=self.overlay.roi_geometry,
                bounds_gpio=self.overlay.roi_bounds,hu_status_gpio=self.overlay.roi_hu_status,
                count_identity_gpio=self.overlay.roi_count_identity,input_buffer=self.buffers[0],output_buffer=self.buffers[1],
                clock_hz=float(self.clocks.fclk0_mhz)*1e6,timeout_seconds=self.timeout,wait_dma=self._wait_roi)
            if row.get('statistics_abi'):
                if row['statistics_abi']!=STATISTICS_ABI:raise ValueError('Unsupported ROI statistics ABI')
                from roi_statistics_processor import RoiStatisticsProcessor
                gpios={name:getattr(self.overlay,name) for name in GPIO_NAMES}
                for name,gpio in gpios.items():
                    gpio.channel1.setdirection('in')
                    if name!='roi_stats_identity':gpio.channel2.setdirection('in')
                self.roi_processor=RoiStatisticsProcessor(statistics_gpio=gpios,**resources)
            else:self.roi_processor=CtRoiProcessor(**resources)
            self.active_bit=row['bit_sha256'];self.switches+=1
            return (time.perf_counter()-started)*1000
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise

    def _wait_roi(self,dma,timeout):
        deadline=time.monotonic()+timeout
        while True:
            statuses=[int(dma.mmio.read(v)) for v in (0x04,0x34)]
            if any(s&0x770 for s in statuses):raise RuntimeError('ROI DMA error: '+str(statuses))
            if all(s&2 for s in statuses):break
            if time.monotonic()>deadline:raise TimeoutError('ROI DMA timeout; owner/buffers retained')
            time.sleep(.0001)
        dma.sendchannel.wait();dma.recvchannel.wait();self.buffers[1].invalidate()

    def roi(self,hu,params,lung):
        switch_ms=self.switch('roi');n=hu.size
        try:
            before=float(self.clocks.fclk0_mhz)
            for buffer in self.buffers:
                buffer[n:n+16]=GUARD;buffer.flush()
            # Requests were fully validated before entering this owner. Once a
            # transfer might be armed, no fresh-DMA shortcut may be used again.
            self._roi_transfer_started=True
            result=self.roi_processor.process_slice(hu,params['roi_yx'],params['hu_range'],params['use_hu'],lung)
            self._dma_status()
            for buffer in self.buffers:
                buffer.invalidate()
                if not np.all(buffer[n:n+16]==GUARD):raise RuntimeError('ROI DMA guard changed')
            after=float(self.clocks.fclk0_mhz)
            self._check_clock(before,100.,.5);self._check_clock(after,100.,.5)
            self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
            result['timing'].update(overlay_switch_ms=switch_ms,actual_pl_mhz_before=before,
                actual_pl_mhz_after=after,cpu_mhz=float(self.clocks.cpu_mhz),guard_verified=True,
                timing_scope=('PS pack/config + DMA/PL/wait + scalar GPIO read/mean/std; overlay switch separate'
                    if result.get('statistics_abi') else
                    'PS pack/config + DMA/PL/wait + returned-mask statistics; overlay switch separate'))
            return result
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise

    def selftest_roi(self):
        from roi_contract import encode
        rng=np.random.default_rng(20261022)
        hu=rng.integers(-32768,32768,(512,512),dtype=np.int16)
        mask=rng.integers(0,2,hu.shape,dtype=np.uint8)
        cases=[('full',hu,None,None,False,None),
               ('roi_hu_mask',hu,(2,510,7,511),(-1000,-400),True,mask),
               ('empty',hu,(512,512,0,512),None,False,None),
               ('endpoint',np.array([[-32768,-1,0,32767]],np.int16),None,(-32768,32767),True,None),
               ('one',np.array([[40]],np.int16),None,(40,40),True,np.ones((1,1),np.uint8)),
               ('odd',hu[:127,:129],(0,126,1,129),(-500,300),True,mask[:127,:129])]
        if self.manifest['roi'].get('statistics_abi'):
            cases.extend([('int16_min',np.full((512,512),-32768,np.int16),None,None,False,None),
                          ('int16_max',np.full((512,512),32767,np.int16),None,None,False,None)])
        arrays={};rows=[];retained=None;stem=self.folder/('studio_roi_startup_'+uuid.uuid4().hex)
        try:
            for repeat in range(3):
                for name,image,roi,limits,use_hu,lung in cases:
                    raw,params,image,lung=encode(image,roi,limits,use_hu,lung)
                    reference=CpuRoiProcessor().process_slice(image,roi,limits,use_hu,lung)
                    result=self.roi(image,params,lung)
                    np.testing.assert_array_equal(result['filtered_mask'],reference['filtered_mask'])
                    if result['valid_count']!=reference['valid_count']:raise AssertionError('ROI count mismatch')
                    if self.manifest['roi'].get('statistics_abi'):
                        actual=validate_statistics(result['hu_statistics'],result['valid_count'])
                        values=image[reference['filtered_mask']!=0].astype(np.int64)
                        for field in ('count','sum','min','max','mean'):
                            if actual[field]!=reference['hu_statistics'][field]:raise AssertionError('RST1 '+field+' differs')
                        if actual['sumsq']!=int((values*values).sum()):raise AssertionError('RST1 sumsq differs')
                        if values.size and not np.isclose(actual['std_population'],values.std(),rtol=1e-12,atol=1e-12):
                            raise AssertionError('RST1 std differs')
                    if retained is not None:np.testing.assert_array_equal(*retained)
                    retained=(result['filtered_mask'],result['filtered_mask'].copy())
                    key=str(repeat)+'_'+name;arrays[key]=result['filtered_mask']
                    rows.append(dict(case=key,parameters=params,input_sha256=digest(raw),valid_count=result['valid_count'],
                                     stats=result['timing'],hardware_status=result['hardware_status'],
                                     hu_statistics=result['hu_statistics'],statistics_abi=result.get('statistics_abi')))
            # Exercise all directions with the same owner and preserved assets.
            self.switch('window');self.basic(np.full((512,512),40,np.int16),base.parameters(dict(operation='window')))
            self.switch('roi')
            raw,params,image,lung=encode(np.array([[40]],np.int16),None,None,False,None)
            assert self.roi(image,params,lung)['valid_count']==1
            self.denoise(np.full((8,8),40,np.int16),'bypass')
            self.switch('roi');assert self.roi(image,params,lung)['valid_count']==1
            self.switch('segmentation')
            arrays['segmentation_after_roi']=self.last_startup_head.copy()
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise
        finally:
            with stem.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
            with stem.with_suffix('.json').open('x') as f:
                json.dump(dict(passed=self.fault is None,error=self.fault,roi=self.manifest['roi'],checks=rows,
                    platform_manifest_sha256=base.sha((self.folder/'platform_manifest.json').read_bytes()),
                    arrays_file=stem.with_suffix('.npz').name,arrays_sha256=digest(stem.with_suffix('.npz').read_bytes()),
                    segmentation_after_roi=getattr(self,'segmentation_identity',None)),f,indent=2)


class Service(studio.Service):
    def health(self):
        value=super().health()
        value.update(roi=self.hardware.manifest['roi'],roi_startup_acceptance=
                     'startup-selftests-passed' if getattr(self,'roi_verified',False) else 'not-run')
        if self.hardware.manifest['roi'].get('statistics_abi'):
            value['roi_statistics_startup_acceptance']='startup-selftests-passed' if getattr(self,'roi_verified',False) else 'not-run'
        return value

    def execute_roi(self,raw,rid,input_hash,params):
        if not re.fullmatch('[0-9a-f]{32}',rid) or digest(raw)!=input_hash:raise ValueError('ROI request/hash mismatch')
        hu,lung=decode(raw,params)
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            try:
                started=time.perf_counter();result=self.hardware.roi(hu,params,lung)
                output=result['filtered_mask'].tobytes();self.requests+=1;row=self.hardware.manifest['roi']
                return dict(schema='ct_studio_roi_result_v1',backend=BACKEND,transport=TRANSPORT,
                    parameters=params,request_id=rid,input_sha256=input_hash,hu_sha256=digest(hu.tobytes()),
                    lung_mask_sha256=digest(lung.tobytes()) if lung is not None else None,
                    output_sha256=digest(output),mask_base64=base64.b64encode(output).decode(),
                    valid_count=result['valid_count'],hu_statistics=result['hu_statistics'],
                    statistics_abi=row.get('statistics_abi'),
                    hardware_status=result['hardware_status'],stats=result['timing'],
                    state=dict(busy=False,faulted=False,dma_may_access=False),sequence=self.requests,
                    service_ms=(time.perf_counter()-started)*1000,platform_manifest_sha256=self.manifest_sha256,
                    bit_sha256=row['bit_sha256'],hwh_sha256=row['hwh_sha256'])
            except BaseException as exc:self.failed=type(exc).__name__+': '+str(exc);raise
        finally:self.gate.release()


def handler_for(service):
    parent=studio.handler_for(service)
    class Handler(parent):
        def do_POST(self):
            if self.path!='/v1/roi':return super().do_POST()
            if not self.allowed():return
            try:
                count=int(self.headers.get('Content-Length','-1'))
                if self.headers.get('Transfer-Encoding') or not 0<count<=512*512*3 or self.headers.get('Content-Type')!='application/octet-stream':
                    raise ValueError('Invalid ROI binary framing')
                text=self.headers.get('X-Parameters','{}')
                if len(text)>2048:raise ValueError('Oversized ROI parameters')
                raw=self.rfile.read(count)
                if len(raw)!=count:raise ValueError('Truncated ROI input')
                self.reply(200,service.execute_roi(raw,self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),json.loads(text)))
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except (ValueError,KeyError) as exc:self.reply(503 if service.failed else 400,dict(error=str(exc)))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder or Path(__file__).resolve().parent).resolve();validate_package(folder)
    token=(folder/'platform_token.txt').read_text().strip()
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=Hardware(folder);hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi()
        service=Service(hardware,token,base.sha((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('CT STUDIO ROI READY: basic + LDCT + segmentation + actual ROI startup checks PASS.',flush=True)
    return service
