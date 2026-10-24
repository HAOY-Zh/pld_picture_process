"""Add the frozen Verilog nodule core to the existing sole board owner.

Legacy endpoints retain their pinned, unchanged platform contract. The new
endpoint separately binds the complete new service bundle. No detection scores
or full-case result are invented here; it returns the actual quantized head.
"""
import base64,json,re,sys,threading,time,uuid
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
import ct_studio_roi_service as roi
from shared_patch_contract import (INTERFACE,BACKEND,INPUT_BYTES,INPUT_SHAPE,OUTPUT_SHAPE,digest,
    validate_assets,validate_receipt_stats,BATCH_INTERFACE,MAX_BATCH,batch_patch_id)


def validate_package(folder):
    folder=Path(folder).resolve();manifest=roi.validate_package(folder)
    validate_assets(folder,manifest['shared_nodule'])
    p=folder/manifest['legacy_contract'];raw=p.read_bytes()
    if digest(raw)!=manifest['legacy_contract_sha256']:raise ValueError('Changed legacy platform contract')
    legacy=json.loads(raw)
    for name,h in legacy['files_sha256'].items():
        target=(folder/name).resolve()
        if folder not in target.parents or digest(target.read_bytes())!=h:raise ValueError('Changed preserved legacy asset: '+name)
    for key in ('basic','ldct','roi','basic_clocks_mhz'):
        if manifest[key]!=legacy[key]:raise ValueError('Legacy contract is not compatible: '+key)
    return manifest


class Hardware(roi.Hardware):
    def _dma_status(self):
        if self.mode=='shared_nodule':
            owner=self.shared_owner
            if owner.failed or owner.closed:raise RuntimeError('Failed/closed shared owner retained')
            owner._idle();return
        return super()._dma_status()

    def release_idle(self):
        owner=getattr(self,'shared_owner',None)
        if owner is not None:
            if self.fault or owner.failed:raise RuntimeError('Faulted shared owner retained; no release/reload')
            self._dma_status();owner.close();self.shared_owner=None
        super().release_idle()

    def switch(self,mode):
        if mode!='shared_nodule':return super().switch(mode)
        if self.fault:raise RuntimeError('Faulted owner retained: '+self.fault)
        if self.mode==mode:self._dma_status();return 0.
        tick=time.perf_counter()
        try:
            validate_package(self.folder);self.release_idle()
            # Private imports prevent new drivers from replacing old segmentation
            # modules in sys.modules. Model/bit/RTL bytes remain frozen.
            from shared_cnn.lung_nodule_shared import SharedCNN,load_model
            from shared_cnn.lung_fast import PreparedConv,NativeLoader,compile_io
            from shared_cnn.lung_tensor import TensorAxil
            row=self.manifest['shared_nodule'];folder=self.folder/'shared_cnn'
            self.overlay=self.pynq.Overlay(str(self.folder/row['bit']))
            self.clocks.fclk0_mhz=100.
            def clocks():
                self._check_clock(float(self.clocks.fclk0_mhz),100.,.5)
                self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
            clocks()
            conv=PreparedConv(self.overlay.lung_conv8_axil_0.mmio,timeout=self.timeout)
            tensor=TensorAxil(self.overlay.lung_tensor_axil_0.mmio,timeout=self.timeout)
            library,build=compile_io(folder);conv.native_loader=NativeLoader(library,conv.mmio)
            self.shared_owner=SharedCNN(conv,tensor,self.pynq.allocate,clocks)
            self.shared_owner.clocks=self.clocks;self.shared_owner.overlay=self.overlay;self.shared_owner.build=build
            self.mode=mode
            # Keep original U-Net weights; the optional pruned U-Net is not selected.
            for label,kind in (('unet_original','unet'),('nodule','nodule')):
                model,params=load_model(folder,label);self.shared_owner.register(label,model,params,kind)
            self.active_bit=row['bit_sha256'];self.switches+=1
            return (time.perf_counter()-tick)*1000
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise

    def nodule_patch(self,codes):
        switch=self.switch('shared_nodule')
        try:
            before=float(self.clocks.fclk0_mhz)
            output,stats=self.shared_owner.predict_codes('nodule',codes)
            self._dma_status()
            stats.update(overlay_switch_ms=switch,actual_pl_mhz_before=before,
                actual_pl_mhz_after=float(self.clocks.fclk0_mhz),cpu_mhz=float(self.clocks.cpu_mhz),
                scope='Frozen nodule graph in PL; codes in/head out. Excludes CT sampling, decoding/ranking and network transfer.')
            validate_receipt_stats(stats)
            return output,stats
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise

    def selftest_shared(self):
        self.switch('shared_nodule')
        folder=self.folder/'shared_cnn';arrays={};rows=[];retained=None
        from shared_cnn.lung_continuous import unpack_words
        stem=self.folder/('studio_shared_startup_'+uuid.uuid4().hex)
        try:
            for repeat in range(2):
                sequence=[('unet_original',0)]+[('nodule',i) for i in range(4)]
                for label,case in sequence:
                    model=json.loads((folder/'models'/label/'model.json').read_text())
                    with np.load(folder/'models'/label/'reference.npz',allow_pickle=False) as refs:
                        if label=='nodule':
                            image=refs['case'+str(case)+'_input']
                            expected={n['name']:refs['case'+str(case)+'_'+n['name']] for n in model['nodes']}
                        else:
                            with np.load(folder/'models'/label/'input.npz',allow_pickle=False) as z:image=z['codes']
                            expected={n['name']:unpack_words(refs[n['name']],n['shape']) for n in model['nodes']}
                    head,stats=self.shared_owner.predict_codes(label,image)
                    np.testing.assert_array_equal(head,expected['head'])
                    bank=self.shared_owner.snapshot(label);key=f'{repeat}_{label}_{case}';nodes=[]
                    for n in model['nodes']:
                        value=unpack_words(bank[n['name']],n['shape']);np.testing.assert_array_equal(value,expected[n['name']])
                        name=key+'__'+n['name'];arrays[name]=value;nodes.append(dict(name=n['name'],array=name,sha256=digest(value.tobytes())))
                    if retained is not None:np.testing.assert_array_equal(*retained)
                    retained=(head,head.copy())
                    rows.append(dict(key=key,label=label,case=case,nodes=nodes,stats=stats,input_sha256=digest(image.tobytes())))
            # Confirm switch out and back through every existing route.
            sample=np.full((512,512),40,np.int16)
            self.basic(sample,roi.base.parameters(dict(operation='window')))
            from roi_contract import encode
            _,params,image,lung=encode(sample,None,None,False,None)
            assert self.roi(image,params,lung)['valid_count']==sample.size
            self.denoise(np.full((8,8),40,np.int16),'cnn')
            self.switch('segmentation');arrays['legacy_segmentation_after_shared']=self.last_startup_head.copy()
            with np.load(folder/'models/nodule/reference.npz') as z:
                actual,stats=self.nodule_patch(z['case0_input']);np.testing.assert_array_equal(actual,z['case0_head'])
            arrays['nodule_after_legacy_routes']=actual
        except BaseException as exc:self.fault=type(exc).__name__+': '+str(exc);raise
        finally:
            with stem.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
            with stem.with_suffix('.json').open('x') as f:
                json.dump(dict(passed=self.fault is None,error=self.fault,rows=rows,
                    arrays_file=stem.with_suffix('.npz').name,arrays_sha256=digest(stem.with_suffix('.npz').read_bytes()),
                    shared_nodule=self.manifest['shared_nodule'],scope='Synthetic frozen inputs; all-node PL parity and actual legacy module switching, not detection quality.'),f,indent=2)


class Service(roi.Service):
    def __init__(self,hardware,token,bundle_sha):
        # The full legacy manifest is separately copied and hash-validated.
        super().__init__(hardware,token,hardware.manifest['legacy_contract_sha256'])
        self.bundle_sha256=bundle_sha

    def health(self):
        value=super().health();value['service_bundle_sha256']=self.bundle_sha256
        return value

    def shared_health(self):
        return dict(schema=INTERFACE,ready=self.failed is None and not self.hardware.fault,
            busy=self.gate.locked(),error=self.failed or self.hardware.fault,
            service_bundle_sha256=self.bundle_sha256,legacy_contract_sha256=self.manifest_sha256,
            shared_nodule=self.hardware.manifest['shared_nodule'],active_operation=self.hardware.mode,
            shared_batch=dict(interface=BATCH_INTERFACE,max_patches=MAX_BATCH),
            startup_acceptance='startup-selftests-passed' if getattr(self,'shared_verified',False) else 'not-run')

    def execute_patch(self,raw,rid,input_hash):
        if len(raw)!=INPUT_BYTES or not re.fullmatch('[0-9a-f]{32}',rid) or digest(raw)!=input_hash:
            raise ValueError('Invalid nodule input/request/hash')
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            try:
                record,packed=self._patch(raw,rid,input_hash)
                return dict(record,head_base64=base64.b64encode(packed).decode())
            except BaseException as exc:self.failed=type(exc).__name__+': '+str(exc);raise
        finally:self.gate.release()

    def _patch(self,raw,rid,input_hash):
        """Caller holds the sole-owner gate; every patch still completes all guards."""
        output,stats=self.hardware.nodule_patch(np.frombuffer(raw,np.uint8).reshape(INPUT_SHAPE))
        if output.dtype!=np.uint8 or output.shape!=OUTPUT_SHAPE:raise ValueError('Invalid actual head')
        packed=output.tobytes();self.requests+=1
        return dict(schema=INTERFACE,backend=BACKEND,request_id=rid,input_sha256=input_hash,
            input_shape=list(INPUT_SHAPE),output_shape=list(OUTPUT_SHAPE),
            output_sha256=digest(packed),shared_nodule=self.hardware.manifest['shared_nodule'],
            service_bundle_sha256=self.bundle_sha256,sequence=self.requests,stats=stats,
            state=dict(busy=False,faulted=False,dma_may_access=False),whole_case_detection=False),packed

    def execute_batch(self,raw,rid,input_hash,count):
        if (type(count)is not int or not 1<=count<=MAX_BATCH or len(raw)!=count*INPUT_BYTES
                or not re.fullmatch('[0-9a-f]{32}',rid) or digest(raw)!=input_hash):
            raise ValueError('Invalid bounded nodule batch/request/hash')
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            try:
                records=[];heads=[];started=time.perf_counter()
                for index in range(count):
                    patch=raw[index*INPUT_BYTES:(index+1)*INPUT_BYTES]
                    record,head=self._patch(patch,batch_patch_id(rid,index),digest(patch))
                    records.append(record);heads.append(head)
                packed=b''.join(heads)
                return dict(schema=BATCH_INTERFACE,backend=BACKEND,request_id=rid,input_sha256=input_hash,
                    patch_count=count,output_sha256=digest(packed),head_base64=base64.b64encode(packed).decode(),
                    records=records,service_bundle_sha256=self.bundle_sha256,
                    shared_nodule=self.hardware.manifest['shared_nodule'],
                    batch_service_ms=(time.perf_counter()-started)*1000,
                    state=dict(busy=False,faulted=False,dma_may_access=False),whole_case_detection=False)
            except BaseException as exc:self.failed=type(exc).__name__+': '+str(exc);raise
        finally:self.gate.release()


def handler_for(service):
    parent=roi.handler_for(service)
    class Handler(parent):
        def do_GET(self):
            if self.path!='/v1/shared/platform':return super().do_GET()
            if self.allowed():self.reply(200,service.shared_health())
        def do_POST(self):
            batch=self.path=='/v1/shared/nodule-batch'
            if self.path!='/v1/shared/nodule-patch' and not batch:return super().do_POST()
            if not self.allowed():return
            try:
                count=int(self.headers.get('X-Patch-Count','0')) if batch else 1
                if not 1<=count<=MAX_BATCH:raise ValueError('Invalid batch count')
                if (self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length')!=str(INPUT_BYTES*count)
                        or self.headers.get('Content-Type')!='application/octet-stream'):raise ValueError('Invalid patch binary framing')
                raw=self.rfile.read(INPUT_BYTES*count)
                args=(raw,self.headers.get('X-Request-ID',''),self.headers.get('X-Input-SHA256',''))
                self.reply(200,service.execute_batch(*args,count) if batch else service.execute_patch(*args))
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except ValueError as exc:self.reply(503 if service.failed else 400,dict(error=str(exc)))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder).resolve();validate_package(folder)
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=Hardware(folder);hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi();hardware.selftest_shared()
        service=Service(hardware,(folder/'platform_token.txt').read_text().strip(),digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=service.shared_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;roi.base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('CT STUDIO SHARED READY: legacy functions + frozen nodule Verilog core startup PASS.',flush=True)
    return service
