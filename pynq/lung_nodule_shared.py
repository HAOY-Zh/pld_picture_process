"""Single owner of one convolution/tensor overlay, multiple frozen models.

No overlay reload or numerical conversion when selecting a different model.
All callers must use this owner; direct access to its drivers is unsupported.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import threading
from time import perf_counter
import numpy as np
from lung_fast import FastUNet,PreparedConv,NativeLoader,compile_io
from lung_tensor import TensorAxil
from nodule_pl import NodulePL
from lung_axil import LungAxil
from lung_continuous import word_count
from lung_model_layers_test import SENTINEL

_RETAINED=[]
_BOARD_OWNER=None
_BOARD_LOCK=threading.Lock()


def checked_package(folder):
    from nodule_plan import sha
    folder=Path(folder).resolve();m=json.loads((folder/'shared_manifest.json').read_text(encoding='utf-8'))
    if m['schema']!='lung_nodule_single_engine_v1':raise ValueError('Wrong shared package')
    for n,h in m['files_sha256'].items():
        p=(folder/n).resolve()
        if folder not in p.parents or not p.is_file() or sha(p)!=h:raise ValueError('Changed shared artifact '+n)
    if m['clock_mhz']!=100 or sha(folder/'lung_network.bit')!=m['bit_sha256']:raise ValueError('Wrong bit/clock')
    return folder,m


def load_model(folder,label):
    if label not in ('unet_original','unet_pruned','nodule'):raise ValueError('Unknown model')
    folder=Path(folder)/'models'/label
    model=json.loads((folder/'model.json').read_text(encoding='utf-8'))
    with np.load(folder/'parameters.npz',allow_pickle=False) as z:params={n:z[n] for n in z.files}
    if label=='nodule':
        from nodule_plan import validate
    else:
        from lung_continuous import validate_model as validate
    validate(model,params)
    return model,params


class SharedCNN:
    def __init__(self,conv,tensor,allocate,clock_check=None):
        self.conv,self.tensor,self.allocate=conv,tensor,allocate
        self._clock_check=clock_check or (lambda:None)
        self._lock=threading.Lock();self._nets={};self._current=None
        self.closed=False;self.failed=False;self._board_claim=False
        self.overlay=None;self.build=None;self.clocks=None
        self._idle()

    def _idle(self):
        if any(d._inflight or d._poisoned for d in (self.conv,self.tensor)):
            raise RuntimeError('Outstanding/failed shared hardware; retain buffers and recover board')
        self.conv._idle();self.tensor._idle()

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(False):raise RuntimeError('Shared CNN is busy; serialize model tasks')
        try:
            if self.closed or self.failed:raise RuntimeError('Closed/failed shared owner')
            self._idle();self._clock_check();yield
        finally:self._lock.release()

    @property
    def cma_bytes(self):return sum(n.cma_bytes for n in self._nets.values())

    def register(self,label,model,parameters,kind):
        if not isinstance(label,str) or not label:raise ValueError('Need model label')
        with self._exclusive():
            if label in self._nets:raise ValueError('Duplicate registered model')
            cls={'unet':FastUNet,'nodule':NodulePL}.get(kind)
            if cls is None:raise ValueError('Unknown model kind')
            net=cls(self.conv,self.tensor,self.allocate,model,parameters)
            prior=[b for n in self._nets.values() for b in n.buffers.values()]
            for buf in net.buffers.values():
                address=LungAxil._buffer_address(buf)
                for other in prior:
                    start=LungAxil._buffer_address(other)
                    if address<start+other.nbytes and start<address+buf.nbytes:
                        self.failed=True;_RETAINED.append((self,net))
                        raise ValueError('Cross-model CMA overlap; no subsequent inference allowed')
            self._nets[label]=net
            return dict(label=label,commands=model['commands'],cma_bytes=net.cma_bytes,
                        preparation_ms=net.preparation_ms,allocation_ms=net.allocation_ms)

    def predict_codes(self,label,image,observer=None):
        with self._exclusive():
            if label not in self._nets:raise ValueError('Unknown registered model')
            net=self._nets[label]
            if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or list(image.shape)!=net.model['input_shape']:
                raise ValueError('Wrong model input codes')
            previous=self._current;start=perf_counter()
            try:
                # Existing schedulers configure EVERY operator and reload EVERY
                # convolution block, so no cross-model cached-weight assumption.
                head,stats=net.predict_codes(image,observer)
                self._idle();self._clock_check()
                self._current=label
                stats=dict(stats,selected_model=label,previous_model=previous,model_changed=previous!=label,
                           overlay_reloaded=False,shared_call_ms=(perf_counter()-start)*1000)
                return head,stats
            except BaseException:
                self.failed=True
                if self not in _RETAINED:_RETAINED.append(self)
                raise

    def snapshot(self,label):
        """Diagnostic detached device payload; never feeds an inference."""
        with self._exclusive():
            net=self._nets[label];result={}
            for name,b in net.buffers.items():
                b.invalidate();count=word_count(net.shapes[name][0])
                if np.any(np.asarray(b)[count:]!=SENTINEL):raise AssertionError('Shared allocation guard overwritten')
                result[name]=np.array(b[:count],copy=True)
            return result

    def close(self):
        if not self._lock.acquire(False):raise RuntimeError('Cannot close running shared CNN')
        try:
            if self.closed:return
            self._idle() # Never free memory while DMA completion is uncertain.
            for net in self._nets.values():net.close()
            self.closed=True
            if self._board_claim:
                global _BOARD_OWNER
                with _BOARD_LOCK:
                    if _BOARD_OWNER is self:_BOARD_OWNER=None
        finally:self._lock.release()


def open_shared(folder,timeout=30.):
    """Download one bit ONCE; reserve a sole in-process hardware owner."""
    global _BOARD_OWNER
    folder,manifest=checked_package(folder)
    with _BOARD_LOCK:
        if _BOARD_OWNER is not None:raise RuntimeError('Another shared overlay owner is active')
        token=object();_BOARD_OWNER=token
    owner=None
    try:
        from pynq import Overlay,allocate,Clocks
        overlay=Overlay(str(folder/'lung_network.bit'))
        Clocks.fclk0_mhz=100.
        def clock_check():
            if abs(float(Clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('Shared engine FCLK changed')
        conv=PreparedConv(overlay.lung_conv8_axil_0.mmio,timeout=timeout)
        tensor=TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=timeout)
        conv._idle();tensor._idle();clock_check()
        lib,build=compile_io(folder);conv.native_loader=NativeLoader(lib,conv.mmio)
        owner=SharedCNN(conv,tensor,allocate,clock_check)
        owner.overlay=overlay;owner.build=build;owner.clocks=Clocks;owner._board_claim=True
        with _BOARD_LOCK:_BOARD_OWNER=owner
        return owner,manifest
    except BaseException:
        # Retain claim after loading may have affected hardware. Do not enable
        # blind overlay retries; recovery belongs to the board owner.
        if owner is not None:_RETAINED.append(owner)
        raise
