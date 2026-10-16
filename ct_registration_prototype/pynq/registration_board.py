"""Board DMA adapter. Requires a separately built/tested matching overlay.

Not executed on hardware in this delivery. Connect 3 dual-channel, 32-bit
AXI GPIO blocks as specified in README. Simple DMA, 32-bit streams.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ct_registration import _image, hardware_packet, quantize_matrix, register_rigid


class RegistrationBoard:
    def __init__(self, bit_path, shape=(128,128)):
        from pynq import Overlay, allocate
        self.shape=tuple(shape)
        self.overlay=Overlay(str(bit_path), download=True)
        self.dma=self.overlay.axi_dma_0
        self.gpios=[getattr(self.overlay,name) for name in ('coeff_ab','coeff_cd','coeff_ef')]
        self.source=None
        self.destination=None
        try:
            self.source=allocate(shape=(int(np.prod(self.shape)),),dtype=np.uint32)
            self.destination=allocate(shape=(self.source.size+4,),dtype=np.uint32)
        except Exception:
            self.close()
            raise

    def close(self):
        for name in ('source','destination'):
            buffer=getattr(self,name,None)
            if buffer is not None:
                buffer.freebuffer()
                setattr(self,name,None)

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()

    def evaluate(self,fixed,moving,matrix):
        fixed,moving=_image(fixed),_image(moving)
        if fixed.shape!=self.shape or moving.shape!=self.shape:
            raise ValueError('Dimensions must match the synthesized frame geometry')
        coefficients=quantize_matrix(matrix)
        # Check the affine coordinate range at the output rectangle corners.
        # No software warp is used to replace board output/cost.
        h,w=self.shape
        corners=np.array([[0,0,1],[w-1,0,1],[0,h-1,1],[w-1,h-1,1]],dtype=np.int64)
        coords=corners@coefficients.T
        if (coords < -2**31).any() or (coords > 2**31-1).any():
            raise ValueError('Q16.16 coordinate overflow')
        for index,value in enumerate(coefficients.ravel()):
            channel=self.gpios[index//2].channel1 if index%2==0 else self.gpios[index//2].channel2
            channel.write(int(value)&0xffffffff,0xffffffff)
        self.source[:]=hardware_packet(fixed,moving)
        # Arm receive first: output contains N+4 words, not N.
        self.dma.recvchannel.transfer(self.destination)
        self.dma.sendchannel.transfer(self.source)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        words=np.array(self.destination,copy=True)
        low,high,count,flags=map(int,words[-4:])
        if flags:
            raise RuntimeError(f'PL input protocol flags: {flags:#x}')
        pixels=words[:-4].reshape(self.shape)
        valid=((pixels>>16)&1)!=0
        if count!=int(valid.sum()) or not np.array_equal(((pixels>>8)&255).astype(np.uint8),fixed):
            raise RuntimeError('PL return geometry/packet mismatch')
        ssd=low+(high<<32)
        return {'warped':(pixels&255).astype(np.uint8),'valid':valid,'ssd':ssd,'count':count,
                'mse':ssd/count if count else float('inf')}

    def register(self,fixed,moving,min_overlap=.65,**search_options):
        def evaluator(matrix):
            result=self.evaluate(fixed,moving,matrix)
            if result['count'] < fixed.size*min_overlap:
                return float('inf')
            return result['mse']
        # PS searches parameters; each candidate warping/interpolation/SSD is PL.
        # This version retransmits both images per candidate; measure overhead.
        return register_rigid(fixed,moving,min_overlap=min_overlap,evaluate=evaluator,**search_options)
