"""Exact single-channel native packing; reuses the hash-bound compiler helper."""
import ctypes
from pathlib import Path
import numpy as np
from .nodule_quantized_pack import QuantizedPacker, compile_pack


class ExactPacker(QuantizedPacker):
    def __init__(self, library, mode):
        super().__init__(library)
        self.mode = mode
        array = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        self.lib.segmentation_pack_exact.argtypes = [array,ctypes.c_size_t,array,ctypes.c_size_t]
        self.lib.segmentation_pack_exact.restype = ctypes.c_int

    def pack(self,image,destination):
        if self.mode == 'nodule':
            return super().pack(image,destination)
        if (not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.shape!=(1,256,256)
                or not image.flags.c_contiguous):
            raise ValueError('Need unchanged single-channel U-Net input')
        out=np.asarray(destination)
        if (out.dtype!=np.dtype('uint64') or not out.flags.c_contiguous or not out.flags.writeable
                or np.shares_memory(image,out)):
            raise ValueError('Need separate writable CMA words')
        raw=out.view(np.uint8).reshape(-1)
        if self.lib.segmentation_pack_exact(image,image.size,raw,raw.nbytes):
            raise ValueError('Native segmentation packing rejected')


def open_packer(folder,mode):
    # Build inputs live in a separately manifest-bound directory, avoiding any
    # modification to the previously accepted native binary or its receipt.
    return ExactPacker(compile_pack(Path(folder)/'exact_io_native'),mode)
