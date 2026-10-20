"""Tiny residual CT denoiser: Conv3x3(1,C), ReLU, Conv3x3(C,C), ReLU,
Conv3x3(C,1), output=input-predicted_noise. HU is scaled by 1024, never clipped.
No pooling, BatchNorm, sigmoid or full-resolution resampling. No PL claim.
"""
import torch
from torch import nn

HU_SCALE = 1024.0
HALO = 3  # three radius-1 convolutions; effective receptive field 7x7


class TinyDenoiser(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        if channels not in (4, 8, 16):
            raise ValueError('Use 4, 8 or 16 channels')
        self.channels = channels
        self.layers = nn.ModuleList([
            nn.Conv2d(1, channels, 3, padding=1, padding_mode='replicate'),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.Conv2d(channels, 1, 3, padding=1, padding_mode='replicate')])
        # A new model starts at identity, not at an arbitrary HU mapping.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        z = torch.relu(self.layers[0](x))
        z = torch.relu(self.layers[1](z))
        return x-self.layers[2](z)


def select_device(name='auto'):
    if name == 'auto': name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install matching PyTorch or select cpu')
    return torch.device(name)


@torch.inference_mode()
def infer_hu(model, image, device, tile=128):
    """Tiled inference with sufficient halo; identical to full inference to float tolerance.

    Global image edges are padded by each convolution, exactly as during training.
    Internal tile edges are cropped away. No stitching average or seams intended.
    """
    import numpy as np
    a = np.asarray(image,dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all(): raise ValueError('Finite 2D HU required')
    if tile < 8: raise ValueError('tile >= 8 required')
    h,w = a.shape
    out = np.empty_like(a)
    model.eval()
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            y2,x2 = min(y+tile,h),min(x+tile,w)
            y0,x0 = max(0,y-HALO),max(0,x-HALO)
            ye,xe = min(h,y2+HALO),min(w,x2+HALO)
            patch = np.ascontiguousarray(a[y0:ye,x0:xe]/HU_SCALE)
            t = torch.from_numpy(patch)[None,None].to(device)
            q = model(t)[0,0].cpu().numpy()*HU_SCALE
            out[y:y2,x:x2] = q[y-y0:y2-y0,x-x0:x2-x0]
    return out


@torch.inference_mode()
def infer_hu_full(model, image, device):
    """Opt-in PC full-frame FP32 inference; same model, scaling and edge padding.

    Requires memory for full intermediate features. Existing tiled inference is
    retained for low-memory use and reproducibility. This is not a PL backend.
    """
    import numpy as np
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('Finite nonempty 2D HU required')
    model.eval()
    t = torch.from_numpy(np.ascontiguousarray(a/HU_SCALE))[None,None].to(device)
    return model(t)[0,0].cpu().numpy()*HU_SCALE
