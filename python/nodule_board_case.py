"""Prepare an explicit full-HU + predicted-mask packet for board-only detection.

No inference, annotations or test selection. Existing native-grid NPY images
and geometry JSON are required; this does not turn a display PNG into HU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def prepare(hu,mask,geometry,out):
    hu,mask,geometry=map(lambda p:Path(p).resolve(strict=True),[hu,mask,geometry]);out=Path(out).resolve()
    image=np.load(hu,mmap_mode='r',allow_pickle=False);lung=np.load(mask,mmap_mode='r',allow_pickle=False)
    g=json.loads(geometry.read_text(encoding='utf-8'));spacing=np.asarray(g['spacing_zyx_mm'],float)
    if (image.dtype!=np.int16 or lung.dtype!=np.bool_ or image.ndim!=3 or image.shape!=lung.shape or len(image)<2
            or spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any()
            or np.asarray(g['basis_lps_zyx']).shape!=(3,3) or np.asarray(g['origin_lps_mm']).shape!=(3,)):
        raise ValueError('Need original native-grid int16 HU, predicted boolean lung mask and LPS geometry')
    out.mkdir()  # Never overwrite an existing packet.
    for src,name in [(hu,'hu.npy'),(mask,'lung_mask.npy'),(geometry,'geometry.json')]:shutil.copy2(src,out/name)
    def digest(p):
        with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(schema='nodule_case_input_v1',shape_zyx=list(image.shape),predicted_lung_mask=True,
        note='Caller supplies a PREDICTED mask. Expert annotations must not be used as detector input.',
        files_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'case.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hu',required=True);p.add_argument('--lung-mask',required=True)
    p.add_argument('--geometry',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(prepare(a.hu,a.lung_mask,a.geometry,a.out))
