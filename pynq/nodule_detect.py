"""Full candidate extraction after supplied lung segmentation, no web/Torch.

Input: HU.npy (int16 ZYX), lung_mask.npy (bool ZYX), geometry.json (LPS).
Both images are memory mapped. The original CT and its predicted mask must
have identical native shape; neither reference annotations nor U-Net is read.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import platform
import time
import numpy as np
from nodule_plan import load_package,sha
from nodule_benchmark import hardware,save
from lung_nodule_cnn import config_from_dict,search_mask,tiles,decode,consolidate
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_continuity import rerank_scores

THRESHOLD=0.6418673229668569


class PlaneCache:
    def __init__(self,hu):self.hu=hu;self.planes=OrderedDict()
    def __len__(self):return len(self.hu)
    def __getitem__(self,i):
        i=int(i)
        if i not in self.planes:self.planes[i]=np.asarray(self.hu[i],np.float32)
        self.planes.move_to_end(i)
        if len(self.planes)>16:self.planes.popitem(last=False)
        return self.planes[i]


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2')
    folder,manifest,model,params=load_package(Path(__file__).resolve().parent)
    source=Path(args.input).resolve();receipt=json.loads((source/'case.json').read_text(encoding='utf-8'))
    if set(receipt['files_sha256'])!={'hu.npy','lung_mask.npy','geometry.json'}:
        raise ValueError('Need complete fixed input receipt')
    for n,h in receipt['files_sha256'].items():
        p=(source/n).resolve()
        if source not in p.parents or sha(p)!=h:raise ValueError('Input receipt mismatch: '+n)
    hu=np.load(source/'hu.npy',mmap_mode='r',allow_pickle=False)
    lung=np.load(source/'lung_mask.npy',mmap_mode='r',allow_pickle=False)
    geometry=json.loads((source/'geometry.json').read_text(encoding='utf-8'))
    spacing=np.asarray(geometry['spacing_zyx_mm'],float)
    if (hu.dtype!=np.int16 or lung.dtype!=np.bool_ or hu.ndim!=3 or hu.shape!=lung.shape
            or len(hu)<2 or spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any()
            or not receipt.get('predicted_lung_mask',False)):
        raise ValueError('Need same-grid full HU and predicted lung mask, with valid spacing')
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('Choose fresh output directory')
    out.mkdir();net,overlay,clocks,io=hardware(folder,model,params,args.timeout)
    cfg=config_from_dict(model['detector_config']);points=[[],[],[]];cache=PlaneCache(hu)
    start=time.perf_counter();timing=dict(search_ms=0.,sample_ms=0.,pl_patch_ms=0.,decode_ms=0.,post_ms=0.)
    try:
        tick=time.perf_counter();search=search_mask(lung,spacing,cfg);jobs,schedule=tiles(search,spacing,cfg)
        timing['search_ms']=(time.perf_counter()-tick)*1000
        pl_cycles=0
        for i,job in enumerate(jobs):
            tick=time.perf_counter();patch=sample_patch_reuse(cache,spacing,job,cfg)
            timing['sample_ms']+=(time.perf_counter()-tick)*1000
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            tick=time.perf_counter();pred,stats=net.predict_hu(patch)
            timing['pl_patch_ms']+=(time.perf_counter()-tick)*1000;pl_cycles+=stats['pl_command_cycles']
            tick=time.perf_counter()
            for dest,a in zip(points,decode(pred[None],[job],geometry,cfg,search)):dest.append(a)
            timing['decode_ms']+=(time.perf_counter()-tick)*1000
            if i%50==0 or i+1==len(jobs):print('PL tiles {}/{}'.format(i+1,len(jobs)),flush=True)
        tick=time.perf_counter()
        candidates=consolidate(*(np.concatenate(a) for a in points),geometry,cfg) if jobs else []
        for c in candidates:
            box=np.asarray(c['bbox_zyx']).reshape(3,2);box[:,0]=np.maximum(0,box[:,0]);box[:,1]=np.minimum(hu.shape,box[:,1]);c['bbox_zyx']=box.ravel().tolist()
        features,_=evidence_for_packet(hu,spacing,candidates)
        cross=rerank_scores(np.asarray([c['score'] for c in candidates]),features,
            json.loads((folder/'nodule_ranker.json').read_text(encoding='utf-8')))
        selected=[]
        for i in np.flatnonzero(cross>=THRESHOLD):
            c=dict(candidates[i]);c['raw_score']=c['score'];c['score']=float(cross[i]);selected.append(c)
        selected.sort(key=lambda c:-c['score']);timing['post_ms']=(time.perf_counter()-tick)*1000
        timing['whole_candidate_ms']=(time.perf_counter()-start)*1000
        timing['pl_active_cycles_ms']=pl_cycles/100000.
        if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed before completion')
        save(out/'prediction.json',dict(status='complete_experimental_pl_int8',backend='PYNQ_PL_INT8',
            candidates=selected,candidate_count=len(selected),raw_candidate_count=len(candidates),
            threshold=THRESHOLD,schedule=schedule,timing=timing,cma_resident_bytes=net.cma_bytes,
            input_receipt_sha256=sha(source/'case.json'),manifest_sha256=sha(folder/'nodule_manifest.json'),
            lung_mask_reused=True,lung_inference_rerun=False,annotations_used=False,
            scope='Includes ARM search/sampling/decode/3D scoring and all PL patches. Excludes prior lung segmentation, input transfer/hash validation, bit download and display.',
            warning='INT8 candidate quality not yet accepted; historical FP32 sensitivity does not apply automatically.'))
        print('Saved:',out/'prediction.json',timing,flush=True)
    finally:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    p.add_argument('--out',required=True);p.add_argument('--timeout',type=float,default=30.);run(p.parse_args())
