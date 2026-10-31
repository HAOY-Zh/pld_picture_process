"""Measure one output plane with CT, lung mask and weights already resident."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import ndimage as ndi
from lung_candidate_data import load_development_case,write_json,sha_file
from lung_candidate_rules_run import ROOT
from lung_nodule_cnn_experiment import verify,records_for,load_mask
from lung_nodule_cnn import config_from_dict,model,tiles,sample_patch,decode,consolidate


def plane_envelope(lung,spacing,z,margin):
    """Exact plane of the reference fill-then-rectangular-3D-dilation envelope."""
    radius=np.ceil(margin/spacing).astype(int)
    plane=np.zeros(lung.shape[1:],bool)
    for index in range(max(0,z-radius[0]),min(len(lung),z+radius[0]+1)):
        plane |= ndi.binary_fill_holes(lung[index])
    return ndi.maximum_filter(plane,size=tuple(2*radius[1:]+1),mode='constant',cval=0)


def main():
    import torch
    out=ROOT/'data/lung_nodule_cnn_20261019_v1';plan=verify(out);cfg=config_from_dict(plan['config'])
    destination=out/'single_slice_benchmark.json'
    if destination.exists():raise FileExistsError('Preserve first measurement')
    split,records=records_for(plan,'calibration_scans');record=records[0]
    tick=perf_counter();hu,geometry=load_development_case(split,record);lung=load_mask(record,split)
    load_ms=(perf_counter()-tick)*1000;spacing=np.asarray(geometry['spacing_zyx_mm'])
    # Reference-free, predeclared busy slice: largest predicted lung area.
    z=int(np.argmax(lung.sum(axis=(1,2))));zm=round(z*spacing[0]/cfg.z_step_mm)*cfg.z_step_mm
    z=int(np.clip(round(zm/spacing[0]),0,len(lung)-1))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    tick=perf_counter();net=model(cfg).cuda().eval();checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    frozen=json.loads((out/'model_freeze.json').read_text('utf-8'))
    if sha_file(out/'model.pt')!=frozen['model_sha256']:raise ValueError('Model changed')
    net.load_state_dict(checkpoint['state'])
    with torch.inference_mode():net(torch.zeros(1,5,cfg.patch,cfg.patch,device='cuda'))
    torch.cuda.synchronize();initialize_ms=(perf_counter()-tick)*1000
    runs=[]
    with torch.inference_mode():
        for _ in range(3):
            start=perf_counter();plane=plane_envelope(lung,spacing,z,cfg.margin_mm)
            jobs,_=tiles(plane[None],spacing,cfg);jobs=[(zm,y,x) for _,y,x in jobs];scheduled=perf_counter()
            image=np.stack([sample_patch(hu,spacing,j,cfg) for j in jobs]);sampled=perf_counter()
            tensor=torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
            prediction=net(tensor).cpu().numpy();network_done=perf_counter()
            points,scores,diameters=decode(prediction,jobs,geometry,cfg)
            if len(points):
                q=np.rint(points).astype(int);valid=(q>=0).all(1)&(q<hu.shape).all(1)
                valid &= plane[np.clip(q[:,1],0,plane.shape[0]-1),np.clip(q[:,2],0,plane.shape[1]-1)]
                points,scores,diameters=points[valid],scores[valid],diameters[valid]
            candidates=consolidate(points,scores,diameters,geometry,cfg)
            end=perf_counter()
            runs.append(dict(schedule_ms=(scheduled-start)*1000,sampling_ms=(sampled-scheduled)*1000,
                network_and_transfer_ms=(network_done-sampled)*1000,decode_and_merge_ms=(end-network_done)*1000,
                total_ms=(end-start)*1000,tiles=len(jobs),candidate_count_at_decode_floor=len(candidates)))
    report=dict(status='MEASURED_WARM_SINGLE_OUTPUT_PLANE',series_uid=record['series_uid'],source_shape_zyx=list(hu.shape),
        native_z=z,target_z_mm=zm,selection='largest predicted lung area; no reference coordinates',
        backend='FP32_CUDA_REFERENCE_NOT_FPGA',scope='CT, cached FP32 lung prediction and model resident; includes per-plane ROI, sampling, network+transfer, decode+merge',
        excluded='CT loading, model initialization, lung segmentation inference, web/network/rendering and cross-slice merging',
        initial_load_ms=load_ms,model_init_and_warmup_ms=initialize_ms,runs=runs,
        median_ms={k:float(np.median([r[k] for r in runs])) for k in runs[0] if k.endswith('_ms')},
        model_sha256=frozen['model_sha256'],heldout_scans_run=0,notes='2.5D needs adjacent slices; selected loaded case only. Not guaranteed latency for every slice or FPGA.')
    write_json(destination,report);print(json.dumps(report['median_ms']))


if __name__=='__main__':main()
