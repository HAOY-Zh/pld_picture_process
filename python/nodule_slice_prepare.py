"""Prepare one real complete CT detection plane, without neural inference.

Select the largest predicted-lung native plane, then the nearest existing
detector Z plane. No annotations enter selection or sampling. A compact HU
slab preserves exactly the original five-context samples for that plane.
"""
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from luna16_multibundle import read_manifest,load_development_volume
from lung_nodule_cnn import config_from_dict,search_mask,tiles,sample_patch

UID='1.3.6.1.4.1.14519.5.2.1.6279.6001.203425588524695836343069893813'


def prepare(out):
    out=Path(out).resolve();out.mkdir();pack=out/'nodule_slice_timing';pack.mkdir()
    package=ROOT/'data/nodule_board_20261023_v1/nodule_board'
    _,manifest,model,_=load_package(package);cfg=config_from_dict(model['detector_config'])
    data=read_manifest(ROOT/'data/luna16_expansion_20261021_v2/manifest.json')
    hu,g=load_development_volume(data,UID,'calibration')
    mask_path=ROOT/'data/lung_nodule_context_20261020_v1/calibration_masks'/(UID+'.npz')
    with np.load(mask_path,allow_pickle=False) as z:
        lung=np.unpackbits(z['body'],count=int(np.prod(z['shape_zyx']))).reshape(z['shape_zyx']).astype(bool)
    if lung.shape!=hu.shape:raise ValueError('Mask shape mismatch')
    # Verify this is the exact predicted mask already used in the saved run.
    import gzip,hashlib
    old=ROOT/'data/lung_nodule_five_bundle_20261021_v1/evaluation_cases'/UID/'expanded.json.gz'
    with gzip.open(old,'rt',encoding='utf-8') as f:old=json.load(f)
    if hashlib.sha256(lung.tobytes()).hexdigest()!=old['lung_mask_sha256']:raise ValueError('Mask identity mismatch')
    spacing=np.asarray(g['spacing_zyx_mm']);search=search_mask(lung,spacing,cfg);jobs,schedule=tiles(search,spacing,cfg)
    target=int(lung.sum((1,2)).argmax());planes=sorted({j[0] for j in jobs})
    center=min(planes,key=lambda z:abs(z-target*spacing[0]));chosen=[j for j in jobs if j[0]==center]
    lo=max(0,int(np.floor((center+min(cfg.context_mm))/spacing[0])))
    hi=min(len(hu),int(np.ceil((center+max(cfg.context_mm))/spacing[0]))+1)
    local_jobs=[(j[0]-lo*spacing[0],j[1],j[2]) for j in chosen]
    slab=np.array(hu[lo:hi],copy=True);gate=np.array(search[lo:hi],copy=True)
    for oldjob,newjob in zip(chosen,local_jobs):
        np.testing.assert_array_equal(sample_patch(hu,spacing,oldjob,cfg),sample_patch(slab,spacing,newjob,cfg))
    geometry=dict(g);geometry['origin_lps_mm']=(np.asarray(g['origin_lps_mm'])+np.asarray(g['basis_lps_zyx'])@np.array([lo,0,0])).tolist()
    nearest=min(len(hu)-1,int(round(center/spacing[0])))
    plane=dict(series_uid=UID,original_ct_shape_zyx=list(hu.shape),xy_shape=list(hu.shape[1:]),
        native_index_nearest=nearest,native_slice_number_one_based=nearest+1,
        detector_z_from_volume_start_mm=float(center),offset_from_nearest_native_mm=float(center-nearest*spacing[0]),
        selected_tiles=len(chosen),context_planes=5,context_offsets_mm=list(cfg.context_mm),
        detector_step_mm=cfg.z_step_mm,native_spacing_zyx_mm=spacing.tolist(),
        selection='Largest predicted-lung native area; nearest existing detector Z plane; no reference annotation',
        context_native_range_half_open=[lo,hi],image_plane_is_full_field_of_view=True)
    np.savez_compressed(pack/'slice_case.npz',hu=slab,search=gate,jobs=np.asarray(local_jobs,float))
    meta=dict(schema='nodule_complete_plane_input_v1',plane=plane,local_geometry=geometry,
        package_manifest_sha256=sha(package/'nodule_manifest.json'),payload_sha256=sha(pack/'slice_case.npz'),
        mask_sha256=sha(mask_path),sampling_equal_all_tiles=True,model_inference_rerun=False,annotations_used=False)
    (pack/'slice_case.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'pynq/nodule_slice_benchmark.py',pack/'nodule_slice_benchmark.py')
    (pack/'README.txt').write_text('Uses the existing /home/xilinx/ct_test/nodule_board package.\nRun nodule_slice_benchmark.py --runs 3 --warmup 1 --out /home/xilinx/ct_test/nodule_slice_speed_01\nComplete plane with five-layer context; NOT a whole volume. Search mask/schedule are precomputed; cross-layer scoring is excluded. See slice_case.json for exact native/detector plane positions.\n',encoding='utf-8')
    with zipfile.ZipFile(out/'nodule_slice_timing.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in pack.iterdir():z.write(f,f.relative_to(out))
    (out/'preparation.json').write_text(json.dumps(dict(status='prepared_no_inference',plane=plane,
        sampling_equal_all_tiles=True,source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'pynq/nodule_slice_benchmark.py']},
        files_sha256={f.name:sha(f) for f in pack.iterdir()},archive_sha256=sha(out/'nodule_slice_timing.zip')),indent=2),encoding='utf-8')
    print(json.dumps(plane,indent=2),flush=True);print('Prepared:',pack,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);prepare(p.parse_args().out)
