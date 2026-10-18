"""Rebuild retired test arrays from immutable sources; require original cache hashes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from lung_external import align_pair,archive_members,normalize_hu,read_member,require_hu
from lung_prune import WORKSPACE
from lung_segmentation import fresh_output,write_json
from lung_unet import _load_source,file_sha256
from verify_lung_axil import check_hashes


def run(outdir):
    held=WORKSPACE/'data/lung_training_20261011/heldout_2d_evaluation'
    external=WORKSPACE/'data/lung_external_20261012/prepared'
    recovery=WORKSPACE/'data/lung_prune_recover_20261017'
    old_held=json.loads((held/'test_dataset.json').read_text(encoding='utf-8'))
    old_external=json.loads((external/'external_dataset.json').read_text(encoding='utf-8'))
    old_protocol=json.loads((external/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((recovery/'freeze.json').read_text(encoding='utf-8'));check_hashes(frozen['files_sha256'])
    for key in ['ct','masks']:
        if file_sha256(old_protocol['source_paths'][key])!=old_protocol['source_sha256'][key]:raise ValueError('Original archive changed')
    output=fresh_output(outdir).resolve()
    if shutil.disk_usage(output).free<8*1024**3:raise ValueError('Need at least 8 GiB free to reconstruct test arrays')
    tracked={str(p):file_sha256(p) for p in [held/'test_dataset.json',external/'external_dataset.json',
        external/'protocol.json',recovery/'freeze.json',Path(__file__).resolve()]}
    for key in ['ct','masks']:tracked[old_protocol['source_paths'][key]]=old_protocol['source_sha256'][key]
    sources=[WORKSPACE/'python'/name for name in ['lung_unet.py','lung_external.py','lungseg/dicom_io.py']]
    tracked.update({str(p):file_sha256(p) for p in sources})
    expected=[dict(r,cohort='lctsc_heldout') for r in old_held['cases']]+[
        dict(r,cohort='covid_external') for r in old_external['cases'] if r['status']=='ready_primary_hu']
    if len(expected)!=27:raise ValueError('Need original 27-case cohort')
    write_json(output/'protocol.json',dict(status='frozen_reconstruction_protocol',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        policy='All original 27 cases, no inference/selection/training. Reconstructed images/reference must match historical SHA256 byte for byte.',
        input_sha256=tracked,candidate_freeze_sha256=file_sha256(recovery/'freeze.json'),
        expected=[dict(case=r['case'],images_sha256=r['images_sha256'],reference_sha256=r['reference_sha256']) for r in expected]))
    records=[]
    try:
        with zipfile.ZipFile(old_protocol['source_paths']['ct']) as ct_zip,zipfile.ZipFile(old_protocol['source_paths']['masks']) as masks_zip:
            ct_members=archive_members(ct_zip,ct=True);mask_members=archive_members(masks_zip)
            for number,record in enumerate(expected,1):
                name=record['case'];print('Reconstruct',number,'/27',name,flush=True)
                if record['cohort']=='lctsc_heldout':
                    hu,reference,spacing,files,patient,frame,_=_load_source(name,Path('D:/CT2/lctsc')/name)
                    if files!=record['source_ct_files_sorted'] or patient!=record['patient_key'] or frame!=record['frame_uid']:
                        raise ValueError('LCTSC source identity or physical order changed')
                    if not np.allclose(spacing,record['spacing_zyx_mm'],rtol=0,atol=1e-5):raise ValueError('LCTSC grid changed')
                else:
                    ct,ct_hash=read_member(ct_zip,ct_members[name]);label,mask_hash=read_member(masks_zip,mask_members[name])
                    if ct_hash!=record['ct_member_sha256'] or mask_hash!=record['mask_member_sha256']:raise ValueError('External archive member changed')
                    hu,reference,metadata=align_pair(ct,label);require_hu(name,hu,metadata['ct_geometry']['source_dtype'])
                    spacing=metadata['reference_geometry']['spacing_zyx'];del ct,label
                if list(hu.shape)!=record['native_shape']:raise ValueError('Native shape differs')
                folder=output/'cache'/name;folder.mkdir(parents=True)
                images=normalize_hu(hu,256);np.save(folder/'images.npy',images)
                np.save(folder/'reference_native.npy',reference.astype(np.uint8))
                for filename,key in [('images.npy','images_sha256'),('reference_native.npy','reference_sha256')]:
                    if file_sha256(folder/filename)!=record[key]:raise ValueError('Historical hash mismatch: '+name+'/'+filename)
                int16_lossless=hu.min()>=-32768 and hu.max()<=32767 and np.equal(hu,np.rint(hu)).all()
                np.save(folder/'hu_native.npy',hu.astype(np.int16) if int16_lossless else hu)
                hu_hash=file_sha256(folder/'hu_native.npy')
                if record['cohort']=='covid_external' and hu_hash!=record['hu_cache_sha256']:raise ValueError('Historical external HU mismatch')
                item=dict(record,cache_root=str(output),cache_folder=str(folder.relative_to(output)),
                    spacing_zyx_mm=np.asarray(spacing).tolist(),hu_cache_sha256=hu_hash)
                write_json(folder/'case.json',item);records.append(item)
                write_json(output/'dataset.json',dict(status='reconstructing',cases=records,expected_cases=27))
                print('  Original normalized input and expert mask SHA256 MATCH;',len(hu),'slices',flush=True)
                del hu,reference,images
        check_hashes(tracked);check_hashes(frozen['files_sha256'])
        write_json(output/'dataset.json',dict(status='complete_hash_matched_reconstruction',cases=records,
            expected_cases=27,slices=sum(r['slices'] for r in records),historical_input_and_reference_byte_exact=True))
        write_json(output/'verification.json',dict(status='complete_hash_matched_reconstruction',cases=27,
            slices=sum(r['slices'] for r in records),source_sha256=tracked,
            arrays_sha256={str(p):file_sha256(p) for p in output.glob('cache/*/*.npy')},
            candidate_unchanged=True,inference_performed=False))
        print('All 27 original test input/reference caches restored byte-exactly',flush=True)
    except Exception as error:
        write_json(output/'failure.json',dict(status='incomplete_reconstruction',error=str(error),completed_cases=len(records)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/lung_prune_test_cache_20261017')
    run(p.parse_args().outdir)
