"""Save ALL new/recovered empty predictions on expert-positive slices, plus previews."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_prune import BASELINE,DATASET
from lung_segmentation import write_json
from lung_unet import file_sha256,mapped_npy,read_manifest
from verify_lung_axil import check_hashes


def run(experiment,candidate):
    experiment=Path(experiment).resolve();destination=experiment/'small_slice_audit.json'
    if destination.exists():raise FileExistsError('Preserve small-slice audit')
    frozen=json.loads((experiment/'freeze.json').read_text());check_hashes(frozen['files_sha256'])
    cache,manifest=read_manifest(DATASET);rows=[];preview_rows=[]
    for record in [r for r in manifest['cases'] if r['split']=='validation']:
        with np.load(BASELINE/record['case']/'predictions_native.npz') as a:old=a['fixed_final']
        path=experiment/candidate/record['case']/'fixed_native.npz'
        with np.load(path) as a:new=a['final']
        with mapped_npy(cache/record['cache_folder']/'reference_native.npy') as reference:
            area=reference.sum((1,2));old_area=old.sum((1,2));new_area=new.sum((1,2))
            for kind,indices in [('newly_empty',(area>0)&(old_area>0)&(new_area==0)),
                                 ('recovered_empty',(area>0)&(old_area==0)&(new_area>0))]:
                for z in np.flatnonzero(indices):
                    row=dict(case=record['case'],slice_index=int(z),kind=kind,expert_pixels=int(area[z]),
                        baseline_prediction_pixels=int(old_area[z]),candidate_prediction_pixels=int(new_area[z]),
                        baseline_true_positive_pixels=int((old[z]&(reference[z]!=0)).sum()),
                        candidate_true_positive_pixels=int((new[z]&(reference[z]!=0)).sum()))
                    rows.append(row)
                    if kind=='newly_empty':preview_rows.append((record,int(z),np.array(reference[z]!=0),old[z].copy(),new[z].copy()))
    if preview_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pydicom
        figure,axes=plt.subplots(len(preview_rows),3,figsize=(12,4*len(preview_rows)),squeeze=False)
        for number,(record,z,reference,old,new) in enumerate(preview_rows):
            ds=pydicom.dcmread(record['source_ct_files_sorted'][z])
            hu=ds.pixel_array.astype(np.float32)*float(ds.RescaleSlope)+float(ds.RescaleIntercept)
            ys,xs=np.nonzero(reference);y0,y1=max(0,int(ys.min())-32),min(len(hu),int(ys.max())+33)
            x0,x1=max(0,int(xs.min())-32),min(hu.shape[1],int(xs.max())+33)
            for column,(predicted,title) in enumerate([(None,'Expert reference'),(old,'Original model'),(new,'Pruned candidate')]):
                ax=axes[number,column];ax.imshow(hu[y0:y1,x0:x1],cmap='gray',vmin=-1000,vmax=400)
                ax.contour(reference[y0:y1,x0:x1],levels=[.5],colors=['lime'],linewidths=1.)
                if predicted is not None and predicted[y0:y1,x0:x1].any():
                    ax.contour(predicted[y0:y1,x0:x1],levels=[.5],colors=['orange'],linewidths=1.)
                ax.set_title(record['case']+' z='+str(z)+'\n'+title);ax.axis('off')
        figure.suptitle('Post-hoc failure review: green=expert, orange=prediction; native CT crop')
        figure.tight_layout();figure.savefig(experiment/'newly_empty_review.png',dpi=160);plt.close(figure)
    write_json(destination,dict(status='complete_all_empty_positive_slice_changes',candidate=candidate,
        newly_empty_slices=sum(r['kind']=='newly_empty' for r in rows),
        recovered_empty_slices=sum(r['kind']=='recovered_empty' for r in rows),changes=rows,
        candidate_freeze_sha256=file_sha256(experiment/'freeze.json'),
        note='Post-hoc audit of all six validation volumes. Expert masks select review images only, never inference or timing inputs.',
        preview_sha256=file_sha256(experiment/'newly_empty_review.png') if preview_rows else None))
    print('All small-slice changes:',rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',default='data/lung_prune_recover_20261017')
    p.add_argument('--candidate',default='balanced_recovered');a=p.parse_args();run(a.experiment,a.candidate)
