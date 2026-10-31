"""Add genuinely new shell features at cached candidate points, development only.

Never runs a CNN, LoG/Hessian detector or full-volume geometric resampling.
The six reused development checks are explicitly excluded from this preparation.
"""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from lung_candidate_cached import load,ROOT
from lung_candidate_data import read_split,load_development_case,write_json,sha_file
from lung_candidate_rules_run import SPLIT
from lung_candidate_shell import measure,FEATURE_NAMES


def run(out):
    if out.exists():raise FileExistsError('Preserve completed features')
    rows,folds,protected=load();split=read_split(SPLIT);out.mkdir(parents=True)
    for path in (Path(__file__),ROOT/'python/lung_candidate_shell.py'):
        protected[str(path)]=sha_file(path)
    protocol=dict(status='RUNNING',feature_names=FEATURE_NAMES,geometry='native anisotropic HU; spacing in mm',
        local_only=True,protected_files_sha256=protected,heldout_scans_run=0,
        network_forwards=0,log_or_hessian_extractions=0,cases=[])
    write_json(out/'protocol.json',protocol)
    for i,row in enumerate(rows):
        start=perf_counter();hu,geom=load_development_case(split,row['record'])
        matrix=np.array([measure(hu,c['center_index_zyx'],geom['spacing_zyx_mm'],np.expm1(x[3]))
            for c,x in zip(row['packet']['candidates'],row['features'])],np.float32).reshape(-1,len(FEATURE_NAMES))
        uid=row['uid'];path=out/(uid+'.npz')
        np.savez_compressed(path,features=matrix,
            candidate_ids=np.array([c['candidate_id'] for c in row['packet']['candidates']]))
        protocol['cases'].append(dict(series_uid=uid,candidates=len(matrix),sha256=sha_file(path),
            elapsed_ms=(perf_counter()-start)*1000))
        write_json(out/'protocol.json',protocol)
        print(i+1,'/24',uid,'candidate patches',len(matrix),'seconds',round((perf_counter()-start),2),flush=True)
        del hu
    for path,digest in protected.items():
        if sha_file(path)!=digest:raise ValueError('Protected input/source changed')
    protocol['status']='COMPLETE';write_json(out/'protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir)
