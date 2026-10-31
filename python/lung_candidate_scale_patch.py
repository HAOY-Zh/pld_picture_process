"""Complement the broad pool with independent 2-mm-scale peaks, development only.

The original max-over-scales image can suppress a broad nodule peak near a narrow
vessel response. This experiment retains one fixed scale across ALL 24 scans,
not reference-directed rescue points. Original v3 and original feature tables
remain untouched. Only compact numeric features are stored.
"""
import argparse
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import hashlib
import numpy as np

from lung_candidate_data import read_split,load_development_case,sha_file,write_json
from lung_candidates import CandidateConfig,detect_candidates
from lung_candidate_structure import hessian_features
from lung_candidate_rules import measure_pool
from lung_candidate_rules_run import SPLIT,ARCHIVE,BASELINES,expected_records,frozen_sources


def unique_complement(candidates,original_table,sigma=2.):
    existing={tuple(p) for p,s in zip(original_table['original_centers_zyx'],original_table['features'][:,5])
              if np.isclose(np.expm1(s),sigma,atol=1e-5)}
    return [dict(c,candidate_id='scale2_'+c['candidate_id']) for c in candidates
            if tuple(c['center_index_zyx']) not in existing]


def augment(features_dir,out):
    if out.exists():raise ValueError('Preserve previous feature experiments')
    split,records=expected_records();by_uid={r['series_uid']:r for r in records}
    old=json.loads((features_dir/'protocol.json').read_text(encoding='utf-8'))
    if old['status']!='COMPLETE' or set(old['selected_uids'])!=set(by_uid):raise ValueError('Need original 24 completed scans')
    frozen=frozen_sources();frozen[str(Path(__file__).resolve())]=sha_file(__file__)
    out.mkdir(parents=True)
    protocol=dict(old,status='RUNNING',cases=[],frozen_sources=frozen,
        parent_feature_protocol_sha256=sha_file(features_dir/'protocol.json'),complementary_sigma_mm=2.,
        scope='24 previously viewed development scans; uniform extra 2-mm-scale peak proposals',
        heldout_scans_run=0)
    write_json(out/'protocol.json',protocol)
    with ZipFile(ARCHIVE) as archive:
        for i,meta in enumerate(old['cases'],1):
            uid=meta['series_uid'];r=by_uid[uid];path=features_dir/(uid+'.npz')
            if sha_file(path)!=meta['feature_sha256']:raise ValueError('Source feature bytes changed')
            with np.load(path,allow_pickle=False) as p:table={k:p[k] for k in p.files}
            hu,geom=load_development_case(split,r)
            if uid in split['pilot_development_uids']:
                packed=np.load(BytesIO(archive.read(f'data/lung_candidates_pilot_20261018_v1/{uid}/masks_packed.npz')))
            else:packed=np.load(BASELINES[1]/uid/'masks_packed.npz')
            body=np.unpackbits(packed['lung_body'],count=hu.size).reshape(hu.shape).astype(bool);packed.close()
            if hashlib.sha256(body.tobytes()).hexdigest()!=meta['lung_mask_sha256']:raise ValueError('Lung mask changed')
            print(f'[{i}/24] complementary scale2 {uid}',flush=True);start=perf_counter()
            result,seed=detect_candidates(hu,body,geom,replace(CandidateConfig(),sigma_mm=(2.,)));del seed
            extra=unique_complement(result['candidates'],table);del result
            features=hessian_features(hu,extra,geom['spacing_zyx_mm'])
            added=measure_pool(hu,extra,features,geom)
            added['source_indices']+=meta['coarse_count']
            joined={k:np.concatenate([table[k],added[k]],axis=0) for k in table}
            np.savez_compressed(out/(uid+'.npz'),**joined)
            updated=dict(meta,pool_count=len(joined['features']),complementary_sigma_mm=2.,
                original_pool_count=len(table['features']),complementary_pool_count=len(added['features']),
                complementary_ms=(perf_counter()-start)*1000,feature_sha256=sha_file(out/(uid+'.npz')))
            write_json(out/(uid+'.json'),updated);protocol['cases'].append(updated)
            write_json(out/'protocol.json',protocol)
            print('  added',len(added['features']),'total',len(joined['features']),flush=True)
            del hu,body,table,extra,features,added,joined
    for path,expected in frozen.items():
        if sha_file(path)!=expected:raise ValueError('Frozen code changed during complementary pass')
    protocol['status']='COMPLETE';write_json(out/'protocol.json',protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True);parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args();augment(args.features,args.outdir)
