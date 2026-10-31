"""Restore only the exact nine-plane arrays retired by the 2026-10-20 cleanup.

Leaves frozen experiment metadata, model weights and original sources unchanged.
Requires the retained original five-plane arrays and the original CT ZIP files.
"""
from collections import defaultdict
import gc
import os
from pathlib import Path
import numpy as np

from lung_candidate_data import sha_file
from lung_nodule_cnn_balanced import read
from lung_nodule_context_comparison import (OUT, TRAIN, ALL_CONTEXT, BASE_CONTEXT,
    sample_context, config_from_dict, FloatPlaneCache)
from luna16_multibundle import read_manifest, OUT as DATA, records_for, load_development_volume


def restore():
    plan=read(OUT/'plan.json')
    if sha_file(OUT/'plan.json')!=(OUT/'plan.sha256').read_text().strip():
        raise ValueError('Original context plan changed')
    manifest=read_manifest(DATA/'manifest.json')
    records={r['series_uid']:r for r in records_for(manifest,'train')}
    old=read(TRAIN/'training_catalogue.json');cat=read(OUT/'training_catalogue.json')
    grouped=defaultdict(list)
    for item in old['items']:grouped[item['series_uid']].append(item)
    cfg=config_from_dict(plan['config'])
    extra=tuple(c for c in ALL_CONTEXT if c not in BASE_CONTEXT)
    for count,(uid,items) in enumerate(grouped.items(),1):
        path=OUT/'patches'/(uid+'.npy')
        expected=cat['cache_sha256'][str(path)]
        if path.exists():
            if sha_file(path)!=expected:raise ValueError('Existing cache differs')
            continue
        hu,g=load_development_volume(manifest,uid,'train')
        cache=FloatPlaneCache(hu);spacing=np.asarray(g['spacing_zyx_mm'])
        maps={p:np.load(p,mmap_mode='r',allow_pickle=False) for p in {i['image_path'] for i in items}}
        temp=path.with_suffix('.tmp.npy')
        images=np.lib.format.open_memmap(temp,mode='w+',dtype=np.int16,
            shape=(len(items),len(ALL_CONTEXT),cfg.patch,cfg.patch))
        for j,item in enumerate(items):
            original=maps[item['image_path']][item['patch_index']]
            added=sample_context(cache,spacing,item['label']['job'],cfg,extra)
            for k,offset in enumerate(ALL_CONTEXT):
                images[j,k]=original[BASE_CONTEXT.index(offset)] if offset in BASE_CONTEXT else added[extra.index(offset)]
        images.flush();del images,maps,cache,hu
        if sha_file(temp)!=expected:
            raise ValueError('Reconstructed array does not match frozen original')
        os.replace(temp,path);gc.collect()
        print('RESTORED',count,'/',len(grouped),uid,flush=True)


if __name__=='__main__':restore()
