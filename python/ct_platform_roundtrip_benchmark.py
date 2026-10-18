"""Measure actual PC<->board latency, checking every output against numpy.

No throughput inferred from DMA alone. Explicit package/model identities are
checked before any operation. Run when the shared board is otherwise idle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ct_basic_processing import process,parameters
from ct_platform_client import PlatformClient
from lung_live_data import read_ct
from lung_live_web import MODEL_SHA


def run(config,ct,z,out,repeats):
    config=Path(config).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve prior benchmark evidence')
    conf=json.loads(config.read_text(encoding='utf-8'))
    if hashlib.sha256((config.parent/conf['model']).read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError('Model differs')
    client=PlatformClient(conf['board_url'],(config.parent/conf['token_file']).read_text(encoding='utf-8').strip(),
        json.loads((config.parent/conf['platform_manifest']).read_text(encoding='utf-8')))
    identity=client.health()
    hu,geom,details=read_ct(ct,confirm_hu=True,slices=str(z));hu=hu[0]
    if hu.shape!=(512,512):raise ValueError('Native 512x512 input required')
    records=[];actual={}
    # Include same-mode repeats and a six-mode sequence on exactly the same HU.
    operations=['window']*repeats+list(client.manifest['basic'])*repeats
    for i,op in enumerate(operations):
        p=parameters(dict(operation=op));reference=process(hu,p)
        image,receipt=client.basic(hu,p)
        if not np.array_equal(image,reference):raise AssertionError('Actual PL mismatch: '+op)
        key=f'{i:03d}_{op}';actual[key]=image
        records.append(dict(key=key,operation=op,parameters=p,reference_bit_exact=True,receipt=receipt))
        print(op,'roundtrip_ms',round(receipt['roundtrip_ms'],3),'DMA_ms',round(receipt['stats']['dma_ms'],3),
              'switch_ms',round(receipt['stats']['overlay_switch_ms'],3),'cache_hit',receipt.get('transport',{}).get('input_cache_hit'),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    arrays=out.with_suffix('.npz')
    if arrays.exists():raise FileExistsError(arrays)
    with arrays.open('xb') as f:np.savez_compressed(f,hu=hu,**actual)
    report=dict(schema='ct_basic_actual_roundtrip_benchmark_v1',actual_board_execution=True,
        config_manifest_sha256=client.manifest['platform_manifest_sha256'],identity=identity,
        source=details,repeats=repeats,records=records,actual_arrays=arrays.name,
        actual_arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        note='真实电脑到板端往返；每次实际执行PL并校验；首次bit加载及冷缓存单列，不代表网页点击至显示或临床性能')
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('Saved actual roundtrip results:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ct',required=True)
    p.add_argument('--slice',type=int,default=44);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.repeats<=20:p.error('repeats must be 1..20')
    run(a.config,a.ct,a.slice,a.out,a.repeats)
