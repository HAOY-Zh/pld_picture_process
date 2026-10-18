"""Actual XSim probes of EVERY pruned convolution using its real new weights."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from lung_cache_vectors import generate as cache_vectors
from lung_flow_prepare import BIN, ROOT as FLOW, RTL, WORKSPACE,checked
from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_window_vectors import emit
from verify_lung_axil import check_hashes

sys.path.insert(0,str(WORKSPACE/'pynq'))


def run(package_root):
    checked(FLOW)
    package_root=Path(package_root).resolve();proof=json.loads((package_root/'verification.json').read_text())
    package=package_root/'lung_pruned'
    check_hashes({str(package/n):h for n,h in proof['package_files_sha256'].items()})
    check_hashes(proof['source_sha256'])
    model=json.loads((package/'model.json').read_text());root=package_root/'rtl_probes';root.mkdir()
    raw=root/'window_vectors';raw.mkdir();cases=[];rng=np.random.default_rng(20261017)
    from lung_continuous import unpack_words
    with np.load(package/'parameters.npz') as parameters,np.load(package/'reference.npz') as references,np.load(package/'input.npz') as image:
        shapes={n['name']:n['shape'] for n in model['nodes']}
        for node in model['nodes']:
            if node['kind']!='conv':continue
            row=node['layer'];prefix=row['prefix'];source=node['sources'][0]
            q=image['codes'] if source=='input' else unpack_words(references[source],shapes[source])
            y,x=q.shape[-2]//2,q.shape[-1]//2
            for tag,a in [('real_feature_crop',q[:,y-2:y+3,x-3:x+4].copy()),
                          ('random_full_range',rng.integers(0,256,(len(q),3,5),dtype=np.uint8))]:
                cases.append(dict(name=node['name']+'_'+tag,q=a,w=parameters[prefix+'_weight'],
                    bias=parameters[prefix+'_bias'],m=parameters[prefix+'_multiplier'],shift=parameters[prefix+'_shift'],
                    iz=row['input_zero'],oz=row['output_zero'],relu=row['relu'],pad=row['padding']))
    report=emit(cases,raw)
    report.update(source_sha256={str(package_root/'verification.json'):file_sha256(package_root/'verification.json')},
        outputs_sha256={p.name:file_sha256(p) for p in raw.glob('*.mem')})
    write_json(raw/'vectors.json',report)
    cache=root/'cache_vectors';cache_vectors(SimpleNamespace(source=raw,outdir=cache,sets=256))
    text=(FLOW/'benches/tb_lung_cached_axi.sv').read_text()
    text=text.replace('D:/pld_picture_process/data/lung_cache_axi_20261015/vectors/',cache.as_posix()+'/')
    bench=root/'tb_lung_cached_axi.sv';bench.write_text(text,encoding='utf-8')
    (root/'run.tcl').write_text('run all\nexit\n')
    files=[FLOW/'rtl'/n for n in RTL]+[bench]
    commands=[[str(BIN/'xvlog.bat'),'--sv',*map(str,files)],
        [str(BIN/'xelab.bat'),'tb_lung_cached_axi','-s','pruned_snapshot','--debug','typical'],
        [str(BIN/'xsim.bat'),'pruned_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    for i,command in enumerate(commands):
        with (root/('tool_'+str(i)+'.log')).open('x',encoding='utf-8') as log:
            if subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode:
                raise RuntimeError('Pruned RTL tool failed')
    log=(root/'rtl_run.log').read_text()
    match=re.search(r'Lung cached AXI convolution RTL: PASS \((\d+) frames, (\d+) outputs',log)
    if not match or int(match[1])!=report['frames'] or int(match[2])!=report['outputs'] or re.search('Fatal:|ERROR:',log):
        raise ValueError('Pruned model RTL coverage failed')
    write_json(root/'verification.json',dict(status='passed_actual_xsim_pruned_convolutions',
        layers=15,probes_per_layer=2,frames=report['frames'],outputs=report['outputs'],
        log_sha256=file_sha256(root/'rtl_run.log'),source_sha256={str(f):file_sha256(f) for f in files+[Path(__file__).resolve()]},
        vector_sha256={str(f):file_sha256(f) for f in cache.iterdir() if f.is_file()},
        package_proof_sha256=file_sha256(package_root/'verification.json'),
        scope='Isolated real new weights/widths with cropped CT features and random full-range codes; NOT full-frame board execution'))
    print('Pruned actual RTL: PASS',report['frames'],'frames',report['outputs'],'output codes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',default='data/lung_prune_board_20261017')
    run(p.parse_args().package_root)
