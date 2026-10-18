"""Isolated window stepping experiment against the accepted burst/cache design."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from lung_cache_fast_prepare import RTL, BENCH, TOPS, MARKERS, cycle_rows
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256
from lung_word_prepare import replace_once
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
OLD=WORKSPACE/'data/lung_tensor_burst_20261016_v3'
CACHE=WORKSPACE/'data/lung_cache_fast_20261016'
BIN=Path('E:/vivado2/Vivado/2022.2/bin')
EXTRA='Lung stepped window RTL: PASS (10 cases;'


def checked_old():
    old=json.loads((OLD/'verification.json').read_text(encoding='utf-8'))
    for key in ['source_sha256','packaged_source_sha256','evidence_sha256']: check_hashes(old[key])
    check_hashes({str(OLD/'lung_tensor_burst'/n):h for n,h in old['package_files_sha256'].items()})
    returned=OLD/'lung_tensor_burst_confirm_result_verification.json'
    v=json.loads(returned.read_text(encoding='utf-8'))
    if v['status']!='verified_returned_prepared_board_actual_head_and_profile' or not v['actual_head_codes_bit_exact']:
        raise ValueError('Need accepted physical full-network baseline')
    check_hashes({str(OLD/n):h for n,h in v['returned_files_sha256'].items()})
    if v['bit_sha256']!=file_sha256(OLD/'lung_tensor_burst/lung_network.bit'):
        raise ValueError('Physical baseline differs')
    return old


def prepare(root):
    root=Path(root).resolve()
    if WORKSPACE/'data' not in root.parents: raise ValueError('Use workspace/data')
    old=checked_old();root=fresh_output(root).resolve();(root/'rtl').mkdir();(root/'benches').mkdir()
    text=(WORKSPACE/'fpga/lung_feature_window_step.v').read_text(encoding='utf-8')
    (root/'rtl/lung_feature_window.v').write_text(replace_once(text,'module lung_feature_window_step','module lung_feature_window'),encoding='utf-8')
    for name in BENCH: shutil.copy2(CACHE/'benches'/name,root/'benches'/name)
    shutil.copy2(WORKSPACE/'fpga/tb_lung_window_step.sv',root/'benches/tb_lung_window_step.sv')
    for source,target in [(OLD/'build_burst.tcl','build_step.tcl'),(CACHE/'synth_cache.tcl','synth_step.tcl')]:
        text=source.read_text(encoding='utf-8')
        text=replace_once(text,(CACHE/'rtl/lung_feature_window.v').as_posix(),(root/'rtl/lung_feature_window.v').as_posix())
        (root/target).write_text(text,encoding='utf-8')
    shutil.copy2(OLD/'ps_settings.tcl',root/'ps_settings.tcl')
    (root/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    sources=[Path(__file__).resolve(),WORKSPACE/'fpga/lung_feature_window_step.v',WORKSPACE/'fpga/tb_lung_window_step.sv']
    sources += [WORKSPACE/'fpga'/n for n in RTL+['tb_lung_write64.sv','tb_lung_axil_control.sv']]
    sources += [CACHE/'rtl'/n for n in ['lung_feature_window.v','lung_feature_cache.v']]
    sources += [CACHE/'benches'/n for n in BENCH]+[OLD/'build_burst.tcl',CACHE/'synth_cache.tcl']
    frozen={str(p):file_sha256(p) for folder in [WORKSPACE/'data/lung_cache_axi_20261015/vectors',
        WORKSPACE/'data/lung_memory_axi_20261015/vectors'] for p in folder.iterdir() if p.is_file()}
    write_json(root/'preparation.json',dict(status='prepared_step_window_experiment',
        previous_proof_sha256=file_sha256(OLD/'verification.json'),
        accepted_board_proof_sha256=file_sha256(OLD/'lung_tensor_burst_confirm_result_verification.json'),
        source_sha256={str(p):file_sha256(p) for p in sources},frozen_vectors_sha256=frozen,
        generated_sha256={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()},
        arithmetic_changed=False,model_changed=False))
    print('Prepared isolated stepped window:',root,flush=True)


def simulate(root,variant,resume=False):
    root=Path(root).resolve();p=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(p['source_sha256']);check_hashes(p['frozen_vectors_sha256'])
    check_hashes({str(root/n):h for n,h in p['generated_sha256'].items()})
    folder=root/('sim_'+variant)
    if folder.exists() and not resume: raise FileExistsError('Preserve previous attempt')
    folder.mkdir(exist_ok=resume)
    if not (folder/'run.tcl').exists(): shutil.copy2(root/'run.tcl',folder/'run.tcl')
    files=[]
    for n in RTL:
        if n=='lung_feature_cache.v': path=CACHE/'rtl'/n
        elif n=='lung_feature_window.v': path=(root/'rtl' if variant=='step' else CACHE/'rtl')/n
        else: path=WORKSPACE/'fpga'/n
        files.append(path)
    files += [WORKSPACE/'fpga'/n for n in ['tb_lung_write64.sv','tb_lung_axil_control.sv']]
    files += [root/'benches'/n for n in BENCH+['tb_lung_window_step.sv']]
    def call(command,name):
        with (folder/(name+'.log')).open('x',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=folder,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('RTL tool failed: '+name)
    start=time.perf_counter()
    if not (folder/'compile.log').exists(): call([str(BIN/'xvlog.bat'),'--sv',*[str(f) for f in files]],'compile')
    results={}
    for top,marker in zip(TOPS+['tb_lung_window_step'],MARKERS+[EXTRA]):
        print(variant+': '+top,flush=True);snapshot=top+'_snapshot';logname=top+'_run.log'
        if not (folder/logname).exists():
            call([str(BIN/'xelab.bat'),top,'-s',snapshot,'--debug','typical'],top+'_elaborate')
            call([str(BIN/'xsim.bat'),snapshot,'-tclbatch','run.tcl','-log',logname],top+'_tool')
        text=(folder/logname).read_text(encoding='utf-8')
        if marker not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
            raise ValueError('Incomplete/failing RTL: '+str(folder/logname))
        rows=cycle_rows(text) if top in ['tb_lung_cached_axi','tb_lung_axil_accel'] else []
        results[top]=dict(log_sha256=file_sha256(folder/logname),cycles=rows)
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_step_window_'+variant,
        results=results,source_sha256=p['source_sha256'],wall_seconds=time.perf_counter()-start))
    print(variant+' all seven actual RTL checks PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_window_step_20261016')
    parser.add_argument('--simulate',choices=['baseline','step']);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    simulate(args.root,args.simulate,args.resume) if args.simulate else prepare(args.root)
