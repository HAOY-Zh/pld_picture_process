"""Isolate one-cycle cache-hit return optimization from accepted window-step."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from lung_window_step_prepare import WORKSPACE, BIN, CACHE, RTL, BENCH, TOPS, MARKERS, EXTRA
from lung_word_prepare import replace_once
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes

OLD=WORKSPACE/'data/lung_window_step_20261016_v2'
ROOT=WORKSPACE/'data/lung_cache_return_20261016_v2'


def checked_previous():
    p=json.loads((OLD/'verification.json').read_text(encoding='utf-8'))
    for key in ['source_sha256','packaged_source_sha256','evidence_sha256']: check_hashes(p[key])
    check_hashes({str(OLD/'lung_window_step'/n):h for n,h in p['package_files_sha256'].items()})
    v=json.loads((OLD/'lung_window_step_result_verification.json').read_text(encoding='utf-8'))
    if v['status']!='verified_returned_prepared_board_actual_head_and_profile' or not v['actual_head_codes_bit_exact']:
        raise ValueError('Need accepted real-board reference')
    check_hashes({str(OLD/n):h for n,h in v['returned_files_sha256'].items()})
    if v['bit_sha256']!=p['routed_design']['bit_sha256']: raise ValueError('Baseline bit mismatch')
    return p


def prepare(root):
    root=Path(root).resolve()
    if root.parent!=WORKSPACE/'data': raise ValueError('Use new direct workspace/data child')
    checked_previous();root=fresh_output(root).resolve();(root/'rtl').mkdir();(root/'benches').mkdir()
    src=WORKSPACE/'fpga/lung_feature_cache_return.v'
    (root/'rtl/lung_feature_cache.v').write_text(replace_once(src.read_text(encoding='utf-8'),
        'module lung_feature_cache_return','module lung_feature_cache'),encoding='utf-8')
    for n in BENCH+['tb_lung_window_step.sv']: shutil.copy2(OLD/'benches'/n,root/'benches'/n)
    shutil.copy2(WORKSPACE/'fpga/tb_lung_cache_return.sv',root/'benches/tb_lung_cache_return.sv')
    shutil.copy2(OLD/'ps_settings.tcl',root/'ps_settings.tcl')
    (root/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    text=(OLD/'build_step.tcl').read_text(encoding='utf-8')
    text=replace_once(text,(CACHE/'rtl/lung_feature_cache.v').as_posix(),(root/'rtl/lung_feature_cache.v').as_posix())
    text=text.split('if {$argc == 2 && [lindex $argv 1] eq "build"} {',1)[0]
    # Vivado propagates PS reset polarity during validate_bd_design. Check the
    # resolved properties AFTER that propagation, never transient IP defaults.
    text=replace_once(text,'validate_bd_design\n','validate_bd_design\nsource [file join $task_root fpga validate_lung_window_step.tcl]\n')
    suffix=(WORKSPACE/'fpga/resume_lung_window_step.tcl').read_text(encoding='utf-8')
    suffix=suffix[suffix.index('launch_runs impl_1 -to_step route_design'):]
    text+='''if {$argc == 2 && [lindex $argv 1] eq "build"} {
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Synthesis failed" }
'''+suffix.replace('close_project\n','')+'}\nclose_project\n'
    (root/'build_return.tcl').write_text(text,encoding='utf-8')
    sources=[Path(__file__).resolve(),src,WORKSPACE/'fpga/tb_lung_cache_return.sv',
             WORKSPACE/'fpga/validate_lung_window_step.tcl',WORKSPACE/'fpga/resume_lung_window_step.tcl']
    sources += [WORKSPACE/'fpga'/n for n in RTL+['tb_lung_write64.sv','tb_lung_axil_control.sv']]
    sources += [OLD/'rtl/lung_feature_window.v',CACHE/'rtl/lung_feature_cache.v',OLD/'build_step.tcl']
    frozen=json.loads((OLD/'preparation.json').read_text(encoding='utf-8'))['frozen_vectors_sha256']
    write_json(root/'preparation.json',dict(status='prepared_direct_cache_return',model_changed=False,arithmetic_changed=False,
        previous_proof_sha256=file_sha256(OLD/'verification.json'),
        accepted_board_proof_sha256=file_sha256(OLD/'lung_window_step_result_verification.json'),
        source_sha256={str(p):file_sha256(p) for p in sources},frozen_vectors_sha256=frozen,
        generated_sha256={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()}))
    print('Prepared:',root,flush=True)


def simulate(root):
    root=Path(root).resolve();p=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    for key in ['source_sha256','frozen_vectors_sha256']: check_hashes(p[key])
    check_hashes({str(root/n):h for n,h in p['generated_sha256'].items()})
    folder=root/'sim_return';folder.mkdir();shutil.copy2(root/'run.tcl',folder/'run.tcl')
    files=[(root/'rtl'/n if n=='lung_feature_cache.v' else OLD/'rtl'/n if n=='lung_feature_window.v'
            else WORKSPACE/'fpga'/n) for n in RTL]
    files += [WORKSPACE/'fpga'/n for n in ['tb_lung_write64.sv','tb_lung_axil_control.sv']]
    files += [root/'benches'/n for n in BENCH+['tb_lung_window_step.sv','tb_lung_cache_return.sv']]
    def call(command,name):
        with (folder/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('RTL tool failed: '+name)
    call([str(BIN/'xvlog.bat'),'--sv',*[str(f) for f in files]],'compile')
    results={}
    for top,marker in zip(TOPS+['tb_lung_window_step','tb_lung_cache_return'],MARKERS+[EXTRA,'Lung direct cache return RTL: PASS']):
        print('Actual XSim:',top,flush=True)
        call([str(BIN/'xelab.bat'),top,'-s',top+'_snapshot','--debug','typical'],top+'_elaborate')
        call([str(BIN/'xsim.bat'),top+'_snapshot','-tclbatch','run.tcl','-log',top+'_run.log'],top+'_tool')
        log=folder/(top+'_run.log');text=log.read_text(encoding='utf-8')
        if marker not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
            raise ValueError('Failed/incomplete actual RTL: '+str(log))
        results[top]=dict(log_sha256=file_sha256(log))
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_cache_return',results=results,
        source_sha256=p['source_sha256']))
    print('All eight actual RTL checks PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default=str(ROOT));ap.add_argument('--simulate',action='store_true')
    a=ap.parse_args();simulate(a.root) if a.simulate else prepare(a.root)
