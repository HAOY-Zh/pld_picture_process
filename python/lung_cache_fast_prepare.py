"""Isolated cache/window timing experiment; unchanged math, fixtures and AXI ABI."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from lung_word_prepare import replace_once
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
BIN=Path('E:/vivado2/Vivado/2022.2/bin')
RTL=['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v','lung_feature_window.v',
     'lung_conv8_window.v','lung_feature_cache.v','lung_axi_read64.v','lung_conv8_cached_axi.v',
     'lung_axi_write64.v','lung_output_writer.v','lung_conv8_memory_axi.v','lung_axil_control.v','lung_conv8_axil.v']
BENCH=['tb_lung_cached_axi.sv','tb_lung_memory_axi.sv','tb_lung_axil_accel.sv',
       'tb_lung_feature_window_bounds.sv','tb_lung_cache_protocol.sv']
TOPS=['tb_lung_cached_axi','tb_lung_memory_axi','tb_lung_axil_accel',
      'tb_lung_feature_window_bounds','tb_lung_cache_protocol','tb_lung_cache_capacity']
MARKERS=['Lung cached AXI convolution RTL: PASS (147 frames, 68849 outputs, 371974 windows, 312018 requests, 2612 AXI bursts, 41606 AXI read beats;',
         'Lung memory AXI convolution RTL: PASS (147 frames, 68849 codes, 8866 writes, 2612 read bursts, 41606 read beats, 14 rejection/error cases;',
         'Lung AXI-Lite accelerator RTL: PASS (147 frames, 8866 words, 6055 weight packets, 2612 memory read bursts, 41606 memory read beats, 5 error cases;',
         'Lung feature window bounds RTL: PASS (4 configurations;',
         'Lung feature cache protocol RTL: PASS (28 reads;',
         'Lung feature cache capacity RTL: PASS (18437 reads;']


def cycle_rows(text):
    if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I): raise ValueError('RTL error/warning')
    rows=[dict(frame=int(f),cycles=int(c)) for f,c in re.findall(r'CACHE_FAST_CYCLES frame=(\d+) cycles=(\d+)',text)]
    if [r['frame'] for r in rows]!=list(range(147)) or any(r['cycles']<=0 for r in rows):
        raise ValueError('Missing complete frame cycle counters')
    return rows


def prepare(root):
    root=Path(root).resolve()
    if WORKSPACE/'data' not in root.parents: raise ValueError('Use a new directory under workspace/data')
    root=fresh_output(root).resolve()
    cache=WORKSPACE/'data/lung_cache_axi_20261015/vectors'
    memory=WORKSPACE/'data/lung_memory_axi_20261015/vectors'
    old=WORKSPACE/'data/lung_word_20261015'
    prior=json.loads((old/'verification.json').read_text(encoding='utf-8'))
    check_hashes(prior['source_sha256']);check_hashes(prior['evidence_sha256'])
    check_hashes({str(old/'lung_word'/n):h for n,h in prior['package_files_sha256'].items()})
    generated=root/'rtl';generated.mkdir()
    for name in ['lung_feature_cache','lung_feature_window']:
        text=(WORKSPACE/'fpga'/(name+'_fast.v')).read_text(encoding='utf-8')
        (generated/(name+'.v')).write_text(replace_once(text,'module '+name+'_fast','module '+name),encoding='utf-8')
    benches=root/'benches';benches.mkdir()
    for name in BENCH:
        text=(WORKSPACE/'fpga'/name).read_text(encoding='utf-8')
        text=text.replace('"../lung_cache_axi_20261015/vectors/','"'+cache.as_posix()+'/')
        text=text.replace('"../lung_memory_axi_20261015/vectors/','"'+memory.as_posix()+'/')
        folder=cache if name=='tb_lung_cached_axi.sv' else memory
        text=text.replace('"vectors/','"'+folder.as_posix()+'/')
        if name=='tb_lung_cached_axi.sv':
            text=replace_once(text,'reg checking=0,out_stall=0','integer frame_start_cycle;\n    reg checking=0,out_stall=0')
            text=replace_once(text,'start_frame();base=','frame_start_cycle=cycle;start_frame();base=')
            text=replace_once(text,'if(f%15==0)$display',
                '$display("CACHE_FAST_CYCLES frame=%0d cycles=%0d",f,cycle-frame_start_cycle);\n            if(f%15==0)$display')
        if name=='tb_lung_axil_accel.sv':
            text=replace_once(text,'if(value==0)$fatal(1,"Missing cycles");',
                'if(value==0)$fatal(1,"Missing cycles");$display("CACHE_FAST_CYCLES frame=%0d cycles=%0d",f,value);')
        (benches/name).write_text(text,encoding='utf-8')
    (root/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    builder=(old/'build_word.tcl').read_text(encoding='utf-8')
    anchor='[file join $task_root fpga $f]'
    # Existing tensor-word conditional has this anchor once. Nest the two
    # shadow source paths only; preserve every IP/BD/reset/timing setting.
    select='[expr {$f eq "lung_feature_cache.v" ? "'+(generated/'lung_feature_cache.v').as_posix()+'" : ($f eq "lung_feature_window.v" ? "'+(generated/'lung_feature_window.v').as_posix()+'" : [file join $task_root fpga $f])}]'
    (root/'build_cache.tcl').write_text(replace_once(builder,anchor,select),encoding='utf-8')
    shutil.copy2(old/'ps_settings.tcl',root/'ps_settings.tcl')
    synth=(WORKSPACE/'fpga/synth_lung_cached_axi.tcl').read_text(encoding='utf-8')
    synth=replace_once(synth,'set task_root [file normalize [file join [file dirname [info script]] ..]]',
                       'set task_root {'+WORKSPACE.as_posix()+'}')
    (root/'synth_cache.tcl').write_text(replace_once(synth,anchor,select),encoding='utf-8')
    sources=[Path(__file__).resolve(),WORKSPACE/'fpga/lung_feature_cache_fast.v',WORKSPACE/'fpga/lung_feature_window_fast.v']
    sources += [WORKSPACE/'fpga'/n for n in RTL+BENCH+['tb_lung_write64.sv','tb_lung_axil_control.sv','synth_lung_cached_axi.tcl']]
    vectors={str(p):file_sha256(p) for folder in [cache,memory] for p in folder.iterdir() if p.is_file()}
    write_json(root/'preparation.json',dict(status='prepared_cache_window_fast_experiment',
        source_sha256={str(p):file_sha256(p) for p in sources},frozen_vectors_sha256=vectors,
        generated_sha256={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()},
        previous_word_proof_sha256=file_sha256(old/'verification.json'),
        model_or_arithmetic_changed=False,board_performance_known=False))
    print('Prepared cache/window experiment:',root,flush=True)


def simulate(root,variant,resume=False):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes(prep['frozen_vectors_sha256'])
    check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    folder=root/('sim_'+variant)
    if folder.exists() and not resume: raise FileExistsError('Preserve prior simulation; --resume validates completed logs')
    folder.mkdir(exist_ok=resume)
    if not (folder/'run.tcl').exists(): shutil.copy2(root/'run.tcl',folder/'run.tcl')
    files=[root/'rtl'/n if variant=='fast' and n in ['lung_feature_cache.v','lung_feature_window.v']
           else WORKSPACE/'fpga'/n for n in RTL]
    files += [WORKSPACE/'fpga'/n for n in ['tb_lung_write64.sv','tb_lung_axil_control.sv']]
    files += [root/'benches'/n for n in BENCH]
    def call(command,name):
        with (folder/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('RTL tool failed: '+name)
    start=time.perf_counter()
    if not (folder/'compile.log').exists():
        call([str(BIN/'xvlog.bat'),'--sv',*[str(p) for p in files]],'compile')
    results={}
    for top,marker in zip(TOPS,MARKERS):
        print(variant+': '+top,flush=True)
        snapshot=top+'_snapshot'
        logname=top+'_run.log'
        if not (folder/logname).exists():
            call([str(BIN/'xelab.bat'),top,'-s',snapshot,'--debug','typical'],top+'_elaborate')
            call([str(BIN/'xsim.bat'),snapshot,'-tclbatch','run.tcl','-log',logname],top+'_tool')
        text=(folder/logname).read_text(encoding='utf-8')
        if marker not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
            raise ValueError('Incomplete/failing RTL: '+str(folder/logname))
        rows=cycle_rows(text) if top in ['tb_lung_cached_axi','tb_lung_axil_accel'] else []
        results[top]=dict(log_sha256=file_sha256(folder/logname),cycles=rows)
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_cache_'+variant,
        results=results,wall_seconds=time.perf_counter()-start,
        source_sha256=prep['source_sha256'],scope='Actual RTL under same stalled AXI BFM; NOT real-board latency'))
    print(variant+' all cache/window/convolution RTL checks PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_cache_fast_20261016')
    parser.add_argument('--simulate',choices=['baseline','fast'])
    parser.add_argument('--resume',action='store_true',help='Only identical frozen RTL/fixtures; validate existing completed logs')
    args=parser.parse_args()
    if args.simulate: simulate(args.root,args.simulate,args.resume)
    else: prepare(args.root)
