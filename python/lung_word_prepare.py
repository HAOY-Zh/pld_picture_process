"""Isolated tensor-word RTL experiment; mechanical build derivatives preserve old proofs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from lung_rtl_vectors import hex_bank
from lung_tensor_vectors import reference,source_words
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_continuous import unpack_words
from lung_fast_benchmark import validate_package
from lung_axil_smoke import feature_words


def replace_once(text,old,new):
    if text.count(old)!=1: raise ValueError('Mechanical derivative anchor changed: '+old)
    return text.replace(old,new,1)


def prepare(root):
    root=fresh_output(root).resolve()
    if WORKSPACE/'data' not in root.parents: raise ValueError('Use a fresh folder under data')
    old=WORKSPACE/'data/lung_tensor_20261015/vectors'
    vectors=json.loads((old/'vectors.json').read_text(encoding='utf-8'))
    check_hashes(vectors['source_sha256']);check_hashes({str(old/n):h for n,h in vectors['files_sha256'].items()})
    cases=[]
    with np.load(old/'vectors.npz',allow_pickle=False) as archive:
        for row in vectors['cases']:
            cases.append({k:row[k] for k in ['name','op','za','zb','zo','ma','sa','mb','sb']} |
                dict(a=archive[row['prefix']+'_a'],b=None if row['b_shape'] is None else archive[row['prefix']+'_b']))
    rng=np.random.default_rng(10516)
    random=lambda c,h,w:rng.integers(0,256,(c,h,w),dtype=np.uint8)
    def add(name,op,a,b,q=None):
        cases.append(dict(name=name,op=op,a=a,b=b,za=128,zb=64,zo=99,ma=1073741824,sa=31,mb=1610612736,sb=30)| (q or {}))
    add('aligned_signed_all_codes',2,np.tile(np.arange(256,dtype=np.uint8),(8,1)).reshape(8,1,256),
        np.tile(np.arange(255,-1,-1,dtype=np.uint8),(8,1)).reshape(8,1,256))
    add('aligned_tails_dirty_shift0_62',2,random(8,3,7),random(5,3,7),dict(ma=1,sa=0,mb=2147483647,sb=62))
    add('aligned_96_channels',3,random(8,3,7),random(88,6,14))
    add('aligned_many_a_blocks',2,random(88,3,7),random(8,3,7))
    add('aligned_wide_512',2,random(16,1,512),random(3,1,512))
    add('aligned_high_512',3,random(8,256,1),random(1,512,2),dict(za=255,zb=0,zo=128,ma=2147483647,sa=31,mb=1,sb=0))
    original=WORKSPACE/'data/lung_fast_20261015/lung_fast'
    manifest,model,params,fast=validate_package(original)
    check_hashes(fast['source_sha256'])
    shapes={n['name']:n['shape'] for n in model['nodes']}
    with np.load(original/'reference.npz',allow_pickle=False) as archive:
        for node in model['nodes']:
            if node['kind']!='fused_concat': continue
            a,b=[unpack_words(archive[n],shapes[n]) for n in node['sources']]
            add('FULL_'+node['name'],3,a,b,node['quant'])
    out=root/'vectors';out.mkdir();abank=[];bbank=[];gold=[];commands=[];rows=[]
    for case in cases:
        a,b=case['a'],case['b'];q={k:case[k] for k in ['za','zb','zo','ma','sa','mb','sb']}
        expected=reference(case['op'],a,b,**q)
        if case['name'].startswith('FULL_'):
            with np.load(original/'reference.npz',allow_pickle=False) as archive:
                if not np.array_equal(feature_words(expected,q['zo']),archive[case['name'][5:]]):
                    raise ValueError('Full-frame independent reference differs from frozen graph')
        aa=source_words(a,q['za']);bb=[] if b is None else source_words(b,q['zb']);gg=feature_words(expected,q['zo'])
        ca,h,w=a.shape;cb=0 if b is None else len(b)
        commands.extend([case['op'],h,w,ca,cb,*q.values(),len(abank),len(aa),len(bbank),len(bb),len(gold),len(gg)])
        abank.extend(aa.tolist());bbank.extend(list(bb));gold.extend(gg.tolist())
        rows.append(dict(name=case['name'],op=case['op'],ca=ca,cb=cb,words=len(gg),vector_path=case['op']>=2 and ca%8==0))
    header=[len(rows),len(abank),len(bbank),len(gold),18]
    for name,values,bits in [('header',header,32),('commands',commands,32),('a',abank,64),('b',bbank,64),('gold',gold,64)]:
        hex_bank(out/(name+'.mem'),values,bits)
    top=(WORKSPACE/'fpga/lung_tensor_axil.v').read_text(encoding='utf-8')
    for variant in ['baseline','word']:
        folder=root/'rtl'/variant;folder.mkdir(parents=True)
        (folder/'lung_tensor_axil.v').write_text(top if variant=='baseline' else replace_once(
            top,'lung_tensor_engine engine','lung_tensor_word_engine engine'),encoding='utf-8')
    bench=(WORKSPACE/'fpga/tb_lung_tensor_axil.sv').read_text(encoding='utf-8')
    bench=replace_once(bench,'module tb_lung_tensor_axil;','module tb_lung_tensor_word_axil;')
    bench=replace_once(bench,'header[0]!==26','header[0]!=='+str(len(rows)))
    bench=bench.replace('[0:99999]','[0:999999]').replace('>100000','>1000000')
    bench=bench.replace('[0:19999]','[0:299999]').replace('<20000','<300000')
    bench=bench.replace('32768','1048576').replace('65536','2097152')
    anchor='host.read_reg(64,0,value);if(value==0)$fatal(1,"Missing tensor cycles");'
    bench=replace_once(bench,anchor,anchor+'\n            $display("WORD_CYCLES frame=%0d op=%0d ca=%0d cycles=%0d words=%0d",f,commands[t],commands[t+3],value,frame_pixels);')
    fault='''response=0;response_id=0;read_fault=0;
        setup(2,2,2,8,8);a_offset=0;a_length=4;b_offset=0;b_length=4;
        for(p=1;p<=3;p=p+1)begin
            prior_acks=acks;read_fault=p;put(4,1);await_completion(1);errors=errors+1;
            host.read_reg(56,0,value);if(value!=0||acks!=prior_acks)$fatal(1,"Vector read fault wrote output");
        end
        read_fault=0;
        for(p=0;p<2;p=p+1)begin
            prior_acks=acks;response=p==0?2:0;response_id=p==1?7:0;
            put(4,1);await_completion(1);errors=errors+1;
            host.read_reg(56,0,value);if(value!=0||acks!=prior_acks+1)$fatal(1,"Vector write fault not reported");
        end
        response=0;response_id=0;force_block=1;put(4,1);'''
    bench=replace_once(bench,'response=0;response_id=0;force_block=1;put(4,1);',fault)
    (root/'tb_lung_tensor_word_axil.sv').write_text(bench,encoding='utf-8')
    (root/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    # Derived scripts are generated artifacts, never rewrites of frozen sources.
    root_anchor='set task_root [file normalize [file join [file dirname [info script]] ..]]'
    read_anchor='[file join $task_root fpga $f]'
    select='[expr {$f eq "lung_tensor_axil.v" ? "'+(root/'rtl/word/lung_tensor_axil.v').as_posix()+'" : [file join $task_root fpga $f]}]'
    for source,target in [('fpga/build_lung_network_board.tcl','build_word.tcl'),('fpga/synth_lung_tensor.tcl','synth_word.tcl')]:
        text=(WORKSPACE/source).read_text(encoding='utf-8')
        text=replace_once(text,root_anchor,'set task_root {'+WORKSPACE.as_posix()+'}')
        text=text.replace('lung_tensor_engine.v','lung_tensor_word_engine.v')
        text=replace_once(text,read_anchor,select)
        (root/target).write_text(text,encoding='utf-8')
    shutil.copy2(WORKSPACE/'data/lung_network_board_20261015/ps_settings.tcl',root/'ps_settings.tcl')
    sources=[Path(__file__).resolve()]+[WORKSPACE/'fpga'/n for n in ['lung_tensor_word_engine.v','lung_tensor_engine.v',
        'lung_tensor_axil.v','lung_tensor_control.v','lung_requant.v','lung_axi_read64.v','lung_axi_write64.v',
        'tb_lung_tensor_axil.sv','tb_lung_write64.sv','tb_lung_axil_control.sv','build_lung_network_board.tcl','synth_lung_tensor.tcl']]
    record=dict(status='prepared_word_parallel_tensor_rtl_experiment',cases=rows,header=header,source_sha256={str(p):file_sha256(p) for p in sources},
        generated_sha256={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()},
        original_continuous_manifest_sha256=file_sha256(original/'continuous_manifest.json'),
        warning='Independent test vectors and generated derivative scripts only; no RTL/route/board proof until those tools pass.')
    write_json(root/'preparation.json',record)
    print('Prepared',len(rows),'frames,',len(gold),'words; full frozen concat frames included.',flush=True)


def simulate(root,variant):
    root=Path(root).resolve(); prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    folder=root/('sim_'+variant)
    if folder.exists(): raise FileExistsError('Preserve prior simulation; use a new experiment')
    folder.mkdir();shutil.copytree(root/'vectors',folder/'vectors');shutil.copy2(root/'run.tcl',folder/'run.tcl')
    rtl=[WORKSPACE/'fpga'/n for n in ['lung_requant.v','lung_axi_read64.v','lung_axi_write64.v','lung_tensor_control.v',
        'lung_tensor_engine.v' if variant=='baseline' else 'lung_tensor_word_engine.v','tb_lung_write64.sv','tb_lung_axil_control.sv']]
    rtl+=[root/'rtl'/variant/'lung_tensor_axil.v',root/'tb_lung_tensor_word_axil.sv']
    binpath=Path('E:/vivado2/Vivado/2022.2/bin')
    commands=[[str(binpath/'xvlog.bat'),'--sv',*[str(p) for p in rtl]],
              [str(binpath/'xelab.bat'),'tb_lung_tensor_word_axil','-s','word_snapshot','--debug','typical'],
              [str(binpath/'xsim.bat'),'word_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    start=time.perf_counter()
    for index,command in enumerate(commands):
        print(variant+': '+command[0],flush=True)
        with (folder/('tool_'+str(index)+'.log')).open('w',encoding='utf-8') as stream:
            proc=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError('RTL tool failed; inspect '+str(folder/('tool_'+str(index)+'.log')))
    log=(folder/'rtl_run.log').read_text(encoding='utf-8')
    rows=[dict(zip(['frame','op','ca','cycles','words'],map(int,m))) for m in re.findall(
        r'WORD_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)',log)]
    if (len(rows)!=len(prep['cases']) or 'Lung tensor AXI-Lite RTL: PASS' not in log
            or '24 error cases;' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I)):
        raise ValueError('Incomplete/failing RTL simulation: '+str(folder/'rtl_run.log'))
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_tensor_'+variant,frames=len(rows),error_cases=24,
        results=rows,wall_seconds=time.perf_counter()-start,performance_scope='Actual RTL cycles under testbench AXI stalls; NOT real-board wall time',
        log_sha256=file_sha256(folder/'rtl_run.log'),source_sha256=prep['source_sha256']))
    print(variant+' actual RTL PASS: '+str(len(rows))+' frames, 24 fault cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_word_20261015')
    parser.add_argument('--simulate',choices=['baseline','word'])
    args=parser.parse_args()
    if args.simulate: simulate(args.root,args.simulate)
    else: prepare(args.root)
