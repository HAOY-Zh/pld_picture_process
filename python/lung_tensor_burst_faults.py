"""Additional mid-command fault injection; no gold or frozen bench is modified."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from lung_tensor_burst_prepare import WORKSPACE,BIN
from lung_word_prepare import replace_once
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes


def run(root):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    folder=root/'sim_mid_faults'
    if folder.exists(): raise FileExistsError('Preserve prior mid-command fault run')
    folder.mkdir();shutil.copytree(root/'vectors',folder/'vectors');shutil.copy2(root/'run.tcl',folder/'run.tcl')
    text=(root/'tb_lung_tensor_burst_axil.sv').read_text(encoding='utf-8')
    text=replace_once(text,'module tb_lung_tensor_burst_axil;','module tb_lung_tensor_burst_faults;')
    text=replace_once(text,'reg[31:0]status,value;',
        'integer bad_read_at=0,bad_b_at=0,start_acks=0,start_writes=0;\n    reg[31:0]status,value;')
    text=replace_once(text,'lung_test_burst_memory mem(.*);',
        'lung_test_burst_memory mem(.response((bad_b_at!=0&&acks>=bad_b_at-1)?2\'b10:response),.*);')
    text=replace_once(text,'read_mode<=read_fault;',
        'read_mode<=(bad_read_at!=0&&bursts+1>=bad_read_at)?1:read_fault;')
    start=text.index('    initial begin\n        fh=$fopen(')
    end=text.index('    initial begin #2000000000;',start)
    text=text[:start]+'''    initial begin
        $readmemh("vectors/header.mem",header,0,4);
        $readmemh("vectors/a.mem",a_bank,0,header[1]-1);
        $readmemh("vectors/b.mem",b_bank,0,header[2]-1);
        a_length=64;b_length=64;a_offset=0;b_offset=0;checking=0;hold_final=0;
        repeat(3)@(negedge clk);rst_n=1;
        setup(2,8,8,8,8);start_acks=acks;bad_read_at=bursts+3;
        put(4,1);await_completion(1);host.read_reg(56,0,value);
        if(value!=32||acks-start_acks!=2)$fatal(1,"Late read fault lost committed count or issued more writes: %0d",value);
        start_writes=writes;repeat(15)@(negedge clk);
        if(writes!=start_writes||read_active||rvalid||mem.pending||bvalid)$fatal(1,"Late read fault did not drain");
        bad_read_at=0;start_acks=acks;bad_b_at=acks+2;
        setup(2,8,8,8,8);put(4,1);await_completion(1);host.read_reg(56,0,value);
        if(value!=16||acks-start_acks!=2)$fatal(1,"Late B fault counted unacknowledged/failed burst: %0d",value);
        start_writes=writes;repeat(15)@(negedge clk);
        if(writes!=start_writes||read_active||rvalid||mem.pending||bvalid)$fatal(1,"Late B fault left transactions");
        bad_b_at=0;setup(2,8,8,8,8);put(4,1);await_completion(0);host.read_reg(56,0,value);
        if(value!=128)$fatal(1,"Healthy retry after late faults failed");
        host.read_reg(68,0,value);if(value!=3)$fatal(1,"Late fault completion count differs");
        $display("Lung tensor burst mid-command faults RTL: PASS (late read after 2 committed bursts; late B after 1 committed burst; no writes after DONE; healthy retry)");$finish;
    end
'''+text[end:]
    path=folder/'tb_mid.sv';path.write_text(text,encoding='utf-8')
    files=[WORKSPACE/'fpga'/n for n in ['lung_requant.v','lung_axi_read64.v','lung_axi_burst_write64.v',
        'lung_tensor_control.v','tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']]
    files += [root/'rtl/lung_tensor_burst_engine.v',root/'rtl/lung_tensor_axil.v',path]
    commands=[[str(BIN/'xvlog.bat'),'--sv',*[str(p) for p in files]],
        [str(BIN/'xelab.bat'),'tb_lung_tensor_burst_faults','-s','mid_snapshot','--debug','typical'],
        [str(BIN/'xsim.bat'),'mid_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    for i,command in enumerate(commands):
        with (folder/('tool_'+str(i)+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('Mid-command RTL tool failed')
    log=(folder/'rtl_run.log').read_text(encoding='utf-8')
    if 'Lung tensor burst mid-command faults RTL: PASS' not in log or re.search(r'Fatal:|ERROR:',log): raise ValueError('Mid-command fault failed')
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_burst_mid_command_faults',
        log_sha256=file_sha256(folder/'rtl_run.log'),generated_bench_sha256=file_sha256(path),
        source_sha256={str(p):file_sha256(p) for p in files+[Path(__file__).resolve()]}))
    print('Late read/B faults and healthy retry: actual RTL PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_tensor_burst_20261016_v3')
    run(parser.parse_args().root)
