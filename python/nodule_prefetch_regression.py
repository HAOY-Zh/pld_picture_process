"""Frozen convolution protocol/error tests plus full-FIFO cancellation."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from nodule_prefetch_build import ROOT, checked, replace
from nodule_plan import sha


def run(root,binary):
    root,_=checked(root);out=root/'protocol';out.mkdir();binary=Path(binary)
    names=['tb_lung_cached_axi','tb_lung_memory_axi','tb_lung_axil_accel']
    for name in names:shutil.copy2(root/'benches'/(name+'.sv'),out/(name+'.sv'))
    f=out/'tb_lung_cached_axi.sv';text=f.read_text()
    marker='        checking=0;reset_core();load_simple();start_frame();while(!read_active)@(negedge clk);reset_core();'
    text=replace(text,marker,'''        // Cancel with a full lookahead queue and a stalled output. Prove old
        // data/coordinates cannot survive reset into the following command.
        checking=0;reset_core();load_simple();width=256;
        force m_ready=0;start_frame();
        while(dut.conv.queued!=64)@(negedge clk);
        repeat(20)begin
            @(negedge clk);
            if(dut.conv.queued!=64||dut.conv.win_ready||!busy||done)
                $fatal(1,"Full FIFO backpressure failed");
        end
        reset_core();
        if(dut.conv.queued!=0||dut.conv.core_valid)$fatal(1,"FIFO stale after reset");
        force m_ready=1;load_simple();start_frame();while(!done)@(negedge clk);
        if(error||busy||dut.conv.queued!=0)$fatal(1,"Post-FIFO-reset frame failed");
        release m_ready;
        $display("Prefetch full FIFO stall/reset/restart: PASS");
        reset_core();load_simple();start_frame();while(!read_active)@(negedge clk);reset_core();''')
    f.write_text(text)
    for name in ['tb_lung_write64.sv','tb_lung_axil_control.sv']:
        shutil.copy2(ROOT/'fpga'/name,out/name)
    (out/'run.tcl').write_text('run all\nexit\n')
    sources=list((root/'rtl').glob('*.v'))+list(out.glob('*.sv'));tick=time.perf_counter()
    def call(cmd,name):
        with (out/(name+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Protocol tool failed: '+name)
    call([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],'compile');rows={}
    for name in names:
        print('Prefetch protocol actual XSim',name,flush=True)
        call([str(binary/'xelab.bat'),name,'-s',name,'--debug','typical','--mt','2'],name+'_elaborate')
        call([str(binary/'xsim.bat'),name,'-tclbatch','run.tcl','-log',name+'_run.log'],name+'_simulate')
        log=(out/(name+'_run.log')).read_text()
        if 'RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Failed protocol: '+name)
        if name=='tb_lung_cached_axi' and 'Prefetch full FIFO stall/reset/restart: PASS' not in log:raise ValueError('Missing FIFO full/reset proof')
        rows[name]=dict(log_sha256=sha(out/(name+'_run.log')),result=re.findall(r'^.*(?:RTL: PASS|FIFO.*PASS).*$',log,re.M))
    result=dict(status='actual_xsim_prefetch_protocol_pass',board_tested=False,results=rows,
        source_sha256={str(p):sha(p) for p in sources},elapsed_sec=time.perf_counter()-tick)
    (out/'verification.json').write_text(json.dumps(result,indent=2));checked(root)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();run(a.root,a.vivado_bin)
