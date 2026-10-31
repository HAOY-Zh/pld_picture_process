"""Actual frozen MAC/reuse arithmetic and AXI regression for pixel overlap."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from nodule_overlap_build import ROOT,checked,replace
from nodule_plan import sha


def run(root,binary):
    root,_=checked(root);out=root/'protocol';out.mkdir();binary=Path(binary);tick=time.perf_counter()
    names=['tb_lung_cached_axi','tb_lung_memory_axi','tb_lung_axil_accel']
    for n in names:shutil.copy2(root/'benches'/(n+'.sv'),out/(n+'.sv'))
    f=out/'tb_lung_cached_axi.sv';t=f.read_text()
    marker='        checking=0;reset_core();load_simple();start_frame();while(!read_active)@(negedge clk);reset_core();'
    t=replace(t,marker,'''        checking=0;reset_core();load_simple();width=256;
        force m_ready=0;start_frame();
        while(dut.conv.queued!=64)@(negedge clk);
        repeat(20)begin
            @(negedge clk);
            if(dut.conv.queued!=64||dut.conv.win_ready||!busy||done)
                $fatal(1,"Full lookahead backpressure failed");
        end
        if(dut.conv.core.tile.inflight!=2)$fatal(1,"Did not fill both result slots");
        reset_core();
        if(dut.conv.queued!=0||dut.conv.core_valid||dut.conv.meta_count!=0||
            dut.conv.core.pending_groups!=0||dut.conv.core.tile.inflight!=0)
            $fatal(1,"Stale ownership after reset");
        force m_ready=1;load_simple();start_frame();while(!done)@(negedge clk);
        if(error||busy||dut.conv.queued!=0||dut.conv.meta_count!=0)
            $fatal(1,"Post-full-pipeline reset frame failed");
        release m_ready;
        $display("Overlap full result slots + input FIFO stall/reset/restart: PASS");
        reset_core();load_simple();start_frame();while(!read_active)@(negedge clk);reset_core();''')
    f.write_text(t)
    for n in ['tb_lung_write64.sv','tb_lung_axil_control.sv']:shutil.copy2(ROOT/'fpga'/n,out/n)
    (out/'run.tcl').write_text('run all\nexit\n')
    sources=list((root/'rtl').glob('*.v'))+list(out.glob('*.sv'))
    def call(cmd,folder,name):
        with (folder/(name+'.log')).open('x') as stream:r=subprocess.run(cmd,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Tool failed: '+str(folder/(name+'.log')))
    def replay(folder,top):
        print('Pixel-overlap actual XSim',top,flush=True)
        call([str(binary/'xelab.bat'),top,'-s',top,'--debug','typical','--mt','2'],folder,top+'_elaborate')
        call([str(binary/'xsim.bat'),top,'-tclbatch','run.tcl','-log',top+'_run.log'],folder,top+'_simulate')
        f=folder/(top+'_run.log');log=f.read_text()
        if 'RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Failed actual regression: '+str(f))
        return dict(log_sha256=sha(f),result=re.findall(r'^.*(?:RTL: PASS|restart: PASS).*$',log,re.M))
    call([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],out,'compile');rows={}
    for n in names:rows[n]=replay(out,n)
    # Original independent integer gold: signed products, ACC32 overflow, RNE,
    # all channel tails, sparse output masks, changing quant parameters, reset.
    for name,bank,top,extra in [('mac','lung_tile8_20261014','tb_lung_mac_tile8',['lung_mac_tile8_cached.v']),
                              ('reuse','lung_weight_reuse_20261015','tb_lung_weight_reuse',[])]:
        folder=out/name;folder.mkdir();old=ROOT/'data'/bank
        proof=json.loads((old/'verification.json').read_text())
        for n,h in proof.get('vectors_sha256',{}).items():
            if sha(old/'vectors'/n)!=h:raise ValueError('Changed arithmetic gold')
        shutil.copytree(old/'vectors',folder/'vectors')
        shutil.copy2(ROOT/'fpga'/(top+'.sv'),folder/(top+'.sv'))
        for n in extra:shutil.copy2(ROOT/'fpga'/n,folder/n)
        (folder/'run.tcl').write_text('run all\nexit\n')
        ss=[root/'rtl'/n for n in ['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v']]+list(folder.glob('*.sv'))+[folder/n for n in extra]
        sources+=ss
        call([str(binary/'xvlog.bat'),'--sv',*map(str,ss)],folder,'compile');rows[name]=replay(folder,top)
    result=dict(status='actual_xsim_pixel_overlap_protocol_pass',board_tested=False,results=rows,
        source_sha256={str(p):sha(p) for p in sources},elapsed_sec=time.perf_counter()-tick,
        log_paths={n:str(out/(n+'_run.log')) for n in names})
    result['log_paths'].update(mac=str(out/'mac/tb_lung_mac_tile8_run.log'),reuse=str(out/'reuse/tb_lung_weight_reuse_run.log'))
    result['vectors_sha256']={str(p):sha(p) for sub in ['mac','reuse'] for p in (out/sub/'vectors').iterdir() if p.is_file()}
    (out/'verification.json').write_text(json.dumps(result,indent=2));checked(root);print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();run(a.root,a.vivado_bin)
