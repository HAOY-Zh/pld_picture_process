"""Synthesize the actual reusable MAC, replay a port-only ownership oracle."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from nodule_overlap_build import ROOT,checked,replace
from nodule_plan import sha


def run(root,binary,export_ip=False,existing=False):
    root,_=checked(root);out=root/('netlist_standalone' if existing else 'netlist_export' if export_ip else 'netlist')
    if not existing:out.mkdir()
    binary=Path(binary)
    sources=[root/'rtl'/n for n in ['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v']]
    tcl='create_project -in_memory -part xc7z020clg400-1\n'
    tcl+=''.join('read_verilog {'+str(p).replace('\\','/')+'}\n' for p in sources)
    tcl+='''synth_design -top lung_mac_tile8_reuse -part xc7z020clg400-1 -mode out_of_context -flatten_hierarchy full
write_verilog -mode funcsim synthesized.v
report_utilization -file utilization.rpt
close_project
'''
    checkpoint=root/'project/lung_network.runs/design_1_lung_conv8_axil_0_0_synth_1/design_1_lung_conv8_axil_0_0.dcp'
    if export_ip:
        tcl='open_checkpoint {'+str(checkpoint).replace('\\','/')+'}\n'
        tcl+='''set cores [get_cells -hier -filter {REF_NAME =~ *lung_mac_tile8_reuse*}]
if {[llength $cores] != 1} { error "Need unique implemented reusable MAC" }
set core [lindex $cores 0]
set f [open top_name.txt w];puts $f [get_property REF_NAME $core];close $f
write_verilog -cell $core -mode funcsim synthesized.v
close_design
'''
    if not existing:(out/'synth.tcl').write_text(tcl)
    def call(cmd,name):
        with (out/(name+'.log')).open('x') as s:r=subprocess.run(cmd,cwd=out,stdout=s,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Tool failed: '+str(out/(name+'.log')))
    print('Actual OOC MAC synthesis and netlist functional replay',flush=True)
    if not existing:call([str(binary/'vivado.bat'),'-mode','batch','-source','synth.tcl','-log','vivado.log','-journal','vivado.jou'],'synthesize')
    log=(out/'vivado.log').read_text()
    if (not export_ip and 'synth_design completed successfully' not in log) or re.search(r'ERROR:|CRITICAL WARNING: \[Synth|multi-driven net',log):raise ValueError('Unsafe synthesized netlist')
    top='tb_nodule_overlap_ownership';t=(ROOT/'fpga'/(top+'.sv')).read_text()
    for line in ['        if(dut.pending_groups>max_pending)max_pending=dut.pending_groups;\n',
                 '        if(dut.tile.accepted&&!dut.tile.input_active&&dut.tile.inflight!=0)overlaps=overlaps+1;\n',
                 '        if(dut.pending_groups>3||dut.tile.inflight>2)$fatal(1,"Ownership overflow");\n',
                 '        if(max_pending!=3||overlaps<1)$fatal(1,"Did not exercise both slots plus the input holding register");\n']:
        t=replace(t,line,'')
    t=replace(t,'        repeat(3)@(negedge clk);rst_n=1;w_valid=1;',
        '        repeat(20)@(negedge clk);rst_n=1;w_valid=1;')
    t=replace(t,'while(dut.tile.inflight!=2)@(negedge clk);','while(!m_valid)@(negedge clk);')
    t=replace(t,'m_valid||s_ready||loaded||dut.pending_groups||dut.tile.inflight','m_valid||s_ready||loaded')
    t=t.replace('Nodule overlap ownership RTL: PASS','Nodule synthesized overlap ownership RTL: PASS')
    if export_ip:
        module=(out/'top_name.txt').read_text().strip()
        if not re.fullmatch(r'\w+',module):raise ValueError('Unexpected exported module name')
        t=replace(t,'lung_mac_tile8_reuse dut(',module+' dut(')
    (out/(top+'.sv')).write_text(t);(out/'run.tcl').write_text('run all\nexit\n')
    glbl=binary.parent/'data/verilog/src/glbl.v';shutil.copy2(glbl,out/'glbl.v')
    call([str(binary/'xvlog.bat'),'--sv','synthesized.v',top+'.sv','glbl.v'],'compile')
    call([str(binary/'xelab.bat'),top,'glbl','-L','unisims_ver','-L','unimacro_ver','-L','secureip','-s','netlist_test','--debug','typical','--mt','2'],'elaborate')
    call([str(binary/'xsim.bat'),'netlist_test','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')
    text=(out/'rtl_run.log').read_text()
    if 'synthesized overlap ownership RTL: PASS' not in text or re.search(r'Fatal:|ERROR:',text):raise ValueError('Synthesized circuit failed port-only oracle')
    result=dict(status='actual_post_synthesis_overlap_ownership_pass',scope='Reusable MAC core functional netlist, not routed timing simulation',
        exported_from_actual_conv_ip=export_ip,
        source_sha256={str(p):sha(p) for p in sources+[out/(top+'.sv'),out/'synthesized.v',out/'glbl.v',out/'synth.tcl']+([checkpoint] if export_ip else [])},
        log_sha256=sha(out/'rtl_run.log'),synthesis_log_sha256=sha(out/'vivado.log'))
    (out/'verification.json').write_text(json.dumps(result,indent=2));checked(root);print(result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');p.add_argument('--export-ip',action='store_true');p.add_argument('--existing',action='store_true');a=p.parse_args();run(a.root,a.vivado_bin,a.export_ip,a.existing)
