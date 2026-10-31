"""Audit reused routed exact-LUT hardware and replay real nodule full canvas."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from verify_nodule_canvas import verify


def prepare(out):
    out=Path(out).resolve();out.mkdir()
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    verify(ROOT/'data/nodule_canvas_speed_01')
    source=ROOT/'data/lung_lut_20261017';proof=json.loads((source/'verification.json').read_text())
    if proof['routed_design']['clock_mhz']!=100 or min(proof['routed_design'][k] for k in ['wns_ns','whs_ns'])<0:
        raise ValueError('LUT route/clock failed')
    for name,digest in proof['packaged_source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed LUT imported source')
    for name,digest in proof['evidence_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed LUT validation evidence')
    for name,key in [('lung_network.bit','bit_sha256'),('lung_network.hwh','hwh_sha256')]:
        if sha(source/'lung_lut'/name)!=proof['routed_design'][key]:raise ValueError('Changed routed LUT artifact')
    difference=[p.name for p in (parent/'rtl').glob('*.v') if sha(p)!=sha(source/'rtl'/p.name)]
    if difference!=['lung_tensor_burst_engine.v']:raise ValueError('Unexpected additional RTL difference')
    for p in (source/'rtl').glob('*.v'):
        imported=[v for n,v in proof['packaged_source_sha256'].items() if Path(n).name==p.name]
        if not imported or any(v!=sha(p) for v in imported):raise ValueError('RTL not bound to routed imported source')
    actual=json.loads((source/'lung_lut_result_threeway_verification.json').read_text())
    if not actual['int8_head_bit_exact']:raise ValueError('LUT lacks old physical exactness evidence')
    sim=out/'rtl_sim';sim.mkdir();shutil.copytree(source/'rtl',out/'rtl')
    old=ROOT/'data/nodule_canvas_20261023_v1/rtl_sim';prior=json.loads((old/'verification.json').read_text())
    for n,h in prior['vectors_sha256'].items():
        if sha(old/'vectors'/n)!=h:raise ValueError('Changed original integer golden vectors')
    for n,h in prior['source_sha256'].items():
        if sha(n)!=h:raise ValueError('Changed accepted canvas simulation source')
    shutil.copytree(old/'vectors',sim/'vectors')
    for name in ['nodule_sim_engine.sv','tb_nodule_network.sv','run.tcl']:shutil.copy2(old/name,sim/name)
    for name in ['tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']:shutil.copy2(ROOT/'fpga'/name,sim/name)
    for name in ['lung_network.bit','lung_network.hwh']:shutil.copy2(source/'lung_lut'/name,out/name)
    files=list((out/'rtl').glob('*.v'))+list(sim.glob('*.sv'))+list((sim/'vectors').iterdir())
    receipt=dict(status='prepared_existing_routed_lut_for_nodule',parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
        model_threshold_unchanged=True,differences=difference,clock_mhz=100,route=proof['routed_design'],
        physical_baseline_result_sha256=sha(ROOT/'data/nodule_canvas_speed_01/result.json'),
        input_files_sha256={str(p):sha(p) for p in files},bit_sha256=sha(out/'lung_network.bit'),hwh_sha256=sha(out/'lung_network.hwh'),
        old_lut_proof_sha256=sha(source/'verification.json'),old_lut_physical_proof_sha256=sha(source/'lung_lut_result_threeway_verification.json'))
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Routed LUT identity and unchanged nodule golden inputs verified',flush=True)


def simulate(out,bin):
    out=Path(out).resolve();sim=out/'rtl_sim';bin=Path(bin)
    receipt=json.loads((out/'preparation.json').read_text())
    for name,digest in receipt['input_files_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed prepared simulation input')
    sources=list((out/'rtl').glob('*.v'))+list(sim.glob('*.sv'))
    calls=[([str(bin/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(bin/'xelab.bat'),'tb_nodule_network','-s','nodule_exact_lut_snapshot','--debug','typical','--mt','2'],'elaborate'),
        ([str(bin/'xsim.bat'),'nodule_exact_lut_snapshot','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]
    for cmd,name in calls:
        print('Nodule existing LUT actual XSim:',name,flush=True)
        with (sim/(name+'.log')).open('x',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=sim,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('LUT XSim failed: '+name)
    log=(sim/'rtl_run.log').read_text()
    if 'Nodule full canvas RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('LUT full graph failed')
    commands=re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)
    if len(commands)!=15 or [int(c[0]) for c in commands]!=list(range(15)):raise ValueError('Incomplete RTL graph')
    result=dict(status='actual_xsim_nodule_448_exact_lut_pass',commands=commands,log_sha256=sha(sim/'rtl_run.log'),
        source_sha256={str(p):sha(p) for p in sources},preparation_sha256=sha(out/'preparation.json'),
        scope='Same original input and all 11 integer golden outputs; only routed tensor LUT RTL differs',board_tested_for_nodule=False)
    (sim/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Nodule exact-LUT full graph actual RTL PASS',flush=True)


def exhaustive(out,bin):
    out=Path(out).resolve();sim=out/'exhaustive';sim.mkdir();bin=Path(bin)
    _,_,model,_=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
    source=ROOT/'fpga/tb_lung_concat_lut.sv';text=source.read_text()
    text=text.replace('test_id<8;','test_id<10;').replace('4096 codes','5120 codes')
    rows=[]
    for i,name in enumerate(['cat1','cat0'],8):
        q=next(n['quant'] for n in model['nodes'] if n['name']==name)
        rows.append(str(i)+':begin '+';'.join(k+'='+str(q[k]) for k in ['za','zb','zo','ma','mb','sa','sb'])+';end')
    text=text.replace('            endcase','\n'.join(rows)+'\n            endcase')
    old='dut.lookup_lanes[0].codes[index]!==expected||dut.lookup_lanes[7].codes[index]!==expected'
    text=text.replace(old,'||'.join('dut.lookup_lanes['+str(i)+'].codes[index]!==expected' for i in range(8)))
    bench=sim/'tb_lung_concat_lut.sv';bench.write_text(text,encoding='utf-8')
    (sim/'run.tcl').write_text('run all\nexit\n')
    sources=[out/'rtl/lung_tensor_burst_engine.v',out/'rtl/lung_requant.v',bench]
    for cmd,name in [([str(bin/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(bin/'xelab.bat'),'tb_lung_concat_lut','-s','nodule_tables_snapshot','--debug','typical'],'elaborate'),
        ([str(bin/'xsim.bat'),'nodule_tables_snapshot','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        with (sim/(name+'.log')).open('x',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=sim,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Exhaustive LUT failed: '+name)
    log=(sim/'rtl_run.log').read_text()
    if 'PASS (5120 codes' not in log or re.search(r'Fatal:|ERROR:',log):raise ValueError('Exhaustive nodule LUT mismatch')
    (sim/'verification.json').write_text(json.dumps(dict(status='actual_xsim_all_nodule_concat_codes_exact',
        codes=5120,lanes_checked=8,quantization_nodes=['cat1','cat0'],log_sha256=sha(sim/'rtl_run.log'),
        source_sha256={str(p):sha(p) for p in sources}),indent=2),encoding='utf-8')
    print('All nodule concat LUT codes actual RTL PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--simulate',action='store_true')
    p.add_argument('--exhaustive',action='store_true')
    p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.exhaustive:exhaustive(a.out,a.vivado_bin)
    elif a.simulate:simulate(a.out,a.vivado_bin)
    else:prepare(a.out)
