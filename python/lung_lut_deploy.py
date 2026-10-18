"""Audit/packaging entry for the isolated exact concat LUT experiment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import lung_flow_prepare as flow
from lung_lut_prepare import ROOT
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes, timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware


def exhaustive(root):
    root=Path(root).resolve();flow.checked(root)
    folder=root/'sim_lut_exhaustive';folder.mkdir()
    files=[root/'rtl/lung_tensor_burst_engine.v',root/'rtl/lung_requant.v',
           flow.WORKSPACE/'fpga/tb_lung_concat_lut.sv']
    shutil.copy2(root/'run.tcl',folder/'run.tcl')
    commands=[[str(flow.BIN/'xvlog.bat'),'--sv',*map(str,files)],
        [str(flow.BIN/'xelab.bat'),'tb_lung_concat_lut','-s','lut_snapshot','--debug','typical'],
        [str(flow.BIN/'xsim.bat'),'lut_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    for i,command in enumerate(commands):
        with (folder/('tool_'+str(i)+'.log')).open('x',encoding='utf-8') as out:
            if subprocess.run(command,cwd=folder,stdout=out,stderr=subprocess.STDOUT).returncode:
                raise RuntimeError('Exhaustive LUT tool failed')
    log=(folder/'rtl_run.log').read_text()
    if 'Lung exhaustive concat LUT RTL: PASS' not in log or re.search('Fatal:|ERROR:',log):
        raise ValueError('Exhaustive LUT test failed')
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_exhaustive_concat_lut',
        tested_codes=4096,log_sha256=file_sha256(folder/'rtl_run.log'),
        source_sha256={str(f):file_sha256(f) for f in files}))
    print('Actual exhaustive LUT: PASS',flush=True)


def rtl_proof(root):
    root=Path(root).resolve();flow.checked(root);runs={}
    for variant in ['baseline','flow']:
        folder=root/('sim_'+variant)
        d=json.loads((folder/'verification.json').read_text())
        if d['status']!='passed_actual_xsim_flow_'+variant or len(d['results'])!=10:
            raise ValueError('Incomplete actual RTL regression')
        for top,row in d['results'].items():
            log=folder/(top+'_run.log')
            if file_sha256(log)!=row['log_sha256'] or re.search('Fatal:|ERROR:',log.read_text()):
                raise ValueError('Changed or failed RTL proof')
        runs[variant]=d
    comparisons={}
    for top in ['tb_lung_cached_axi','tb_lung_axil_accel','tb_lung_tensor_burst_axil']:
        a=runs['baseline']['results'][top]['cycles'];b=runs['flow']['results'][top]['cycles']
        if [r['frame'] for r in a]!=[r['frame'] for r in b]:raise ValueError('Different cases')
        before=sum(r['cycles'] for r in a);after=sum(r['cycles'] for r in b)
        if after>before or (top=='tb_lung_tensor_burst_axil' and after==before):
            raise ValueError('No RTL gain or convolution regression')
        comparisons[top]=dict(baseline_cycles=before,lut_cycles=after,reduction_percent=100*(before-after)/before)
        if top=='tb_lung_tensor_burst_axil':
            comparisons[top]['by_op']={str(op):dict(baseline_cycles=sum(r['cycles'] for r in a if r['op']==op),
                lut_cycles=sum(r['cycles'] for r in b if r['op']==op)) for op in range(4)}
    for folder,status in [('sim_lut_exhaustive','passed_actual_xsim_exhaustive_concat_lut'),
                          ('sim_mid_faults','passed_actual_xsim_burst_mid_command_faults')]:
        proof=json.loads((root/folder/'verification.json').read_text())
        if proof['status']!=status:raise ValueError('Missing extra numerical/fault proof')
        check_hashes(proof['source_sha256'])
        if file_sha256(root/folder/'rtl_run.log')!=proof['log_sha256']:raise ValueError('Changed extra proof')
    return dict(status='verified_actual_xsim_concat_lut_gains',comparisons=comparisons,
        scope='Identical frozen outputs and traffic. Simulation cycles are not board latency.')


def package(root):
    root=Path(root).resolve();prep=flow.checked(root);sim=rtl_proof(root)
    prior=json.loads((flow.ROOT/'verification.json').read_text())
    target=root/'lung_lut'
    if target.exists() or (root/'verification.json').exists():raise FileExistsError('Preserve previous package/proof')
    wns=timing(root/'timing_routed.rpt')
    m=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',(root/'timing_routed.rpt').read_text(),re.S)
    if not m or float(m[1].split()[4])<0:raise ValueError('Hold timing failed')
    whs=float(m[1].split()[4]);resources=utilization(root/'utilization_routed.rpt')
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text()):raise ValueError('DRC failed')
    if resources['DSPs']>=prior['utilization']['DSPs']:raise ValueError('Expected measured DSP saving')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot());imported={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        expected={Path(n).name for n in prior['packaged_source_sha256'] if top in Path(n).parts}
        files=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if len(files)!=len(expected) or {f.name for f in files}!=expected:raise ValueError('IP dependency mismatch')
        for f in files:
            if file_sha256(f)!=file_sha256(root/'rtl'/f.name):raise ValueError('Imported source changed')
            imported[str(f)]=file_sha256(f)
    test=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',(root/'python_tests.log').read_text())
    if not test or int(test[1])<341:raise ValueError('Need complete Python regression')
    shutil.copytree(flow.ROOT/'lung_flow',target)
    for name in ['lung_network.bit','lung_network.hwh']:shutil.copy2(root/'deploy'/name,target/name)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    def update(name,extra):
        d=json.loads((target/name).read_text());d.update(extra)
        d['files_sha256']={n:file_sha256(target/n) for n in d['files_sha256']};write_json(target/name,d)
    old=flow.ROOT/'lung_flow'
    update('continuous_manifest.json',dict(bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'],original_continuous_manifest_sha256=file_sha256(old/'continuous_manifest.json')))
    update('cpu_compare_manifest.json',dict(original_continuous_package_sha256=file_sha256(target/'continuous_manifest.json')))
    update('fast_manifest.json',dict(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),hardware_revision='concat_lut_v1',bit_changed=True,board_tested=False,
        warning='Exact per-command requantization lookup. Actual board head and timing still required.'))
    update('clock_manifest.json',dict(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route))
    update('selection_manifest.json',dict(target_folder='/home/xilinx/ct_test/lung_lut'))
    update('fp32_manifest.json',dict(combined_hardware_concat_lut=True,board_tested=False))
    sys.path.insert(0,str(flow.WORKSPACE/'pynq'))
    from lung_fast_benchmark import validate_package
    from lung_clock_benchmark import validate_addon
    from lung_selected_benchmark import validate_selection
    from lung_threeway_benchmark import validate
    validate_package(target);validate_addon(target);validate_selection(target);validate(target)
    evidence=[root/n for n in ['preparation.json','python_tests.log','build_flow.tcl','timing_routed.rpt',
        'utilization_routed.rpt','drc.rpt','configuration_exceptions.rpt']]
    for folder in ['sim_baseline','sim_flow','sim_mid_faults','sim_lut_exhaustive']:
        evidence += [root/folder/'verification.json']+list((root/folder).glob('*_run.log'))
    write_json(root/'verification.json',dict(status='built_routed_concat_lut_not_board_tested',board_tested=False,
        model_changed=False,arithmetic_changed=False,routed_design=route,hardware=hardware,utilization=resources,
        simulation=sim,python_tests=int(test[1]),source_sha256=prep['source_sha256'],packaged_source_sha256=imported,
        evidence_sha256={str(f):file_sha256(f) for f in evidence},package_files_sha256={f.name:file_sha256(f) for f in target.iterdir() if f.is_file()}))
    flow.checked(root);print('Audited exact LUT package:',target,route,resources,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT))
    p.add_argument('--exhaustive',action='store_true');p.add_argument('--rtl-only',action='store_true');a=p.parse_args()
    if a.exhaustive:exhaustive(a.root)
    elif a.rtl_only:print(json.dumps(rtl_proof(a.root),indent=2))
    else:package(a.root)
