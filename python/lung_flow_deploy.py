"""Validate isolated RTL gains and timing before packaging an untested board bit."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from lung_flow_prepare import WORKSPACE, BASE, ROOT, PACKAGE, checked, baseline
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes, timing
from verify_lung_axil_board import utilization
from verify_lung_network_board import check_hardware


def rtl_proof(root):
    root=Path(root).resolve();p=checked(root);results={}
    for variant in ['baseline','flow']:
        folder=root/('sim_'+variant)
        d=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        if d['status']!='passed_actual_xsim_flow_'+variant or d['source_sha256']!=p['source_sha256'] or len(d['results'])!=10:
            raise ValueError('Incomplete actual RTL regression')
        for name, row in d['results'].items():
            log=folder/(name+'_run.log');text=log.read_text(encoding='utf-8')
            if file_sha256(log)!=row['log_sha256'] or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
                raise ValueError('Failed or changed RTL log')
        results[variant]=d
    comparisons={}
    for top in ['tb_lung_cached_axi','tb_lung_axil_accel','tb_lung_tensor_burst_axil']:
        a=results['baseline']['results'][top]['cycles'];b=results['flow']['results'][top]['cycles']
        if len(a)!=len(b) or [r['frame'] for r in a]!=[r['frame'] for r in b]:
            raise ValueError('Different comparison cases')
        before=sum(r['cycles'] for r in a);after=sum(r['cycles'] for r in b)
        if after>=before:raise ValueError('No cycle improvement: '+top)
        row=dict(baseline_cycles=before,flow_cycles=after,reduction_percent=100*(before-after)/before)
        if top=='tb_lung_tensor_burst_axil':
            row['by_op']={str(op):dict(baseline_cycles=sum(r['cycles'] for r in a if r['op']==op),
                                      flow_cycles=sum(r['cycles'] for r in b if r['op']==op)) for op in range(4)}
        comparisons[top]=row
    return dict(status='verified_actual_xsim_flow_gains',comparisons=comparisons,
                traffic=p['independent_traffic'],scope='Same BFM, unchanged exact outputs; NOT board latency')


def package(root):
    root=Path(root).resolve();p=checked(root);prior,_=baseline();sim=rtl_proof(root)
    target=root/'lung_flow';out=root/'verification.json'
    if target.exists() or out.exists():raise FileExistsError('Preserve package/proof')
    wns=timing(root/'timing_routed.rpt')
    text=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    m=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',text,re.S)
    if not m or float(m[1].split()[4])<0:raise ValueError('Hold timing failed')
    whs=float(m[1].split()[4]);resources=utilization(root/'utilization_routed.rpt')
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text()):raise ValueError('DRC failed')
    if resources['DSPs']>prior['utilization']['DSPs'] or resources['Block RAM Tile']>prior['utilization']['Block RAM Tile']:
        raise ValueError('Unexpected multiplier/memory increase')
    hardware=check_hardware(ET.parse(root/'deploy/lung_network.hwh').getroot())
    imported={}
    for top in ['lung_conv8_axil','lung_tensor_axil']:
        expected={Path(n).name for n in prior['packaged_source_sha256'] if top in Path(n).parts}
        files=list((root/'ip_repo'/top).rglob('lung_*.v'))
        if len(files)!=len(expected) or {f.name for f in files}!=expected:raise ValueError('IP dependencies changed')
        for f in files:
            if file_sha256(f)!=file_sha256(root/'rtl'/f.name):raise ValueError('IP source differs')
            imported[str(f)]=file_sha256(f)
    log=(root/'python_tests.log').read_text(encoding='utf-8')
    m=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',log)
    if not m or int(m[1])<337 or 'skipped=' in log:raise ValueError('Need complete Python regression')
    mid=root/'sim_mid_faults/verification.json'
    faults=json.loads(mid.read_text())
    if faults['status']!='passed_actual_xsim_burst_mid_command_faults':raise ValueError('Need late read/write fault proof')
    check_hashes(faults['source_sha256'])
    if file_sha256(root/'sim_mid_faults/rtl_run.log')!=faults['log_sha256']:raise ValueError('Changed fault result')
    target.mkdir()
    for f in PACKAGE.iterdir():
        if f.is_file():shutil.copy2(f,target/f.name)
    for name in ['lung_network.bit','lung_network.hwh']:shutil.copy2(root/'deploy'/name,target/name)
    route=dict(clock_mhz=100,wns_ns=wns,whs_ns=whs,bit_sha256=file_sha256(target/'lung_network.bit'),hwh_sha256=file_sha256(target/'lung_network.hwh'))
    def update(name,extra):
        d=json.loads((target/name).read_text())
        d.update(extra)
        d['files_sha256']={n:file_sha256(target/n) for n in d['files_sha256']}
        write_json(target/name,d)
    update('continuous_manifest.json',dict(bit_sha256=route['bit_sha256'],hwh_sha256=route['hwh_sha256'],original_continuous_manifest_sha256=file_sha256(PACKAGE/'continuous_manifest.json')))
    update('cpu_compare_manifest.json',dict(original_continuous_package_sha256=file_sha256(target/'continuous_manifest.json')))
    update('fast_manifest.json',dict(original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),hardware_revision='flow_v1',bit_changed=True,board_tested=False,
        warning='Exact frozen math; response bypass and pool row cache. Actual board head and timing still required.'))
    update('clock_manifest.json',dict(original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=route))
    update('selection_manifest.json',dict(target_folder='/home/xilinx/ct_test/lung_flow'))
    update('fp32_manifest.json',dict(combined_hardware_flow=True,board_tested=False))
    sys.path.insert(0,str(WORKSPACE/'pynq'))
    from lung_fast_benchmark import validate_package
    from lung_clock_benchmark import validate_addon
    from lung_selected_benchmark import validate_selection
    from lung_threeway_benchmark import validate
    validate_package(target);validate_addon(target);validate_selection(target);validate(target)
    evidence=[root/n for n in ['preparation.json','build_flow.tcl','python_tests.log','timing_routed.rpt','utilization_routed.rpt','drc.rpt','configuration_exceptions.rpt',
                               'sim_baseline/verification.json','sim_flow/verification.json','sim_mid_faults/verification.json']]
    evidence += list((root/'sim_baseline').glob('*_run.log'))+list((root/'sim_flow').glob('*_run.log'))+[root/'sim_mid_faults/rtl_run.log']
    write_json(out,dict(status='built_routed_flow_not_board_tested',board_tested=False,model_changed=False,arithmetic_changed=False,
        routed_design=route,hardware=hardware,utilization=resources,simulation=sim,python_tests=int(m[1]),
        source_sha256=p['source_sha256'],packaged_source_sha256=imported,evidence_sha256={str(f):file_sha256(f) for f in evidence},
        package_files_sha256={f.name:file_sha256(f) for f in target.iterdir() if f.is_file()}))
    checked(root)
    print('Audited new board package:',target,route,resources,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default=str(ROOT));ap.add_argument('--rtl-only',action='store_true')
    a=ap.parse_args()
    print(json.dumps(rtl_proof(a.root),indent=2)) if a.rtl_only else package(a.root)
