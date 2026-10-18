"""Bind initial timing failure, physical-only repair, routed checkpoint and bit."""
import argparse
import json
from pathlib import Path
import re

from lung_flow_prepare import checked,WORKSPACE
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import timing


def run(root):
    root=Path(root).resolve();checked(root);destination=root/'physopt_verification.json'
    if destination.exists():raise FileExistsError('Preserve physical optimization proof')
    script=WORKSPACE/'fpga/resume_lung_lut_physopt.tcl'
    if file_sha256(script)!=file_sha256(root/script.name):raise ValueError('Different executed physical optimization script')
    initial=(root/'route_initial_failed/timing_routed.rpt').read_text()
    final=(root/'timing_routed.rpt').read_text();clock='100.000'
    if clock not in initial or clock not in final:raise ValueError('Clock target differs')
    wns=timing(root/'timing_routed.rpt')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',final,re.S)
    if not match or float(match[1].split()[4])<0:raise ValueError('Hold timing failed')
    whs=float(match[1].split()[4])
    for key in ['checking no_clock (0)','checking unconstrained_internal_endpoints (0)']:
        if key not in final:raise ValueError('Incomplete timing constraints')
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',(root/'drc.rpt').read_text()):raise ValueError('DRC failed')
    raw=(root/'physopt_console.log').read_bytes()
    log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8').replace('\r\n','\n')
    if not re.search(r'^Unchanged 100MHz constraints: post-route physical optimization and bit generation PASS\.$',log,re.M):
        raise ValueError('Missing successful current-design bit generation')
    files=[script,root/script.name,root/'build_console.log',root/'physopt_console.log',root/'routed_physopt.dcp',
        root/'route_status_physopt.rpt',root/'timing_routed.rpt',root/'utilization_routed.rpt',root/'drc.rpt',
        root/'configuration_exceptions.rpt',root/'deploy/lung_network.bit',root/'deploy/lung_network.hwh']
    files+=list((root/'route_initial_failed').glob('*.rpt'))
    write_json(destination,dict(status='verified_100mhz_physical_only_repair',clock_mhz=100,wns_ns=wns,whs_ns=whs,
        rtl_changed_after_simulation=False,constraints_relaxed=False,
        current_design_bit_written_directly=True,evidence_sha256={str(f):file_sha256(f) for f in files},
        note='Supplementary proof; final package audit and actual board acceptance are separate'))
    print('Physical-only 100MHz repair audited:',wns,whs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_lut_20261017')
    run(parser.parse_args().root)
