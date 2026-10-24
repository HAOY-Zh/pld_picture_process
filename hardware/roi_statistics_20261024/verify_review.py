"""Verify reviewed source identity, returned integer statistics and OOC reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source, result):
    provenance=json.loads((source/'source_provenance.json').read_text(encoding='utf-8'))
    for name,expected in provenance['files_sha256'].items():
        if sha(source/name)!=expected:raise ValueError('Imported source/evidence changed: '+name)
    for name in ('fpga/pynq_z2_roi.v','fpga/pynq_z2_roi_statistics.v','tests/tb_roi_statistics.sv'):
        if sha(result/name)!=provenance['files_sha256'][name]:raise ValueError('Simulated source mismatch')
    expected=json.loads((result/'vectors/statistics_manifest.json').read_text(encoding='utf-8'))
    with (result/'results/fpga_statistics/actual_statistics.csv').open(newline='') as stream:
        actual=list(csv.DictReader(stream))
    if len(actual)!=len(expected['cases']):raise ValueError('Missing returned frames')
    for i,(a,e) in enumerate(zip(actual,expected['cases'])):
        minmax=0 if not e['count'] else (e['min']&65535)|((e['max']&65535)<<16)
        if (int(a['case_id']),int(a['count']),int(a['sum']),int(a['sumsq']),int(a['minmax_hex'],16),int(a['valid']))!=(
                i,e['count'],e['sum'],e['sumsq'],minmax,int(not e['bad_keep'])):
            raise ValueError('Returned statistics mismatch: '+e['name'])
    log=(result/'build/sim/xsim.log').read_text(encoding='utf-8',errors='replace')
    if not re.search(r'ROI_STATISTICS_XSIM_ALL_PASS frames=9 pixels=803180',log) or re.search(r'Fatal:|ERROR:',log):
        raise ValueError('Incomplete/failed simulation')
    timing=(result/'results/fpga_statistics/timing_status.txt').read_text()
    wns=float(re.search(r'WNS_NS=([-+.\deE]+)',timing)[1]);whs=float(re.search(r'WHS_NS=([-+.\deE]+)',timing)[1])
    if min(wns,whs)<0 or 'SCOPE=OUT_OF_CONTEXT_ONLY' not in timing:raise ValueError('OOC timing failed')
    report=(result/'results/fpga_statistics/utilization_routed.rpt').read_text(errors='replace')
    resources={}
    for label,pattern in [('LUT',r'Slice LUTs\*?'),('FF',r'Slice Registers'),('DSP',r'DSPs'),('BRAM',r'Block RAM Tile')]:
        match=re.search(r'\|\s*'+pattern+r'\s*\|\s*([\d.]+)\s*\|',report)
        if match:resources[label]=float(match[1])
    record=dict(status='REVIEWED_RTL_SIMULATION_AND_OOC_PASS',source_archive_sha256=provenance['archive_sha256'],
        frames=len(actual),pixels=expected['total_pixels'],pixel_and_integer_stat_differences=0,
        clock_mhz=100,internal_ooc_wns_ns=wns,internal_ooc_whs_ns=whs,resources_routed=resources,
        scope='Standalone OOC; no full-system external timing or real clock tree',
        new_integrated_bitstream_built=False,board_validated=False,
        existing_cnn_model_changed=False,existing_roi_bit_changed=False)
    (result/'review_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    args=p.parse_args();print(json.dumps(verify(args.source,args.result),indent=2))
