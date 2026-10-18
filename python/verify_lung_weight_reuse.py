"""Fail-closed audit of real XSim, synthesized resource reports and packed math."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from lung_fpga_contract import requantize
from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


def signed32(v):
    return v-(1<<32) if v&(1<<31) else v


def audit(root):
    root=Path(root).resolve();workspace=Path(__file__).resolve().parents[1]
    result=root/'verification_v2.json'
    if result.exists():raise ValueError('Audit output exists; do not overwrite provenance')
    manifest=root/'vectors/vectors.json';info=json.loads(manifest.read_text(encoding='utf-8'))
    if file_sha256(workspace/'python/lung_weight_reuse_vectors.py')!=info['generated_by_sha256']:
        raise ValueError('Vector generator changed since generation')
    tracked=dict(info['source_sha256'])
    for name,h in info['outputs_sha256'].items():tracked[str(root/'vectors'/name)]=h
    tracked[str(manifest)]=file_sha256(manifest)
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Vector/contract hash mismatch: '+p)
    previous=workspace/'data/lung_tile8_20261014/verification.json'
    old=json.loads(previous.read_text(encoding='utf-8'))
    for p,h in old['source_sha256'].items():
        if file_sha256(p)!=h:raise ValueError('Previously verified source changed: '+p)
    protected=json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    for p,h in protected.items():
        if file_sha256(p)!=h:raise ValueError('Protected medical/model source changed: '+p)
    log=(root/'reuse_run_v3.log').read_text(encoding='utf-8')
    if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Simulation has failure/file warning')
    match=re.search(r'Lung weight reuse RTL: PASS \((\d+) outputs, (\d+) groups, (\d+) weight loads, (\d+) activation beats;',log)
    wanted=(info['outputs'],info['groups'],info['weight_load_beats'],info['activation_beats'])
    if not match or tuple(map(int,match.groups()))!=wanted or not all(wanted):raise ValueError('Missing exact nonempty XSim PASS')
    if 'pending reload progress, sparse, reset, lengths)' not in log:raise ValueError('Missing final protocol test suite')
    banks=root/'vectors';h=read_bank(banks,'header');desc=read_bank(banks,'descriptors')
    acts=read_bank(banks,'activations');weights=read_bank(banks,'weights');wd=read_bank(banks,'weight_descriptors')
    bias=read_bank(banks,'bias');mult=read_bank(banks,'multipliers');shift=read_bank(banks,'shifts')
    acc=read_bank(banks,'accumulators');codes=read_bank(banks,'codes');lanes=read_bank(banks,'lanes');lasts=read_bank(banks,'lasts')
    weight_arrays={}
    for block in range(h[5]):
        start,n,pbase=wd[block*3:block*3+3]
        weight_arrays[block]=np.array([[(word>>(8*i)&255) for i in range(64)] for word in weights[start:start+n]],np.int64).reshape(n,8,8)
        a=weight_arrays[block];a[a>=128]-=256
    checked=0;seen=set();loads=0
    for g in range(h[0]):
        start,n,mask,iz,oz,relu,pbase,rbase,block,new,y,x=desc[g*12:g*12+12]
        if (g==0 or desc[(g-1)*12+8]!=block)!=bool(new):raise ValueError('Incorrect resident block marker')
        if new:loads+=n;seen.add(block)
        words=acts[start:start+n]
        if len(words)!=n or wd[block*3+1]!=n or wd[block*3+2]!=pbase:raise ValueError('Activation/weight length/config mismatch')
        for i,word in enumerate(words):
            if word>>74 or (word>>72&1)!=(i==0) or (word>>73&1)!=(i==n-1):raise ValueError('Bad activation markers')
        q=np.array([[(word>>(8*i)&255)-iz if word>>(64+i)&1 else 0 for i in range(8)] for word in words],np.int64)
        values=(q[:,None,:]*weight_arrays[block]).sum(axis=(0,2))
        indices=[i for i in range(8) if mask>>i&1]
        expected=values[indices]+np.array([signed32(bias[pbase+i]) for i in indices],np.int64)
        expected_code=requantize(expected,np.array([mult[pbase+i] for i in indices],np.int64),
            np.array([shift[pbase+i] for i in indices],np.int64),oz,bool(relu))
        nout=len(indices)
        if (expected.tolist()!=[signed32(v) for v in acc[rbase:rbase+nout]] or
            expected_code.tolist()!=codes[rbase:rbase+nout] or indices!=lanes[rbase:rbase+nout] or
            lasts[rbase:rbase+nout]!=[0]*(nout-1)+[1]):raise ValueError('Independent packed arithmetic failed at group '+str(g))
        checked+=nout
    if checked!=info['outputs'] or loads!=info['weight_load_beats'] or len(seen)!=info['weight_blocks']:
        raise ValueError('Incomplete arithmetic/block coverage')
    utilization=(root/'synthesis_v2/utilization.rpt').read_text(encoding='utf-8')
    resources={}
    for label in ['Slice LUTs*','Slice Registers','DSPs','Block RAM Tile','RAMB36/FIFO*','RAMB18']:
        m=re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)',utilization)
        if not m:raise ValueError('Missing resource '+label)
        resources[label]=float(m[1])
    if resources['DSPs']!=68 or resources['Block RAM Tile']!=7.5 or resources['RAMB36/FIFO*']!=7 or resources['RAMB18']!=1:
        raise ValueError('Unexpected DSP/BRAM synthesis mapping')
    hierarchy=(root/'synthesis_v2/utilization_hier.rpt').read_text(encoding='utf-8')
    for label,expected in [('(tile)',64),('shared_quant',4)]:
        rows=[line for line in hierarchy.splitlines() if line.startswith('|') and label in line]
        if len(rows)!=1 or int(rows[0].split('|')[-2].strip())!=expected:raise ValueError('Unexpected MAC/quant DSP hierarchy')
    timing=(root/'synthesis_v2/timing_synth.rpt').read_text(encoding='utf-8')
    m=re.search(r'WNS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)',timing,re.S)
    if not m:raise ValueError('Missing timing summary')
    wns=float(m[1])
    if wns<0 or not re.search(r'core_clk\s+.*?10\.000',timing):raise ValueError('100 MHz internal timing not met')
    files=[workspace/'fpga'/name for name in ['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v',
        'tb_lung_weight_reuse.sv','synth_lung_weight_reuse.tcl']]
    files += [workspace/'python'/name for name in ['lung_weight_reuse_vectors.py','verify_lung_weight_reuse.py','tests/test_lung_weight_reuse_vectors.py']]
    evidence=[root/'reuse_run_v3.log',root/'synthesis_v2/utilization.rpt',root/'synthesis_v2/utilization_hier.rpt',
        root/'synthesis_v2/timing_synth.rpt',root/'synthesis_v2/weight_reuse_synth.dcp']
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Source changed during audit: '+p)
    report=dict(status='passed_persistent_weight_core',outputs=checked,groups=info['groups'],
        weight_blocks=info['weight_blocks'],weight_load_beats=loads,activation_beats=info['activation_beats'],
        weight_transfer_reduction=info['weight_transfer_reduction'],independent_packed_math=True,
        utilization=resources,post_synth_wns_ns=wns,internal_post_synth_100mhz_timing_met=True,
        protected_medical_source_count=len(protected),medical_model_sources_unchanged=True,
        previous_tile_sources_unchanged=True,source_sha256={str(p):file_sha256(p) for p in files},
        pending_reload_progress_tested=True,reload_exclusion_tested=True,reset_and_length_errors_tested=True,
        evidence_sha256={str(p):file_sha256(p) for p in evidence},
        whole_image_window_controller=False,axi_dma_integrated=False,full_network_rtl=False,
        routed_timing=False,bitstream_generated=False,fpga_deployed=False)
    write_json(result,report);print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_weight_reuse_20261015')
    audit(parser.parse_args().root)
