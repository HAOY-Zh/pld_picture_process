"""Audit cached AXI traffic against a second FIFO model and frozen math proofs."""
import argparse
import json
from pathlib import Path
import re

from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


def audit(root):
    root=Path(root).resolve();workspace=Path(__file__).resolve().parents[1];out=root/'verification.json'
    if out.exists():raise ValueError('Audit exists; preserve evidence')
    manifest=root/'vectors/vectors.json';info=json.loads(manifest.read_text(encoding='utf-8'))
    if info['sets']!=256 or info['ways']!=3 or info['line_words']!=16:raise ValueError('Unexpected cache geometry')
    if file_sha256(workspace/'python/lung_cache_vectors.py')!=info['generated_by_sha256']:raise ValueError('Generator changed')
    tracked={str(manifest):file_sha256(manifest),**info['source_sha256']}
    for name,digest in info['outputs_sha256'].items():tracked[str(root/'vectors'/name)]=digest
    previous=workspace/'data/lung_window_20261015/verification.json'
    old=json.loads(previous.read_text(encoding='utf-8'))
    if old['status']!='passed_feature_window_convolution' or not old['independent_address_window_convolution']:
        raise ValueError('Missing prior independent math proof')
    tracked.update(old['source_sha256'])
    old_weights=json.loads((workspace/'data/lung_weight_reuse_20261015/verification_v2.json').read_text(encoding='utf-8'))
    tracked.update(old_weights['source_sha256'])
    old_manifest=workspace/'data/lung_window_20261015/vectors/vectors.json'
    old_info=json.loads(old_manifest.read_text(encoding='utf-8'))
    # All mathematical fixtures, including reference windows/coordinates, must
    # be byte-identical to the independently recomputed previous-stage banks.
    for name,digest in old_info['outputs_sha256'].items():
        if name!='header.mem' and info['outputs_sha256'].get(name)!=digest:raise ValueError('Frozen math bank changed: '+name)
    protected=json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    tracked.update(protected)
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Frozen source/vector changed: '+p)
    logs={name:(root/name).read_text(encoding='utf-8') for name in ['cached_axi_run.log','cache_protocol_run_v2.log','axi_protocol_run_v2.log','cache_capacity_run.log']}
    for name,log in logs.items():
        if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Failure in '+name)
    m=re.search(r'Lung cached AXI convolution RTL: PASS \((\d+) frames, (\d+) outputs, (\d+) windows, (\d+) requests, (\d+) AXI bursts, (\d+) AXI read beats;',logs['cached_axi_run.log'])
    h=read_bank(root/'vectors','header')
    want=(h[0],h[4],h[5],h[6],h[7],h[8])
    if not m or tuple(map(int,m.groups()))!=want or not all(want):raise ValueError('Missing exact nonempty AXI integration PASS')
    for name,marker in [('cache_protocol_run_v2.log','Lung feature cache protocol RTL: PASS (28 reads;'),
        ('axi_protocol_run_v2.log','Lung AXI read64 protocol RTL: PASS (10 cases;'),
        ('cache_capacity_run.log','Lung feature cache capacity RTL: PASS (18437 reads;')]:
        if marker not in logs[name]:raise ValueError('Missing protocol/capacity proof: '+name)
    frames=read_bank(root/'vectors','frames');requests=read_bank(root/'vectors','requests')
    burst_addresses=read_bank(root/'vectors','burst_addresses');burst_beats=read_bank(root/'vectors','burst_beats');traffic=read_bank(root/'vectors','traffic')
    request_index=burst_index=read_beats=all_hits=0
    for f in range(h[0]):
        base,height,width,cin,kernel,pad=frames[f*16:f*16+6];blocks=(cin+7)//8;end=base+height*width*blocks
        # A queue per set: second model uses FIFO lists, not the generator's
        # indexed way/round-robin implementation. Hits never reorder the queue.
        queues=[[] for _ in range(256)];hits=misses=frame_requests=frame_beats=0;start=burst_index
        for y in range(height+2*pad-kernel+1):
            for x in range(width+2*pad-kernel+1):
                for ky in range(kernel):
                    for kx in range(kernel):
                        sy=y+ky-pad;sx=x+kx-pad
                        if not (0<=sy<height and 0<=sx<width):continue
                        for cb in range(blocks):
                            address=base+(sy*width+sx)*blocks+cb
                            if requests[request_index]!=address:raise ValueError('Independent upstream address mismatch')
                            request_index+=1;frame_requests+=1
                            line=address//16;queue=queues[line%256];tag=line//256
                            if tag in queue:hits+=1
                            else:
                                if len(queue)==3:queue.pop(0)
                                queue.append(tag);misses+=1
                                a=max(base,line*16);n=min(end,line*16+16)-a
                                if not 1<=n<=16 or a+n>end or a<base or a>=2**29 or (a%512)+n>512:
                                    raise ValueError('Unsafe burst span/physical address/4KB boundary')
                                if (a,n)!=(burst_addresses[burst_index],burst_beats[burst_index]):raise ValueError('Independent FIFO burst mismatch')
                                burst_index+=1;frame_beats+=n
        if traffic[f*6:f*6+6]!=[frame_requests,hits,misses,start,misses,frame_beats]:raise ValueError('Per-frame traffic mismatch')
        all_hits+=hits;read_beats+=frame_beats
    if (request_index,burst_index,read_beats)!=(h[6],h[7],h[8]) or all_hits!=info['cache_hits']:
        raise ValueError('Incomplete cache traffic coverage')
    util=(root/'synthesis/utilization.rpt').read_text(encoding='utf-8');resources={}
    for label in ['Slice LUTs*','Slice Registers','DSPs','Block RAM Tile','RAMB36/FIFO*','RAMB18']:
        m=re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)',util)
        if not m:raise ValueError('Missing resource '+label)
        resources[label]=float(m[1])
    if resources['DSPs']!=70 or resources['Block RAM Tile']!=31.5:raise ValueError('Unexpected synthesis mapping')
    hierarchy=(root/'synthesis/utilization_hier.rpt').read_text(encoding='utf-8')
    for label,dsp,bram in [('lung_feature_cache',0,24),('lung_axi_read64',0,0)]:
        rows=[line for line in hierarchy.splitlines() if line.startswith('|') and label in line]
        if len(rows)!=1:raise ValueError('Missing synthesis hierarchy '+label)
        cells=[v.strip() for v in rows[0].split('|')[1:-1]]
        if int(cells[-1])!=dsp or int(cells[-3])+int(cells[-2])/2!=bram:raise ValueError('Cache/reader hierarchy resource mismatch')
    timing=(root/'synthesis/timing_synth.rpt').read_text(encoding='utf-8')
    m=re.search(r'WNS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)',timing,re.S)
    if not m or float(m[1])<0 or not re.search(r'core_clk\s+.*?10\.000',timing):raise ValueError('100MHz internal timing not met')
    sources=[workspace/'fpga'/name for name in ['lung_axi_read64.v','lung_feature_cache.v','lung_conv8_cached_axi.v',
        'tb_lung_cached_axi.sv','tb_lung_cache_protocol.sv','tb_lung_axi_read64.sv','synth_lung_cached_axi.tcl']]
    sources += [workspace/'python'/name for name in ['lung_cache_vectors.py','verify_lung_cached_axi.py','tests/test_lung_cache_vectors.py']]
    evidence=[root/name for name in logs]+[root/'synthesis'/name for name in ['utilization.rpt','utilization_hier.rpt','timing_synth.rpt','cached_axi_synth.dcp']]
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Source changed during audit: '+p)
    report=dict(status='passed_cached_axi_read_convolution',frames=h[0],outputs=h[4],window_beats=h[5],internal_requests=request_index,
        cache_hits=all_hits,axi_bursts=burst_index,axi_read_beats=read_beats,read_payload_reduction=1-read_beats/request_index,
        cache_sets=256,cache_ways=3,line_words=16,cache_payload_bytes=98304,
        independent_fifo_traffic=True,previous_independent_math_banks_unchanged=True,
        protocol_error_retry_and_capacity_tests=True,utilization=resources,post_synth_wns_ns=float(m[1]),
        internal_post_synth_100mhz_timing_met=True,protected_medical_source_count=len(protected),medical_and_previous_sources_unchanged=True,
        previous_math_proof_sha256=file_sha256(previous),source_sha256={str(p):file_sha256(p) for p in sources},
        evidence_sha256={str(p):file_sha256(p) for p in evidence},
        axi_read_rtl_simulated=True,output_writer=False,axi_lite_control=False,full_network_rtl=False,
        real_ddr_measured=False,routed_timing=False,bitstream_generated=False,fpga_deployed=False)
    write_json(out,report);print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_cache_axi_20261015')
    audit(parser.parse_args().root)
