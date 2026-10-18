"""Independent address/window/convolution audit plus actual XSim/OOC evidence."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from lung_fpga_contract import conv_accumulator,requantize
from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank
from verify_lung_weight_reuse import signed32


def audit(root):
    root=Path(root).resolve();workspace=Path(__file__).resolve().parents[1];result=root/'verification.json'
    if result.exists():raise ValueError('Audit exists; preserve provenance')
    manifest=root/'vectors/vectors.json';info=json.loads(manifest.read_text(encoding='utf-8'))
    if file_sha256(workspace/'python/lung_window_vectors.py')!=info['generated_by_sha256']:raise ValueError('Generator changed')
    tracked={str(manifest):file_sha256(manifest),**info['source_sha256']}
    for name,h in info['outputs_sha256'].items():tracked[str(root/'vectors'/name)]=h
    old=json.loads((workspace/'data/lung_weight_reuse_20261015/verification_v2.json').read_text(encoding='utf-8'))
    tracked.update(old['source_sha256'])
    protected=json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    tracked.update(protected)
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Frozen input/source changed: '+p)
    log=(root/'window_run.log').read_text(encoding='utf-8');bounds=(root/'bounds_run.log').read_text(encoding='utf-8')
    if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log+'\n'+bounds,re.I):raise ValueError('RTL log failure/file warning')
    m=re.search(r'Lung window convolution RTL: PASS \((\d+) frames, (\d+) outputs, (\d+) windows, (\d+) memory reads, (\d+) weight loads;',log)
    want=(info['frames'],info['outputs'],info['window_beats'],info['memory_requests'],info['weight_beats'])
    if not m or tuple(map(int,m.groups()))!=want or not all(want):raise ValueError('Incomplete nonempty RTL coverage')
    if 'shared reset, memory errors)' not in log or 'Lung feature window bounds RTL: PASS (4 configurations;' not in bounds:
        raise ValueError('Missing protocol/address boundary tests')
    banks=root/'vectors';names=['header','frames','features','weights','bias','multipliers','shifts','windows',
        'window_coords','requests','accumulators','codes','lanes','lasts','coords']
    b={name:read_bank(banks,name) for name in names};h=b['header'];windows=reads=outputs=0
    if h!=info['header']:raise ValueError('Header differs from manifest')
    for frame in range(h[0]):
        base,height,width,cin,kernel,pad,iz,oz,mask,relu,wstart,n,pbase,rbase,vbase,pixels=b['frames'][frame*16:frame*16+16]
        blocks=(cin+7)//8;cout=mask.bit_count();oh=height+2*pad-kernel+1;ow=width+2*pad-kernel+1
        if mask!=(1<<cout)-1 or n!=blocks*kernel*kernel or pixels!=oh*ow or vbase!=windows or rbase!=outputs:
            raise ValueError('Frame count/coverage mismatch')
        q=np.empty((cin,height,width),np.uint8)
        for y in range(height):
            for x in range(width):
                for c in range(cin):q[c,y,x]=b['features'][base+(y*width+x)*blocks+c//8]>>(8*(c%8))&255
        w=np.empty((cout,cin,kernel,kernel),np.int8)
        for ky in range(kernel):
            for kx in range(kernel):
                for c in range(cin):
                    word=b['weights'][wstart+(ky*kernel+kx)*blocks+c//8]
                    for o in range(cout):
                        value=word>>(8*(o*8+c%8))&255;w[o,c,ky,kx]=value-256 if value>=128 else value
        bias=np.array([signed32(v) for v in b['bias'][pbase:pbase+cout]],np.int32)
        acc=conv_accumulator(q,w,bias,iz,pad);codes=requantize(acc,
            np.array(b['multipliers'][pbase:pbase+cout],np.int64),np.array(b['shifts'][pbase:pbase+cout],np.int64),oz,bool(relu))
        ordered=acc.transpose(1,2,0).ravel();ordered_codes=codes.transpose(1,2,0).ravel();nout=ordered.size
        if ordered.tolist()!=[signed32(v) for v in b['accumulators'][outputs:outputs+nout]] or ordered_codes.tolist()!=b['codes'][outputs:outputs+nout]:
            raise ValueError('Independent convolution failed: '+str(frame))
        for y in range(oh):
            for x in range(ow):
                for lane in range(cout):
                    last=lane==cout-1;final=last and y==oh-1 and x==ow-1
                    if b['lanes'][outputs]!=lane or b['lasts'][outputs]!=last or b['coords'][outputs]!=((int(final)<<20)|(y<<10)|x):raise ValueError('Output coordinate/tag failed')
                    outputs+=1
                for ky in range(kernel):
                    for kx in range(kernel):
                        for cb in range(blocks):
                            sy=y+ky-pad;sx=x+kx-pad;inside=0<=sy<height and 0<=sx<width
                            keep=(1<<min(8,cin-cb*8))-1;first=ky==0 and kx==0 and cb==0
                            last=ky==kernel-1 and kx==kernel-1 and cb==blocks-1;final=last and y==oh-1 and x==ow-1
                            if inside:
                                address=base+(sy*width+sx)*blocks+cb
                                if address!=b['requests'][reads]:raise ValueError('Independent address mismatch')
                                data=b['features'][address];reads+=1
                            else:data=sum(iz<<(8*i) for i in range(8))
                            expected=(int(final)<<74)|(int(last)<<73)|(int(first)<<72)|(keep<<64)|data
                            if b['windows'][windows]!=expected or b['window_coords'][windows]!=((y<<10)|x):raise ValueError('Independent window/padding mismatch')
                            windows+=1
    if (outputs,windows,reads)!=(h[4],h[5],h[6]):raise ValueError('Incomplete independent math coverage')
    util=(root/'synthesis/utilization.rpt').read_text(encoding='utf-8');resources={}
    for label in ['Slice LUTs*','Slice Registers','DSPs','Block RAM Tile','RAMB36/FIFO*','RAMB18']:
        m=re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)',util)
        if not m:raise ValueError('Missing resource '+label)
        resources[label]=float(m[1])
    if resources['DSPs']!=70 or resources['Block RAM Tile']!=7.5:raise ValueError('Unexpected actual DSP/BRAM mapping')
    hierarchy=(root/'synthesis/utilization_hier.rpt').read_text(encoding='utf-8')
    for label,expected in [('(tile)',64),('shared_quant',4),('lung_feature_window',2)]:
        rows=[line for line in hierarchy.splitlines() if line.startswith('|') and label in line]
        if len(rows)!=1 or int(rows[0].split('|')[-2].strip())!=expected:raise ValueError('Unexpected hierarchy for '+label)
    timing=(root/'synthesis/timing_synth.rpt').read_text(encoding='utf-8')
    m=re.search(r'WNS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)',timing,re.S)
    if not m or float(m[1])<0 or not re.search(r'core_clk\s+.*?10\.000',timing):raise ValueError('Internal 100MHz timing failed')
    files=[workspace/'fpga'/name for name in ['lung_feature_window.v','lung_conv8_window.v','tb_lung_conv8_window.sv',
        'tb_lung_feature_window_bounds.sv','synth_lung_conv8_window.tcl']]
    files += [workspace/'python'/name for name in ['lung_window_vectors.py','verify_lung_window.py','tests/test_lung_window_vectors.py']]
    evidence=[root/name for name in ['window_run.log','bounds_run.log','synthesis/utilization.rpt','synthesis/utilization_hier.rpt',
        'synthesis/timing_synth.rpt','synthesis/conv8_window_synth.dcp']]
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Input changed during audit: '+p)
    report=dict(status='passed_feature_window_convolution',frames=h[0],outputs=outputs,window_beats=windows,memory_requests=reads,
        padding_without_read=windows-reads,independent_address_window_convolution=True,utilization=resources,
        post_synth_wns_ns=float(m[1]),internal_post_synth_100mhz_timing_met=True,
        protected_medical_source_count=len(protected),medical_and_previous_core_sources_unchanged=True,
        bounds512_row_transition_and_address_overflow_tested=True,full_512x512_frame_exhaustively_tested=False,
        variable_memory_latency_backpressure_error_and_shared_reset_tested=True,
        source_sha256={str(p):file_sha256(p) for p in files},evidence_sha256={str(p):file_sha256(p) for p in evidence},
        line_cache=False,feature_frame_storage_onchip=False,axi_dma_integrated=False,full_network_rtl=False,
        routed_timing=False,bitstream_generated=False,fpga_deployed=False)
    write_json(result,report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_window_20261015')
    audit(parser.parse_args().root)
