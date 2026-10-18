"""Independent three-way feature-cache model and frozen AXI traffic fixtures."""
import argparse
import json
from pathlib import Path

from lung_rtl_vectors import hex_bank
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


class FeatureCacheModel:
    def __init__(self,base,words,sets=256):
        if sets<2 or sets>512 or sets&(sets-1) or words<=0 or base<0 or base+words>2**32:
            raise ValueError('Bad cache configuration')
        self.base=base;self.end=base+words;self.sets=sets
        self.tags=[[None]*3 for _ in range(sets)];self.next=[0]*sets
        self.hits=0;self.misses=0;self.bursts=[]

    def read(self,address,fill_ok=True):
        if not self.base<=address<self.end:raise ValueError('Address outside configured feature span')
        block=address//16;index=block%self.sets;tag=block//self.sets
        if tag in self.tags[index]:self.hits+=1;return None
        self.misses+=1;victim=self.next[index];self.tags[index][victim]=None
        begin=max(block*16,self.base);end=min(block*16+16,self.end);burst=(begin,end-begin)
        if fill_ok:
            self.tags[index][victim]=tag;self.next[index]=(victim+1)%3
        self.bursts.append(burst);return burst


def generate(args):
    src=Path(args.source).resolve();manifest=src/'vectors.json';info=json.loads(manifest.read_text(encoding='utf-8'))
    tracked={str(manifest):file_sha256(manifest),**info['source_sha256']}
    for name,h in info['outputs_sha256'].items():tracked[str(src/name)]=h
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Frozen window vector changed: '+p)
    out=fresh_output(args.outdir)
    # Runtime-generated fixture banks, not edits of any old source or result.
    for name in info['outputs_sha256']:
        if name!='header.mem':hex_bank(out/name,read_bank(src,name[:-4]),{'features':64,'weights':512,'frames':32,'bias':32,
            'multipliers':32,'shifts':6,'windows':75,'window_coords':20,'requests':32,'accumulators':32,
            'codes':8,'lanes':3,'lasts':1,'coords':21}[name[:-4]])
    h=read_bank(src,'header');frames=read_bank(src,'frames');requests=read_bank(src,'requests')
    read_index=0;bursts=[];traffic=[];frame_reports=[];total_hits=0;total_beats=0
    for f in range(h[0]):
        row=frames[f*16:f*16+16];base,height,width,cin,kernel,pad=row[:6];blocks=(cin+7)//8
        model=FeatureCacheModel(base,height*width*blocks,args.sets);before=len(bursts);needed=0
        for y in range(height+2*pad-kernel+1):
            for x in range(width+2*pad-kernel+1):
                for ky in range(kernel):
                    for kx in range(kernel):
                        sy=y+ky-pad;sx=x+kx-pad
                        if 0<=sy<height and 0<=sx<width:
                            for cb in range(blocks):
                                address=base+(sy*width+sx)*blocks+cb
                                if requests[read_index]!=address:raise ValueError('Independent request order mismatch')
                                model.read(address);read_index+=1;needed+=1
        beats=sum(n for _,n in model.bursts)
        bursts.extend(model.bursts);traffic.extend([needed,model.hits,model.misses,before,len(model.bursts),beats])
        total_hits+=model.hits;total_beats+=beats
        frame_reports.append(dict(frame=f,requests=needed,hits=model.hits,misses=model.misses,bursts=len(model.bursts),axi_read_beats=beats))
    if read_index!=h[6]:raise ValueError('Incomplete request coverage')
    hex_bank(out/'header.mem',h+[len(bursts),total_beats],32)
    hex_bank(out/'burst_addresses.mem',[address for address,_ in bursts],32)
    hex_bank(out/'burst_beats.mem',[beats for _,beats in bursts],8);hex_bank(out/'traffic.mem',traffic,32)
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Input changed during generation: '+p)
    report=dict(status='complete_cached_axi_vectors',header=h+[len(bursts),total_beats],sets=args.sets,ways=3,line_words=16,
        internal_requests=h[6],cache_hits=total_hits,axi_bursts=len(bursts),axi_read_beats=total_beats,
        read_payload_reduction=1-total_beats/h[6],frames=frame_reports,
        source_sha256=tracked,generated_by_sha256=file_sha256(__file__),
        outputs_sha256={p.name:file_sha256(p) for p in out.glob('*.mem')},
        warning='Traffic of frozen probes with per-frame flush; not real DDR/FPS/full network')
    write_json(out/'vectors.json',report)
    print({k:v for k,v in report.items() if k not in ['source_sha256','outputs_sha256','frames']},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/lung_window_20261015/vectors')
    parser.add_argument('--outdir',required=True);parser.add_argument('--sets',type=int,default=256)
    generate(parser.parse_args())
