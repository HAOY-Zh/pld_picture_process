"""Reorder frozen tile vectors into resident-weight blocks, without CT/GT access."""
import argparse
import json
from pathlib import Path

from lung_rtl_vectors import hex_bank
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256


def read_bank(root,name):
    return [int(line,16) for line in (root/(name+'.mem')).read_text(encoding='ascii').splitlines()]


def convert_banks(source,output):
    banks={name:read_bank(source,name) for name in ['header','descriptors','packets','bias',
        'multipliers','shifts','accumulators','codes','lanes','lasts']}
    h=banks['header']
    if len(h)!=5 or not all(h) or len(banks['descriptors'])!=h[0]*12:
        raise ValueError('Invalid/empty source header')
    for name,n in [('packets',h[1]),('bias',h[2]),('multipliers',h[2]),('shifts',h[2]),
                   ('accumulators',h[3]),('codes',h[3]),('lanes',h[3]),('lasts',h[3])]:
        if len(banks[name])!=n:raise ValueError('Wrong source bank length: '+name)
    rows=[banks['descriptors'][i:i+12] for i in range(0,h[0]*12,12)]
    rows.sort(key=lambda r:(r[8],r[11],r[9],r[10]))
    out={n:[] for n in ['weights','weight_descriptors','activations','descriptors','bias',
        'multipliers','shifts','accumulators','codes','lanes','lasts']}
    previous=None;resident=None;blocks=0;original_outputs=[];keys=set()
    for row in rows:
        start,n,mask,iz,oz,relu,pbase,rbase,case,y,x,oc=row
        if not 1<=n<=128 or not 1<=mask<=255 or not 0<=iz<=255 or not 0<=oz<=255 or relu not in (0,1):
            raise ValueError('Invalid frame metadata/depth')
        if (case,oc,y,x) in keys:raise ValueError('Duplicate pixel group')
        keys.add((case,oc,y,x))
        words=banks['packets'][start:start+n]
        params=[banks[name][pbase:pbase+8] for name in ['bias','multipliers','shifts']]
        if len(words)!=n or any(len(p)!=8 for p in params):raise ValueError('Source offset outside bank')
        for i,w in enumerate(words):
            if w<0 or w>>586 or (w>>584&1)!=(i==0) or (w>>585&1)!=(i==n-1):
                raise ValueError('Invalid packet marker/width')
        weights=[w>>64&((1<<512)-1) for w in words]
        identity=(n,mask,iz,oz,relu,params,weights)
        key=(case,oc);new=key!=previous
        if new:
            previous=key;resident=identity;block=blocks;blocks+=1
            param_start=len(out['bias']);out['weight_descriptors'].extend([len(out['weights']),n,param_start])
            out['weights'].extend(weights)
            for name,p in zip(['bias','multipliers','shifts'],params):out[name].extend(p)
        elif identity!=resident:raise ValueError('Weights/configuration changed inside resident block')
        out['descriptors'].extend([len(out['activations']),n,mask,iz,oz,relu,param_start,
            len(out['codes']),block,int(new),y,x])
        for i,w in enumerate(words):
            # 74 bits: LAST / FIRST / input keep / 64-bit activations.
            out['activations'].append((int(i==n-1)<<73)|(int(i==0)<<72)|
                ((w>>576&255)<<64)|(w&((1<<64)-1)))
        count=mask.bit_count()
        if rbase+count>h[3]:raise ValueError('Result offset outside bank')
        for lane in range(count):
            if banks['lasts'][rbase+lane]!=(lane==count-1):raise ValueError('Invalid result LAST')
        for name in ['accumulators','codes','lanes','lasts']:out[name].extend(banks[name][rbase:rbase+count])
        original_outputs.extend(range(rbase,rbase+count))
    if sorted(original_outputs)!=list(range(h[3])):raise ValueError('Result coverage changed')
    header=[h[0],len(out['activations']),blocks*8,len(out['codes']),h[4],blocks,len(out['weights'])]
    widths=dict(weights=512,weight_descriptors=32,activations=74,descriptors=32,bias=32,
        multipliers=32,shifts=6,accumulators=32,codes=8,lanes=3,lasts=1)
    hex_bank(output/'header.mem',header,32)
    for name,values in out.items():hex_bank(output/(name+'.mem'),values,widths[name])
    return dict(header=header,groups=h[0],outputs=h[3],weight_blocks=blocks,
        old_weight_beats=h[1],weight_load_beats=len(out['weights']),activation_beats=h[1],
        weight_transfer_reduction=1-len(out['weights'])/h[1],
        old_payload_bytes=h[1]*73,new_payload_bytes=h[1]*9+len(out['weights'])*64,
        warning='Payload estimate excludes markers, parameters, DMA framing; not measured DDR/FPS')


def generate(args):
    source=Path(args.source).resolve();manifest=source/'vectors.json'
    info=json.loads(manifest.read_text(encoding='utf-8'))
    tracked={str(manifest):file_sha256(manifest)}
    for name,digest in info['outputs_sha256'].items():
        p=source/name
        if file_sha256(p)!=digest:raise ValueError('Frozen tile bank changed: '+name)
        tracked[str(p)]=digest
    for p,digest in info['source_sha256'].items():
        if file_sha256(p)!=digest:raise ValueError('Frozen contract changed: '+p)
        tracked[p]=digest
    output=fresh_output(args.outdir);report=convert_banks(source,output)
    for p,digest in tracked.items():
        if file_sha256(p)!=digest:raise ValueError('Source changed during conversion: '+p)
    report.update(status='complete_weight_reuse_vectors',source_sha256=tracked,
        generated_by_sha256=file_sha256(__file__),outputs_sha256={p.name:file_sha256(p) for p in output.glob('*.mem')})
    write_json(output/'vectors.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','outputs_sha256']},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/lung_tile8_20261014/vectors')
    parser.add_argument('--outdir',required=True)
    generate(parser.parse_args())
