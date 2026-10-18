"""Board tensor fixtures; run only through the artifact-checked dual-engine test."""
import json
import time

import numpy as np

from lung_axil_smoke import feature_words
from lung_tensor import layout

_TIMED_OUT_BUFFERS=[]


def expected_reads(row):
    ca,h,w=row['input_shape'];cb=0 if row['b_shape'] is None else row['b_shape'][0]
    config=layout(row['op'],h,w,ca,cb)
    if row['op']<2: return config['words']*(4 if row['op']==0 else 1)
    previous=None;reads=0
    for y in range(config['oh']):
        for x in range(config['ow']):
            for channel in range(ca+cb):
                second=channel>=ca;c=channel-ca if second else channel
                sy,sx=(y//2,x//2) if row['op']==3 and not second else (y,x)
                key=(second,sy,sx,c//8)
                if key!=previous: reads+=1;previous=key
    return reads


def run_cases(driver,allocate,archive,rows):
    reports=[]
    for index,row in enumerate(rows):
        prefix=row['prefix'];a=archive[prefix+'_a'];b=archive[prefix+'_b'];gold=archive[prefix+'_expected']
        ca,h,w=a.shape;cb=0 if row['b_shape'] is None else b.shape[0]
        spec=layout(row['op'],h,w,ca,cb)
        if (a.dtype!=np.uint8 or list(a.shape)!=row['input_shape'] or gold.dtype!=np.uint8 or
            list(gold.shape)!=[spec['cout'],spec['oh'],spec['ow']] or
            (cb and (b.dtype!=np.uint8 or list(b.shape)!=[cb,spec['oh'],spec['ow']]))):
            raise ValueError('Wrong tensor fixture array')
        src=dst=src_b=None
        print('Tensor {}/{} {}: allocate/configure/start'.format(index+1,len(rows),row['name']),flush=True)
        try:
            packed=feature_words(a,row['za']);expected=feature_words(gold,row['zo'])
            src=allocate(shape=(len(packed),),dtype=np.uint64);src[:]=packed
            if cb:
                packed_b=feature_words(b,row['zb']);src_b=allocate(shape=(len(packed_b),),dtype=np.uint64);src_b[:]=packed_b
            dst=allocate(shape=(len(expected)+16,),dtype=np.uint64);dst[:]=np.uint64(0xDEADBEEFCAFEF00D)
            begin=time.perf_counter()
            stats=driver.run(src,dst,row['op'],h,w,ca,src_b=src_b,channels_b=cb,
                             **{k:row[k] for k in ['za','zb','zo','ma','sa','mb','sb']})
            elapsed=(time.perf_counter()-begin)*1000
            different=int(np.count_nonzero(np.asarray(dst)[:len(expected)]!=expected))
            if different or np.any(np.asarray(dst)[len(expected):]!=np.uint64(0xDEADBEEFCAFEF00D)):
                raise AssertionError('Tensor output/canary mismatch: '+row['name'])
            if stats['read_words']!=expected_reads(row) or stats['completed']!=index+1:
                raise AssertionError('Tensor read/completion counters differ: '+row['name'])
            result=dict(case=row['name'],op=row['op'],same=True,different_words=0,suffix_canary_words=16,
                        cpu_call_ms=elapsed,**stats)
            reports.append(result);print(json.dumps(result),flush=True)
        finally:
            if driver._inflight or driver._poisoned:
                _TIMED_OUT_BUFFERS.extend([src,src_b,dst])
            else:
                for buffer in [dst,src_b,src]:
                    if buffer is not None: buffer.freebuffer()
    print('Real-board tensor operators: PASS',flush=True)
    return reports
