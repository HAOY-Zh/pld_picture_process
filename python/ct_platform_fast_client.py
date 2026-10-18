"""Strict lossless binary transport; compatible with the PL receipt contract."""
import base64
from collections import OrderedDict
import hashlib
import json
import time
from urllib import request,error
import uuid
import zlib

import numpy as np
from ct_basic_processing import parameters

TRANSPORT='ct_basic_zlib_cache_v1'


def basic(client,hu,supplied):
    params=parameters(supplied);op=params['operation']
    if op not in client.manifest['basic']:raise ValueError('No PL module')
    rounded=np.rint(hu)
    if (rounded.shape!=(512,512) or not np.isfinite(rounded).all() or
            rounded.min()<-32768 or rounded.max()>32767):raise ValueError('Need native int16 HU')
    raw=rounded.astype('<i2').tobytes();input_hash=hashlib.sha256(raw).hexdigest()
    known=getattr(client,'_fast_known_inputs',None)
    if known is None:known=client._fast_known_inputs=OrderedDict()
    cache_only=input_hash in known
    start=time.perf_counter();cache_miss=False
    while True:
        wire=b'' if cache_only else zlib.compress(raw,1)
        rid=uuid.uuid4().hex
        req=request.Request(client.url+'/v2/basic',data=wire,headers={
            'Authorization':'Bearer '+client.token,'Content-Type':'application/octet-stream',
            'X-Request-ID':rid,'X-Input-SHA256':input_hash,
            'X-Parameters':json.dumps(params),'X-Input-Encoding':'cache' if cache_only else 'zlib-int16'})
        try:
            with client.opener.open(req,timeout=120) as response:
                header=response.headers.get('X-Receipt','')
                if len(header)>16384:raise ValueError('Oversized receipt')
                if response.headers.get_content_type()!='application/octet-stream':raise ValueError('Wrong output type')
                compressed=response.read(262144+1025)
                if len(compressed)>262144+1024:raise ValueError('Oversized compressed output')
                if response.headers.get('Content-Length')!=str(len(compressed)):raise ValueError('Truncated output')
            break
        except error.HTTPError as exc:
            # A typed cache miss is rejected BEFORE any hardware command. No
            # ambiguous timeout, busy or hardware fault may enter this branch.
            payload=exc.read(8193)
            try:code=json.loads(payload).get('error_code') if len(payload)<=8192 else None
            except (ValueError,AttributeError):code=None
            if not(cache_only and exc.code==409 and code=='INPUT_CACHE_MISS'):raise
            cache_only=False;cache_miss=True;known.pop(input_hash,None)
    receipt=json.loads(base64.b64decode(header,validate=True))
    expected=client.manifest['basic'][op]
    if (receipt.get('schema')!='ct_basic_pl_result_v1' or receipt.get('backend')!='FPGA_BASIC' or
            receipt.get('request_id')!=rid or receipt.get('input_sha256')!=input_hash or
            receipt.get('parameters')!=params or receipt.get('shape')!=[512,512] or
            receipt.get('platform_manifest_sha256')!=client.manifest['platform_manifest_sha256'] or
            receipt.get('bit_sha256')!=expected['bit_sha256'] or receipt.get('hwh_sha256')!=expected['hwh_sha256']):
        raise ValueError('Fast PL identity/request/parameter mismatch')
    decoder=zlib.decompressobj();output=decoder.decompress(compressed,262145)
    if (len(output)!=262144 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or
            hashlib.sha256(output).hexdigest()!=receipt.get('output_sha256')):
        raise ValueError('Actual output hash/length mismatch')
    stats=receipt['stats'];transport=receipt['transport']
    names=('dma_ms','preparation_ms','overlay_switch_ms','fclk0_mhz','cpu_mhz')
    if (not all(type(stats.get(k)) in (int,float) and np.isfinite(stats[k]) for k in names) or
            min(stats[k] for k in names[:3])<0 or stats['fclk0_mhz']<=0 or stats['cpu_mhz']<=0 or
            stats.get('guard_verified') is not True or stats.get('shared_basic_overlay') is not True or
            stats.get('input_bytes')!=1048576 or stats.get('output_bytes')!=1048576 or
            transport.get('schema')!=TRANSPORT or transport.get('input_cache_hit') is not cache_only or
            transport.get('request_body_bytes')!=len(wire) or transport.get('response_body_bytes')!=len(compressed)):
        raise ValueError('Incomplete fast PL timing/transport receipt')
    receipt['roundtrip_ms']=(time.perf_counter()-start)*1000
    receipt['transport']['cache_miss_safe_upload']=cache_miss
    known[input_hash]=True;known.move_to_end(input_hash)
    while len(known)>8:known.popitem(last=False)
    return np.frombuffer(output,np.uint8).reshape(512,512).copy(),receipt
