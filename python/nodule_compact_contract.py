"""Lossless five-plane transport and bounded binary full-head response."""
import hashlib
import json
import struct
import zlib
import numpy as np
from nodule_canvas_contract import shapes,inflate,validate_stats

INTERFACE='ct_shared_nodule_compact_v2'
ROUTE='/v2/shared/nodule-canvas'
MAX_INPUT=5*1024*1024
MAX_HEAD=5*512*512
MAX_META=65536


def digest(raw):return hashlib.sha256(raw).hexdigest()


def encode_response(meta,head):
    raw=json.dumps(meta,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_META:raise ValueError('Oversized response metadata')
    return struct.pack('!I',len(raw))+raw+zlib.compress(head,1)


def decode_response(raw,expected):
    if len(raw)<4:raise ValueError('Truncated binary response')
    size=struct.unpack('!I',raw[:4])[0]
    if not 0<size<=MAX_META or len(raw)<4+size:raise ValueError('Invalid binary metadata length')
    meta=json.loads(raw[4:4+size])
    head=inflate(raw[4+size:],expected)
    return meta,head
