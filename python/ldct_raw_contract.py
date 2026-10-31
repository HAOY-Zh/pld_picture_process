"""Bounded uncompressed binary LDCT framing; bytes and HU never changed."""
import json
import struct

INTERFACE='ct_ldct_raw_exact_v1'
ROUTE='/v4/ldct'
MAX_META=65536
MAX_RAW=512*512*2


def encode_raw(meta,raw):
    if not isinstance(raw,bytes)or not 0<len(raw)<=MAX_RAW:raise ValueError('Invalid raw LDCT payload')
    header=json.dumps(meta,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')
    if not 0<len(header)<=MAX_META:raise ValueError('Oversized raw LDCT metadata')
    return struct.pack('<I',len(header))+header+raw


def decode_raw(encoded,expected):
    if type(expected)is not int or not 0<expected<=MAX_RAW or len(encoded)<4:raise ValueError('Invalid raw LDCT length')
    size=struct.unpack('<I',encoded[:4])[0]
    if not 0<size<=MAX_META or len(encoded)!=4+size+expected:raise ValueError('Truncated or oversized raw LDCT frame')
    meta=json.loads(encoded[4:4+size])
    if not isinstance(meta,dict):raise ValueError('Invalid raw LDCT metadata')
    return meta,encoded[4+size:]
