"""Bounded, lossless transport. Cached planes are verified by content, never position."""
import re
import zlib
from collections import OrderedDict
from nodule_compact_contract import digest, encode_response, decode_response, MAX_HEAD, MAX_META

INTERFACE = 'ct_dual_exact_io_v3'
NODULE_ROUTE = '/v3/shared/nodule-canvas'
SEG_ROUTE = '/v3/shared/segmentation'
CACHE_BYTES = 8 * 1024 * 1024
CACHE_PLANES = 10


def inflate_exact(packed, size):
    if type(size) is not int or not 0 <= size <= 5*1024*1024:
        raise ValueError('Invalid uncompressed length')
    d = zlib.decompressobj()
    raw = d.decompress(packed, size+1)
    if len(raw) != size or not d.eof or d.unused_data or d.unconsumed_tail:
        raise ValueError('Truncated, oversized or trailing compressed input')
    return raw


class PlaneCache:
    def __init__(self):
        self.session = None
        self.planes = OrderedDict()

    def reconstruct(self, session, hashes, indices, raw, plane_bytes, full_hash):
        # Validate everything before cache mutation or hardware execution.
        if not re.fullmatch('[0-9a-f]{32}', session):
            raise ValueError('Invalid cache session')
        if (not isinstance(hashes, list) or len(hashes) != 5 or
                any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes)):
            raise ValueError('Need five exact plane identities')
        if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < 5 for i in indices)
                or indices != sorted(set(indices)) or len(raw) != len(indices)*plane_bytes):
            raise ValueError('Invalid transmitted plane indices/length')
        previous = self.planes if session == self.session else {}
        supplied = {}
        for j, i in enumerate(indices):
            plane = raw[j*plane_bytes:(j+1)*plane_bytes]
            if digest(plane) != hashes[i]:
                raise ValueError('Plane hash mismatch')
            supplied[hashes[i]] = plane
        planes = [supplied.get(h, previous.get(h)) for h in hashes]
        if any(p is None or len(p) != plane_bytes for p in planes):
            raise ValueError('Cache miss; command was not executed')
        full = b''.join(planes)
        if digest(full) != full_hash:
            raise ValueError('Reconstructed full input hash mismatch')
        if session != self.session:
            self.planes.clear()
            self.session = session
        for h, plane in zip(hashes, planes):
            self.planes[h] = plane
            self.planes.move_to_end(h)
        while len(self.planes) > CACHE_PLANES or sum(map(len, self.planes.values())) > CACHE_BYTES:
            self.planes.popitem(last=False)
        return full

    def keys(self):
        return list(self.planes)
