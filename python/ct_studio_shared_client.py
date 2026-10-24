"""Explicit shared-core client; preserves the existing ROI/LDCT web client ABI."""
import base64,time,uuid
import numpy as np
from lung_live_web import BoardClient
from shared_patch_contract import (INTERFACE,BACKEND,INPUT_SHAPE,OUTPUT_SHAPE,OUTPUT_BYTES,digest,input_codes,
    validate_receipt_stats,BATCH_INTERFACE,MAX_BATCH,batch_patch_id)


class SharedPatchClient(BoardClient):
    def __init__(self,url,token,bundle):
        super().__init__(url,token);self.bundle=bundle

    def health(self):
        h=self.call('/v1/shared/platform',timeout=10)
        if (h.get('schema')!=INTERFACE or h.get('service_bundle_sha256')!=self.bundle['service_bundle_sha256']
                or h.get('shared_nodule')!=self.bundle['shared_nodule']):raise ValueError('Shared service identity differs')
        if not h.get('ready') or h.get('busy') or h.get('error') or h.get('startup_acceptance')!='startup-selftests-passed':
            raise RuntimeError('Shared core is not accepted/idle; no retry/fallback')
        return h

    def nodule_patch(self,codes):
        image=input_codes(codes);raw=image.tobytes();rid=uuid.uuid4().hex;tick=time.perf_counter()
        r=self.call('/v1/shared/nodule-patch',raw,{'Content-Type':'application/octet-stream',
            'X-Request-ID':rid,'X-Input-SHA256':digest(raw)},timeout=120)
        if (r.get('schema')!=INTERFACE or r.get('backend')!=BACKEND or r.get('request_id')!=rid
                or r.get('input_sha256')!=digest(raw) or r.get('input_shape')!=list(INPUT_SHAPE)
                or r.get('output_shape')!=list(OUTPUT_SHAPE) or r.get('shared_nodule')!=self.bundle['shared_nodule']
                or r.get('service_bundle_sha256')!=self.bundle['service_bundle_sha256']
                or r.get('whole_case_detection')is not False or r.get('state')!={'busy':False,'faulted':False,'dma_may_access':False}):
            raise ValueError('Actual shared-core receipt identity/state mismatch')
        packed=base64.b64decode(r.pop('head_base64'),validate=True)
        if len(packed)!=OUTPUT_BYTES or digest(packed)!=r.get('output_sha256'):raise ValueError('Head payload hash/shape mismatch')
        validate_receipt_stats(r['stats']);r['roundtrip_ms']=(time.perf_counter()-tick)*1000
        return np.frombuffer(packed,np.uint8).reshape(OUTPUT_SHAPE).copy(),r

    def nodule_batch(self,images):
        if not 1<=len(images)<=MAX_BATCH:raise ValueError('Bounded batch requires 1..8 patches')
        parts=[input_codes(image).tobytes()for image in images];raw=b''.join(parts)
        rid=uuid.uuid4().hex;tick=time.perf_counter()
        r=self.call('/v1/shared/nodule-batch',raw,{'Content-Type':'application/octet-stream',
            'X-Request-ID':rid,'X-Input-SHA256':digest(raw),'X-Patch-Count':str(len(parts))},timeout=120)
        count=len(parts)
        if (r.get('schema')!=BATCH_INTERFACE or r.get('backend')!=BACKEND or r.get('request_id')!=rid
                or r.get('input_sha256')!=digest(raw) or r.get('patch_count')!=count
                or r.get('service_bundle_sha256')!=self.bundle['service_bundle_sha256']
                or r.get('shared_nodule')!=self.bundle['shared_nodule']
                or r.get('state')!={'busy':False,'faulted':False,'dma_may_access':False}
                or r.get('whole_case_detection')is not False or len(r.get('records',[]))!=count):
            raise ValueError('Batch identity/state/count mismatch; no retry')
        packed=base64.b64decode(r.pop('head_base64'),validate=True)
        if len(packed)!=count*OUTPUT_BYTES or digest(packed)!=r.get('output_sha256'):
            raise ValueError('Batch output hash/shape mismatch')
        result=[];previous=None
        for index,(part,record) in enumerate(zip(parts,r['records'])):
            head=packed[index*OUTPUT_BYTES:(index+1)*OUTPUT_BYTES]
            if (record.get('schema')!=INTERFACE or record.get('backend')!=BACKEND
                    or record.get('request_id')!=batch_patch_id(rid,index) or record.get('input_sha256')!=digest(part)
                    or record.get('output_sha256')!=digest(head) or record.get('input_shape')!=list(INPUT_SHAPE)
                    or record.get('output_shape')!=list(OUTPUT_SHAPE) or record.get('shared_nodule')!=self.bundle['shared_nodule']
                    or record.get('service_bundle_sha256')!=self.bundle['service_bundle_sha256']
                    or record.get('state')!=r['state'] or record.get('whole_case_detection')is not False
                    or type(record.get('sequence'))is not int or (previous is not None and record['sequence']!=previous+1)):
                raise ValueError('Batch patch binding/order mismatch')
            validate_receipt_stats(record['stats']);previous=record['sequence']
            result.append((np.frombuffer(head,np.uint8).reshape(OUTPUT_SHAPE).copy(),record))
        elapsed=(time.perf_counter()-tick)*1000
        for index,(_,record) in enumerate(result):
            record.update(roundtrip_ms=elapsed/count,transport_scope='amortized_batch_roundtrip',
                batch_id=rid,batch_index=index,batch_count=count,batch_roundtrip_ms=elapsed)
        return result
