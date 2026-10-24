"""Authenticated ROI1 client. No CPU fallback or FPGA retry."""
import base64
import time
import uuid
import numpy as np
from ct_studio_client import StudioClient
from roi_contract import BACKEND, TRANSPORT, digest, encode, hu_statistics, make_result
from roi_statistics_contract import STATISTICS_ABI, validate_statistics


class ROIStudioClient(StudioClient):
    def health(self):
        identity=super().health()
        if (identity['platform'].get('roi')!=self.manifest.get('roi') or not self.manifest.get('roi')
                or identity['platform'].get('roi_startup_acceptance')!='startup-selftests-passed'):
            raise ValueError('ROI deployment/startup identity differs')
        if self.manifest['roi'].get('statistics_abi') and identity['platform'].get('roi_statistics_startup_acceptance')!='startup-selftests-passed':
            raise ValueError('ROI statistics startup tests not accepted')
        return identity

    def roi(self,hu_slice_i16,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
        raw,params,hu,mask=encode(hu_slice_i16,roi_yx,hu_range,use_hu,lung_mask)
        rid=uuid.uuid4().hex;started=time.perf_counter()
        receipt=self.call('/v1/roi',raw,{'Content-Type':'application/octet-stream',
            'X-Request-ID':rid,'X-Input-SHA256':digest(raw),'X-Parameters':__import__('json').dumps(params)},timeout=120)
        row=self.manifest['roi'];shape=params['shape']
        if (receipt.get('schema')!='ct_studio_roi_result_v1' or receipt.get('backend')!=BACKEND
                or receipt.get('transport')!=TRANSPORT or receipt.get('request_id')!=rid
                or receipt.get('parameters')!=params or receipt.get('input_sha256')!=digest(raw)
                or receipt.get('hu_sha256')!=digest(hu.tobytes())
                or receipt.get('lung_mask_sha256')!=(digest(mask.tobytes()) if mask is not None else None)
                or receipt.get('platform_manifest_sha256')!=self.manifest['platform_manifest_sha256']
                or receipt.get('bit_sha256')!=row['bit_sha256'] or receipt.get('hwh_sha256')!=row['hwh_sha256']):
            raise ValueError('ROI response identity/request/input mismatch')
        packed=base64.b64decode(receipt.pop('mask_base64'),validate=True)
        if len(packed)!=int(np.prod(shape)) or digest(packed)!=receipt.get('output_sha256'):
            raise ValueError('ROI output hash/size differs')
        output=np.frombuffer(packed,np.uint8).reshape(shape).copy()
        if (output>1).any() or type(receipt.get('valid_count'))is not int or int(output.sum())!=receipt['valid_count']:
            raise ValueError('ROI returned mask/count differs')
        stats=receipt['stats']
        if (receipt.get('hardware_status')!=0x01000002 or receipt.get('state')!={'busy':False,'faulted':False,'dma_may_access':False}
                or stats.get('guard_verified')is not True):raise ValueError('ROI resources/status not confirmed idle')
        for key in ('actual_pl_mhz_before','actual_pl_mhz_after'):
            if not isinstance(stats.get(key),(int,float)) or not np.isfinite(stats[key]) or abs(stats[key]-100)>.5:
                raise ValueError('ROI actual clock differs')
        for key in ('total_seconds','dma_compute_seconds','prepare_seconds','postprocess_seconds','overlay_switch_ms'):
            if not isinstance(stats.get(key),(int,float)) or not np.isfinite(stats[key]) or stats[key]<0:
                raise ValueError('Incomplete ROI timing receipt')
        hardware_statistics=None
        if row.get('statistics_abi'):
            if row['statistics_abi']!=STATISTICS_ABI or receipt.get('statistics_abi')!=STATISTICS_ABI:
                raise ValueError('Missing/mismatched RST1 statistics receipt')
            hardware_statistics=validate_statistics(receipt.get('hu_statistics'),receipt['valid_count'])
        elif receipt.get('hu_statistics')!=hu_statistics(hu,output):raise ValueError('ROI HU statistics differ')
        receipt['roundtrip_ms']=(time.perf_counter()-started)*1000
        timing={k:stats.get(k) for k in ('total_seconds','dma_compute_seconds','prepare_seconds','postprocess_seconds')}
        if hardware_statistics is None:
            result=make_result(hu,params['roi_yx'],params['hu_range'],params['use_hu'],mask,output,BACKEND,timing,100e6)
        else:
            from roi_vendor.ct_roi_reference import MODULE_VERSION,ALGORITHM_ID,PARAMS_VERSION
            result=dict(filtered_mask=output,valid_count=receipt['valid_count'],hu_statistics=hardware_statistics,
                backend=BACKEND,status='SUCCESS',module_version=MODULE_VERSION,algorithm_id=ALGORITHM_ID,
                params_version=PARAMS_VERSION,shape_yx=shape,roi_yx=params['roi_yx'],use_hu=params['use_hu'],
                hu_range=params['hu_range'],use_lung_mask=params['use_lung_mask'],clock_hz=100e6,
                statistics_abi=STATISTICS_ABI,timing=timing)
        result['hardware_receipt']=receipt
        return result


class RemoteRoiProcessor:
    def __init__(self,client):self.client=client
    def process_slice(self,*args,**kwargs):return self.client.roi(*args,**kwargs)
