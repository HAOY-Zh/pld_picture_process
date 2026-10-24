"""Call from B's exclusive manager session with an injected CtRoiProcessor.

Does not load an Overlay, change clocks, allocate/free CMA or launch a service.
This test has not been executed on a physical board by this delivery.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ct_roi_reference import CpuRoiProcessor


def verify_roi_processor(processor,output_dir,*,bit_path,hwh_path,repetitions=5):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    bit_hash=hashlib.sha256(Path(bit_path).read_bytes()).hexdigest()
    hwh_hash=hashlib.sha256(Path(hwh_path).read_bytes()).hexdigest()
    rng=np.random.default_rng(20261019)
    hu=rng.integers(-1200,2000,(512,512),dtype=np.int16)
    lung=rng.integers(0,2,hu.shape,dtype=np.uint8)
    cases=[('full',None,None,False,None),('roi_hu_mask',(2,510,7,511),(-500,300),True,lung),
           ('empty',(512,512,0,512),None,False,None),('single',(0,1,0,1),(int(hu[0,0]),int(hu[0,0])),True,None)]
    rows=[]; retained=None
    for repeat in range(repetitions):
        for name,roi,limits,use_hu,mask in cases:
            reference=CpuRoiProcessor().process_slice(hu,roi,limits,use_hu,mask)
            result=processor.process_slice(hu,roi,limits,use_hu,mask)
            if result['backend']!='FPGA' or result['status']!='SUCCESS':raise AssertionError('Expected FPGA success')
            np.testing.assert_array_equal(result['filtered_mask'],reference['filtered_mask'])
            if result['valid_count']!=reference['valid_count']:raise AssertionError('Count mismatch')
            if retained is not None:np.testing.assert_array_equal(retained[0],retained[1])
            retained=(result['filtered_mask'],result['filtered_mask'].copy())
            stem=f'{repeat:02d}_{name}'
            np.savez_compressed(output_dir/f'{stem}.npz',hu_i16=hu,
                lung_mask=np.zeros((0,),np.uint8) if mask is None else mask,
                actual_mask=result['filtered_mask'],expected_mask=reference['filtered_mask'])
            rows.append({'case':stem,'count':result['valid_count'],'different_pixels':0,
                         'timing':result['timing'],'clock_hz':result['clock_hz']})
    manifest={'status':'PASS','bit_sha256':bit_hash,'hwh_sha256':hwh_hash,'rows':rows,
              'synthetic_only':True,'timing_scope':'DMA+compute as reported by adapter; not pure PL',
              'cma_check':'Manager must additionally record CMA before/after and module switches.'}
    (output_dir/'board_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
