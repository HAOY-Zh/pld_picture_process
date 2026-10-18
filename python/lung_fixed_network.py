"""Complete U-Net numerical reference for the frozen proposed FPGA contract.

Codes/requantization are integer. Exact integer-valued convolution is accelerated
using FP64 GEMM, with integral-output and ACC32 checks; NOT a floating NN model,
RTL execution, new calibration, or a board-performance benchmark.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_fpga_contract import SCHEMA
from lung_int8_final import final_masks, removed_counts, score_mask, summarize_rows
from lung_segmentation import fresh_output, write_json
from lung_unet import (_case_prediction, file_sha256, load_cached_source, mapped_npy,
                       read_manifest, torch_module)


NAMES=['enc1.0','enc1.3','enc2.0','enc2.3','enc3.0','enc3.3','neck.0','neck.3',
       'dec3.0','dec3.3','dec2.0','dec2.3','dec1.0','dec1.3','head']


def requant_tensor(acc, multiplier, shift, zero_point, relu=False):
    torch=torch_module()
    if acc.dtype!=torch.int64 or acc.ndim!=4:raise ValueError('Need NCHW INT64 accumulator')
    m=torch.as_tensor(multiplier,dtype=torch.int64,device=acc.device).reshape(1,-1,1,1)
    s=torch.as_tensor(shift,dtype=torch.int64,device=acc.device).reshape(1,-1,1,1)
    if m.numel() not in [1,acc.shape[1]] or s.numel()!=m.numel() or not 0<=zero_point<=255:
        raise ValueError('Invalid scalar/channel requantization shape or zero point')
    if (m<=0).any() or (m>=2**31).any() or (s<0).any() or (s>62).any():
        raise ValueError('Invalid Q31 multiplier/shift')
    if (acc<-(2**31)).any() or (acc>=2**31).any():raise ValueError('ACC32 overflow')
    product=acc*m;quotient=product>>s
    mask=(torch.ones_like(s)<<s)-1;remainder=product&mask
    half=torch.ones_like(s)<<(s.clamp_min(1)-1)
    increment=(s>0)&((remainder>half)|((remainder==half)&((quotient&1)!=0)))
    return (quotient+increment+int(zero_point)).clamp(zero_point if relu else 0,255).to(torch.uint8)


class FixedUNet:
    context_slices=1

    def __init__(self,contract,parameters,device='cpu'):
        torch=torch_module();self.device=torch.device(device);self.contract=contract
        if contract['schema']!=SCHEMA or [r['name'] for r in contract['layers']]!=NAMES:
            raise ValueError('Expected frozen original 15-layer single-slice graph')
        self.rows={r['name']:r for r in contract['layers']};self.weights={};self.params={}
        for name,row in self.rows.items():
            prefix=name.replace('.','_')
            p={key:np.array(parameters[prefix+'_'+key],copy=True) for key in
               ['weight_int8','bias_int32','multiplier_int32','right_shift_int32']}
            if (p['weight_int8'].dtype!=np.int8 or list(p['weight_int8'].shape)!=row['weight_shape'] or
                p['bias_int32'].dtype!=np.int32 or p['bias_int32'].shape!=(row['weight_shape'][0],) or
                row['stride']!=[1,1] or row['padding'] not in [[0,0],[1,1]]):
                raise ValueError('Unsupported layer layout')
            bound=np.abs(p['weight_int8'].astype(np.int64)).sum((1,2,3))*max(row['input_zero_point'],255-row['input_zero_point'])+np.abs(p['bias_int32'].astype(np.int64))
            if int(bound.max())!=row['conservative_accumulator_bound'] or int(bound.max())>=2**31:
                raise ValueError('Invalid mathematical exactness/ACC32 bound')
            self.weights[name]=torch.as_tensor(p['weight_int8'].astype(np.float64),device=self.device)
            self.params[name]={key:torch.as_tensor(value.astype(np.int64),device=self.device)
                for key,value in p.items() if key!='weight_int8'}

    def conv(self,q,name):
        torch=torch_module();row=self.rows[name];p=self.params[name]
        if q.dtype!=torch.uint8 or q.ndim!=4 or q.shape[1]!=row['weight_shape'][1]:
            raise ValueError('Need correctly shaped NCHW UINT8 input')
        # Each input/weight/product and EVERY partial sum is integer and far
        # below 2**53. Disable algorithmic transforms; direct FP64 GEMM is exact.
        with torch.backends.cudnn.flags(enabled=False):
            acc=torch.nn.functional.conv2d(q.to(torch.float64)-row['input_zero_point'],
                self.weights[name],padding=row['padding'][0])
        acc=acc+p['bias_int32'].reshape(1,-1,1,1).to(torch.float64)
        if not torch.isfinite(acc).all() or not torch.equal(acc,acc.round()):
            raise ValueError('FP64 transport was not exact integer arithmetic')
        return requant_tensor(acc.to(torch.int64),p['multiplier_int32'],p['right_shift_int32'],
            row['output_zero_point'],row['fused_relu'])

    def pool(self,q):
        torch=torch_module()
        if q.dtype!=torch.uint8 or q.shape[-1]%2 or q.shape[-2]%2:raise ValueError('Invalid 2x2 pool input')
        return torch.nn.functional.max_pool2d(q.to(torch.float64),2).to(torch.uint8)

    def concat(self,q,skip,number):
        torch=torch_module();item=self.contract['concat'][number];results=[]
        for a,branch in zip([q,skip],item['branches']):
            results.append(requant_tensor(a.to(torch.int64)-branch['input_zero_point'],
                [branch['multiplier']],[branch['right_shift']],item['output_zero_point']))
        return torch.cat(results,dim=1)

    def forward_codes(self,q,capture=False):
        torch=torch_module()
        if (q.dtype!=torch.uint8 or q.ndim!=4 or q.shape[1]!=1 or min(q.shape[-2:])<8 or
            q.shape[-2]%8 or q.shape[-1]%8):raise ValueError('Need N1HW UINT8, spatial sizes divisible by 8')
        q=q.to(self.device);trace={}
        def block(a,prefix):
            for suffix in ['0','3']:
                name=prefix+'.'+suffix;a=self.conv(a,name)
                if capture:trace[name]=a.detach().cpu().numpy().copy()
            return a
        e1=block(q,'enc1');e2=block(self.pool(e1),'enc2');e3=block(self.pool(e2),'enc3')
        x=block(self.pool(e3),'neck')
        for number,(skip,name) in enumerate([(e3,'dec3'),(e2,'dec2'),(e1,'dec1')]):
            x=x.repeat_interleave(2,dim=2).repeat_interleave(2,dim=3)
            x=self.concat(x,skip,number)
            if capture:trace['cats.'+str(number)]=x.detach().cpu().numpy().copy()
            x=block(x,name)
        x=self.conv(x,'head')
        if capture:trace['head']=x.detach().cpu().numpy().copy()
        return (x,trace) if capture else x

    def __call__(self,image):
        torch=torch_module();image=image.detach().cpu().to(torch.float32)
        if image.ndim!=4 or image.shape[1]!=1 or not torch.isfinite(image).all():
            raise ValueError('Need finite normalized N1HW cached inputs')
        grid=self.contract['input_quantization']
        # Original CPU input quantizer remains in preprocessing, outside PL.
        q=torch.quantize_per_tensor(image,grid['scale'],grid['zero_point'],torch.quint8).int_repr()
        with torch.inference_mode():codes=self.forward_codes(q).cpu().to(torch.float32)
        head=self.rows['head']
        return (codes-head['output_zero_point'])*head['output_scale']


def checked_contract(path):
    path=Path(path).resolve(strict=True);contract=json.loads(path.read_text(encoding='utf-8'))
    if contract.get('status')!='complete_proposed_integer_contract' or contract['schema']!=SCHEMA:
        raise ValueError('Need completed frozen integer contract')
    tracked=dict(contract['input_sha256'])
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('Contract source changed: '+p)
    for name,h in contract['outputs'].items():
        file=path.parent/name
        if file_sha256(file)!=h:raise ValueError('Contract output changed: '+name)
        tracked[str(file.resolve())]=h
    tracked[str(path)]=file_sha256(path)
    return path,contract,tracked


def run(args):
    torch=torch_module();torch.set_num_threads(4)
    path,contract,tracked=checked_contract(args.contract)
    previous_path=Path(args.cpu_screen).resolve(strict=True)
    previous=json.loads(previous_path.read_text(encoding='utf-8'))
    root,manifest=read_manifest(Path(args.dataset).resolve(strict=True))
    records=[r for r in manifest['cases'] if r['split']=='validation']
    names=[r['case'] for r in records]
    if (previous['status']!='complete_cpu_int8_final_pipeline' or not previous['accepted_for_hardware_design'] or
        len(previous['cases'])!=len(records) or {r['case'] for r in previous['cases']}!=set(names) or
        previous['protocol']['validation_cases']!=names):raise ValueError('Need exact accepted six-case CPU final screen')
    tracked.update(previous['protocol']['input_sha256'])
    tracked[str(previous_path)]=file_sha256(previous_path)
    tracked[str(Path(__file__).resolve())]=file_sha256(__file__)
    if tracked[str(Path(args.dataset).resolve())]!=file_sha256(args.dataset):raise ValueError('Dataset identity differs')
    for p,h in tracked.items():
        if file_sha256(p)!=h:raise ValueError('CPU screen source changed: '+p)
    with np.load(path.parent/'parameters.npz') as parameters:model=FixedUNet(contract,parameters,args.device)
    output=fresh_output(args.outdir)
    protocol=dict(input_sha256=tracked,validation_cases=names,retraining=False,recalibration=False,test_used=False,
        numerical_contract=SCHEMA,threshold=.5,reference_device=str(model.device),batch_size=args.batch_size,
        exact_transport='FP64 direct convolution of integer codes with integral/ACC32 checks; Q31 signed integer requantization',
        warning='Numerical reference on computer, NOT hardware execution or board speed')
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',protocol=protocol,cases=[])
    try:
        for r in records:
            name=r['case'];print('Complete fixed-network VALIDATION '+name,flush=True)
            raw,elapsed=_case_prediction(model,root,r,torch.device(args.device),args.batch_size)
            hu=load_cached_source(r);body,final,plans,_,geo=final_masks(hu,raw,r['spacing_zyx_mm'])
            prev=next(row for row in previous['cases'] if row['case']==name)
            old_path=previous_path.parent/name/'predictions_native.npz'
            if file_sha256(old_path)!=prev['prediction_sha256']:raise ValueError('Frozen CPU mask artifact changed')
            tracked[str(old_path.resolve())]=prev['prediction_sha256']
            with np.load(old_path) as a:cpu_raw,cpu_final=a['int8_raw'],a['int8_final']
            folder=output/name;folder.mkdir()
            np.savez_compressed(folder/'predictions_native.npz',fixed_raw=raw,fixed_body=body,fixed_final=final)
            write_json(folder/'body_geometry.json',dict(plans=plans,expert_used=False))
            write_json(folder/'island_geometry.json',geo)
            with mapped_npy(root/r['cache_folder']/'reference_native.npy') as ref:
                scores={key:score_mask(a,ref) for key,a in [('fixed_raw',raw),('fixed_final',final),('cpu_raw',cpu_raw),('cpu_final',cpu_final)]}
                scores['fp32_final']=prev['scores']['fp32_final']
                for stage,key in [('cpu_raw','int8_raw'),('cpu_final','int8_final')]:
                    if scores[stage]['metrics']!=prev['scores'][key]['metrics']:raise ValueError('CPU baseline scores mismatch')
                removed=removed_counts(raw,final,ref)
                inspected=dict(index=186,expert=int(np.count_nonzero(ref[186])),raw=int(raw[186].sum()),final=int(final[186].sum())) if name=='LCTSC-Train-S3-002' else None
            row=dict(case=name,split='validation',scores=scores,removed=removed,
                different_raw_voxels=int((raw!=cpu_raw).sum()),different_final_voxels=int((final!=cpu_final).sum()),
                computer_reference_ms=elapsed,timing_warning='Computer FP64 accelerated exact-integer reference; NOT FPGA latency',
                inspected_slice=inspected,prediction_sha256=file_sha256(folder/'predictions_native.npz'))
            report['cases'].append(row);write_json(output/'evaluation.json',report)
            print(f'  CPU INT8 final {scores["cpu_final"]["metrics"]["dice"]:.6f}; fixed final {scores["fixed_final"]["metrics"]["dice"]:.6f}; postprocess removed TP {removed["true_positive"]}',flush=True)
            del hu,raw,body,final,cpu_raw,cpu_final
        summaries=summarize_rows(report['cases'],['fixed_raw','fixed_final','cpu_raw','cpu_final','fp32_final'])
        delta=summaries['fixed_final']['mean_case_dice']-summaries['cpu_final']['mean_case_dice']
        small=summaries['fixed_final']['slice_quality']['groups']['small']['mean_slice_dice']-summaries['cpu_final']['slice_quality']['groups']['small']['mean_slice_dice']
        fp_delta=summaries['fixed_final']['mean_case_dice']-summaries['fp32_final']['mean_case_dice']
        accepted=delta>=-.005 and fp_delta>=-.005 and small>=-.03 and all(r['removed']['true_positive']==0 for r in report['cases'])
        for p,h in tracked.items():
            if file_sha256(p)!=h:raise ValueError('Protected input changed: '+p)
        report.update(status='complete_proposed_fixed_network_validation',summaries=summaries,
            protected_inputs_unchanged=True,accepted_for_rtl=accepted,mean_dice_delta_vs_cpu_int8=delta,
            mean_dice_delta_vs_fp32=fp_delta,small_slice_dice_delta=small,fpga_deployed=False)
        write_json(output/'evaluation.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['protocol','cases','summaries']},indent=2),flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'evaluation.json',report);raise
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    p.add_argument('--cpu-screen',default='data/lung_fpga_readiness_20261014/final_pipeline_v2/evaluation.json')
    p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
