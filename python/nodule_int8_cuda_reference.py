"""Batched exact-integer semantics of the deployed nodule graph, for quality QA.

FP32 CUDA GEMM transports integer MACs exactly: absolute partial-sum bounds
must be below 2**24, TF32/autocast/cuDNN transforms are disabled. Requantization
uses INT64, identical to RTL. This is NOT GPU INT8 execution or board timing.
"""
import numpy as np
import torch
from nodule_plan import validate


class IntegerNodule:
    def __init__(self,plan,parameters,device='cuda'):
        validate(plan,parameters);self.plan=plan;self.device=torch.device(device)
        self.weights={};self.params={};self.bounds={}
        for node in plan['nodes']:
            if node['kind']!='conv':continue
            r=node['layer'];p=r['prefix'];w=parameters[p+'_weight'];bias=parameters[p+'_bias']
            bound=np.abs(w.astype(np.int64)).sum((1,2,3))*max(r['input_zero'],255-r['input_zero'])+np.abs(bias.astype(np.int64))
            if bound.max()>=2**24:raise ValueError('FP32 integer transport cannot be proven exact')
            self.bounds[node['name']]=int(bound.max())
            if node['name']=='stem0':w=w.reshape(8,5,3,3)
            self.weights[node['name']]=torch.as_tensor(w.astype(np.float32),device=self.device)
            self.params[node['name']]={k:torch.as_tensor(parameters[p+'_'+k].astype(np.int64),device=self.device).reshape(1,-1,1,1)
                for k in ['bias','multiplier','shift']}
        a=np.arange(-32768,32768,dtype=np.float64)
        self.lut=np.rint((a.clip(-1000,400)+1000)*(255./1400.)).astype(np.uint8)
        self.head_scales=torch.as_tensor(plan['head_scales'],dtype=torch.float32,device=self.device).reshape(1,5,1,1)

    @staticmethod
    def requant(acc,mult,shift,zero,relu=False):
        product=acc*mult;quotient=product>>shift
        if isinstance(shift,int):
            rem=product&((1<<shift)-1);half=1<<(max(shift,1)-1)
        else:
            rem=product&((torch.ones_like(shift)<<shift)-1)
            half=torch.ones_like(shift)<<(shift.clamp_min(1)-1)
        inc=(shift>0)&((rem>half)|((rem==half)&((quotient&1)!=0)))
        return (quotient+inc+zero).clamp(zero if relu else 0,255).to(torch.uint8)

    def forward_codes(self,input_codes,capture=False):
        if (input_codes.dtype!=torch.uint8 or input_codes.ndim!=4 or input_codes.shape[1]!=5
                or any(v<128 or v>1024 or v%8 for v in input_codes.shape[2:])):
            raise ValueError('Need N5xHxW UINT8 codes; aligned dimensions in 128..1024')
        # Set explicitly on every call; callers cannot silently enable TF32 or
        # autocast and invalidate the exact-integer arithmetic proof.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        values={'input':input_codes.to(self.device)};trace={}
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=False),torch.backends.cudnn.flags(enabled=False):
            for node in self.plan['nodes']:
                name=node['name'];a=values[node['sources'][0]]
                if node['kind']=='conv':
                    r=node['layer'];p=self.params[name]
                    y=torch.nn.functional.conv2d(a.to(torch.float32)-r['input_zero'],self.weights[name],
                        stride=2 if name=='stem0' else 1,padding=1 if name=='stem0' else r['padding'])
                    if capture and (not torch.equal(y,y.round()) or y.abs().max()>=2**24):
                        raise ArithmeticError('Integer transport lost exactness')
                    out=self.requant(y.to(torch.int64)+p['bias'],p['multiplier'],p['shift'],r['output_zero'],r['relu'])
                elif node['kind']=='pool':
                    out=torch.nn.functional.max_pool2d(a.float(),2).to(torch.uint8)
                else:
                    q=node['quant'];b=values[node['sources'][1]]
                    a=a.repeat_interleave(2,2).repeat_interleave(2,3)
                    out=torch.cat([self.requant(a.to(torch.int64)-q['za'],q['ma'],q['sa'],q['zo']),
                        self.requant(b.to(torch.int64)-q['zb'],q['mb'],q['sb'],q['zo'])],1)
                values[name]=out
                if capture:trace[name]=out.cpu().numpy()
        return (values['head'],trace) if capture else values['head']

    def __call__(self,hu):
        hu=np.asarray(hu)
        if hu.dtype!=np.int16 or hu.ndim!=4 or hu.shape[1]!=5:raise ValueError('Need INT16 HU patches/canvases')
        q=self.lut[hu.astype(np.int32)+32768]
        codes=self.forward_codes(torch.from_numpy(q))
        return ((codes.float()-128)*self.head_scales).cpu().numpy()
