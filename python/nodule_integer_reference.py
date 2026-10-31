"""Independent integer nodule graph on the development computer, not PL timing."""
import numpy as np
import torch
from nodule_plan import validate, requant


def forward(model,params,codes,device='cuda',capture=False):
    validate(model,params)
    q=np.asarray(codes)
    if q.dtype!=np.uint8 or q.shape!=(45,64,64): raise ValueError('Wrong lowered input')
    values={'input':q};accumulators={}
    with torch.inference_mode(),torch.backends.cudnn.flags(enabled=False):
        for node in model['nodes']:
            name=node['name'];a=values[node['sources'][0]]
            if node['kind']=='conv':
                r=node['layer'];p=r['prefix'];w=params[p+'_weight']
                # Each exact integer partial sum is far below FP64's 2**53.
                y=torch.nn.functional.conv2d(torch.as_tensor(a.astype(np.float64)-r['input_zero'],device=device)[None],
                    torch.as_tensor(w.astype(np.float64),device=device),padding=r['padding'])[0].cpu().numpy()
                if not np.array_equal(y,np.rint(y)): raise ArithmeticError('Non-integral accumulator transport')
                acc=y.astype(np.int64)+params[p+'_bias'][:,None,None]
                out=requant(acc,params[p+'_multiplier'][:,None,None],params[p+'_shift'][:,None,None],r['output_zero'],r['relu'])
                if capture: accumulators[name]=acc
            elif node['kind']=='pool':
                c,h,w=a.shape;out=a.reshape(c,h//2,2,w//2,2).max((2,4))
            else:
                qn=node['quant'];b=values[node['sources'][1]]
                aa=np.repeat(np.repeat(a,2,1),2,2)
                out=np.concatenate([requant(aa.astype(np.int64)-qn['za'],qn['ma'],qn['sa'],qn['zo']),
                    requant(b.astype(np.int64)-qn['zb'],qn['mb'],qn['sb'],qn['zo'])])
            values[name]=out
    return (values['head'],values,accumulators) if capture else values['head']
