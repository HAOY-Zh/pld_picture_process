"""Independent NumPy full-graph and CPU/CUDA exact-code regression checks."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_fixed_network import FixedUNet,NAMES,requant_tensor
from lung_fpga_contract import SCHEMA,conv_accumulator,requantize,multiplier,maxpool2,nearest2,concatenate
from lung_unet import torch_module


def synthetic_contract():
    rng=np.random.default_rng(553);rows=[];parameters={}
    shapes=[(2,1),(2,2),(4,2),(4,4),(8,4),(8,8),(16,8),(16,16),
            (8,24),(8,8),(4,12),(4,4),(2,6),(2,2),(1,2)]
    for name,(cout,cin) in zip(NAMES,shapes):
        k=1 if name=='head' else 3
        w=rng.integers(-4,5,(cout,cin,k,k),dtype=np.int8)
        bias=(np.arange(cout)*7-5).astype(np.int32)
        m,s=multiplier(np.linspace(.1,.13,cout));iz=128 if name=='enc1.0' else (13 if name in ['dec3.0','dec2.0','dec1.0'] else 17)
        bound=int((np.abs(w.astype(np.int64)).sum((1,2,3))*max(iz,255-iz)+np.abs(bias.astype(np.int64))).max())
        row=dict(name=name,weight_shape=list(w.shape),input_zero_point=iz,
            output_zero_point=128 if name=='head' else 17,output_scale=.1,
            stride=[1,1],padding=[k//2,k//2],fused_relu=name!='head',conservative_accumulator_bound=bound)
        rows.append(row);prefix=name.replace('.','_')
        for key,v in [('weight_int8',w),('bias_int32',bias),('multiplier_int32',m),('right_shift_int32',s)]:parameters[prefix+'_'+key]=v
    m,s=multiplier(.5)
    cats=[dict(output_scale=.2,output_zero_point=13,branches=[dict(input_scale=.1,input_zero_point=17,multiplier=int(m),right_shift=int(s)) for _ in range(2)]) for _ in range(3)]
    return dict(schema=SCHEMA,layers=rows,concat=cats,input_quantization=dict(scale=.02,zero_point=128)),parameters


def independent_graph(q,contract,p):
    rows={r['name']:r for r in contract['layers']};trace={}
    def conv(a,name):
        row=rows[name];prefix=name.replace('.','_')
        acc=conv_accumulator(a,p[prefix+'_weight_int8'],p[prefix+'_bias_int32'],row['input_zero_point'],row['padding'][0])
        result=requantize(acc,p[prefix+'_multiplier_int32'],p[prefix+'_right_shift_int32'],row['output_zero_point'],row['fused_relu'])
        trace[name]=result[None];return result
    def block(a,name):return conv(conv(a,name+'.0'),name+'.3')
    e1=block(q,'enc1');e2=block(maxpool2(e1),'enc2');e3=block(maxpool2(e2),'enc3');x=block(maxpool2(e3),'neck')
    for number,(skip,name) in enumerate([(e3,'dec3'),(e2,'dec2'),(e1,'dec1')]):
        x=nearest2(x);c=contract['concat'][number]
        x=concatenate([(a,b['input_scale'],b['input_zero_point']) for a,b in zip([x,skip],c['branches'])],c['output_scale'],c['output_zero_point'])
        trace['cats.'+str(number)]=x[None];x=block(x,name)
    x=conv(x,'head');return x[None],trace


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class FixedNetworkTests(unittest.TestCase):
    def setUp(self):
        self.torch=torch_module();self.threads=self.torch.get_num_threads();self.torch.set_num_threads(2)
        self.contract,self.p=synthetic_contract()
    def tearDown(self):self.torch.set_num_threads(self.threads)

    def test_signed_tensor_requantization_matches_numpy(self):
        a=np.array([[[[-2147483648,-7,-3,-1,0,1,3,7,2147483647]],[[9,-9,3,-3,0,5,-5,1,-1]],[[1,-1,2,-2,0,3,-3,4,-4]]]],np.int64)
        m=np.array([2147483647,1073741824,1073741824]);s=np.array([0,31,62])
        expected=requantize(a[0],m,s,128)
        actual=requant_tensor(self.torch.from_numpy(a),m,s,128).numpy()[0]
        np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(ValueError):requant_tensor(self.torch.tensor([[[[2**31]]]],dtype=self.torch.int64),[1],[0],0)

    def test_complete_cpu_graph_all_layers_and_skip_order_match_independent_numpy(self):
        q=np.random.default_rng(221).integers(0,256,(1,1,32,32),dtype=np.uint8)
        expected,trace=independent_graph(q[0],self.contract,self.p)
        actual,observed=FixedUNet(self.contract,self.p).forward_codes(self.torch.from_numpy(q),True)
        self.assertEqual(len(trace),18);self.assertEqual(set(trace),set(observed))
        for name in trace:np.testing.assert_array_equal(observed[name],trace[name],err_msg=name)
        np.testing.assert_array_equal(actual.numpy(),expected)

    def test_batch_invariance_and_cpu_preprocessing_dequantization(self):
        model=FixedUNet(self.contract,self.p);q=np.random.default_rng(55).integers(0,256,(2,1,32,32),dtype=np.uint8)
        result=model.forward_codes(self.torch.from_numpy(q)).numpy()
        individual=np.concatenate([model.forward_codes(self.torch.from_numpy(q[i:i+1])).numpy() for i in range(2)])
        np.testing.assert_array_equal(result,individual)
        x=self.torch.from_numpy(q.astype(np.float32)/255)
        grid=self.contract['input_quantization'];codes=self.torch.quantize_per_tensor(x,grid['scale'],grid['zero_point'],self.torch.quint8).int_repr()
        expected=(model.forward_codes(codes).to(self.torch.float32)-128)*.1
        self.torch.testing.assert_close(model(x),expected,rtol=0,atol=0)

    def test_graph_bound_shape_and_nonfinite_guards(self):
        changed=dict(self.contract,layers=list(reversed(self.contract['layers'])))
        with self.assertRaises(ValueError):FixedUNet(changed,self.p)
        bad={k:v.copy() for k,v in self.p.items()};bad['enc1_0_bias_int32'][0]+=1
        # Make bound identity invalid rather than relying on sampled activations.
        changed=dict(self.contract,layers=[dict(r) for r in self.contract['layers']]);changed['layers'][0]['conservative_accumulator_bound']+=1
        with self.assertRaises(ValueError):FixedUNet(changed,self.p)
        model=FixedUNet(self.contract,self.p)
        with self.assertRaises(ValueError):model.forward_codes(self.torch.zeros(1,1,32,32))
        with self.assertRaises(ValueError):model(self.torch.full((1,1,32,32),float('nan')))

    def test_complete_cuda_graph_equals_cpu_and_numpy(self):
        if not self.torch.cuda.is_available():self.skipTest('CUDA required for exact transport regression')
        q=np.random.default_rng(311).integers(0,256,(1,1,32,32),dtype=np.uint8)
        expected,_=independent_graph(q[0],self.contract,self.p)
        result=FixedUNet(self.contract,self.p,'cuda').forward_codes(self.torch.from_numpy(q)).cpu().numpy()
        np.testing.assert_array_equal(result,expected)


if __name__=='__main__':unittest.main()
