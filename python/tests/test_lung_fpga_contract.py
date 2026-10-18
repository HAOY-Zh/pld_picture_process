"""Exact integer arithmetic checks; not FPGA resource or medical proof."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_fpga_contract import (conv_accumulator,round_shift_even,multiplier,
    requantize,maxpool2,nearest2,concatenate,run,parser)
from lung_quantize import (make_quantizable,fold_batchnorm,prepare_static,
    convert_static,export_core,save_quantized)
from lung_unet import make_model,file_sha256,torch_module


class IntegerContractTests(unittest.TestCase):
    def test_signed_ties_even(self):
        values=np.arange(-13,14,dtype=np.int64)
        np.testing.assert_array_equal(round_shift_even(values,1),np.rint(values/2).astype(np.int64))
        np.testing.assert_array_equal(round_shift_even(values,2),np.rint(values/4).astype(np.int64))
        np.testing.assert_array_equal(round_shift_even(values,0),values)
        with self.assertRaises(ValueError):round_shift_even(values,-1)
        with self.assertRaises(ValueError):round_shift_even(values.astype(float),2)
        with self.assertRaises(ValueError):round_shift_even(np.array([2**63],np.uint64),2)

    def test_q31_multiplier_and_saturation_relu(self):
        ratios=np.array([1.,.5,.001,2.,.75])
        m,s=multiplier(ratios)
        np.testing.assert_allclose(m.astype(float)*np.exp2(-s.astype(float)),ratios,rtol=1e-9)
        m,s=multiplier([.5])
        a=np.array([[-1000,-3,-1,1,3,1000]],np.int32)
        np.testing.assert_array_equal(requantize(a,m,s,128),[[0,126,128,128,130,255]])
        np.testing.assert_array_equal(requantize(a,m,s,128,True),[[128,128,128,128,130,255]])
        with self.assertRaises(ValueError):multiplier([0])
        with self.assertRaises(ValueError):multiplier([np.nan])
        with self.assertRaises(ValueError):requantize(a.astype(float),m,s,128)

    def test_nonzero_padding_and_exact_manual_cross_correlation(self):
        q=np.array([[[127,128],[129,130]]],np.uint8)
        weights=np.arange(-4,5,dtype=np.int8).reshape(1,1,3,3)
        bias=np.array([7],np.int32)
        acc=conv_accumulator(q,weights,bias,128)
        padded=np.pad(q.astype(np.int64)-128,((0,0),(1,1),(1,1)))
        expected=np.array([[[7+(padded[:,y:y+3,x:x+3]*weights[0]).sum() for x in range(2)] for y in range(2)]])
        np.testing.assert_array_equal(acc,expected)
        zeros=np.full((1,2,2),128,np.uint8)
        np.testing.assert_array_equal(conv_accumulator(zeros,weights,bias,128),np.full((1,2,2),7))
        with self.assertRaises(ValueError):conv_accumulator(q,np.full((1,1,3,3),127,np.int8),np.array([2**31-1],np.int32),0)

    def test_pool_resize_concat_order_and_grids(self):
        q=np.arange(16,dtype=np.uint8).reshape(1,4,4)
        np.testing.assert_array_equal(maxpool2(q),[[[5,7],[13,15]]])
        np.testing.assert_array_equal(nearest2(maxpool2(q)),np.repeat(np.repeat(maxpool2(q),2,1),2,2))
        a=np.array([[[10,12],[14,16]]],np.uint8)
        b=np.array([[[5,6],[7,8]]],np.uint8)
        result=concatenate([(a,.5,10),(b,1.,5)],.5,20)
        np.testing.assert_array_equal(result[0],[[20,22],[24,26]])
        np.testing.assert_array_equal(result[1],[[20,22],[24,26]])
        with self.assertRaises(ValueError):maxpool2(q[:,:,:3])
        with self.assertRaises(ValueError):concatenate([],.5,0)

    def test_int32_domain_and_shape_guards(self):
        m,s=multiplier([.5])
        for a in [np.array([[2**31]],np.int64),np.zeros((2,2),np.int32)]:
            with self.assertRaises(ValueError):requantize(a,m,s,0)
        with self.assertRaises(ValueError):conv_accumulator(np.zeros((1,4,4),np.uint8),np.zeros((2,2,3,3),np.int8),np.zeros(2,np.int32),0)

    def test_real_frozen_export_all_layers_concat_vectors_and_hashes(self):
        torch=torch_module();old_threads=torch.get_num_threads();old_engine=torch.backends.quantized.engine
        try:
            torch.set_num_threads(2);torch.manual_seed(55)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir()
                prepared=prepare_static(fold_batchnorm(make_quantizable(make_model(2).eval())))
                with torch.inference_mode():prepared(torch.rand(1,1,32,32))
                model=convert_static(prepared)
                export_core(model,32,source)
                protocol=dict(checkpoint_sha256='synthetic-frozen')
                checkpoint=dict(base_channels=2,image_size=32,dataset_fingerprint='synthetic')
                save_quantized(model,source/'int8_model.pt',checkpoint,protocol)
                (source/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
                before={str(p):file_sha256(p) for p in source.iterdir()}
                args=parser().parse_args(['--quantization',str(source),'--outdir',str(root/'contract')])
                report=run(args)
                self.assertEqual(len(report['layers']),15);self.assertEqual(len(report['concat']),3)
                self.assertFalse(report['hardware_ready']);self.assertFalse(report['whole_network_fixed_point_validated'])
                self.assertEqual(report['concat'][0]['branches'][0]['source'],'neck.3')
                with np.load(root/'contract/operator_vectors.npz') as a:
                    self.assertEqual(len(a.files),180)
                    with np.load(root/'contract/parameters.npz') as params:
                        for layer in report['layers']:
                            prefix=layer['name'].replace('.','_')
                            q=a[prefix+'_ramp255_input_uint8']
                            acc=conv_accumulator(q,params[prefix+'_weight_int8'],params[prefix+'_bias_int32'],layer['input_zero_point'],layer['padding'][0])
                            np.testing.assert_array_equal(acc,a[prefix+'_ramp255_accumulator_int32'])
                            result=requantize(acc,params[prefix+'_multiplier_int32'],params[prefix+'_right_shift_int32'],layer['output_zero_point'],layer['fused_relu'])
                            np.testing.assert_array_equal(result,a[prefix+'_ramp255_output_uint8'])
                for p,h in before.items():self.assertEqual(file_sha256(p),h)
        finally:
            torch.set_num_threads(old_threads);torch.backends.quantized.engine=old_engine


if __name__=='__main__':unittest.main()
