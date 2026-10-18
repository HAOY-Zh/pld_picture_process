import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_lung_continuous_board import validate_report,validate_head,planned_peak
from lung_unet import file_sha256
from test_lung_continuous import fixture
from lung_axil_smoke import feature_words


def example():
    model,_,_=fixture()
    manifest=dict(case='synthetic',slice_index=0,files_sha256={'lung_network.bit':'bit','lung_network.hwh':'hwh'})
    report=dict(status='passed_real_board_continuous_unet_codes',continuous_network=True,core_forward_on_pl=True,
                package_sha256='package',bit_sha256='bit',hwh_sha256='hwh',case='synthetic',slice_index=0,
                head_shape=[1,8,8],head_codes_sha256='0'*64,command_count=51,convolution_commands=45,
                tensor_commands=6,clock_mhz=100,cma_peak_bytes=planned_peak(model),host_total_ms=100.,
                commands=[],nodes=[],node_checks=[])
    shapes={'input':model['input_shape']}; counters={'conv':0,'tensor':0}
    for node in model['nodes']:
        name=node['name']; c,h,w=node['shape']; conv=node['kind']=='conv'; engine='conv' if conv else 'tensor'
        words=h*w*((c+7)//8); count=(c+7)//8 if conv else 1
        report['nodes'].append(dict(node=name,output_words=words,host_node_ms=3.,observer_ms=.1))
        report['node_checks'].append(dict(node=name,output_words=words,same=True,different_words=0))
        for block in range(count):
            counters[engine]+=1
            row=dict(node=name,engine=engine,output_block=block if conv else None,completed=counters[engine],
                     written_words=h*w if conv else words,cycles=100,driver_call_ms=.1)
            if conv:
                cin,ih,iw=shapes[node['sources'][0]]; k=node['layer']['kernel']; pad=node['layer']['padding']
                rows=sum(max(0,min(ih,y-pad+k)-max(0,y-pad)) for y in range(ih))
                cols=sum(max(0,min(iw,x-pad+k)-max(0,x-pad)) for x in range(iw))
                row.update(cache_hits=rows*cols*((cin+7)//8)-1,cache_misses=1)
            else: row['read_words']=words*(4 if node['kind']=='pool' else 1)
            report['commands'].append(row)
        shapes[name]=node['shape']
    report['pl_command_cycles']=5100
    return report,manifest,model


class ContinuousBoardTests(unittest.TestCase):
    def test_valid_report_decomposes_without_double_counting(self):
        report,manifest,model=example(); stats=validate_report(report,manifest,model,'package')
        self.assertAlmostEqual(stats['hardware_active_ms'],.051)
        self.assertAlmostEqual(stats['driver_call_ms'],5.1)
        self.assertAlmostEqual(stats['observer_ms'],2.1)
        self.assertAlmostEqual(sum(stats[k] for k in ['hardware_active_ms','driver_time_outside_engine_ms',
                                                      'observer_ms','other_host_unattributed_ms']),100.)

    def test_wrong_package_bit_or_case_rejected(self):
        for key in ['package_sha256','bit_sha256','hwh_sha256','case','slice_index']:
            report,manifest,model=example(); report[key]='wrong'
            with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')

    def test_missing_reordered_or_false_node_result_rejected(self):
        for mode in ['missing','order','same','different']:
            report,manifest,model=example()
            if mode=='missing': report['node_checks'].pop()
            elif mode=='order': report['node_checks'].reverse()
            elif mode=='same': report['node_checks'][0]['same']=False
            else: report['node_checks'][0]['different_words']=1
            with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')

    def test_wrong_commands_counters_reads_and_cycles_rejected(self):
        for field,value in [('completed',2),('cycles',0),('cycles',0xffffffff),('cache_hits',0),
                            ('written_words',True),('engine','tensor'),('output_block',1)]:
            report,manifest,model=example(); report['commands'][0][field]=value
            with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')
        report,manifest,model=example()
        next(c for c in report['commands'] if c['engine']=='tensor')['read_words']+=1
        with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')

    def test_impossible_or_nonfinite_timing_rejected(self):
        for key,value in [('clock_mhz',125),('host_total_ms',float('nan')),('pl_command_cycles',5101)]:
            report,manifest,model=example(); report[key]=value
            with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')
        report,manifest,model=example(); report['commands'][0]['driver_call_ms']=float('inf')
        with self.assertRaises(ValueError): validate_report(report,manifest,model,'package')

    def test_actual_head_checked_against_both_digest_and_codes(self):
        report,manifest,model=example(); image=np.arange(64,dtype=np.uint8).reshape(1,8,8)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.npz'; np.savez_compressed(path,head_codes=image)
            report['head_codes_sha256']=file_sha256(path)
            self.assertTrue(validate_head(path,report,model,feature_words(image,128))['actual_head_independently_verified'])
            with self.assertRaises(ValueError): validate_head(path,report,model,feature_words(image+1,128))
            report['head_codes_sha256']='0'*64
            with self.assertRaises(ValueError): validate_head(path,report,model,feature_words(image,128))

    def test_actual_head_wrong_dtype_rejected_even_with_matching_file_hash(self):
        report,manifest,model=example()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.npz'; np.savez_compressed(path,head_codes=np.zeros((1,8,8),np.int16))
            report['head_codes_sha256']=file_sha256(path)
            with self.assertRaises(ValueError): validate_head(path,report,model,np.zeros(64,np.uint64))


if __name__=='__main__': unittest.main()
