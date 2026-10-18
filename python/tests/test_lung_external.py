"""Synthetic external-input engineering checks, not medical accuracy validation."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_external import (align_pair, archive_members, build_parser, case_id,
                           check_frozen, evaluate, lps_zyx, md5_file, normalize_hu,
                           prepare, require_hu)
from lung_quantize import (convert_static, fold_batchnorm, make_quantizable,
                           prepare_static, save_quantized)
from lung_unet import file_sha256, make_model, read_manifest
from lungseg.metrics import overlap_metrics


def phantom():
    hu = np.full((3,32,32), -1000, dtype=np.int16)
    labels = np.zeros_like(hu, dtype=np.uint16)
    hu[1:,5:15,7:16] = -700
    hu[1:,5:15,18:26] = -650
    hu[:,25:30,5:30] = 100
    labels[1:,5:15,7:16] = 1
    labels[1:,5:15,18:26] = 2
    affine = np.diag([-.7,-.8,1.5,1.])
    affine[:3,3] = [25,30,-10]
    ct = nib.Nifti1Image(hu.transpose(2,1,0), affine)
    mask = nib.Nifti1Image(labels.transpose(2,1,0), affine)
    return hu, labels, ct, mask


class ExternalTests(unittest.TestCase):
    def test_case_names_and_duplicate_archive_members(self):
        self.assertEqual(case_id('Covid-19/ct_scans/coronacases_org_001.nii'), 'coronacases_001')
        self.assertEqual(case_id('radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii'),
                         'radiopaedia_10_85902_1')
        with self.assertRaises(ValueError):
            case_id('other.nii.gz')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'duplicate.zip'
            with zipfile.ZipFile(path,'w') as stream:
                stream.writestr('ct_scans/coronacases_001.nii', b'fake')
                stream.writestr('ct_scans/coronacases_org_001.nii', b'fake')
            with zipfile.ZipFile(path) as stream, self.assertRaisesRegex(ValueError,'Duplicate'):
                archive_members(stream,ct=True)

    def test_lossless_orientation_permutations_and_real_physical_alignment(self):
        hu, labels, ct, mask = phantom()
        variants = [np.array([[1,-1],[0,1],[2,-1]]),np.array([[0,1],[1,-1],[2,1]])]
        for variant in variants:
            changed_ct = ct.as_reoriented(variant)
            result, reference, metadata = align_pair(changed_ct,mask)
            np.testing.assert_array_equal(result,hu)
            np.testing.assert_array_equal(reference,labels!=0)
            self.assertEqual(metadata['label_counts']['1'],180)
        affine = mask.affine.copy()
        affine[0,3] += 5
        wrong = nib.Nifti1Image(np.asanyarray(mask.dataobj), affine)
        with self.assertRaisesRegex(ValueError,'physical voxel grids'):
            align_pair(ct,wrong)

    def test_reject_bad_coordinates_labels_nonfinite_and_grayscale_as_hu(self):
        hu, labels, ct, mask = phantom()
        affine = ct.affine.copy()
        affine[0,1] = .2
        with self.assertRaisesRegex(ValueError,'Oblique/sheared'):
            lps_zyx(nib.Nifti1Image(hu.transpose(2,1,0),affine))
        mask.set_qform(mask.affine+np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),code=1)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            align_pair(ct,mask)
        _,_,ct,mask = phantom()
        invalid = labels.copy();invalid[0,0,0]=3
        with self.assertRaisesRegex(ValueError,'labels 0/1/2'):
            align_pair(ct,nib.Nifti1Image(invalid.transpose(2,1,0),ct.affine))
        data=hu.astype(np.float32);data[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            lps_zyx(nib.Nifti1Image(data.transpose(2,1,0),ct.affine))
        with self.assertRaisesRegex(ValueError,'HU sanity'):
            require_hu('coronacases_001',np.full_like(hu,100),'uint8')
        with self.assertRaisesRegex(ValueError,'predeclared'):
            require_hu('radiopaedia_10_85902_1',hu,'int16')
        normalized = normalize_hu(hu,32)
        self.assertEqual(normalized.dtype,np.float16)
        self.assertEqual(float(normalized[0,0,0]),0.)

    def test_no_case_threshold_or_training_options(self):
        parser = build_parser()
        sub = next(a for a in parser._actions if a.dest=='command')
        for child in sub.choices.values():
            self.assertFalse({'case','threshold','epochs','calibrate'} & {a.dest for a in child._actions})

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
    def test_frozen_zip_to_real_fp32_int8_inference_roundtrip_and_failures(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest={'status':'complete','failures':[], 'fingerprint':'synthetic-only',
                'image_size':32,'hu_clip':[-1000,400],
                'input_normalization':'(clip(HU,-1000,400)+1000)/1400','cases':[]}
            for number,split in enumerate(('train','validation')):
                folder=root/f'dev-{number}';folder.mkdir()
                np.save(folder/'images.npy',np.full((1,32,32),.25+number*.1,dtype=np.float16))
                manifest['cases'].append({'case':f'LCTSC-Train-S2-{number+1:03d}',
                    'patient_key':str(number),'split':split,'cache_folder':folder.name})
            dataset=root/'dataset.json';dataset.write_text(json.dumps(manifest),encoding='utf-8')
            prior=root/'prior.json';prior.write_text(json.dumps({'cases':[]}),encoding='utf-8')
            hu,labels,ct,mask=phantom()
            ct_path=root/'ct.zip';mask_path=root/'masks.zip'
            with zipfile.ZipFile(ct_path,'w',zipfile.ZIP_DEFLATED) as stream:
                stream.writestr('ct_scans/coronacases_org_001.nii',ct.to_bytes())
                gray=nib.Nifti1Image(np.full((32,32,3),100,dtype=np.uint8),ct.affine)
                stream.writestr('ct_scans/radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii',gray.to_bytes())
            with zipfile.ZipFile(mask_path,'w',zipfile.ZIP_DEFLATED) as stream:
                for name in ('coronacases_001','radiopaedia_10_85902_1'):
                    stream.writestr(name+'.nii.gz',gzip.compress(mask.to_bytes()))
            torch.manual_seed(52)
            original=make_model(2).eval()
            saved={'model':original.state_dict(),'dataset_fingerprint':'synthetic-only','image_size':32,
                   'base_channels':2,'context_slices':1,'threshold':.5,'epoch':1}
            checkpoint=root/'best.pt';torch.save(saved,checkpoint)
            old_engine=torch.backends.quantized.engine
            prepared=prepare_static(fold_batchnorm(make_quantizable(original)))
            with torch.inference_mode():prepared(torch.rand(1,1,32,32))
            quantized=convert_static(prepared)
            integer=root/'int8.pt'
            save_quantized(quantized,integer,saved,{'checkpoint_sha256':file_sha256(checkpoint)})
            torch.backends.quantized.engine=old_engine
            args=build_parser().parse_args(['prepare','--ct',str(ct_path),'--masks',str(mask_path),
                '--dataset',str(dataset),'--prior-test',str(prior),'--checkpoint',str(checkpoint),
                '--int8-model',str(integer),'--outdir',str(root/'prepared')])
            with self.assertRaisesRegex(ValueError,'official published MD5'):
                prepare(args)
            self.assertFalse((root/'prepared').exists())
            with patch('lung_external.OFFICIAL_MASK_MD5',md5_file(mask_path)),patch('lung_external.save_alignment'):
                prepare(args)
            cache=json.loads((root/'prepared'/'external_dataset.json').read_text())
            self.assertEqual(cache['primary_prepared'],1)
            self.assertEqual(cache['primary_slices'],3)
            self.assertEqual(len(cache['failures']),9)
            self.assertEqual(len(cache['cohort_exclusions']),1)
            with self.assertRaises(ValueError):read_manifest(root/'prepared'/'external_dataset.json')
            run=build_parser().parse_args(['evaluate','--prepared',str(root/'prepared'),
                '--outdir',str(root/'results'),'--device','cpu','--threads','2','--batch-size','1'])
            tf32=torch.backends.cudnn.allow_tf32
            engine=torch.backends.quantized.engine
            with patch('lung_external.save_learned_preview'):
                evaluate(run)
            report=json.loads((root/'results'/'evaluation.json').read_text())
            self.assertEqual(report['status'],'complete_available_primary_subset')
            self.assertEqual(report['evaluated_volumes'],1)
            self.assertEqual(report['evaluated_slices'],3)
            self.assertEqual(torch.backends.cudnn.allow_tf32,tf32)
            self.assertEqual(torch.backends.quantized.engine,engine)
            for kind,filename in [('fp32','float_prediction_native.npy'),('int8','int8_prediction_native.npy')]:
                pred=np.load(root/'results'/'coronacases_001'/filename)
                self.assertEqual(overlap_metrics(pred,labels!=0),report['cases'][0][kind])
            with self.assertRaisesRegex(ValueError,'not empty'):evaluate(run)
            run.outdir=str(root/'aborted')
            with patch('lung_external._case_prediction',side_effect=RuntimeError('synthetic inference failure')):
                with self.assertRaisesRegex(RuntimeError,'inference failure'):evaluate(run)
            aborted=json.loads((root/'aborted'/'evaluation.json').read_text())
            self.assertEqual(aborted['status'],'aborted_incomplete_external_evaluation')
            self.assertEqual(aborted['cases'],[])
            with checkpoint.open('ab') as stream:stream.write(b'changed')
            protocol=json.loads((root/'prepared'/'protocol.json').read_text())
            with self.assertRaisesRegex(ValueError,'changed'):check_frozen(protocol)


if __name__=='__main__':
    unittest.main()
