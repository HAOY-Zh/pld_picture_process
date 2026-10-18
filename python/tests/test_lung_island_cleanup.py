"""Geometry-only filtering, physical units, scoring and frozen-run safeguards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_island_cleanup import (analyze, clean_prediction, evaluation_tables, materialize,
    parser, removed_ids, run, score, summarize)
from lungseg.metrics import overlap_metrics
from lung_unet import file_sha256, make_model, torch_module
from test_lung_outer_contour import fixture


class CleanupTests(unittest.TestCase):
    def test_far_small_removed_near_small_and_large_preserved_input_unchanged(self):
        p=np.zeros((50,50,80),bool);p[5:45,5:45,5:45]=1
        p[20,20,47]=1;p[20,20,75]=1
        before=p.copy()
        cleaned,info=clean_prediction(p,[1,1,1],max_island_ml=5,min_distance_mm=20)
        np.testing.assert_array_equal(p,before)
        self.assertTrue(cleaned[20,20,47]);self.assertFalse(cleaned[20,20,75])
        np.testing.assert_array_equal(cleaned[5:45,5:45,5:45],p[5:45,5:45,5:45])
        self.assertEqual(len(info['removed_component_ids']),1)

    def test_distance_exact_physical_units_and_all_large_components_protected(self):
        p=np.zeros((50,50,90),bool);p[5:45,5:45,5:45]=1;p[20,20,60]=1
        labels,info=analyze(p,[1,1,2])
        island=[c for c in info['components'] if c['voxels']==1][0]
        self.assertEqual(island['nearest_anchor_mm'],32.)
        self.assertEqual(removed_ids(info,5,32),[]) # Strictly greater, not >=.
        self.assertEqual(removed_ids(info,5,31),[island['id']])
        p[5:45,5:45,50:90]=1
        _,info=analyze(p,[1,1,1]);self.assertEqual(len(info['anchors']),2)

    def test_tiny_unreliable_empty_and_single_lung_no_forced_deletion(self):
        for p in [np.zeros((20,20,20),bool),np.ones((20,20,20),bool)]:
            cleaned,info=clean_prediction(p,[1,1,1])
            np.testing.assert_array_equal(cleaned,p);self.assertFalse(info['reliable_anchor'])
        p=np.zeros((50,50,70),bool);p[5:45,5:45,5:45]=1;p[20,20,65]=1
        _,info=clean_prediction(p,[1,1,1]);self.assertEqual(len(info['anchors']),1)

    def test_26_connected_diagonal_preserved_and_no_hole_filling(self):
        p=np.zeros((50,50,80),bool);p[5:45,5:45,5:45]=1;p[45,45,45]=1
        p[20:22,20:22,20:22]=0
        cleaned,info=clean_prediction(p,[1,1,1])
        self.assertEqual(len(info['components']),1)
        np.testing.assert_array_equal(cleaned,p)

    def test_validation_scoring_matches_native_mask_metrics_and_true_removal(self):
        p=np.zeros((8,10,12),bool);p[1:5,2:7,2:7]=1;p[7,8,10]=1
        g=p.copy();g[6,2,2]=1;g[1,2,2]=0
        labels,n=ndimage.label(p)
        tables=evaluation_tables(labels,g)
        small=int(labels[7,8,10]);s=score(tables,[small]);cleaned=materialize(labels,[small])
        expected=overlap_metrics(cleaned,g)
        for key in ['dice','true_positive','false_positive','false_negative']:self.assertEqual(s[key],expected[key])
        self.assertEqual(s['removed_true_lung_voxels'],1)
        self.assertEqual(summarize([s,s])['removed_true_lung_voxels'],2)
        self.assertEqual(s['lost_positive_slices'],1)

    def test_invalid_parameters_and_masks_rejected(self):
        p=np.zeros((10,10,10),bool)
        for spacing in [[1,1,0],[1,1], [1,float('nan'),1]]:
            with self.assertRaises(ValueError):analyze(p,spacing)
        with self.assertRaises(ValueError):analyze(np.ones((3,3)),[1,1,1])
        with self.assertRaises(ValueError):analyze(np.ones((3,3,3))*.5,[1,1,1])
        _,info=analyze(p,[1,1,1])
        with self.assertRaises(ValueError):removed_ids(info,6,20)
        with self.assertRaises(ValueError):removed_ids(info,1,-2)

    def test_real_frozen_run_validation_only_original_files_unchanged(self):
        torch=torch_module();torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,manifest=fixture(root)
            ck=root/'original.pt'
            torch.save(dict(model=make_model(2).state_dict(),base_channels=2,image_size=64,
                context_slices=1,hu_clip=[-1000,400],threshold=.5,epoch=12,dataset_fingerprint='original'),ck)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['--dataset',str(dataset),'--checkpoint',str(ck),'--outdir',str(root/'screen'),'--device','cpu'])
            with patch('lung_island_cleanup.comparison_preview'):
                run(args)
            for path,digest in before.items():self.assertEqual(file_sha256(path),digest)
            report=json.loads((root/'screen/evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'complete_development_screen')
            self.assertTrue(report['protected_inputs_unchanged'])
            self.assertEqual([r['case'] for r in report['cases']],[manifest['cases'][1]['case']])
            self.assertFalse((root/'screen'/manifest['cases'][0]['case']).exists())


if __name__=='__main__':unittest.main()
