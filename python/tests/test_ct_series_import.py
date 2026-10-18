"""Series geometry, immutable uploads and real multi-plane PC dataflow."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pydicom
from pydicom.uid import generate_uid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'python/tests')]
from ct_series_import import FolderImports, inspect_series
from ct_workspace_web import WorkspaceController, workspace_handler
from lung_live_data import read_ct, parse_slices
from lung_live_reference import read_reference
from test_lung_segmentation import make_case
from test_lung_live import RunningServer, TOKEN, wait


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.source=self.root/'source';self.source.mkdir()
        make_case(self.source)
        self.paths=sorted(self.source.glob('slice_*.dcm'))
        for path in self.paths:
            ds=pydicom.dcmread(path);ds.PatientID='SYNTHETIC-DEMO';ds.StudyInstanceUID='1.2.826.0.1.3680043.10.999.1';ds.save_as(path)

    def inconsistent(self):
        for path in self.paths:
            ds=pydicom.dcmread(path);ds.FrameOfReferenceUID=generate_uid();ds.save_as(path)

    def test_strict_default_and_compatibility_retain_uid_and_exact_hu(self):
        normal,geom,_=read_ct(self.source);self.inconsistent()
        hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths}
        with self.assertRaisesRegex(ValueError,'坐标系'):read_ct(self.source)
        hu,g,info=read_ct(self.source,slices='全部',allow_inconsistent_frame_uids=True)
        np.testing.assert_array_equal(hu,normal);self.assertEqual(g,geom)
        self.assertTrue(info['frame_uid_exception_applied']);self.assertEqual(info['frame_uid'],'')
        self.assertEqual(len(set(info['frame_uids'])),len(self.paths))
        self.assertEqual(hashes,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths})
        with self.assertRaisesRegex(ValueError,'专家标注'):read_reference('',hu.shape,g,info)
        self.assertEqual(read_reference('',hu.shape,g,info,enabled=False),(None,None))

    def test_compatibility_rejects_mixed_or_missing_patient_and_study(self):
        self.inconsistent();path=self.paths[0];original=pydicom.dcmread(path)
        for tag,value in [('PatientID','OTHER'),('PatientID',''),('StudyInstanceUID',generate_uid()),('StudyInstanceUID','')]:
            ds=copy.deepcopy(original);setattr(ds,tag,value);ds.save_as(path)
            with self.assertRaisesRegex(ValueError,'患者或检查'):read_ct(self.source,allow_inconsistent_frame_uids=True)
        original.save_as(path)

    def test_geometry_failures_are_not_bypassed(self):
        self.inconsistent();path=self.paths[0];original=pydicom.dcmread(path)
        changes=[('ImagePositionPatient',[10,20,30]),('ImagePositionPatient',[11,20,33]),('PixelSpacing',[2,2]),('ImageOrientationPatient',[0,1,0,1,0,0]),('SOPInstanceUID',pydicom.dcmread(self.paths[1]).SOPInstanceUID)]
        for tag,value in changes:
            ds=copy.deepcopy(original);setattr(ds,tag,value);ds.save_as(path)
            with self.assertRaises(ValueError):read_ct(self.source,allow_inconsistent_frame_uids=True)
        original.save_as(path)
        extra=copy.deepcopy(original);extra.SOPInstanceUID=generate_uid();extra.ImagePositionPatient=[10,20,37];extra.save_as(self.source/'gap.dcm')
        with self.assertRaisesRegex(ValueError,'nonuniform'):read_ct(self.source,allow_inconsistent_frame_uids=True)

    def test_inspection_is_header_only_and_distinguishes_series(self):
        self.inconsistent();read=pydicom.dcmread
        def header_only(*args,**kwargs):
            self.assertTrue(kwargs.get('stop_before_pixels'));return read(*args,**kwargs)
        with patch.object(pydicom,'dcmread',side_effect=header_only):value=inspect_series(str(self.source))
        self.assertEqual(len(value['series']),1);self.assertTrue(value['series'][0]['valid']);self.assertTrue(value['series'][0]['requires_compatibility'])
        ds=read(self.paths[0]);ds.SeriesInstanceUID=generate_uid();ds.save_as(self.source/'other.dcm')
        self.assertEqual(len(inspect_series(str(self.source))['series']),2)

    def test_all_alias_and_selected_range_keep_physical_origin(self):
        for text in ['全部','all','*','']:self.assertEqual(parse_slices(text,2,True),[0,1])
        all_hu,_,_=read_ct(self.source)
        hu,g,info=read_ct(self.source,slices='1')
        np.testing.assert_array_equal(hu,all_hu[1:2]);self.assertEqual(g['origin_lps_mm'],[10,20,33])
        self.assertTrue(info['partial_input'])

    def test_upload_is_bounded_complete_and_never_uses_client_path(self):
        manager=FolderImports(self.root/'imports');raw=[p.read_bytes() for p in self.paths]
        key=manager.start(dict(file_count=len(raw),total_bytes=sum(map(len,raw))))['import_id']
        manager.receive(key,0,'../../outside.dcm',raw[0])
        with self.assertRaises(ValueError):manager.finish(key)
        with self.assertRaises(ValueError):manager.receive(key,0,'duplicate',raw[0])
        for i,body in enumerate(raw[1:],1):manager.receive(key,i,'source.dcm',body)
        value=manager.finish(key);self.assertEqual(value['series'][0]['slices'],len(raw))
        self.assertFalse((self.root/'outside.dcm').exists())
        np.testing.assert_array_equal(read_ct(value['path'])[0],read_ct(self.source)[0])
        with self.assertRaises(ValueError):manager.finish(key)
        for values in [dict(file_count=True,total_bytes=1),dict(file_count=1,total_bytes=0),dict(file_count=5000,total_bytes=5000)]:
            with self.assertRaises(ValueError):manager.start(values)
        with self.assertRaises(ValueError):manager.finish([])

    def test_upload_rejects_non_dicom_without_mutating_case(self):
        manager=FolderImports(self.root/'imports');key=manager.start(dict(file_count=1,total_bytes=5))['import_id']
        with self.assertRaises(ValueError):manager.receive(key,0,'bad.dcm',b'hello')
        self.assertEqual(manager.sessions[key]['received'],0)

    def test_http_folder_import_to_multiplane_and_failed_load_preserves_case(self):
        self.inconsistent();controller=WorkspaceController(None,{},self.root/'results')
        server=RunningServer(workspace_handler(controller,TOKEN,b''));self.addCleanup(server.close)
        def post(route,value,token=TOKEN,kind='application/json'):
            data=json.dumps(value).encode() if kind=='application/json' else value
            return json.load(urlopen(Request(server.url+route,data=data,headers={'Content-Type':kind,'X-UI-Token':token}),timeout=15))
        with self.assertRaises(HTTPError) as context:post('/api/import/start',dict(file_count=1,total_bytes=1),token='wrong')
        self.assertEqual(context.exception.code,403);self.assertFalse(controller.folder_imports.sessions)
        raw=[p.read_bytes() for p in self.paths];key=post('/api/import/start',dict(file_count=len(raw),total_bytes=sum(map(len,raw))))['import_id']
        from urllib.parse import urlencode
        for i,body in enumerate(raw):post('/api/import/file?'+urlencode({'import':key,'index':i,'name':'slice.dcm'}),body,kind='application/dicom')
        value=post('/api/import/finish',dict(import_id=key));row=value['series'][0]
        post('/api/load',dict(path=value['path'],series_uid=row['series_uid'],allow_inconsistent_frame_uids=True,load_expert=False))
        state=wait(controller);self.assertEqual(state['phase'],'idle',state['error']);self.assertEqual(state['shape'],[2,20,30])
        hu=controller.hu.copy();case=state['case_id']
        payload=json.load(urlopen(Request(server.url+'/api/ldct/image?'+urlencode(dict(series=state['series_id'],group='',index=0)),headers={'X-UI-Token':TOKEN}),timeout=15))
        self.assertIsNone(payload['enhanced'])
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload['original']['pixels']),'<i2').reshape(20,30),np.rint(hu[0]).astype(np.int16))
        for axis in range(3):
            plane=controller.plane(case,axis,0,'body')
            actual=np.frombuffer(base64.b64decode(plane['hu']),'<f4').reshape(plane['shape'])
            np.testing.assert_array_equal(actual,np.take(hu,0,axis))
        post('/api/load',dict(path=value['path'],load_expert=False))
        state=wait(controller);self.assertEqual(state['phase'],'error');self.assertEqual(state['case_id'],case)
        np.testing.assert_array_equal(controller.hu,hu)


if __name__=='__main__':unittest.main()
