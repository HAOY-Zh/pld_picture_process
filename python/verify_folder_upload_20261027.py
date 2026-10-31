"""300 anonymous synthetic DICOMs through the actual HTTP import handler.

Independent ephemeral localhost server: never calls PL or disturbs 8099.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from urllib import request, parse, error
from http.server import ThreadingHTTPServer
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
OUT=ROOT/'data/folder_upload_recovery_20261027'
sys.path.insert(0,str(APP/'python'))
from ct_workspace_web import workspace_handler
from ct_series_import import FolderImports


def main():
    OUT.mkdir(exist_ok=True);source=OUT/'synthetic_300';source.mkdir(exist_ok=False)
    series,study,frame=[generate_uid()for _ in range(3)]
    raws=[]
    for z in range(300):
        meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
        ds=FileDataset('',{},file_meta=meta,preamble=b'\0'*128)
        ds.Modality='CT';ds.SOPClassUID=CTImageStorage;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID=series;ds.StudyInstanceUID=study;ds.FrameOfReferenceUID=frame
        ds.PatientID='SYNTHETIC_UPLOAD_TEST';ds.SeriesDescription='SYNTHETIC 300 - UPLOAD TEST ONLY'
        ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,z]
        ds.PixelSpacing=[1,1];ds.SliceThickness=1;ds.Rows=16;ds.Columns=16
        ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
        ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
        ds.RescaleSlope=1;ds.RescaleIntercept=0;ds.RescaleType='HU'
        ds.PixelData=np.full((16,16),-1000+z,dtype='<i2').tobytes()
        output=io.BytesIO();pydicom.dcmwrite(output,ds,enforce_file_format=True)
        raw=output.getvalue();raws.append(raw);(source/('%04d.dcm'%z)).write_bytes(raw)
    with tempfile.TemporaryDirectory(dir=OUT)as temp:
        controller=SimpleNamespace(folder_imports=FolderImports(temp))
        server=ThreadingHTTPServer(('127.0.0.1',0),workspace_handler(controller,'synthetic-ui-test',b''))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        url='http://127.0.0.1:'+str(server.server_port)
        op=request.build_opener(request.ProxyHandler({}))
        def post(path,data,mime='application/json'):
            payload=json.dumps(data).encode()if mime=='application/json'else data
            req=request.Request(url+path,data=payload,headers={'X-UI-Token':'synthetic-ui-test','Content-Type':mime})
            with op.open(req,timeout=10)as r:return json.load(r)
        try:
            task=post('/api/import/start',dict(file_count=300,total_bytes=sum(map(len,raws))))
            key=task['import_id']
            def send(z,raw=None):
                query=parse.urlencode({'import':key,'index':z,'name':'synthetic/%04d.dcm'%z})
                return post('/api/import/file?'+query,raws[z]if raw is None else raw,'application/dicom')
            for z in range(300):
                reply=send(z);assert reply['completed_files']==z+1
            assert send(251)['already_received']and send(299)['already_received']
            try:send(251,raws[251]+b'\0')
            except error.HTTPError as exc:
                assert exc.code==400;assert '不一致'in json.load(exc)['error']
            else:raise AssertionError('Changed retry accepted')
            for z,raw in enumerate(raws):
                assert hashlib.sha256((Path(temp)/key/('%06d.dcm'%z)).read_bytes()).digest()==hashlib.sha256(raw).digest()
            result=post('/api/import/finish',dict(import_id=key))
            assert len(result['series'])==1 and result['series'][0]['slices']==300 and result['series'][0]['valid']
            report=dict(status='REAL_HTTP_IMPORT_300_PASS',synthetic_data_only=True,board_accessed=False,
                production_8099_accessed=False,files=300,bytes=sum(map(len,raws)),all_file_hashes_exact=True,
                lost_reply_retries_no_duplicate=True,changed_retry_rejected=True,complete_series_geometry_valid=True,
                source_folder=str(source))
            (OUT/'http_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps(report,ensure_ascii=False,indent=2))
        finally:server.shutdown();server.server_close();worker.join()


if __name__=='__main__':main()
