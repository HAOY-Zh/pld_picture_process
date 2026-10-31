"""Real DICOM upload retry semantics; no board or medical data is used."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

APP=Path(__file__).resolve().parents[2]/'data/CT_Studio_ExactSpeed_20261027_04'
sys.path.insert(0,str(APP/'python'))
from ct_series_import import FolderImports


def synthetic_dicom():
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset('',{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality='CT';ds.SOPClassUID=CTImageStorage;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    output=io.BytesIO();pydicom.dcmwrite(output,ds,enforce_file_format=True)
    return output.getvalue()


class UploadRecovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.upload=FolderImports(self.temp.name);self.raw=synthetic_dicom()
        self.key=self.upload.start(dict(file_count=2,total_bytes=2*len(self.raw)))['import_id']

    def test_lost_reply_retry_does_not_duplicate_or_charge_bytes(self):
        self.upload.receive(self.key,0,'CT/one.dcm',self.raw)
        retry=self.upload.receive(self.key,0,'CT/one.dcm',self.raw)
        self.assertTrue(retry['already_received']);self.assertEqual(retry['completed_files'],1)
        self.assertEqual(self.upload.sessions[self.key]['received'],len(self.raw))

    def test_retry_cannot_replace_received_file(self):
        self.upload.receive(self.key,0,'one.dcm',self.raw)
        for name,raw in [('one.dcm',self.raw+b'\0'),('other.dcm',self.raw)]:
            with self.assertRaisesRegex(ValueError,'不一致'):
                self.upload.receive(self.key,0,name,raw)
        self.assertEqual((Path(self.temp.name)/self.key/'000000.dcm').read_bytes(),self.raw)

    def test_last_file_retry_is_valid_at_total_limit(self):
        for i in range(2):self.upload.receive(self.key,i,str(i)+'.dcm',self.raw)
        self.assertTrue(self.upload.receive(self.key,1,'1.dcm',self.raw)['already_received'])
        self.assertEqual(self.upload.sessions[self.key]['received'],2*len(self.raw))

    def test_overlapping_identical_retries_write_only_once(self):
        with ThreadPoolExecutor(max_workers=2)as workers:
            replies=list(workers.map(lambda _:self.upload.receive(self.key,0,'one.dcm',self.raw),range(2)))
        self.assertEqual(sum(bool(r.get('already_received'))for r in replies),1)
        self.assertEqual(len(list((Path(self.temp.name)/self.key).glob('*.dcm'))),1)

    def test_partial_upload_cannot_be_loaded_as_complete(self):
        self.upload.receive(self.key,0,'one.dcm',self.raw)
        with self.assertRaisesRegex(ValueError,'完整'):self.upload.finish(self.key)


if __name__=='__main__':unittest.main()
