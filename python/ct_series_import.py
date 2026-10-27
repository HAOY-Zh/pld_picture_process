"""Local series inspection and bounded folder uploads; sources stay unchanged."""
import hashlib
import json
from pathlib import Path
import threading
import uuid
import numpy as np
import pydicom
from lungseg.dicom_io import discover, checked_ct_headers

MAX_FILE = 64 * 1024 * 1024
MAX_BATCH = 2 * 1024 * 1024 * 1024


def inspect_series(path):
    source = Path(path.strip().strip('"')).expanduser().resolve(strict=True)
    series, structures, skipped = discover(source)
    rows = []
    for uid, entries in sorted(series.items()):
        first = entries[0][1]
        count = len({str(getattr(ds, 'FrameOfReferenceUID', '')) for _, ds in entries})
        row = dict(series_uid=uid, slices=len(entries), rows=int(first.Rows), columns=int(first.Columns),
                   description=str(getattr(first, 'SeriesDescription', 'CT 序列'))[:160],
                   frame_uid_count=count, requires_compatibility=count > 1, valid=False)
        try:
            info = checked_ct_headers(entries, allow_inconsistent_frame_uids=count > 1)
            row.update(valid=True, spacing_zyx_mm=info['spacing'].tolist(),
                       note='坐标系标识不一致；几何检查通过，需明确选择兼容导入' if count > 1 else '序列与空间几何检查通过')
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            row['note'] = str(exc)
        rows.append(row)
    return dict(path=str(source), series=rows, rtstruct_files=len(structures), skipped_files=skipped,
                note='按 DICOM 空间位置排序；加载时再次校验，不按文件名拼接。')


class FolderImports:
    def __init__(self, root):
        self.root = Path(root)
        self.sessions = {}
        self.lock = threading.RLock()

    def start(self, values):
        count, total = values.get('file_count'), values.get('total_bytes')
        if type(count) is not int or not 1 <= count <= 4096 or type(total) is not int or not count <= total <= MAX_BATCH:
            raise ValueError('请选择 1～4096 个 DICOM 文件，总大小不超过 2GB')
        with self.lock:
            key = uuid.uuid4().hex
            folder = self.root / key; folder.mkdir(parents=True)
            self.sessions[key] = dict(folder=folder, count=count, total=total, received=0, files={})
            return dict(import_id=key, max_file_bytes=MAX_FILE)

    def receive(self, key, index, name, raw):
        with self.lock:
            if not isinstance(key, str) or key not in self.sessions:
                raise ValueError('文件夹导入任务不存在或已完成')
            session = self.sessions[key]
            if (type(index) is not int or not 0 <= index < session['count']
                    or not isinstance(name, str) or len(name) > 1024 or not 0 < len(raw) <= MAX_FILE):
                raise ValueError('文件编号、大小或导入状态无效')
            checksum = hashlib.sha256(raw).hexdigest()
            prior = session['files'].get(index)
            if prior is not None:
                if prior != dict(original_name=name, size=len(raw), sha256=checksum):
                    raise ValueError('同一文件编号的重试内容不一致；未覆盖已接收文件')
                return dict(completed_files=len(session['files']), total_files=session['count'], already_received=True)
            if session['received'] + len(raw) > session['total']:
                raise ValueError('文件总大小超出导入声明')
            # Client filenames never determine a filesystem path.
            path = session['folder'] / ('%06d.dcm' % index)
            import io
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                if str(getattr(ds, 'Modality', '')) not in ('CT', 'RTSTRUCT'):
                    raise ValueError('请选择包含 CT 的 DICOM 文件夹')
            except pydicom.errors.InvalidDicomError as exc:
                raise ValueError('文件不是可识别的 DICOM；请只选择 CT 序列文件夹') from exc
            with path.open('xb') as out:
                out.write(raw)
            session['files'][index] = dict(original_name=name, size=len(raw), sha256=checksum)
            session['received'] += len(raw)
            return dict(completed_files=len(session['files']), total_files=session['count'])

    def finish(self, key):
        with self.lock:
            if not isinstance(key, str) or key not in self.sessions:
                raise ValueError('文件夹导入任务不存在或已完成')
            session = self.sessions[key]
            if len(session['files']) != session['count'] or session['received'] != session['total']:
                raise ValueError('文件尚未完整传入，不能加载为 CT 序列')
            result = inspect_series(str(session['folder']))
            with (session['folder'] / 'import.json').open('x', encoding='utf-8') as out:
                json.dump(dict(import_id=key, files=session['files']), out, ensure_ascii=False, indent=2)
            del self.sessions[key]
            return result
