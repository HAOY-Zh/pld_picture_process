"""Optional host-side expert reference. Never an input to FPGA or postprocessing."""
from pathlib import Path
import hashlib
import zipfile

import numpy as np

from lung_live_data import MAX_VOXELS, checked_shape


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def align_mask(array, source_basis, source_origin, shape, geom, details):
    """Exact voxel permutation/flip and selected Z crop; no label interpolation."""
    checked_shape(array.shape)
    if not np.isfinite(array).all() or not np.isin(array, [0, 1, 2, 255]).all():
        raise ValueError('专家标注须为 0/1 二值或 0/1/2 左右肺标签，不接受概率或灰度图')
    if (array == 2).any() and (array == 255).any():
        raise ValueError('Ambiguous reference label encoding (2 and 255)')
    basis, origin = np.asarray(source_basis, float), np.asarray(source_origin, float)
    if (basis.shape != (3, 3) or origin.shape != (3,) or
            not np.isfinite(basis).all() or not np.isfinite(origin).all()):
        raise ValueError('Invalid reference geometry')
    try:
        transform = np.linalg.solve(basis, geom['basis_lps_zyx'])
    except np.linalg.LinAlgError as exc:
        raise ValueError('Invalid reference affine') from exc
    integer = np.rint(transform).astype(int)
    if (not np.allclose(transform, integer, atol=1e-5, rtol=0) or
            not np.array_equal(np.abs(integer).sum(axis=0), [1, 1, 1]) or
            not np.array_equal(np.abs(integer).sum(axis=1), [1, 1, 1])):
        raise ValueError('专家标注与 CT 方向/层距不匹配；不会插值或缩放人工标签')
    axes = np.argmax(np.abs(integer), axis=0)
    aligned = array.transpose(tuple(axes))
    corner = np.zeros(3)
    for axis, source_axis in enumerate(axes):
        if integer[source_axis, axis] < 0:
            corner[source_axis] = array.shape[source_axis] - 1
            aligned = np.flip(aligned, axis)
    aligned_origin = origin + basis @ corner
    offset = np.linalg.solve(np.asarray(geom['basis_lps_zyx']),
                             np.asarray(geom['origin_lps_mm']) - aligned_origin)
    start = np.rint(offset).astype(int)
    if (not np.allclose(offset, start, atol=1e-4, rtol=0) or
            np.any(start < 0) or np.any(start + np.asarray(shape) > aligned.shape) or
            tuple(aligned.shape[1:]) != tuple(shape[1:]) or np.any(start[1:] != 0)):
        raise ValueError('专家标注与 CT 位置/尺寸不匹配')
    single_dicom = details.get('format', '').startswith('DICOM') and Path(details['source']).is_file()
    if not single_dicom and (aligned.shape[0] != details['source_total_slices'] or
                            start[0] != details['source_slice_indices'][0]):
        raise ValueError('专家标注须覆盖同一源 CT；部分输入按原序列层号截取')
    return np.ascontiguousarray(aligned[start[0]:start[0]+shape[0]] != 0)


def dicom_reference(path, shape, geom, details):
    import pydicom
    from lungseg.dicom_io import discover, CTVolume, rasterize_lungs, orientation_vectors
    from lung_volume_viewer import geometry
    series, _, _ = discover(details['source'])
    entries = series.get(details['series_uid'], [])
    if not entries: raise ValueError('CT series disappeared while loading reference')
    _, _, normal = orientation_vectors(entries[0][1].ImageOrientationPatient)
    entries.sort(key=lambda item: float(np.asarray(item[1].ImagePositionPatient) @ normal))
    selected = details['source_slice_indices']
    if ([str(entries[z][1].SOPInstanceUID) for z in selected] != details['sop_uids'] or
            len(entries) != details['source_total_slices']):
        raise ValueError('CT SOP identity changed during annotation loading')
    first = entries[0][1]
    positions = np.array([ds.ImagePositionPatient for _, ds in entries], float)
    full_shape = (len(entries), *shape[1:]); checked_shape(full_shape)
    for _, ds in entries:
        if (str(ds.FrameOfReferenceUID) != details['frame_uid'] or
                (int(ds.Rows), int(ds.Columns)) != tuple(shape[1:]) or
                not np.allclose(ds.ImageOrientationPatient, first.ImageOrientationPatient, atol=1e-5, rtol=0) or
                not np.allclose(ds.PixelSpacing, geom['spacing_zyx_mm'][1:], atol=1e-5, rtol=0)):
            raise ValueError('Inconsistent CT headers for expert reference')
    if len(set(str(ds.SOPInstanceUID) for _, ds in entries)) != len(entries):
        raise ValueError('Duplicate CT SOP identity')
    if len(entries) > 1:
        deltas = np.diff(positions, axis=0); gaps = deltas @ normal
        dz = float(np.median(gaps))
        if (dz <= 0 or not np.allclose(gaps, dz, atol=.05, rtol=.01) or
                np.max(np.linalg.norm(deltas-gaps[:, None]*normal, axis=1)) > .05):
            raise ValueError('Expert reference requires a complete uniform CT grid')
    expected_origin = np.asarray(geom['origin_lps_mm'])
    header_geom = geometry(positions[selected[0]], first.ImageOrientationPatient, geom['spacing_zyx_mm'])
    expected_positions = (expected_origin + np.arange(len(selected))[:, None] *
                          np.asarray(geom['basis_lps_zyx'])[:, 0])
    if (not np.allclose(positions[selected], expected_positions, atol=.05, rtol=0) or
            not np.allclose(header_geom['basis_lps_zyx'], geom['basis_lps_zyx'], atol=1e-5, rtol=0)):
        raise ValueError('CT geometry changed during annotation loading')
    # Full header grid lets the unchanged strict rasterizer validate every SOP.
    # No unselected CT pixel data is decoded or allocated.
    volume = CTVolume(np.broadcast_to(np.uint8(0), full_shape), positions,
        np.array(first.ImageOrientationPatient, float), np.array(geom['spacing_zyx_mm']),
        tuple(str(ds.SOPInstanceUID) for _, ds in entries), details['series_uid'], details['frame_uid'])
    masks, metadata = rasterize_lungs(volume, pydicom.dcmread(path), allow_overlap=True)
    # Preserve the established union convention, including documented overlap.
    mask = (masks['left'] | masks['right'])[selected].copy()
    return mask, metadata


def read_reference(path, shape, geom, details, enabled=True):
    """Auto-find one series-bound RTSTRUCT, or load an explicit geometry-bearing mask."""
    if not enabled: return None, None
    if details.get('frame_uid_exception_applied'):
        raise ValueError('匿名化坐标系兼容导入仅用于影像处理；请关闭专家标注加载，不能自动匹配 RTSTRUCT')
    if not isinstance(path, str): raise ValueError('专家标注路径须为文本')
    automatic = not path.strip()
    if automatic:
        if not details.get('format', '').startswith('DICOM'): return None, None
        from lungseg.dicom_io import discover, referenced_series
        _, structures, _ = discover(details['source'])
        candidates = [p for p, ds in structures if details['series_uid'] in referenced_series(ds)]
        if not candidates: return None, None
        if len(candidates) != 1:
            raise ValueError('找到多份匹配 RTSTRUCT；请填写专家标注文件路径，或关闭标注加载')
        path = candidates[0]
    path = Path(path).expanduser().resolve(strict=True)
    before = file_sha(path)
    metadata = None
    if path.name.lower().endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        image = nib.load(str(path))
        checked_shape(tuple(reversed(image.shape)))
        if image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('专家 NIfTI 标注须声明 mm 空间单位')
        q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
        if qc and sc and not np.allclose(q, s, atol=1e-4, rtol=0):
            raise ValueError('Conflicting reference NIfTI qform/sform')
        affine = np.diag([-1., -1., 1., 1.]) @ image.affine
        mask = align_mask(image.get_fdata(dtype=np.float32).transpose(2, 1, 0),
            affine[:3, [2, 1, 0]], affine[:3, 3], shape, geom, details)
        kind = 'NIfTI expert lung union'
    elif path.suffix.lower() == '.npz':
        from lung_volume_viewer import geometry
        with zipfile.ZipFile(path) as archive:
            if sum(v.file_size for v in archive.infolist()) > MAX_VOXELS * 4 + 65536:
                raise ValueError('Reference NPZ decompressed size limit exceeded')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'mask', 'spacing_zyx_mm', 'origin_lps_mm', 'orientation'}:
                raise ValueError('专家 NPZ 须仅含 mask、spacing_zyx_mm、origin_lps_mm、orientation')
            g = geometry(archive['origin_lps_mm'], archive['orientation'], archive['spacing_zyx_mm'])
            mask = align_mask(archive['mask'], g['basis_lps_zyx'], g['origin_lps_mm'], shape, geom, details)
        kind = 'Explicit expert lung union NPZ'
    else:
        if not details.get('format', '').startswith('DICOM'):
            raise ValueError('RTSTRUCT 标注须与原 DICOM CT 配合加载')
        mask, metadata = dicom_reference(path, shape, geom, details)
        kind = 'DICOM RTSTRUCT expert lung union'
    if file_sha(path) != before: raise ValueError('Expert annotation changed while reading')
    if mask.shape != tuple(shape): raise ValueError('Reference shape mismatch')
    return mask, dict(path=str(path), file_sha256=before, format=kind, automatic=automatic,
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), roi_metadata=metadata,
        source_slice_indices=details['source_slice_indices'], geometry=geom,
        note='用户提供的人工肺区参考；仅显示和评价，不参与网络输入、人体约束或三维清理。')
