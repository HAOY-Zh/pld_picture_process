"""Host-side CT I/O and frozen pre/postprocessing; NO neural forward here."""
import hashlib
from pathlib import Path
import re
import zipfile

import numpy as np

from lung_unet import resize_stack, torch_module
from lung_body_gate import gate_slice
from lung_volume_viewer import geometry

MAX_VOXELS = 128 * 1024 * 1024


def checked_shape(shape):
    if (len(shape) != 3 or not 1 <= shape[0] <= 2048 or
            min(shape[1:]) < 16 or max(shape[1:]) > 2048 or np.prod(shape) > MAX_VOXELS):
        raise ValueError('CT must be ZYX, >=16x16, <=2048 per axis and <=128M voxels')


def parse_slices(text, count, empty_all=False):
    """Zero-based, inclusive ranges; sorted/deduplicated, never eval or clipping."""
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError('层号须为文本，例如 60-90 或 60,65,70-80')
    if not text.strip():
        if empty_all: return list(range(count))
        raise ValueError('请填写要计算的层号')
    result = set()
    for item in text.replace('，', ',').split(','):
        match = re.fullmatch(r'\s*([0-9]+)\s*(?:-\s*([0-9]+)\s*)?', item)
        if not match: raise ValueError('层号格式错误；例如 60-90 或 60,65,70-80')
        first = int(match[1]); last = int(match[2]) if match[2] is not None else first
        if not 0 <= first <= last < count:
            raise ValueError('层号越界或倒序；有效范围 0-' + str(count-1))
        result.update(range(first, last+1))
    return sorted(result)


def read_ct(path, confirm_hu=False, series_uid='', slices=''):
    """A single DICOM series, an axial NIfTI, or explicitly described HU NPZ.

    No annotations or stored predictions are opened. Non-DICOM input requires
    the operator to confirm HU/mm; intensity ranges alone cannot prove HU.
    """
    path = Path(path).expanduser().resolve(strict=True)
    details = dict(source=str(path), annotations_loaded=False)
    def selection(count):
        chosen = parse_slices(slices, count, empty_all=True)
        if chosen != list(range(chosen[0], chosen[-1]+1)):
            raise ValueError('部分加载须选连续范围，例如 60-90；零散层请加载后用“计算指定层”')
        details.update(source_total_slices=count, source_slice_indices=chosen,
                       partial_input=len(chosen) != count)
        return chosen
    if path.is_dir() or path.suffix.lower() in ('.dcm', '.dicom'):
        from lungseg.dicom_io import discover, load_ct, orientation_vectors
        series, _, _ = discover(path)
        if series_uid:
            if series_uid not in series: raise ValueError('Requested CT series UID not found')
            entries = series[series_uid]
        elif len(series) == 1:
            entries = next(iter(series.values()))
        else:
            raise ValueError('Select one CT series folder or specify series UID. Found: ' + ', '.join(series))
        _, _, normal = orientation_vectors(entries[0][1].ImageOrientationPatient)
        entries = sorted(entries, key=lambda item: float(np.asarray(item[1].ImagePositionPatient) @ normal))
        chosen = selection(len(entries))
        checked_shape((len(chosen), int(entries[0][1].Rows), int(entries[0][1].Columns)))
        # Discover reads headers only; decode pixel data only for selected planes.
        volume = load_ct([entries[z] for z in chosen])
        if path.is_file(): details['partial_input'] = True
        hu = volume.hu
        geom = geometry(volume.positions[0], volume.orientation, volume.spacing)
        details.update(series_uid=volume.series_uid, frame_uid=volume.frame_uid,
                       sop_uids=list(volume.sop_uids), format='DICOM CT (rescale to HU)')
    elif path.name.lower().endswith(('.nii', '.nii.gz')):
        if not confirm_hu: raise ValueError('Confirm this NIfTI contains CT HU with millimetre geometry')
        import nibabel as nib
        image = nib.load(str(path))
        checked_shape(tuple(reversed(image.shape)))
        units = image.header.get_xyzt_units()[0]
        if units not in ('mm', 'unknown'): raise ValueError('NIfTI spatial units must be mm')
        image = nib.as_closest_canonical(image)  # XYZ RAS+, no interpolation.
        chosen = selection(image.shape[2])
        image = image.slicer[:, :, chosen[0]:chosen[-1]+1]
        affine = np.diag([-1., -1., 1., 1.]) @ image.affine
        spacing_xyz = np.linalg.norm(affine[:3, :3], axis=0)
        expected = np.diag([-spacing_xyz[0], -spacing_xyz[1], spacing_xyz[2]])
        if not np.allclose(affine[:3, :3], expected, atol=1e-5, rtol=0):
            raise ValueError('Oblique/sheared CT needs explicit resampling; not silently supported')
        hu = image.get_fdata(dtype=np.float32).transpose(2, 1, 0).copy()
        geom = geometry(affine[:3, 3], [-1, 0, 0, 0, -1, 0], spacing_xyz[::-1])
        details.update(format='NIfTI HU', original_units=units, operator_confirmed_hu_mm=True)
    elif path.suffix.lower() == '.npz':
        if not confirm_hu: raise ValueError('Confirm NPZ hu is HU, spacing/origin are millimetres')
        with zipfile.ZipFile(path) as archive:
            if sum(v.file_size for v in archive.infolist()) > MAX_VOXELS * 4 + 65536:
                raise ValueError('NPZ decompressed size limit exceeded')
        with np.load(path, allow_pickle=False) as archive:
            required = {'hu', 'spacing_zyx_mm', 'origin_lps_mm', 'orientation'}
            if set(archive.files) != required:
                raise ValueError('NPZ keys must be exactly: ' + ', '.join(sorted(required)))
            source = archive['hu']  # Compressed NPZ needs decompression before slicing.
            chosen = selection(source.shape[0])
            checked_shape((len(chosen), *source.shape[1:]))
            hu = source[chosen[0]:chosen[-1]+1].astype(np.float32)
            geom = geometry(archive['origin_lps_mm'], archive['orientation'], archive['spacing_zyx_mm'])
            geom['origin_lps_mm'] = (np.asarray(geom['origin_lps_mm']) +
                np.asarray(geom['basis_lps_zyx'])[:, 0] * chosen[0]).tolist()
        details.update(format='Explicit HU NPZ', operator_confirmed_hu_mm=True)
    else:
        raise ValueError('Use a DICOM series directory, .dcm, .nii/.nii.gz or explicit HU .npz; PNG/JPEG are not HU')
    checked_shape(hu.shape)
    if not np.isfinite(hu).all() or hu.min() < -32768 or hu.max() > 65535:
        raise ValueError('Invalid/nonfinite HU values')
    hu = np.ascontiguousarray(hu, dtype=np.float32)
    details.update(shape=list(hu.shape), geometry=geom,
        hu_sha256=hashlib.sha256(hu.tobytes()).hexdigest())
    return hu, geom, details


def input_codes(hu, model):
    hu = np.asarray(hu)
    if hu.ndim != 2 or not np.isfinite(hu).all(): raise ValueError('Need a finite HU slice')
    # EXACT historical order: HU clip -> pixel-centre bilinear -> normalize ->
    # float16 cache roundtrip -> original Torch affine input quantizer.
    image = ((resize_stack(np.clip(hu, -1000, 400)[None], 256, 1) + 1000) / 1400)
    image = image.astype(np.float16).astype(np.float32)
    torch = torch_module()
    q = model['input_quantization']
    return torch.quantize_per_tensor(torch.from_numpy(image), q['scale'], q['zero_point'],
                                    torch.quint8).int_repr().numpy().copy()


def native_masks(codes, hu, geom, model):
    if codes.dtype != np.uint8 or codes.shape != (1, 256, 256): raise ValueError('Wrong FPGA head')
    torch = torch_module()
    # Sigmoid BEFORE native bilinear resize, exactly as the accepted pipeline.
    with torch.inference_mode():
        logits = (torch.from_numpy(codes.astype(np.float32))[None] - model['output_zero']) * model['output_scale']
        probability = torch.nn.functional.interpolate(torch.sigmoid(logits), size=hu.shape,
                                                      mode='bilinear', align_corners=False)
        raw = (probability[0, 0] >= .5).numpy()
    body, plan = gate_slice(hu, raw, geom['spacing_zyx_mm'][1:])
    return raw, body, plan
