"""Strict single-frame CT and RTSTRUCT conversion without resampling.

Coordinates follow DICOM PS3.3 C.7.6.2 and C.8.8.6. Slice association uses
SOP Instance UID, not filenames, InstanceNumber or rounded z coordinates.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class CTVolume:
    hu: np.ndarray
    positions: np.ndarray
    orientation: np.ndarray
    spacing: np.ndarray  # slice, row, column, in mm
    sop_uids: tuple
    series_uid: str
    frame_uid: str


def discover(root):
    """Read headers recursively; leave source images untouched."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    series, structures, skipped = {}, [], 0
    paths = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
    for path in paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            skipped += 1
            continue
        modality = str(getattr(ds, 'Modality', ''))
        if modality == 'CT':
            uid = str(ds.SeriesInstanceUID)
            series.setdefault(uid, []).append((path, ds))
        elif modality == 'RTSTRUCT':
            structures.append((path, ds))
        else:
            skipped += 1
    return series, structures, skipped


def referenced_series(ds):
    result = set()
    for frame in getattr(ds, 'ReferencedFrameOfReferenceSequence', []):
        for study in getattr(frame, 'RTReferencedStudySequence', []):
            for series in getattr(study, 'RTReferencedSeriesSequence', []):
                result.add(str(series.SeriesInstanceUID))
    return result


def orientation_vectors(orientation):
    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (6,) or not np.isfinite(orientation).all():
        raise ValueError('Invalid ImageOrientationPatient')
    col_direction, row_direction = orientation[:3], orientation[3:]
    if (not np.isclose(np.linalg.norm(col_direction), 1, atol=1e-4) or
            not np.isclose(np.linalg.norm(row_direction), 1, atol=1e-4) or
            not np.isclose(col_direction @ row_direction, 0, atol=1e-4)):
        raise ValueError('Image orientation must be orthonormal')
    return col_direction, row_direction, np.cross(col_direction, row_direction)


def load_ct(entries):
    if not entries:
        raise ValueError('No CT slices selected')
    first = entries[0][1]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    _, _, normal = orientation_vectors(orientation)
    spacing2 = np.asarray(first.PixelSpacing, dtype=np.float64)
    if spacing2.shape != (2,) or not np.isfinite(spacing2).all() or (spacing2 <= 0).any():
        raise ValueError('Invalid PixelSpacing')
    series_uid, frame_uid = str(first.SeriesInstanceUID), str(first.FrameOfReferenceUID)
    shape = (int(first.Rows), int(first.Columns))
    ordered = []
    seen = set()
    for path, ds in entries:
        if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Enhanced/multiframe CT is not supported in this first version')
        if str(ds.SeriesInstanceUID) != series_uid or str(ds.FrameOfReferenceUID) != frame_uid:
            raise ValueError('Mixed CT series or frames of reference')
        if (int(ds.Rows), int(ds.Columns)) != shape:
            raise ValueError('CT slice sizes differ')
        if not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-5, rtol=0):
            raise ValueError('CT slice orientations differ')
        if not np.allclose(ds.PixelSpacing, spacing2, atol=1e-5, rtol=0):
            raise ValueError('CT pixel spacing differs between slices')
        uid = str(ds.SOPInstanceUID)
        if uid in seen:
            raise ValueError('Duplicate SOP Instance UID; select one copy of the series')
        seen.add(uid)
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid ImagePositionPatient')
        ordered.append((float(position @ normal), path, position, uid))
    ordered.sort(key=lambda item: item[0])
    positions = np.stack([item[2] for item in ordered])
    if len(ordered) > 1:
        deltas = np.diff(positions, axis=0)
        gaps = deltas @ normal
        dz = float(np.median(gaps))
        if dz <= 0 or not np.allclose(gaps, dz, atol=.05, rtol=.01):
            raise ValueError('Missing, duplicate or nonuniform CT planes; inspect/resample explicitly')
        if np.max(np.linalg.norm(deltas - gaps[:, None] * normal, axis=1)) > .05:
            raise ValueError('Sheared CT grid is not supported without explicit resampling')
    else:
        dz = float(getattr(first, 'SliceThickness', 0))
        if not np.isfinite(dz) or dz <= 0:
            raise ValueError('Single-slice input needs a positive SliceThickness')
    hu = np.empty((len(ordered), *shape), dtype=np.float32)
    for index, (_, path, _, uid) in enumerate(ordered):
        ds = pydicom.dcmread(path)
        if str(ds.SOPInstanceUID) != uid:
            raise ValueError('CT changed during reading')
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        if not np.isfinite([slope, intercept]).all() or slope == 0:
            raise ValueError('Invalid HU rescale parameters')
        if str(getattr(ds, 'RescaleType', 'HU')).upper() != 'HU':
            raise ValueError('CT rescale units are not HU')
        pixels = ds.pixel_array
        if pixels.shape != shape:
            raise ValueError('Unexpected decoded CT shape')
        hu[index] = pixels.astype(np.float32) * slope + intercept
    return CTVolume(hu, positions, orientation, np.array([dz, *spacing2]),
                    tuple(item[3] for item in ordered), series_uid, frame_uid)


def patient_to_pixel(points, origin, orientation, pixel_spacing, plane_tolerance=.1):
    """Return (row, column) float coordinates; reject a different slice plane."""
    col_direction, row_direction, normal = orientation_vectors(orientation)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Contour must contain finite XYZ points')
    delta = points - np.asarray(origin)
    if np.max(np.abs(delta @ normal), initial=0) > plane_tolerance:
        raise ValueError('Contour plane does not match its referenced CT slice')
    return np.column_stack((delta @ row_direction / pixel_spacing[0],
                            delta @ col_direction / pixel_spacing[1]))


def polygon_mask(vertices, shape):
    """Even-odd polygon fill at integer pixel centers, including its boundary.

    This rasterization convention is recorded in the case metadata. Subpixel
    rasterization may differ from a treatment-planning vendor by edge pixels.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError('A closed contour needs at least three row/column vertices')
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite polygon coordinates')
    result = np.zeros(shape, dtype=bool)
    lo = np.maximum(np.ceil(vertices.min(axis=0) - 1e-8).astype(int), 0)
    hi = np.minimum(np.floor(vertices.max(axis=0) + 1e-8).astype(int), np.array(shape) - 1)
    if (lo > hi).any():
        # A valid subpixel contour can cover no integer pixel centers. Field-of-
        # view and physical-plane checks are performed by rasterize_lungs.
        return result
    yy, xx = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
    inside, boundary = np.zeros(xx.shape, bool), np.zeros(xx.shape, bool)
    for (y1, x1), (y2, x2) in zip(vertices, np.roll(vertices, -1, axis=0)):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        cross = (xx - x1) * dy - (yy - y1) * dx
        boundary |= ((np.abs(cross) <= 1e-7 * max(1, np.hypot(dx, dy))) &
                     (xx >= min(x1, x2) - 1e-8) & (xx <= max(x1, x2) + 1e-8) &
                     (yy >= min(y1, y2) - 1e-8) & (yy <= max(y1, y2) + 1e-8))
        if dy != 0:
            inside ^= ((y1 > yy) != (y2 > yy)) & (xx < (yy - y1) * dx / dy + x1)
    result[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] = inside | boundary
    return result


def _choose_roi(ds, side, override):
    rois = list(getattr(ds, 'StructureSetROISequence', []))
    aliases = {'left': {'lungl', 'lungleft', 'leftlung'},
               'right': {'lungr', 'lungright', 'rightlung'}}
    candidates = []
    for roi in rois:
        name = str(roi.ROIName)
        normalized = re.sub(r'[^a-z0-9]', '', name.lower())
        match = (override is not None and override in (name, str(roi.ROINumber)))
        if match or (override is None and normalized in aliases[side]):
            candidates.append(roi)
    if len(candidates) != 1:
        available = ', '.join(f'{r.ROINumber}:{r.ROIName}' for r in rois)
        raise ValueError(f'Need one explicit {side} lung ROI; available: {available}')
    return candidates[0]


def rasterize_lungs(volume, rt, left_roi=None, right_roi=None, allow_overlap=False):
    if str(getattr(rt, 'Modality', '')) != 'RTSTRUCT':
        raise ValueError('Annotation must be DICOM RTSTRUCT')
    refs = referenced_series(rt)
    if refs and volume.series_uid not in refs:
        raise ValueError('RTSTRUCT references a different CT series')
    uid_to_index = {uid: index for index, uid in enumerate(volume.sop_uids)}
    results, metadata = {}, {}
    selected = set()
    for side, override in [('left', left_roi), ('right', right_roi)]:
        roi = _choose_roi(rt, side, override)
        number = int(roi.ROINumber)
        if number in selected:
            raise ValueError('Left and right lung cannot use the same ROI')
        selected.add(number)
        if str(roi.ReferencedFrameOfReferenceUID) != volume.frame_uid:
            raise ValueError('Lung ROI and CT have different frames of reference')
        items = [c for c in getattr(rt, 'ROIContourSequence', [])
                 if int(c.ReferencedROINumber) == number]
        if len(items) != 1:
            raise ValueError(f'Missing or duplicate ROIContourSequence for {side} lung')
        contours = list(getattr(items[0], 'ContourSequence', []))
        types = {str(c.ContourGeometricType) for c in contours}
        if not contours or not types <= {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'} or len(types) != 1:
            raise ValueError('Lung ROI needs consistent closed planar contours')
        mask = np.zeros(volume.hu.shape, dtype=bool)
        counts = np.zeros(len(volume.hu), dtype=np.int32)
        subpixel_empty = 0
        for contour in contours:
            image_refs = {str(c.ReferencedSOPInstanceUID)
                          for c in getattr(contour, 'ContourImageSequence', [])}
            if len(image_refs) != 1 or not image_refs <= uid_to_index.keys():
                raise ValueError('Contour needs one SOP reference in the selected CT series; '
                                 'download the complete referenced series')
            index = uid_to_index[next(iter(image_refs))]
            raw = np.asarray(contour.ContourData, dtype=np.float64)
            if raw.size != 3 * int(contour.NumberOfContourPoints):
                raise ValueError('Contour point count mismatch')
            rc = patient_to_pixel(raw.reshape(-1, 3), volume.positions[index],
                                  volume.orientation, volume.spacing[1:])
            if ((rc < -.5 - 1e-4).any() or
                    (rc > np.array(volume.hu.shape[1:]) - .5 + 1e-4).any()):
                raise ValueError('Contour extends outside its CT field of view')
            filled = polygon_mask(rc, volume.hu.shape[1:])
            subpixel_empty += int(not filled.any())
            if 'CLOSEDPLANAR_XOR' in types:
                mask[index] ^= filled
            else:
                mask[index] |= filled
            counts[index] += 1
        if not mask.any():
            raise ValueError(f'{side} lung annotation is empty')
        results[side] = mask
        metadata[side] = {'roi_number': number, 'roi_name': str(roi.ROIName),
                          'contour_type': next(iter(types)),
                          'contour_count': len(contours),
                          'subpixel_empty_contours': subpixel_empty,
                          'annotated_slices': np.flatnonzero(counts).tolist()}
    overlap = results['left'] & results['right']
    count = int(overlap.sum())
    metadata['overlap_check'] = {'voxels': count,
        'slice_indices': np.flatnonzero(overlap.any(axis=(1, 2))).tolist(),
        'explicitly_allowed': bool(allow_overlap),
        'policy': 'Reference labels are preserved unchanged; combined-lung evaluation uses their union.'}
    if count and not allow_overlap:
        raise ValueError(f'Left/right lung reference masks overlap at {count} voxels; '
                         'inspect annotation alignment before explicitly allowing union evaluation')
    return results, metadata
