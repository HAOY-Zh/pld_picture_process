"""2-D same-modality rigid registration and exact FPGA arithmetic reference.

Coordinates are pixels, x=column/y=row. Every matrix maps FIXED output
coordinates into MOVING input coordinates, not the opposite direction.
No claim of CT-MRI registration, clinical validation, or FPGA execution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

Q = 65536  # signed 32-bit Q16.16 geometry; interpolation uses 8 fraction bits


def rigid_matrix(shape, angle_deg=0., tx=0., ty=0.):
    """Output-to-input rotation about image center, then input displacement."""
    h, w = shape
    cx, cy = (w-1)/2, (h-1)/2
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, cx-c*cx+s*cy+tx],
                     [s, c, cy-s*cx-c*cy+ty]], dtype=np.float64)


def quantize_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError('Need a finite 2x3 output-to-input matrix')
    scaled = np.rint(matrix * Q)
    if (scaled < -2**31).any() or (scaled > 2**31-1).any():
        raise ValueError('Fixed-point coefficient overflow')
    return scaled.astype(np.int64)


def _image(image):
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 2 or image.dtype != np.uint8:
        raise ValueError('Need a 2-D uint8 image, both dimensions >= 2')
    return image


def warp_float(moving, matrix, output_shape=None):
    """Bilinear sampling. Outside source domain: zero AND excluded by mask."""
    moving = _image(moving)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError('Need a finite 2x3 matrix')
    h, w = output_shape or moving.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    sx = matrix[0, 0]*xx + matrix[0, 1]*yy + matrix[0, 2]
    sy = matrix[1, 0]*xx + matrix[1, 1]*yy + matrix[1, 2]
    valid = (sx >= 0) & (sy >= 0) & (sx <= moving.shape[1]-1) & (sy <= moving.shape[0]-1)
    x0 = np.floor(np.clip(sx, 0, moving.shape[1]-1)).astype(int)
    y0 = np.floor(np.clip(sy, 0, moving.shape[0]-1)).astype(int)
    x1, y1 = np.minimum(x0+1, moving.shape[1]-1), np.minimum(y0+1, moving.shape[0]-1)
    fx, fy = sx-x0, sy-y0
    a = moving[y0, x0].astype(float)*(1-fx) + moving[y0, x1]*fx
    b = moving[y1, x0].astype(float)*(1-fx) + moving[y1, x1]*fx
    result = np.where(valid, a*(1-fy) + b*fy, 0)
    return result, valid


def warp_fixed(moving, coefficients, output_shape=None):
    """RTL reference: Q16.16 geometry, Q8 bilinear weights, half-up rounding."""
    moving = _image(moving)
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (2, 3) or not np.issubdtype(coefficients.dtype, np.integer):
        raise ValueError('Need 2x3 integer Q16.16 coefficients')
    if (coefficients < -2**31).any() or (coefficients > 2**31-1).any():
        raise ValueError('Coefficient overflow')
    coefficients = coefficients.astype(np.int64)
    yy, xx = np.indices(output_shape or moving.shape, dtype=np.int64)
    sx = coefficients[0, 0]*xx + coefficients[0, 1]*yy + coefficients[0, 2]
    sy = coefficients[1, 0]*xx + coefficients[1, 1]*yy + coefficients[1, 2]
    if (sx < -2**31).any() or (sx > 2**31-1).any() or (sy < -2**31).any() or (sy > 2**31-1).any():
        raise ValueError('Coordinate accumulator overflow')
    valid = (sx >= 0) & (sy >= 0) & (sx <= (moving.shape[1]-1)*Q) & (sy <= (moving.shape[0]-1)*Q)
    x0 = np.clip(sx >> 16, 0, moving.shape[1]-1).astype(int)
    y0 = np.clip(sy >> 16, 0, moving.shape[0]-1).astype(int)
    x1, y1 = np.minimum(x0+1, moving.shape[1]-1), np.minimum(y0+1, moving.shape[0]-1)
    fx, fy = (sx >> 8) & 255, (sy >> 8) & 255
    a = moving[y0, x0].astype(np.int64)*(256-fx) + moving[y0, x1].astype(np.int64)*fx
    b = moving[y1, x0].astype(np.int64)*(256-fx) + moving[y1, x1].astype(np.int64)*fx
    value = (a*(256-fy) + b*fy + 32768) >> 16
    return np.where(valid, value, 0).astype(np.uint8), valid


def score(fixed, warped, valid, min_overlap=.65):
    """Mean squared error on valid overlap; reject too-small overlap."""
    fixed = _image(fixed)
    if warped.shape != fixed.shape or valid.shape != fixed.shape:
        raise ValueError('Image/mask geometry mismatch')
    count = int(valid.sum())
    if count == 0 or count < fixed.size*min_overlap:
        return float('inf')
    diff = fixed.astype(np.float64)[valid] - warped[valid]
    return float(np.mean(diff*diff))


def register_rigid(fixed, moving, max_angle=8., max_shift=10., min_overlap=.65, evaluate=None):
    """Bounded coarse-to-fine coordinate search, NOT a global guarantee.

    evaluate(matrix) -> numeric cost can later call the board DMA core.
    Default evaluator is floating-point CPU warp/MSE. Same-modality only.
    """
    fixed, moving = _image(fixed), _image(moving)
    if fixed.shape != moving.shape:
        raise ValueError('Prototype requires identical image dimensions')
    if not np.isfinite([max_angle, max_shift, min_overlap]).all() or max_angle < 0 or max_shift < 0 or not 0 < min_overlap <= 1:
        raise ValueError('Invalid search limits/overlap')
    cache = {}
    def objective(parameters):
        key = tuple(round(float(p), 7) for p in parameters)
        if abs(key[0]) > max_angle or max(abs(key[1]), abs(key[2])) > max_shift:
            return float('inf')
        if key not in cache:
            matrix = rigid_matrix(fixed.shape, *key)
            if evaluate is None:
                warped, valid = warp_float(moving, matrix)
                cache[key] = score(fixed, warped, valid, min_overlap)
            else:
                cache[key] = float(evaluate(matrix))
        return cache[key]
    best = np.zeros(3, dtype=float)
    best_cost = objective(best)
    # Translation coarse grid reduces a common local-minimum failure mode.
    shifts = sorted(set(np.linspace(-max_shift, max_shift, 5).tolist()+[0.]))
    for tx in shifts:
        for ty in shifts:
            candidate = np.array([0., tx, ty])
            value = objective(candidate)
            if value < best_cost:
                best, best_cost = candidate, value
    for angle_step, pixel_step in ((2., 2.), (.5, .5), (.125, .125)):
        for _ in range(30):
            next_best, next_cost = best.copy(), best_cost
            for axis, step in enumerate((angle_step, pixel_step, pixel_step)):
                for sign in (-1, 1):
                    candidate = best.copy()
                    candidate[axis] += sign*step
                    value = objective(candidate)
                    if value < next_cost:
                        next_best, next_cost = candidate, value
            if next_cost >= best_cost-1e-10:
                break
            best, best_cost = next_best, next_cost
    matrix = rigid_matrix(fixed.shape, *best)
    return {'parameters': best.tolist(), 'matrix_fixed_to_moving': matrix.tolist(),
            'objective_mse': best_cost, 'evaluations': len(cache),
            'limits': {'angle_deg': max_angle, 'shift_pixels': max_shift, 'min_overlap': min_overlap}}


def hardware_packet(fixed, moving):
    fixed, moving = _image(fixed), _image(moving)
    if fixed.shape != moving.shape:
        raise ValueError('Image geometry mismatch')
    return (fixed.astype(np.uint32) | moving.astype(np.uint32) << 8).ravel()


def hardware_expected(fixed, moving, coefficients, flags=0):
    warped, valid = warp_fixed(moving, coefficients, fixed.shape)
    pixels = warped.astype(np.uint32) | fixed.astype(np.uint32) << 8 | valid.astype(np.uint32) << 16
    diff = fixed.astype(np.int64)-warped.astype(np.int64)
    ssd = int((diff[valid]**2).sum())
    trailer = np.array([ssd & 0xffffffff, ssd >> 32, int(valid.sum()), flags], dtype=np.uint32)
    return np.concatenate([pixels.ravel(), trailer])


def phantom(size=128):
    """Asymmetric synthetic phantom, NOT patient CT or an anatomical model."""
    yy, xx = np.indices((size, size), dtype=float)
    x, y = (xx-(size-1)/2)/(size/2), (yy-(size-1)/2)/(size/2)
    image = np.full((size, size), 12., dtype=float)
    body = (x/.82)**2+(y/.9)**2 < 1
    image[body] = 95
    for cx, cy, rx, ry, value in ((-.3,-.08,.22,.55,30),(.29,-.02,.25,.48,43),
                                  (.02,.53,.12,.12,210),(-.08,.05,.09,.14,145)):
        image[((x-cx)/rx)**2+((y-cy)/ry)**2 < 1] = value
    image += 14*np.exp(-((x+.39)**2+(y+.35)**2)/.006)
    image += 8*np.sin(19*x+7*y)*np.exp(-3*(x*x+y*y))*body
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def read_gray(path, wl=-600., ww=1500.):
    """PNG/NPY uint8 or optional DICOM. One 2D image, no hidden resizing."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return _image(np.load(path, allow_pickle=False))
    if path.suffix.lower() in ('.dcm', '.ima'):
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError('DICOM input requires pydicom in the project environment') from exc
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array.astype(np.float64)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        if hu.ndim != 2 or not np.isfinite(hu).all() or not np.isfinite([wl,ww]).all() or ww <= 0:
            raise ValueError('Need finite 2D CT and positive finite window width')
        # Display preprocessing only; raw CT must be preserved separately.
        return np.clip(np.floor((hu-(wl-ww/2))*255/ww),0,255).astype(np.uint8)
    from PIL import Image
    with Image.open(path) as image:
        return _image(np.array(image.convert('L')))


def preview(fixed, moving, warped, path, synthetic=False):
    from PIL import Image, ImageDraw
    h, w = fixed.shape
    scale = max(1, 256//max(h,w))
    pw, ph = w*scale, h*scale
    canvas = Image.new('RGB',(pw*3, (ph+32)*2+36),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,8),'SYNTHETIC PHANTOM / CPU ONLY' if synthetic else 'IMAGE REGISTRATION / CPU ONLY',fill='black')
    rgb_before = np.stack([fixed,moving,fixed],axis=-1)
    rgb_after = np.stack([fixed,warped,fixed],axis=-1)
    difference = np.abs(fixed.astype(np.int16)-warped.astype(np.int16)).astype(np.uint8)
    for i,(label, array) in enumerate((('Fixed',fixed),('Moving before',moving),('Registered',warped),
                                      ('Before: magenta/green',rgb_before),('After: magenta/green',rgb_after),('Absolute difference',difference))):
        x,y = (i%3)*pw,36+(i//3)*(ph+32)
        draw.text((x+6,y+6),label,fill='black')
        tile = Image.fromarray(array).convert('RGB').resize((pw,ph),Image.Resampling.NEAREST)
        canvas.paste(tile,(x,y+30))
    canvas.save(path)


def run_demo(output, fixed_path=None, moving_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    synthetic = fixed_path is None and moving_path is None
    if (fixed_path is None) != (moving_path is None):
        raise ValueError('Supply both --fixed and --moving')
    if synthetic:
        moving = phantom()
        truth = rigid_matrix(moving.shape, 3., 4., -3.)
        sample, _ = warp_float(moving, truth)
        fixed = np.clip(np.floor(sample+.5),0,255).astype(np.uint8)
    else:
        fixed, moving = read_gray(fixed_path), read_gray(moving_path)
        truth = None
    start = perf_counter()
    result = register_rigid(fixed,moving)
    seconds = perf_counter()-start
    matrix = np.array(result['matrix_fixed_to_moving'])
    qmatrix = quantize_matrix(matrix)
    warped, valid = warp_fixed(moving,qmatrix,fixed.shape)
    before, before_valid = warp_float(moving,np.eye(3)[:2])
    common = valid & before_valid
    result.update(synthetic=synthetic, software_only=True, image_shape=list(fixed.shape),
                  seconds_registration_cpu=seconds, q16_coefficients=qmatrix.tolist(),
                  before_mse_common_overlap=score(fixed,before,common,0),
                  after_mse_common_overlap=score(fixed,warped,common,0),
                  common_overlap_pixels=int(common.sum()),
                  warning='Synthetic recovery is not clinical validation, real paired CT validation, or FPGA performance.')
    if truth is not None:
        corners=np.array([[0,0,1],[fixed.shape[1]-1,0,1],[0,fixed.shape[0]-1,1],[fixed.shape[1]-1,fixed.shape[0]-1,1]],float)
        result['known_matrix_fixed_to_moving']=truth.tolist()
        result['max_corner_coordinate_error_pixels']=float(np.linalg.norm(corners@matrix.T-corners@truth.T,axis=1).max())
    np.save(output/'fixed.npy',fixed)
    np.save(output/'moving.npy',moving)
    np.save(output/'registered.npy',warped)
    preview(fixed,moving,warped,output/'preview.png',synthetic)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


def export_vectors(output, shape=(4,5)):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261015)
    matrices=[np.array([[1,0,0],[0,1,0.]]), np.array([[1,0,.5],[0,1,.25]]),
              np.array([[1,0,-.5],[0,1,-1.]]),np.array([[0,-1,shape[0]-1],[1,0,0.]]),
              np.array([[1,0,20],[0,1,0.]]),np.array([[1,0,0],[0,1,0.]])]
    config,inputs,expected=[],[],[]
    for index,matrix in enumerate(matrices):
        fixed=rng.integers(0,256,shape,dtype=np.uint8)
        moving=rng.integers(0,256,shape,dtype=np.uint8)
        qmatrix=quantize_matrix(matrix)
        config.extend(qmatrix.ravel().tolist())
        inputs.extend(hardware_packet(fixed,moving).tolist())
        expected.extend(hardware_expected(fixed,moving,qmatrix,3 if index==5 else 0).tolist())
    for name,values in [('config.mem',config),('input.mem',inputs),('expected.mem',expected)]:
        (output/name).write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in values),encoding='ascii')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo')
    demo.add_argument('--outdir',required=True)
    demo.add_argument('--fixed')
    demo.add_argument('--moving')
    vectors=sub.add_parser('vectors')
    vectors.add_argument('--outdir',required=True)
    args=parser.parse_args()
    if args.command=='demo':
        print(json.dumps(run_demo(args.outdir,args.fixed,args.moving),ensure_ascii=False,indent=2))
    else:
        export_vectors(args.outdir)


if __name__=='__main__':
    main()
