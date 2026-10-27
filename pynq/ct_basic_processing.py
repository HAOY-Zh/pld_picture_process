"""Explicit PC reference previews. These outputs are never segmentation inputs."""
import numpy as np

OPERATIONS = {
    'window': '窗宽 / 窗位', 'threshold': 'HU 范围阈值',
    'erosion': '二值腐蚀', 'dilation': '二值膨胀',
    'opening': '二值开运算', 'closing': '二值闭运算',
    'histogram': '直方图均衡', 'pseudocolor': '伪彩映射',
}


def parameters(values):
    operation = values.get('operation', 'window')
    if operation not in OPERATIONS: raise ValueError('不支持的基础处理')
    result = {'operation': operation}
    for name, default, low, high in [('wl', -600, -32768, 32767),
        ('ww', 1500, 1, 65535), ('lower', -1000, -32768, 32767),
        ('upper', -400, -32768, 32767)]:
        value = values.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(name + ' 参数必须为范围内整数')
        result[name] = value
    if result['lower'] > result['upper']: raise ValueError('HU 下限不能大于上限')
    return result


def morphology(mask, dilation=False):
    """Same centered 3x3, zero padding at EACH stage as existing PL references."""
    h, w = mask.shape
    padded = np.pad(mask != 0, 1, constant_values=False)
    out = np.zeros((h, w), bool) if dilation else np.ones((h, w), bool)
    for y in range(3):
        for x in range(3):
            if dilation: out |= padded[y:y+h, x:x+w]
            else: out &= padded[y:y+h, x:x+w]
    return out


def process(hu, params):
    hu = np.asarray(hu)
    if hu.ndim != 2 or not np.isfinite(hu).all(): raise ValueError('需要有效二维 HU 图像')
    p = parameters(params); operation = p['operation']
    # Existing basic PL consumes signed int16 HU after rounding, unlike neural preprocessing.
    rounded = np.rint(hu)
    if rounded.min() < -32768 or rounded.max() > 32767:
        raise ValueError('基础处理参考要求 HU 在 int16 范围内，不做静默截断')
    image = rounded.astype(np.int32)
    low = p['wl'] - p['ww']//2; high = p['wl'] + p['ww']//2
    # RTL explicitly saturates HU>=high to 255. For an odd WW, high-low is
    # WW-1, so clipping then dividing alone would incorrectly stop below 255.
    gray = np.where(image <= low, 0, np.where(image >= high, 255,
        ((image-low)*255)//p['ww'])).astype(np.uint8)
    if operation == 'window': return gray
    if operation in ('threshold', 'erosion', 'dilation', 'opening', 'closing'):
        mask = (image >= p['lower']) & (image <= p['upper'])
        if operation == 'erosion': mask = morphology(mask)
        if operation == 'dilation': mask = morphology(mask, True)
        if operation == 'opening': mask = morphology(morphology(mask), True)
        if operation == 'closing': mask = morphology(morphology(mask, True))
        return mask.astype(np.uint8)*255
    if operation == 'histogram':
        counts = np.bincount(gray.ravel(), minlength=256); cdf = counts.cumsum()
        occupied = np.flatnonzero(counts)
        if len(occupied) <= 1: return gray.copy()
        first = cdf[occupied[0]]
        table = np.rint(np.maximum(cdf-first, 0)*255/(gray.size-first)).astype(np.uint8)
        return table[gray]
    # Fixed piecewise color ramp, explicitly a display transform, not LDCT enhancement.
    knots = np.array([0, .25, .5, .75, 1.])
    colors = np.array([[16, 20, 67], [27, 103, 161], [34, 190, 163],
                       [241, 218, 94], [250, 111, 73]])
    x = gray.astype(float)/255
    return np.stack([np.interp(x, knots, colors[:, c]) for c in range(3)], -1).astype(np.uint8)
