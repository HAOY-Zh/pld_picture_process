"""2-D HU filters. No windowing/clipping before filtering; replicated borders."""
import numpy as np

RANGE_LUT = np.array([256, 226, 155, 83, 35, 11, 3, 1, 0], dtype=np.int64)
SPATIAL = ((1, 2, 1), (2, 4, 2), (1, 2, 1))


def image2d(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected a finite, nonempty 2-D HU image')
    return a


def quantize_hu(image):
    """Explicit round-to-nearest-even to signed int16, never silently saturate."""
    a = np.rint(image2d(image))
    if a.min() < -32768 or a.max() > 32767:
        raise ValueError('HU exceeds signed 16-bit hardware range')
    return a.astype(np.int16)


def pack_hu(image):
    return quantize_hu(image).view(np.uint16).astype(np.uint32)


def unpack_hu(words):
    words = np.asarray(words)
    if np.any(words.astype(np.uint64) >> 16):
        raise ValueError('Hardware output upper 16 bits must be zero')
    return (words.astype(np.uint32) & 0xffff).astype(np.uint16).view(np.int16)


def fixed_filter(image, mode='bilateral', range_shift=4):
    """Bit-exact contract of ct_denoise_axis.v, including border and rounding."""
    if mode not in ('bypass', 'gaussian', 'bilateral'):
        raise ValueError('Unknown hardware mode')
    if isinstance(range_shift, bool) or not isinstance(range_shift, (int, np.integer)) or not 0 <= range_shift <= 15:
        raise ValueError('range_shift must be an integer from 0 to 15')
    hu = quantize_hu(image)
    if mode == 'bypass':
        return hu.copy()
    a = hu.astype(np.int64) + 32768
    padded = np.pad(a, 1, mode='edge')
    numerator = np.zeros(a.shape, dtype=np.int64)
    denominator = np.zeros_like(numerator)
    for y in range(3):
        for x in range(3):
            neighbor = padded[y:y+a.shape[0], x:x+a.shape[1]]
            weight = SPATIAL[y][x] * (RANGE_LUT[np.minimum(np.abs(neighbor-a) >> range_shift, 8)]
                                      if mode == 'bilateral' else 256)
            numerator += neighbor * weight
            denominator += weight
    return ((numerator + denominator//2)//denominator - 32768).astype(np.int16)


def bilateral_float(image, sigma_hu=32.0):
    """Same 3x3 spatial kernel; unquantized Gaussian range weights for comparison."""
    a = image2d(image)
    if not np.isfinite(sigma_hu) or sigma_hu <= 0:
        raise ValueError('sigma_hu must be positive')
    p = np.pad(a, 1, mode='edge')
    numerator, denominator = np.zeros_like(a), np.zeros_like(a)
    for y in range(3):
        for x in range(3):
            n = p[y:y+a.shape[0], x:x+a.shape[1]]
            w = SPATIAL[y][x] * np.exp(-0.5*((n-a)/sigma_hu)**2)
            numerator += n*w
            denominator += w
    return numerator/denominator


def guided_float(image, radius=2, epsilon_hu2=1024.0):
    """Self-guided filter: q = mean(a)*I + mean(b), epsilon in HU squared."""
    from scipy.ndimage import uniform_filter
    a = image2d(image)
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError('radius must be a positive integer')
    if not np.isfinite(epsilon_hu2) or epsilon_hu2 <= 0:
        raise ValueError('epsilon_hu2 must be positive')
    size = 2*radius+1
    mean = uniform_filter(a, size=size, mode='nearest')
    variance = np.maximum(uniform_filter(a*a, size=size, mode='nearest') - mean*mean, 0)
    coefficient = variance/(variance+epsilon_hu2)
    intercept = mean*(1-coefficient)
    return uniform_filter(coefficient, size=size, mode='nearest')*a + uniform_filter(intercept, size=size, mode='nearest')


def synthetic_case(size=128, seed=20261014, noise_hu=25.0):
    """Engineering phantom + additive Gaussian noise, NOT clinical LDCT simulation."""
    if size < 16 or not np.isfinite(noise_hu) or noise_hu < 0:
        raise ValueError('size >= 16 and finite noise_hu >= 0 required')
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    reference = np.full((size, size), -1000.0)
    reference[(x/.8)**2+(y/.9)**2 < 1] = 40
    reference[((x+.28)/.22)**2+(y/.55)**2 < 1] = -700
    reference[((x-.28)/.22)**2+(y/.55)**2 < 1] = -700
    reference[(x/.1)**2+((y-.5)/.13)**2 < 1] = 800
    reference[((x-.08)/.05)**2+((y+.2)/.05)**2 < 1] = 70
    noisy = reference + np.random.default_rng(seed).normal(0, noise_hu, reference.shape)
    return noisy, reference
