"""Generate a synthetic HU phantom for desktop UI checks; no patient data."""
import argparse
from pathlib import Path

import numpy as np


def create(path):
    path = Path(path)
    if path.exists():
        raise ValueError('Output already exists; choose a new path')
    z, y, x = np.mgrid[:8, :128, :128]
    body = ((x-64)/49)**2 + ((y-64)/43)**2 <= 1
    lungs = (((x-43)/16)**2 + ((y-62)/27)**2 <= 1) | (((x-85)/16)**2 + ((y-62)/27)**2 <= 1)
    hu = np.full(z.shape, -1000, np.int16)
    hu[body] = 35
    hu[lungs & body] = -780
    hu += np.random.default_rng(20261018).integers(-12, 13, size=hu.shape, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        np.savez_compressed(out, hu=hu, spacing_zyx_mm=[2., 1., 1.],
                            origin_lps_mm=[0., 0., 0.], orientation=[1., 0., 0., 0., 1., 0.])
    print(f'Synthetic teaching phantom (not a patient CT): {path.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/demo/synthetic_ct.npz'))
    create(parser.parse_args().output)
