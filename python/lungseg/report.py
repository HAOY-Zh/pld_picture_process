"""Offline result plots. No FPGA execution is claimed by these plots."""
import numpy as np


def _contour(axis, mask, color):
    # Pad to show contours for masks touching image edges, including all-white.
    if mask.any():
        padded = np.pad(mask.astype(float), 1)
        axis.contour(np.arange(mask.shape[1] + 2) - 1,
                     np.arange(mask.shape[0] + 2) - 1,
                     padded, levels=[.5], colors=[color], linewidths=.8)


def save_preview(hu, reference, stages, filename, slice_index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    titles = ['CT: expert lung boundary (cyan)', 'Low-HU candidate (includes external air)',
              'Interior candidates (border-connected removed)',
              'Selected components', 'Automatic baseline: green boundary',
              'Errors: red extra / blue missed / gray correct']
    images = [hu, stages['candidate'], stages['interior'], stages['chosen'], hu, None]
    pred = stages['prediction']
    error = np.zeros((*pred.shape, 3), dtype=float)
    error[pred & reference] = [.65, .65, .65]
    error[pred & ~reference] = [1, 0, 0]
    error[~pred & reference] = [0, .4, 1]
    images[-1] = error
    for index, (axis, image, title) in enumerate(zip(axes.flat, images, titles)):
        if index in (0, 4):
            axis.imshow(image, cmap='gray', vmin=-1000, vmax=400, interpolation='nearest')
        elif image.ndim == 2:
            axis.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        else:
            axis.imshow(image, interpolation='nearest')
        axis.set_title(title)
        axis.axis('off')
    _contour(axes[0, 0], reference, 'cyan')
    _contour(axes[1, 1], pred, 'lime')
    figure.suptitle(f'Slice {slice_index} | CPU baseline, NOT FPGA inference | Education only')
    figure.tight_layout()
    figure.savefig(filename, dpi=130)
    plt.close(figure)
