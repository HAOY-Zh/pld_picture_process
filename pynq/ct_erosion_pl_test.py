"""Standalone 512x512 PL erosion test; input is a binary mask, not HU.

%run /home/xilinx/ct_test/ct_erosion_pl_test.py
Optional: --mask /home/xilinx/ct_test/mask.npy
CT demonstration: --ct --lower -500 --upper 300
CT thresholding here only prepares an input mask on PS for the standalone
erosion test. Actual erosion is performed by PL; software is the reference.
Requires matching ct_erosion.bit and ct_erosion.hwh.
"""
import argparse
import numpy as np


def software_erosion(mask):
    height, width = mask.shape
    padded = np.pad(mask != 0, 1, mode="constant", constant_values=False)
    result = np.ones((height, width), dtype=bool)
    for row in range(3):
        for col in range(3):
            result &= padded[row:row + height, col:col + width]
    return result.astype(np.uint8) * 255


def test_masks():
    white = np.full((512, 512), 255, dtype=np.uint8)
    black = np.zeros_like(white)
    blocks = black.copy()
    blocks[100:200, 100:200] = 255
    blocks[250:350, 250:350] = 255
    blocks[290, 290] = 0
    blocks[50, 50] = 255
    blocks[400, 100:400] = 255
    return [("All white", white), ("All black", black), ("Blocks and noise", blocks)]


def removed_overlay(mask, actual):
    """Retained foreground is white; removed foreground is red."""
    rgb = np.repeat((actual != 0)[..., None], 3, axis=2).astype(np.float32)
    rgb[(mask != 0) & (actual == 0)] = [1, 0, 0]
    return rgb


def zoom_region(mask, actual, size=48):
    """Pick the crop containing the most removed pixels, not an empty crop."""
    removed = (mask != 0) & (actual == 0)
    height, width = mask.shape
    best_count, best_y, best_x = -1, 0, 0
    stride = max(1, size // 2)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y0 = min(y, max(0, height - size))
            x0 = min(x, max(0, width - size))
            count = int(removed[y0:y0 + size, x0:x0 + size].sum())
            if count > best_count:
                best_count, best_y, best_x = count, y0, x0
    return (slice(best_y, best_y + size), slice(best_x, best_x + size))


def show_result(name, mask, actual):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    region = zoom_region(mask, actual)
    overlay = removed_overlay(mask, actual)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    images = [mask, actual, overlay, mask[region], actual[region], overlay[region]]
    titles = ["Input Mask", "FPGA Erosion", "Red = removed pixels",
              "Input (zoom)", "FPGA (zoom)", "Removed pixels (zoom)"]
    for axis, image, title in zip(axes.flat, images, titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            axis.imshow(image, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[0]:
        axis.add_patch(Rectangle((region[1].start - .5, region[0].start - .5),
                                 region[1].stop - region[1].start,
                                 region[0].stop - region[0].start,
                                 fill=False, edgecolor="cyan", linewidth=1))
    figure.suptitle(name + " | cyan box = zoom region")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def main():
    from pynq import Overlay, allocate

    parser = argparse.ArgumentParser(description="Standalone PL binary erosion")
    parser.add_argument("--mask", help="Optional 512x512 binary .npy mask (0/1 or 0/255)")
    parser.add_argument("--ct", action="store_true", help="Use a mask prepared from the CT slice")
    parser.add_argument("--ct-path", default="/home/xilinx/ct_test/test_ct.IMA")
    parser.add_argument("--lower", type=int, default=-500)
    parser.add_argument("--upper", type=int, default=300)
    args = parser.parse_args()
    if args.mask and args.ct:
        parser.error("Choose --mask or --ct, not both")
    cases = test_masks()
    if args.mask:
        mask = np.load(args.mask, allow_pickle=False)
        if mask.shape != (512, 512) or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Input must be a 512x512 binary mask (0/1 or 0/255)")
        cases = [("Input mask", np.where(mask != 0, 255, 0).astype(np.uint8))]
    elif args.ct:
        import pydicom
        if not -32768 <= args.lower <= args.upper <= 32767:
            parser.error("Require -32768 <= lower <= upper <= 32767")
        dataset = pydicom.dcmread(args.ct_path)
        hu = np.rint(dataset.pixel_array.astype(np.float32) *
                     float(getattr(dataset, "RescaleSlope", 1)) +
                     float(getattr(dataset, "RescaleIntercept", 0)))
        if hu.shape != (512, 512):
            raise ValueError("The erosion bitstream requires a 512x512 slice")
        mask = np.where((hu >= args.lower) & (hu <= args.upper), 255, 0).astype(np.uint8)
        print("PS prepares CT mask in range [{}, {}] HU; PL performs erosion.".format(
            args.lower, args.upper))
        cases = [("CT mask [{}, {}] HU".format(args.lower, args.upper), mask)]

    overlay = Overlay("/home/xilinx/ct_test/ct_erosion.bit")
    dma = overlay.axi_dma_0
    input_buffer = allocate(shape=(512, 512), dtype=np.uint32)
    try:
        output_buffer = allocate(shape=(512, 512), dtype=np.uint32)
        try:
            for name, mask in cases:
                expected = software_erosion(mask)
                input_buffer[:] = mask
                dma.recvchannel.transfer(output_buffer)
                dma.sendchannel.transfer(input_buffer)
                dma.sendchannel.wait()
                dma.recvchannel.wait()
                words = np.array(output_buffer, copy=True)
                actual = (words & 0xff).astype(np.uint8)
                same = bool(np.array_equal(words, expected.astype(np.uint32)))
                error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                print("{}: same={}, different pixels={}, max abs error={}".format(
                    name, same, np.count_nonzero(words != expected), int(error.max())))
                before = int(np.count_nonzero(mask))
                after = int(np.count_nonzero(actual))
                removed = int(np.count_nonzero((mask != 0) & (actual == 0)))
                print("White pixels: {} -> {}; removed: {}".format(before, after, removed))
                show_result(name, mask, actual)
                if not same:
                    raise RuntimeError("PL erosion mismatch: " + name)
        finally:
            output_buffer.freebuffer()
    finally:
        input_buffer.freebuffer()
    print("Standalone PL erosion test: PASS")


if __name__ == "__main__":
    main()
