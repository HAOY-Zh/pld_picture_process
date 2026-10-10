"""Test the HU -> range threshold -> 3x3 erosion PL pipeline.

Jupyter: %run /home/xilinx/ct_test/ct_threshold_erosion_pl_test.py
Requires matching ct_threshold_erosion.bit/.hwh and 512x512 geometry.
"""
import argparse
import numpy as np


def software_erosion(mask):
    """Centered binary erosion; pixels outside the image are zero."""
    height, width = mask.shape
    padded = np.pad(mask != 0, 1, mode="constant", constant_values=False)
    result = np.ones((height, width), dtype=bool)
    for row in range(3):
        for col in range(3):
            result &= padded[row:row + height, col:col + width]
    return result.astype(np.uint8) * 255


def main():
    import matplotlib.pyplot as plt
    import pydicom
    from pynq import Overlay, allocate

    parser = argparse.ArgumentParser(description="PL threshold + erosion test")
    parser.add_argument("--lower", type=int, default=-500)
    parser.add_argument("--upper", type=int, default=300)
    args = parser.parse_args()
    if not -32768 <= args.lower <= args.upper <= 32767:
        raise ValueError("Require -32768 <= lower <= upper <= 32767")

    dataset = pydicom.dcmread("/home/xilinx/ct_test/test_ct.IMA")
    hu = np.rint(dataset.pixel_array.astype(np.float32) *
                 float(getattr(dataset, "RescaleSlope", 1)) +
                 float(getattr(dataset, "RescaleIntercept", 0)))
    if hu.ndim != 2 or hu.shape != (512, 512):
        raise ValueError("This bitstream requires one 512x512 CT slice")
    if hu.min() < -32768 or hu.max() > 32767:
        raise ValueError("HU exceeds the RTL int16 range")
    hu = hu.astype(np.int16)
    threshold_mask = np.where((hu >= args.lower) & (hu <= args.upper),
                              255, 0).astype(np.uint8)
    expected = software_erosion(threshold_mask)

    overlay = Overlay("/home/xilinx/ct_test/ct_threshold_erosion.bit")
    dma = overlay.axi_dma_0
    packed = ((args.upper & 0xffff) << 16) | (args.lower & 0xffff)
    overlay.axi_gpio_0.channel1.write(packed, 0xffffffff)
    input_buffer = allocate(shape=hu.shape, dtype=np.int32)
    try:
        output_buffer = allocate(shape=hu.shape, dtype=np.uint32)
        try:
            input_buffer[:] = hu
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.transfer(input_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            words = np.array(output_buffer, copy=True)
        finally:
            output_buffer.freebuffer()
    finally:
        input_buffer.freebuffer()

    actual = (words & 0xff).astype(np.uint8)
    # Compare entire words too, detecting garbage in the upper 24 bits.
    same = np.array_equal(words, expected.astype(np.uint32))
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    print("Range = [{}, {}] HU".format(args.lower, args.upper))
    print("same = {}".format(same))
    print("different pixels = {}".format(np.count_nonzero(words != expected)))
    print("max abs error = {}".format(error.max()))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, image, title in zip(axes, [threshold_mask, expected, actual],
                                 ["Threshold Mask", "Software Erosion", "FPGA Erosion"]):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    if not same:
        raise RuntimeError("PL output differs from threshold + erosion reference")
    print("PL threshold + erosion test: PASS")


if __name__ == "__main__":
    main()
