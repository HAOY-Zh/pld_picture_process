"""PYNQ-Z2 threshold-segmentation test.

The script reads one CT DICOM/IMA image, converts it to HU, sends the whole
image through AXI DMA, and compares the PL result with the software result.

The threshold core implements:

    HU >= threshold  -> 255
    HU <  threshold  -> 0

Run on the board, for example:

    python3 ct_threshold_pl_test.py --threshold -500

The matching ``ct_threshold.hwh`` file must be in the same directory as the
bitstream so that PYNQ can load the overlay metadata automatically.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pynq import Overlay, allocate


DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_threshold.bit"
DEFAULT_OUTPUT_PATH = "/home/xilinx/ct_test/ct_threshold_result.png"


def read_ct_hu(filename):
    """Read a DICOM/IMA file and convert stored pixels to int16 HU."""
    if not os.path.isfile(filename):
        raise FileNotFoundError("Input file not found: {}".format(filename))

    dataset = pydicom.dcmread(filename)
    pixels = dataset.pixel_array
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    hu_image = np.rint(
        pixels.astype(np.float32) * slope + intercept
    ).astype(np.int16)
    return hu_image


def software_threshold(hu_image, threshold):
    """Reference implementation matching the RTL comparison."""
    return np.where(hu_image.astype(np.int32) >= int(threshold), 255, 0).astype(
        np.uint8
    )


def write_threshold(gpio, threshold):
    """Write a signed int16 threshold through the 16-bit AXI GPIO output."""
    threshold = int(threshold)
    if not -32768 <= threshold <= 32767:
        raise ValueError("threshold must be in the int16 range")

    # The RTL interprets threshold_in[15:0] as signed int16.
    gpio.channel1.write(threshold & 0xFFFF, 0xFFFF)


def run_pl_threshold(hu_image, threshold, output_path):
    """Run the full image through PL and compare it with software."""
    overlay = Overlay(BIT_PATH)
    dma = overlay.axi_dma_0
    gpio = overlay.axi_gpio_0
    write_threshold(gpio, threshold)

    pixel_count = int(hu_image.size)
    # The AXI-Stream data width is 32 bits.  The RTL consumes the low 16 bits
    # of each word as a signed HU value and produces the low 8 bits as a mask.
    input_buffer = allocate(shape=(pixel_count,), dtype=np.uint32)
    output_buffer = allocate(shape=(pixel_count,), dtype=np.uint32)

    try:
        input_buffer[:] = hu_image.reshape(-1).astype(np.int32).astype(np.uint32)
        output_buffer[:] = 0
        expected = software_threshold(hu_image, threshold)

        # Start the receive channel first so the first output word is accepted.
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        rtl_output = (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
        rtl_output = rtl_output.reshape(hu_image.shape)
        difference = rtl_output.astype(np.int16) - expected.astype(np.int16)
        same = bool(np.array_equal(expected, rtl_output))

        print("full image shape = {}".format(hu_image.shape))
        print("HU min = {}".format(int(np.min(hu_image))))
        print("HU max = {}".format(int(np.max(hu_image))))
        print("threshold = {} HU".format(int(threshold)))
        print("RTL output dtype = {}".format(rtl_output.dtype))
        print("same = {}".format(same))
        print("different pixels = {}".format(int(np.count_nonzero(difference))))
        print("max abs error = {}".format(int(np.max(np.abs(difference)))))
        print("mean abs error = {}".format(float(np.mean(np.abs(difference)))))

        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(hu_image, cmap="gray", vmin=-1024, vmax=1359)
        axes[0].set_title("HU image")
        axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Software threshold")
        axes[2].imshow(rtl_output, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("PL DMA output")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("Threshold = {} HU".format(int(threshold)))
        figure.tight_layout()

        if output_path:
            figure.savefig(output_path, dpi=150, bbox_inches="tight")
            print("result image saved to {}".format(output_path))
        plt.show()

        if not same:
            raise RuntimeError("PL threshold output differs from software reference")
        print("PL threshold DMA test: PASS")
        return rtl_output
    finally:
        input_buffer.freebuffer()
        output_buffer.freebuffer()


def parse_args():
    parser = argparse.ArgumentParser(description="PYNQ-Z2 PL threshold test")
    parser.add_argument(
        "--input", default=DICOM_PATH, help="DICOM/IMA path on the PYNQ board"
    )
    parser.add_argument(
        "--threshold", type=int, default=-500, help="signed HU threshold"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="PNG result path; use --output '' to disable saving",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hu_image = read_ct_hu(args.input)
    output_path = args.output or None
    run_pl_threshold(hu_image, args.threshold, output_path)


if __name__ == "__main__":
    main()
