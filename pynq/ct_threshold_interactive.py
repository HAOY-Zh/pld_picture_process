"""Jupyter interactive CT threshold-segmentation display.

Run on the PYNQ-Z2 board with:

    %run /home/xilinx/ct_test/ct_threshold_interactive.py

The threshold can be changed with the slider or entered directly in the
input box. Press Apply after manual input. Every update writes the value
through AXI GPIO and runs the complete image through the PL DMA pipeline.
"""

import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from IPython.display import clear_output, display
from pynq import Overlay, allocate


DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_threshold.bit"


def read_ct_hu(filename):
    """Read a DICOM/IMA file and convert stored pixels to int16 HU."""
    if not os.path.isfile(filename):
        raise FileNotFoundError("Input file not found: {}".format(filename))

    dataset = pydicom.dcmread(filename)
    pixels = dataset.pixel_array
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return np.rint(
        pixels.astype(np.float32) * slope + intercept
    ).astype(np.int16)


def software_threshold(hu_image, threshold):
    """Reference result matching the RTL: HU >= threshold becomes 255."""
    return np.where(hu_image.astype(np.int32) >= int(threshold), 255, 0).astype(
        np.uint8
    )


def write_threshold(gpio, threshold):
    """Write signed int16 threshold to the single-channel AXI GPIO."""
    gpio.channel1.write(int(threshold) & 0xFFFF, 0xFFFF)


print("Loading FPGA threshold bitstream...")
overlay = Overlay(BIT_PATH, download=True)
dma = overlay.axi_dma_0
gpio = overlay.axi_gpio_0

print("Reading CT image...")
hu_image = read_ct_hu(DICOM_PATH)
print("Image shape = {}".format(hu_image.shape))
print("HU range = {} to {}".format(int(np.min(hu_image)), int(np.max(hu_image))))

print("Allocating DDR buffers...")
pixel_count = int(hu_image.size)
input_buffer = allocate(shape=(pixel_count,), dtype=np.uint32)
output_buffer = allocate(shape=(pixel_count,), dtype=np.uint32)
input_buffer[:] = hu_image.reshape(-1).astype(np.int32).astype(np.uint32)

result_output = widgets.Output()
threshold_slider = widgets.IntSlider(
    value=-500,
    min=-2000,
    max=3000,
    step=10,
    description="阈值:",
    continuous_update=False,
    style={"description_width": "45px"},
    layout=widgets.Layout(width="520px"),
)
threshold_input = widgets.BoundedIntText(
    value=-500,
    min=-32768,
    max=32767,
    step=1,
    description="手动输入:",
    style={"description_width": "70px"},
)
apply_button = widgets.Button(
    description="应用阈值",
    button_style="primary",
    tooltip="将输入的阈值写入 FPGA，并重新处理整幅图像",
)
updating_controls = False


def update_image(threshold):
    """Run the full image through PL and refresh the three result panels."""
    write_threshold(gpio, threshold)
    output_buffer[:] = 0

    # Receive must be started before transmit to avoid losing the first word.
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    rtl_image = (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
    rtl_image = rtl_image.reshape(hu_image.shape)
    expected = software_threshold(hu_image, threshold)
    difference = rtl_image.astype(np.int16) - expected.astype(np.int16)
    same = bool(np.array_equal(expected, rtl_image))

    with result_output:
        clear_output(wait=True)
        print("Threshold = {} HU".format(int(threshold)))
        print("same = {}".format(same))
        print("different pixels = {}".format(int(np.count_nonzero(difference))))
        print("max abs error = {}".format(int(np.max(np.abs(difference)))))

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(hu_image, cmap="gray", vmin=-1024, vmax=1359)
        axes[0].set_title("HU Original")
        axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Software Threshold")
        axes[2].imshow(rtl_image, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("FPGA PL Threshold")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("Threshold = {} HU".format(int(threshold)))
        figure.tight_layout()
        plt.show()


def on_slider_change(change):
    global updating_controls
    if updating_controls or change["name"] != "value":
        return

    updating_controls = True
    threshold_input.value = int(change["new"])
    updating_controls = False
    update_image(int(change["new"]))


def apply_manual_value(_button):
    global updating_controls
    threshold = int(threshold_input.value)
    updating_controls = True
    # Keep the slider in range when a manually entered value is outside it.
    if threshold < threshold_slider.min:
        threshold_slider.min = threshold
    if threshold > threshold_slider.max:
        threshold_slider.max = threshold
    threshold_slider.value = threshold
    updating_controls = False
    update_image(threshold)


threshold_slider.observe(on_slider_change, names="value")
apply_button.on_click(apply_manual_value)

display(
    widgets.VBox(
        [
            widgets.HBox([threshold_slider, threshold_input]),
            apply_button,
            result_output,
        ]
    )
)
update_image(int(threshold_slider.value))
