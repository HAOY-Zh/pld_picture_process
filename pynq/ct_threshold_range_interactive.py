"""Jupyter interactive CT HU range-segmentation display.

Run on the PYNQ-Z2 board with:

    %run /home/xilinx/ct_test/ct_threshold_range_interactive.py

The PL rule is lower HU <= pixel HU <= upper HU -> 255, otherwise 0.
The two thresholds are packed into one 32-bit AXI GPIO output:
bits [15:0] = lower threshold, bits [31:16] = upper threshold.
"""

import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from IPython.display import clear_output, display
from pynq import Overlay, allocate


DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_threshold_range.bit"


def read_ct_hu(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError("Input file not found: {}".format(filename))
    dataset = pydicom.dcmread(filename)
    pixels = dataset.pixel_array
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return np.rint(
        pixels.astype(np.float32) * slope + intercept
    ).astype(np.int16)


def software_range_threshold(hu_image, lower, upper):
    hu = hu_image.astype(np.int32)
    return np.where((hu >= int(lower)) & (hu <= int(upper)), 255, 0).astype(
        np.uint8
    )


def write_threshold_range(gpio, lower, upper):
    lower = int(lower)
    upper = int(upper)
    if not -32768 <= lower <= 32767:
        raise ValueError("lower threshold must be in the int16 range")
    if not -32768 <= upper <= 32767:
        raise ValueError("upper threshold must be in the int16 range")
    if lower > upper:
        raise ValueError("lower threshold cannot be greater than upper threshold")

    packed = ((upper & 0xFFFF) << 16) | (lower & 0xFFFF)
    gpio.channel1.write(packed, 0xFFFFFFFF)


print("Loading FPGA range-threshold bitstream...")
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
lower_slider = widgets.IntSlider(
    value=-500,
    min=-2000,
    max=2000,
    step=10,
    description="下限:",
    continuous_update=False,
    style={"description_width": "45px"},
    layout=widgets.Layout(width="500px"),
)
upper_slider = widgets.IntSlider(
    value=300,
    min=-2000,
    max=3000,
    step=10,
    description="上限:",
    continuous_update=False,
    style={"description_width": "45px"},
    layout=widgets.Layout(width="500px"),
)
lower_input = widgets.BoundedIntText(
    value=-500,
    min=-32768,
    max=32767,
    description="下限输入:",
    style={"description_width": "70px"},
)
upper_input = widgets.BoundedIntText(
    value=300,
    min=-32768,
    max=32767,
    description="上限输入:",
    style={"description_width": "70px"},
)
apply_button = widgets.Button(
    description="应用范围",
    button_style="primary",
    tooltip="将上下限写入 FPGA，并重新处理整幅图像",
)
updating_controls = False


def update_image(lower, upper):
    if int(lower) > int(upper):
        with result_output:
            clear_output(wait=True)
            print("错误：下限不能大于上限")
        return

    write_threshold_range(gpio, lower, upper)
    output_buffer[:] = 0
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    rtl_image = (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
    rtl_image = rtl_image.reshape(hu_image.shape)
    expected = software_range_threshold(hu_image, lower, upper)
    difference = rtl_image.astype(np.int16) - expected.astype(np.int16)
    same = bool(np.array_equal(expected, rtl_image))

    with result_output:
        clear_output(wait=True)
        print("Range = [{} HU, {} HU]".format(int(lower), int(upper)))
        print("same = {}".format(same))
        print("different pixels = {}".format(int(np.count_nonzero(difference))))
        print("max abs error = {}".format(int(np.max(np.abs(difference)))))

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(hu_image, cmap="gray", vmin=-1024, vmax=1359)
        axes[0].set_title("HU Original")
        axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Software Range")
        axes[2].imshow(rtl_image, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("FPGA PL Range")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("HU range: {} to {}".format(int(lower), int(upper)))
        figure.tight_layout()
        plt.show()


def on_lower_slider_change(change):
    global updating_controls
    if updating_controls or change["name"] != "value":
        return
    updating_controls = True
    lower_input.value = int(change["new"])
    updating_controls = False
    update_image(int(change["new"]), int(upper_slider.value))


def on_upper_slider_change(change):
    global updating_controls
    if updating_controls or change["name"] != "value":
        return
    updating_controls = True
    upper_input.value = int(change["new"])
    updating_controls = False
    update_image(int(lower_slider.value), int(change["new"]))


def apply_manual_values(_button):
    global updating_controls
    lower = int(lower_input.value)
    upper = int(upper_input.value)
    updating_controls = True
    lower_slider.value = lower
    upper_slider.value = upper
    updating_controls = False
    update_image(lower, upper)


lower_slider.observe(on_lower_slider_change, names="value")
upper_slider.observe(on_upper_slider_change, names="value")
apply_button.on_click(apply_manual_values)

display(
    widgets.VBox(
        [
            widgets.HBox([lower_slider, lower_input]),
            widgets.HBox([upper_slider, upper_input]),
            apply_button,
            result_output,
        ]
    )
)
update_image(int(lower_slider.value), int(upper_slider.value))
