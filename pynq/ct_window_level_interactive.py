"""Jupyter 交互版 CT 窗位窗宽显示。

运行方式：
    %run /home/xilinx/ct_test/ct_window_level_interactive.py

界面中的 WL、WW 滑块修改后，会通过 AXI GPIO 更新 FPGA 参数，
然后使用一次 DMA 处理整幅 512x512 图像并刷新 PL 输出。
"""

import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from IPython.display import clear_output, display
from pynq import Overlay, allocate


DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_window_level.bit"


def read_ct_hu(filename):
    """读取 DICOM/IMA 并转换为 int16 HU。"""
    if not os.path.isfile(filename):
        raise FileNotFoundError("找不到输入文件：{}".format(filename))

    dataset = pydicom.dcmread(filename)
    pixels = dataset.pixel_array
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return np.rint(
        pixels.astype(np.float32) * slope + intercept
    ).astype(np.int16)


def software_window_level(hu_image, wl, ww):
    """生成与 RTL 整数除法一致的软件参考图像。"""
    low = int(wl - ww // 2)
    high = int(wl + ww // 2)
    clipped = np.clip(hu_image.astype(np.int32), low, high)
    gray = ((clipped - low) * 255) // int(ww)
    return gray.astype(np.uint8)


def set_window_level(gpio, wl, ww):
    """通过 AXI GPIO 写入有符号 WL 和无符号 WW。"""
    gpio.channel1.write(int(wl) & 0xFFFF, 0xFFFF)
    gpio.channel2.write(int(ww) & 0xFFFF, 0xFFFF)


print("正在加载 FPGA bitstream...")
overlay = Overlay(BIT_PATH, download=True)
dma = overlay.axi_dma_0
gpio = overlay.axi_gpio_0

print("正在读取 CT 图像...")
hu_image = read_ct_hu(DICOM_PATH)
if hu_image.shape != (512, 512):
    raise ValueError("当前交互版要求 512x512 图像，实际为 {}".format(hu_image.shape))

print("正在分配整图 DDR buffer...")
input_buffer = allocate(shape=hu_image.shape, dtype=np.int32)
output_buffer = allocate(shape=hu_image.shape, dtype=np.uint32)
input_buffer[:] = hu_image.astype(np.int32)

result_output = widgets.Output()
wl_slider = widgets.IntSlider(
    value=40,
    min=-1000,
    max=1000,
    step=10,
    description="WL:",
    continuous_update=False,
    style={"description_width": "40px"},
)
wl_input = widgets.BoundedIntText(
    value=40,
    min=-1000,
    max=1000,
    step=10,
    description="WL 输入:",
    style={"description_width": "55px"},
)
ww_slider = widgets.IntSlider(
    value=400,
    min=10,
    max=3000,
    step=10,
    description="WW:",
    continuous_update=False,
    style={"description_width": "40px"},
)
ww_input = widgets.BoundedIntText(
    value=400,
    min=10,
    max=3000,
    step=10,
    description="WW 输入:",
    style={"description_width": "55px"},
)
apply_button = widgets.Button(
    description="应用参数",
    button_style="primary",
    tooltip="将输入框中的 WL/WW 写入 FPGA 并重新处理整图",
)
updating_controls = False


def update_image(*_args):
    """滑块值变化后，重新运行一次整图 PL 处理并刷新显示。"""
    wl = int(wl_slider.value)
    ww = int(ww_slider.value)
    set_window_level(gpio, wl, ww)

    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    rtl_image = (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
    expected = software_window_level(hu_image, wl, ww)
    difference = rtl_image.astype(np.int16) - expected.astype(np.int16)

    with result_output:
        clear_output(wait=True)
        print("WL = {}, WW = {}".format(wl, ww))
        print("same = {}".format(np.array_equal(expected, rtl_image)))
        print("max abs error = {}".format(int(np.max(np.abs(difference)))))

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(hu_image, cmap="gray")
        axes[0].set_title("HU Original")
        axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Software")
        axes[2].imshow(rtl_image, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("FPGA PL")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        plt.show()


def on_slider_change(change):
    """滑块变化时同步输入框，并立即处理整图。"""
    global updating_controls
    if updating_controls:
        return
    if change["name"] == "value":
        if change["owner"] is wl_slider:
            wl_input.value = change["new"]
        else:
            ww_input.value = change["new"]
        update_image()


def apply_manual_values(_button):
    """点击按钮后，将输入框中的参数同步到滑块并处理整图。"""
    global updating_controls
    updating_controls = True
    wl_slider.value = int(wl_input.value)
    ww_slider.value = int(ww_input.value)
    updating_controls = False
    update_image()


wl_slider.observe(on_slider_change, names="value")
ww_slider.observe(on_slider_change, names="value")
apply_button.on_click(apply_manual_values)

display(
    widgets.VBox(
        [
            widgets.HBox([wl_slider, wl_input]),
            widgets.HBox([ww_slider, ww_input]),
            apply_button,
            result_output,
        ]
    )
)
update_image()
