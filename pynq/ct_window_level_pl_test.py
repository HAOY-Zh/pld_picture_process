"""PYNQ-Z2 实测：DICOM/IMA -> HU -> DDR -> AXI DMA -> PL 窗位窗宽 -> DDR。

这份脚本就是之前在 Jupyter Notebook 中验证成功的代码的可重复版本。
注意：当前 Vivado 工程里的 WL/WW 由 xlconstant 固定为 40/400，
所以这里的 wl、ww 也必须保持为 40、400。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pynq import Overlay, allocate


# PYNQ 板卡上的文件路径，不是 Windows 路径。
DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_window_level.bit"

# 必须与 Vivado 中两个 xlconstant 的值一致。
WL = 40
WW = 400

# 先用 32x32 小图验证，成功后再改为整幅图。
PATCH_HEIGHT = 32
PATCH_WIDTH = 32


def read_ct_hu(filename):
    """读取 DICOM/IMA，并根据斜率、截距转换为 int16 HU。"""
    if not os.path.isfile(filename):
        raise FileNotFoundError("找不到输入文件：{}".format(filename))

    dataset = pydicom.dcmread(filename)
    pixels = dataset.pixel_array
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))

    hu_image = np.rint(
        pixels.astype(np.float32) * slope + intercept
    ).astype(np.int16)
    return hu_image


def software_window_level(hu_image, wl, ww):
    """软件参考结果，便于和 PL 输出逐像素比较。"""
    # RTL 中 WW 为无符号整数，low/high 和中间结果都是整数。
    # 这里完全使用 int32 和 //，逐像素复现 Verilog 的整数除法。
    low = int(wl - ww // 2)
    high = int(wl + ww // 2)
    clipped = np.clip(hu_image.astype(np.int32), low, high)
    gray = ((clipped - low) * 255) // int(ww)
    return gray.astype(np.uint8)


def center_patch(image, height, width):
    """截取图像中心的小块，供第一阶段 DMA 验证。"""
    start_y = (image.shape[0] - height) // 2
    start_x = (image.shape[1] - width) // 2
    return image[start_y:start_y + height, start_x:start_x + width]


def set_window_level(gpio, wl, ww):
    """通过 AXI GPIO 写入 PL 使用的窗位和窗宽。"""
    if not -32768 <= int(wl) <= 32767:
        raise ValueError("WL 超出 int16 范围")
    if not 1 <= int(ww) <= 65535:
        raise ValueError("WW 必须在 1 到 65535 之间")

    # WL 是有符号 16 bit，负数用二进制补码写入 GPIO。
    gpio.channel1.write(int(wl) & 0xFFFF, 0xFFFF)
    # WW 是无符号 16 bit。
    gpio.channel2.write(int(ww) & 0xFFFF, 0xFFFF)


def run_pl_test(hu_patch, wl, ww):
    """把小块送入 PL，再取回输出并与软件结果比较。"""
    # 下载 bitstream；同目录的 .hwh 会被 PYNQ 自动读取。
    overlay = Overlay(BIT_PATH)
    dma = overlay.axi_dma_0
    gpio = overlay.axi_gpio_0
    set_window_level(gpio, wl, ww)

    # AXI-Stream 数据宽度为 32 bit：输入低 16 bit 是有符号 HU，
    # 输出低 8 bit 是灰度值。因此输入/输出 buffer 都按 32 bit 分配。
    input_buffer = allocate(shape=hu_patch.shape, dtype=np.int32)
    output_buffer = allocate(shape=hu_patch.shape, dtype=np.uint32)

    try:
        input_buffer[:] = hu_patch.astype(np.int32)
        expected = software_window_level(hu_patch, wl, ww)

        # 接收通道要先启动，再启动发送通道，避免丢失最早的数据。
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        # PL 输出 32 bit word 的低 8 bit 是有效灰度值。
        rtl_output = (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
        difference = rtl_output.astype(np.int16) - expected.astype(np.int16)

        print("HU patch shape = {}".format(hu_patch.shape))
        print("RTL output shape = {}".format(rtl_output.shape))
        print("RTL output dtype = {}".format(rtl_output.dtype))
        print("same = {}".format(np.array_equal(expected, rtl_output)))
        print("different pixels = {}".format(int(np.count_nonzero(difference))))
        print("max abs error = {}".format(int(np.max(np.abs(difference)))))
        print("mean abs error = {}".format(float(np.mean(np.abs(difference)))))

        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(hu_patch, cmap="gray")
        axes[0].set_title("HU {}x{}".format(*hu_patch.shape))
        axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Software Expected")
        axes[2].imshow(rtl_output, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("PL DMA Output")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        plt.show()

        return np.array_equal(expected, rtl_output)
    finally:
        # 释放板卡 DDR buffer，避免 Notebook 重复运行时逐渐占满内存。
        input_buffer.freebuffer()
        output_buffer.freebuffer()


def main():
    hu_image = read_ct_hu(DICOM_PATH)
    print("full image shape = {}".format(hu_image.shape))
    print("HU min = {}".format(int(np.min(hu_image))))
    print("HU max = {}".format(int(np.max(hu_image))))
    print("dtype = {}".format(hu_image.dtype))

    hu_patch = center_patch(hu_image, PATCH_HEIGHT, PATCH_WIDTH)
    passed = run_pl_test(hu_patch, WL, WW)
    if not passed:
        raise RuntimeError("PL 输出与软件参考结果不一致")
    print("PL window-level DMA test: PASS")


if __name__ == "__main__":
    main()
