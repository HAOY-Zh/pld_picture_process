"""PYNQ-Z2 整幅 CT 图像一次 DMA 传输测试。

流程：IMA -> HU -> DDR -> AXI DMA -> PL 窗位窗宽 -> DDR -> 整图输出。
当前 AXI DMA 的 Buffer Length Register 已改为 23 bit，
因此 512x512x4 Byte 的整幅图可以一次传输完成。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pynq import Overlay, allocate


DICOM_PATH = "/home/xilinx/ct_test/test_ct.IMA"
BIT_PATH = "/home/xilinx/ct_test/ct_window_level.bit"
OUTPUT_PNG = "/home/xilinx/ct_test/ct_window_level_full.png"

WL = 40
WW = 400
IMAGE_WIDTH = 512


def read_ct_hu(filename):
    """读取 DICOM/IMA，并转换为 int16 HU。"""
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
    """使用与 RTL 完全相同的整数除法生成参考结果。"""
    low = int(wl - ww // 2)
    high = int(wl + ww // 2)
    clipped = np.clip(hu_image.astype(np.int32), low, high)
    gray = ((clipped - low) * 255) // int(ww)
    return gray.astype(np.uint8)


def set_window_level(gpio, wl, ww):
    """通过 AXI GPIO 设置 PL 的窗位和窗宽。"""
    gpio.channel1.write(int(wl) & 0xFFFF, 0xFFFF)
    gpio.channel2.write(int(ww) & 0xFFFF, 0xFFFF)


def run_full_image(hu_image, wl, ww):
    """一次 DMA 处理整幅图。"""
    if hu_image.ndim != 2:
        raise ValueError("当前只支持二维 CT 图像")
    if hu_image.shape[1] != IMAGE_WIDTH:
        raise ValueError(
            "当前 DMA 分块脚本按宽度 512 编写，实际宽度为 {}".format(
                hu_image.shape[1]
            )
        )

    overlay = Overlay(BIT_PATH, download=True)
    dma = overlay.axi_dma_0
    gpio = overlay.axi_gpio_0
    set_window_level(gpio, wl, ww)

    input_buffer = allocate(shape=hu_image.shape, dtype=np.int32)
    output_buffer = allocate(shape=hu_image.shape, dtype=np.uint32)
    try:
        input_buffer[:] = hu_image.astype(np.int32)
        print("DMA transfer bytes = {}".format(input_buffer.nbytes))

        # 接收先启动，再启动发送，保证整帧数据不丢失。
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        return (np.asarray(output_buffer) & 0xFF).astype(np.uint8)
    finally:
        input_buffer.freebuffer()
        output_buffer.freebuffer()


def main():
    hu_image = read_ct_hu(DICOM_PATH)
    print("full image shape = {}".format(hu_image.shape))
    print("HU min = {}".format(int(np.min(hu_image))))
    print("HU max = {}".format(int(np.max(hu_image))))
    print("WL = {}, WW = {}".format(WL, WW))

    expected = software_window_level(hu_image, WL, WW)
    rtl_image = run_full_image(hu_image, WL, WW)
    difference = rtl_image.astype(np.int16) - expected.astype(np.int16)

    print("RTL output shape = {}".format(rtl_image.shape))
    print("same = {}".format(np.array_equal(expected, rtl_image)))
    print("different pixels = {}".format(int(np.count_nonzero(difference))))
    print("max abs error = {}".format(int(np.max(np.abs(difference)))))
    print("mean abs error = {}".format(float(np.mean(np.abs(difference)))))

    plt.imsave(OUTPUT_PNG, rtl_image, cmap="gray", vmin=0, vmax=255)
    print("saved PL image = {}".format(OUTPUT_PNG))

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(hu_image, cmap="gray")
    axes[0].set_title("HU 512x512")
    axes[1].imshow(expected, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Software Expected")
    axes[2].imshow(rtl_image, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("PL Full Image")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()

    if not np.array_equal(expected, rtl_image):
        raise RuntimeError("整幅图 PL 输出与软件参考结果不一致")
    print("FULL IMAGE WINDOW-LEVEL TEST: PASS")


if __name__ == "__main__":
    main()
