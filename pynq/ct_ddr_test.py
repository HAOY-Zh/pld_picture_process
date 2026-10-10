"""Mayo CT -> HU -> PYNQ DDR Buffer -> readback test.

本脚本当前只验证 PS 端的医学图像读取、HU 转换、DDR 缓存和显示。
不使用 DMA、Overlay 或 window_level.v。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom


# 请修改为 PYNQ-Z2 上实际可访问的 Mayo CT 文件路径。
dicom_path = "D:\pld_picture_process\data\mayo\L067\L067_QD_3_SHARP_1.CT.0003.0001.2016.01.21.18.12.01.921560.405032914.IMA"


def read_ct_hu(filename):
    """读取 DICOM/IMA，并将原始像素转换为 int16 HU。"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "请先将 dicom_path 修改为实际的 Mayo DICOM/IMA 文件路径。"
        )

    ds = pydicom.dcmread(filename)
    pixel_array = ds.pixel_array

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # 先在浮点数中完成 DICOM 重标定，再四舍五入为 FPGA 接口使用的 int16。
    hu_image = np.rint(
        pixel_array.astype(np.float32) * slope + intercept
    ).astype(np.int16)

    return hu_image


def window_level_for_display(hu_image, wl, ww):
    """仅用于 PS 显示的软件窗宽窗位，输出 uint8 灰度图。"""
    low = wl - ww / 2.0
    high = wl + ww / 2.0

    clipped = np.clip(hu_image.astype(np.float32), low, high)
    gray = (clipped - low) * 255.0 / (high - low)
    return np.rint(gray).astype(np.uint8)


def display_ct(hu_image, window_image, wl, ww):
    """显示原始 HU 图像和软件窗宽窗位图像。"""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(hu_image, cmap="gray")
    axes[0].set_title("Original HU")
    axes[0].axis("off")

    axes[1].imshow(window_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Windowed: WL={}, WW={}".format(wl, ww))
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


def run_ddr_test(hu_image):
    """在 PYNQ-Z2 上写入 DDR Buffer，再读回并逐元素比较。"""
    try:
        from pynq import allocate
    except ImportError as exc:
        print("当前环境没有 pynq，DDR allocate 测试需要在 PYNQ-Z2 上运行。")
        print("普通 DICOM/HU/显示部分可以在 PC 上检查。")
        print("ImportError: {}".format(exc))
        return False

    input_buffer = None
    try:
        # allocate 返回位于 PYNQ 可访问 DDR 中的连续缓冲区。
        input_buffer = allocate(shape=hu_image.shape, dtype=np.int16)

        # Python numpy 数组 -> PYNQ DDR Buffer。
        input_buffer[:] = hu_image

        # 复制 DDR 内容回普通 numpy 数组，再进行完整比较。
        readback = np.array(input_buffer)
        identical = np.array_equal(hu_image, readback)
        max_abs_error = int(np.max(np.abs(hu_image.astype(np.int32) - readback.astype(np.int32))))

        print("DDR readback identical: {}".format(identical))
        print("DDR readback max_abs_error: {}".format(max_abs_error))

        if not identical:
            error_count = int(np.count_nonzero(hu_image != readback))
            print("DDR readback error pixels: {}".format(error_count))

        return identical and max_abs_error == 0
    finally:
        if input_buffer is not None:
            # 释放 CMA/DDR 缓冲区，避免重复运行 Notebook 时持续占用内存。
            freebuffer = getattr(input_buffer, "freebuffer", None)
            if callable(freebuffer):
                freebuffer()
            else:
                del input_buffer


def main():
    hu_image = read_ct_hu(dicom_path)

    print("shape = {}".format(hu_image.shape))
    print("HU min = {}".format(int(np.min(hu_image))))
    print("HU max = {}".format(int(np.max(hu_image))))
    print("dtype = {}".format(hu_image.dtype))

    wl = 40
    ww = 400
    window_image = window_level_for_display(hu_image, wl, ww)

    # 先显示 PS 端软件结果；它不代表 FPGA window_level.v 的运行结果。
    display_ct(hu_image, window_image, wl, ww)

    # 只有 PYNQ-Z2 环境才执行 DDR Buffer 验证。
    run_ddr_test(hu_image)


if __name__ == "__main__":
    main()
