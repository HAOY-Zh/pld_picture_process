# 3x3 binary erosion

The current implementation uses a centered, all-ones 3x3 kernel. Nonzero
mask pixels are foreground. Pixels outside the image are treated as zero;
output has the same dimensions as the input. The outermost output border
is therefore black, including for an all-white input.

## Sources

- Design sources: `fpga/erosion_3x3.v`, `fpga/erosion_axis.v`.
- Simulation source: `fpga/tb_erosion_axis.v` (simulation top).
- Board test: `pynq/ct_threshold_erosion_pl_test.py`.

`erosion_axis` defaults to IMAGE_WIDTH=512, IMAGE_HEIGHT=512. It accepts
one 32-bit AXI Stream word per pixel, interpreting the low byte as a binary
mask, and produces one 32-bit word containing 0 or 255. TKEEP is 0xf.
Send exactly WIDTH*HEIGHT pixels per frame with TLAST on the final input
word. The wrapper counts geometry and regenerates output TLAST after its
right/bottom padding steps. It does not support variable frame geometry
or arbitrary input TKEEP values.

Two one-bit line stores and horizontal taps construct the neighborhood.
Line-store reads are asynchronous in this version; synthesis may use
distributed memory rather than BRAM. The output pipeline conservatively
allows one pending core result. It is functional and backpressure-safe,
but is not a one-pixel-per-clock design. Measure board throughput before
claiming a frame rate. Optimizing pipelining/BRAM can be a later step.

## Vivado connection

Preserve the existing range-threshold project using Vivado Save Project As
to a separate threshold-plus-erosion project. Keep design_1_wrapper as top.
Add the two design sources and add erosion_axis as a module in the BD.

Replace the threshold-to-DMA output connection with:

```
DMA M_AXIS_MM2S -> threshold_range_axis s_axis
threshold_range_axis m_axis -> erosion_axis s_axis
erosion_axis m_axis -> DMA S_AXIS_S2MM
```

Connect erosion_axis clk to PS FCLK_CLK0 and rst_n to the existing
proc_sys_reset peripheral_aresetn. Keep the 32-bit threshold GPIO,
control/DDR SmartConnect networks, DMA buffer-length width 23, and the
existing single 100 MHz clock unchanged. Set erosion geometry to 512x512.

Validate Design, regenerate output products, update the managed wrapper,
then rerun synthesis, implementation and bitstream generation. Upload the
matching bitstream and HWH under the same base name:

```
/home/xilinx/ct_test/ct_threshold_erosion.bit
/home/xilinx/ct_test/ct_threshold_erosion.hwh
```

Upload the test script, restart the Jupyter kernel and run:

```
%run /home/xilinx/ct_test/ct_threshold_erosion_pl_test.py --lower -500 --upper 300
```

Expected: same=True, different pixels=0, max abs error=0, PASS.
The board test requires the combined pipeline, not the old threshold-only
bitstream. Local RTL simulation is not a substitute for board verification.

## Local verification

Vivado 2022.2 XSim passed a 5x4 test and five consecutive 512x512 frames,
with input gaps and repeated output stalls. Cases cover all-white,
all-black, isolated foreground, a hole in foreground and a mixed mask.
The scoreboard compares pixels, TKEEP, frame TLAST, and stable output under
backpressure. The core separately passed all 512 binary neighborhoods.
