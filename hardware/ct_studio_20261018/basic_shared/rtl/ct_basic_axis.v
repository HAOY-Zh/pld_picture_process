`timescale 1ns/1ps
// Six basic operations in one resident overlay. Only change mode when the DMA
// is idle. Inactive branches stay reset; no frame data reaches an unused core.
// Morphology consumes an explicitly prepared binary mask, not CT grey values.
module ct_basic_axis #(
    parameter integer IMAGE_WIDTH=512, IMAGE_HEIGHT=512
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET rst_n, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) input wire clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *) input wire rst_n,
    input wire [2:0] mode,
    input wire [31:0] window_params, thresholds,
    input wire [31:0] s_axis_tdata,
    input wire [3:0] s_axis_tkeep,
    input wire s_axis_tlast, s_axis_tvalid,
    output wire s_axis_tready,
    output wire [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tlast, m_axis_tvalid,
    input wire m_axis_tready
);
    wire [5:0] ready, valid, last;
    wire [31:0] data [0:5];
    wire [3:0] keep [0:5];
    assign s_axis_tready=(mode<6) ? ready[mode] : 1'b0;
    assign m_axis_tdata=(mode<6) ? data[mode] : 32'd0;
    assign m_axis_tkeep=(mode<6) ? keep[mode] : 4'hf;
    assign m_axis_tvalid=(mode<6) ? valid[mode] : 1'b0;
    assign m_axis_tlast=(mode<6) ? last[mode] : 1'b0;
    ct_window_pipeline window_core (
        .clk(clk),.rst_n(rst_n && mode==0),.wl_in(window_params[15:0]),.ww_in(window_params[31:16]),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==0),.s_axis_tready(ready[0]),
        .m_axis_tdata(data[0]),.m_axis_tkeep(keep[0]),.m_axis_tlast(last[0]),
        .m_axis_tvalid(valid[0]),.m_axis_tready(m_axis_tready && mode==0));
    threshold_range_axis threshold_core (
        .clk(clk),.rst_n(rst_n && mode==1),.thresholds_in(thresholds),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==1),.s_axis_tready(ready[1]),
        .m_axis_tdata(data[1]),.m_axis_tkeep(keep[1]),.m_axis_tlast(last[1]),
        .m_axis_tvalid(valid[1]),.m_axis_tready(m_axis_tready && mode==1));
    erosion_axis #(.IMAGE_WIDTH(IMAGE_WIDTH),.IMAGE_HEIGHT(IMAGE_HEIGHT)) erosion_core (
        .clk(clk),.rst_n(rst_n && mode==2),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==2),.s_axis_tready(ready[2]),
        .m_axis_tdata(data[2]),.m_axis_tkeep(keep[2]),.m_axis_tlast(last[2]),
        .m_axis_tvalid(valid[2]),.m_axis_tready(m_axis_tready && mode==2));
    dilation_axis #(.IMAGE_WIDTH(IMAGE_WIDTH),.IMAGE_HEIGHT(IMAGE_HEIGHT)) dilation_core (
        .clk(clk),.rst_n(rst_n && mode==3),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==3),.s_axis_tready(ready[3]),
        .m_axis_tdata(data[3]),.m_axis_tkeep(keep[3]),.m_axis_tlast(last[3]),
        .m_axis_tvalid(valid[3]),.m_axis_tready(m_axis_tready && mode==3));
    opening_axis #(.IMAGE_WIDTH(IMAGE_WIDTH),.IMAGE_HEIGHT(IMAGE_HEIGHT)) opening_core (
        .clk(clk),.rst_n(rst_n && mode==4),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==4),.s_axis_tready(ready[4]),
        .m_axis_tdata(data[4]),.m_axis_tkeep(keep[4]),.m_axis_tlast(last[4]),
        .m_axis_tvalid(valid[4]),.m_axis_tready(m_axis_tready && mode==4));
    closing_axis #(.IMAGE_WIDTH(IMAGE_WIDTH),.IMAGE_HEIGHT(IMAGE_HEIGHT)) closing_core (
        .clk(clk),.rst_n(rst_n && mode==5),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid && mode==5),.s_axis_tready(ready[5]),
        .m_axis_tdata(data[5]),.m_axis_tkeep(keep[5]),.m_axis_tlast(last[5]),
        .m_axis_tvalid(valid[5]),.m_axis_tready(m_axis_tready && mode==5));
endmodule
