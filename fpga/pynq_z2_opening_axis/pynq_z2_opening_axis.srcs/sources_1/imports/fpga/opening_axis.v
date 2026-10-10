`timescale 1ns/1ps

// Binary opening: centered 3x3 erosion followed by centered 3x3 dilation.
// Both stages use zero padding and retain the configured image dimensions.
// Inter-stage pixels stay in PL; no intermediate DDR/DMA transfer is needed.
module opening_axis #(
    parameter integer IMAGE_WIDTH = 512,
    parameter integer IMAGE_HEIGHT = 512
) (
    input wire clk,
    input wire rst_n,
    input wire [31:0] s_axis_tdata,
    input wire [3:0] s_axis_tkeep,
    input wire s_axis_tlast,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    output wire [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tlast,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);
    wire [31:0] eroded_data;
    wire [3:0] eroded_keep;
    wire eroded_last;
    wire eroded_valid;
    wire eroded_ready;

    erosion_axis #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT)) erode (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tlast(s_axis_tlast), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(eroded_data), .m_axis_tkeep(eroded_keep),
        .m_axis_tlast(eroded_last), .m_axis_tvalid(eroded_valid),
        .m_axis_tready(eroded_ready)
    );

    dilation_axis #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT)) dilate (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(eroded_data), .s_axis_tkeep(eroded_keep),
        .s_axis_tlast(eroded_last), .s_axis_tvalid(eroded_valid),
        .s_axis_tready(eroded_ready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tlast(m_axis_tlast), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );
endmodule
