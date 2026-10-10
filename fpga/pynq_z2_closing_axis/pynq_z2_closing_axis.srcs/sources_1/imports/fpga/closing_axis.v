`timescale 1ns/1ps

// Binary closing: centered 3x3 dilation followed by centered 3x3 erosion.
// Each stage retains the image size and treats pixels outside it as zero.
// Consequently, the final outermost border is black (erosion zero padding).
// The intermediate stream stays in PL; no DDR transfer between stages.
module closing_axis #(
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
    wire [31:0] dilated_data;
    wire [3:0] dilated_keep;
    wire dilated_last;
    wire dilated_valid;
    wire dilated_ready;

    dilation_axis #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT)) dilate (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tlast(s_axis_tlast), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(dilated_data), .m_axis_tkeep(dilated_keep),
        .m_axis_tlast(dilated_last), .m_axis_tvalid(dilated_valid),
        .m_axis_tready(dilated_ready)
    );

    erosion_axis #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT)) erode (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(dilated_data), .s_axis_tkeep(dilated_keep),
        .s_axis_tlast(dilated_last), .s_axis_tvalid(dilated_valid),
        .s_axis_tready(dilated_ready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tlast(m_axis_tlast), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );
endmodule
