`timescale 1ns/1ps
// Existing window/convolution + bounded feature cache + read-only AXI4 master.
// One output-channel block per command; no output writer/full-network scheduler.
module lung_conv8_cached_axi(
    input wire clk,input wire rst_n,
    input wire w_valid,output wire w_ready,input wire w_first,input wire w_last,
    input wire[511:0]w_weights,input wire[7:0]w_output_keep,
    input wire[7:0]w_input_zero_point,input wire[7:0]w_output_zero_point,
    input wire[255:0]w_biases,input wire[255:0]w_multipliers,input wire[47:0]w_shifts,input wire w_relu,
    output wire weights_loaded,input wire c_valid,output wire c_ready,input wire[31:0]c_base,
    input wire[9:0]c_height,input wire[9:0]c_width,input wire[6:0]c_channels,input wire c_kernel3,input wire c_padding,
    output wire m_valid,input wire m_ready,output wire[2:0]m_lane,output wire m_last,output wire m_frame_last,
    output wire[8:0]m_x,output wire[8:0]m_y,output wire signed[31:0]m_acc,output wire[7:0]m_code,output wire m_error,
    output wire busy,output wire done,output wire error,output wire[31:0]cache_hits,output wire[31:0]cache_misses,
    output wire[3:0]m_axi_arid,output wire[31:0]m_axi_araddr,output wire[7:0]m_axi_arlen,
    output wire[2:0]m_axi_arsize,output wire[1:0]m_axi_arburst,output wire m_axi_arlock,
    output wire[3:0]m_axi_arcache,output wire[2:0]m_axi_arprot,output wire m_axi_arvalid,input wire m_axi_arready,
    input wire[3:0]m_axi_rid,input wire[63:0]m_axi_rdata,input wire[1:0]m_axi_rresp,input wire m_axi_rlast,
    input wire m_axi_rvalid,output wire m_axi_rready
);
    wire conv_ready,cfg_ready,req_valid,req_ready,rsp_valid,rsp_ready,rsp_error;
    wire[31:0]req_addr;wire[63:0]rsp_data;
    wire burst_valid,burst_ready,q_valid,q_ready,q_error,q_last;
    wire[31:0]burst_addr;wire[7:0]burst_beats;wire[63:0]q_data;
    wire[3:0]blocks=({1'b0,c_channels}+7)/8;
    wire[23:0]words=c_height*c_width*blocks;
    assign c_ready=conv_ready&&cfg_ready;
    wire command=c_valid&&c_ready;
    lung_conv8_window conv(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),.w_first(w_first),.w_last(w_last),
        .w_weights(w_weights),.w_output_keep(w_output_keep),.w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),
        .w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),.weights_loaded(weights_loaded),
        .c_valid(command),.c_ready(conv_ready),.c_base(c_base),.c_height(c_height),.c_width(c_width),.c_channels(c_channels),
        .c_kernel3(c_kernel3),.c_padding(c_padding),.req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(rsp_data),.rsp_error(rsp_error),
        .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),.m_frame_last(m_frame_last),
        .m_x(m_x),.m_y(m_y),.m_acc(m_acc),.m_code(m_code),.m_error(m_error),.busy(busy),.done(done),.error(error));
    lung_feature_cache cache(.clk(clk),.rst_n(rst_n),.cfg_valid(command),.cfg_ready(cfg_ready),.cfg_base(c_base),.cfg_words(words),
        .s_valid(req_valid),.s_ready(req_ready),.s_addr(req_addr),.m_valid(rsp_valid),.m_ready(rsp_ready),.m_data(rsp_data),.m_error(rsp_error),
        .b_valid(burst_valid),.b_ready(burst_ready),.b_addr(burst_addr),.b_beats(burst_beats),
        .q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),.hits(cache_hits),.misses(cache_misses));
    lung_axi_read64 reader(.clk(clk),.rst_n(rst_n),.b_valid(burst_valid),.b_ready(burst_ready),.b_addr(burst_addr),.b_beats(burst_beats),
        .q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),
        .m_axi_arid(m_axi_arid),.m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock),.m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready));
endmodule
