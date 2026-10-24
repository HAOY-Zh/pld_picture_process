`timescale 1ns/1ps
// Same-clock AXI-Lite control plus unchanged single-block memory compute engine.
module lung_conv8_axil(
    input wire clk,input wire rst_n,
    input wire[9:0]s_axi_awaddr,input wire[2:0]s_axi_awprot,input wire s_axi_awvalid,output wire s_axi_awready,
    input wire[31:0]s_axi_wdata,input wire[3:0]s_axi_wstrb,input wire s_axi_wvalid,output wire s_axi_wready,
    output wire[1:0]s_axi_bresp,output wire s_axi_bvalid,input wire s_axi_bready,
    input wire[9:0]s_axi_araddr,input wire[2:0]s_axi_arprot,input wire s_axi_arvalid,output wire s_axi_arready,
    output wire[31:0]s_axi_rdata,output wire[1:0]s_axi_rresp,output wire s_axi_rvalid,input wire s_axi_rready,
    output wire[3:0]m_axi_arid,output wire[31:0]m_axi_araddr,output wire[7:0]m_axi_arlen,
    output wire[2:0]m_axi_arsize,output wire[1:0]m_axi_arburst,output wire m_axi_arlock,
    output wire[3:0]m_axi_arcache,output wire[2:0]m_axi_arprot,output wire m_axi_arvalid,input wire m_axi_arready,
    input wire[3:0]m_axi_rid,input wire[63:0]m_axi_rdata,input wire[1:0]m_axi_rresp,input wire m_axi_rlast,
    input wire m_axi_rvalid,output wire m_axi_rready,
    output wire[3:0]m_axi_awid,output wire[31:0]m_axi_awaddr,output wire[7:0]m_axi_awlen,
    output wire[2:0]m_axi_awsize,output wire[1:0]m_axi_awburst,output wire m_axi_awlock,
    output wire[3:0]m_axi_awcache,output wire[2:0]m_axi_awprot,output wire m_axi_awvalid,input wire m_axi_awready,
    output wire[63:0]m_axi_wdata,output wire[7:0]m_axi_wstrb,output wire m_axi_wlast,
    output wire m_axi_wvalid,input wire m_axi_wready,
    input wire[3:0]m_axi_bid,input wire[1:0]m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready
);
    wire w_valid,w_ready,w_first,w_last,w_relu,weights_loaded,c_valid,c_ready,c_kernel3,c_padding;
    wire[511:0]w_weights;wire[7:0]w_output_keep,w_input_zero_point,w_output_zero_point;
    wire[255:0]w_biases,w_multipliers;wire[47:0]w_shifts;
    wire[31:0]c_base,c_output_base;wire[9:0]c_height,c_width;wire[6:0]c_channels;wire[3:0]c_output_blocks,c_output_block;
    wire core_busy,core_done,core_error;wire[31:0]written_words,cache_hits,cache_misses;
    lung_axil_control control(.clk(clk),.rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),.s_axi_awprot(s_axi_awprot),.s_axi_awvalid(s_axi_awvalid),.s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),.s_axi_wstrb(s_axi_wstrb),.s_axi_wvalid(s_axi_wvalid),.s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),.s_axi_bvalid(s_axi_bvalid),.s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),.s_axi_arprot(s_axi_arprot),.s_axi_arvalid(s_axi_arvalid),.s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),.s_axi_rresp(s_axi_rresp),.s_axi_rvalid(s_axi_rvalid),.s_axi_rready(s_axi_rready),
        .w_valid(w_valid),.w_ready(w_ready),.w_first(w_first),.w_last(w_last),.w_weights(w_weights),.w_output_keep(w_output_keep),
        .w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),.w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),
        .weights_loaded(weights_loaded),.c_valid(c_valid),.c_ready(c_ready),.c_base(c_base),.c_height(c_height),.c_width(c_width),.c_channels(c_channels),
        .c_kernel3(c_kernel3),.c_padding(c_padding),.c_output_base(c_output_base),.c_output_blocks(c_output_blocks),.c_output_block(c_output_block),
        .core_busy(core_busy),.core_done(core_done),.core_error(core_error),.written_words(written_words),.cache_hits(cache_hits),.cache_misses(cache_misses));
    lung_conv8_memory_axi engine(.clk(clk),.rst_n(rst_n),
        .w_valid(w_valid),.w_ready(w_ready),.w_first(w_first),.w_last(w_last),.w_weights(w_weights),.w_output_keep(w_output_keep),
        .w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),.w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),
        .weights_loaded(weights_loaded),.c_valid(c_valid),.c_ready(c_ready),.c_base(c_base),.c_height(c_height),.c_width(c_width),.c_channels(c_channels),
        .c_kernel3(c_kernel3),.c_padding(c_padding),.c_output_base(c_output_base),.c_output_blocks(c_output_blocks),.c_output_block(c_output_block),
        .busy(core_busy),.done(core_done),.error(core_error),.written_words(written_words),.cache_hits(cache_hits),.cache_misses(cache_misses),
        .m_axi_arid(m_axi_arid),.m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock),.m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready),
        .m_axi_awid(m_axi_awid),.m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(m_axi_awlock),.m_axi_awcache(m_axi_awcache),.m_axi_awprot(m_axi_awprot),.m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready));
endmodule
