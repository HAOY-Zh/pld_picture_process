`timescale 1ns/1ps
// Read-cache/convolution -> HWC8 packer -> AXI write, one output block/command.
// No AXI-Lite/weight DMA/network scheduler. Input/output full spans must not overlap.
module lung_conv8_memory_axi(
    input wire clk,input wire rst_n,
    input wire w_valid,output wire w_ready,input wire w_first,input wire w_last,
    input wire[511:0]w_weights,input wire[7:0]w_output_keep,
    input wire[7:0]w_input_zero_point,input wire[7:0]w_output_zero_point,
    input wire[255:0]w_biases,input wire[255:0]w_multipliers,input wire[47:0]w_shifts,input wire w_relu,
    output wire weights_loaded,input wire c_valid,output wire c_ready,input wire[31:0]c_base,
    input wire[9:0]c_height,input wire[9:0]c_width,input wire[6:0]c_channels,input wire c_kernel3,input wire c_padding,
    input wire[31:0]c_output_base,input wire[3:0]c_output_blocks,input wire[3:0]c_output_block,
    output wire busy,output wire done,output wire error,output wire[31:0]written_words,
    output wire[31:0]cache_hits,output wire[31:0]cache_misses,
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
    wire conv_ready,conv_weight_ready,conv_busy,conv_done,conv_error,write_ready,write_busy,write_done,write_error,write_invalid;
    wire code_valid,code_ready,code_last,code_final,code_error;wire[2:0]code_lane;wire[8:0]code_x,code_y;
    wire[7:0]code;wire signed[31:0]acc;
    reg[7:0]keep,fill;
    wire[3:0]input_blocks=({1'b0,c_channels}+7)/8;
    wire[23:0]input_words=c_height*c_width*input_blocks;
    wire[32:0]input_end={1'b0,c_base}+{9'd0,input_words};
    wire[9:0]out_height=c_kernel3&&!c_padding?c_height-2:c_height;
    wire[9:0]out_width=c_kernel3&&!c_padding?c_width-2:c_width;
    wire[23:0]output_words=out_height*out_width*c_output_blocks;
    wire[32:0]output_end={1'b0,c_output_base}+{9'd0,output_words};
    wire bad_input=c_height==0||c_height>512||c_width==0||c_width>512||c_channels==0||c_channels>96||
        (!c_kernel3&&c_padding)||(c_kernel3&&!c_padding&&(c_height<3||c_width<3))||input_end>33'h020000000||
        ({1'b0,c_base}<output_end&&{1'b0,c_output_base}<input_end);
    assign w_ready=conv_weight_ready&&!write_busy;
    assign c_ready=conv_ready&&write_ready&&!w_valid;
    wire command=c_valid&&c_ready;
    assign busy=conv_busy||write_busy;assign done=write_done;assign error=write_error||conv_error;
    always @(posedge clk)begin
        if(!rst_n)begin keep<=0;fill<=0;end
        else if(w_valid&&w_ready&&w_first)begin keep<=w_output_keep;fill<=w_output_zero_point;end
    end
    lung_conv8_cached_axi conv(.clk(clk),.rst_n(rst_n),.w_valid(w_valid&&!write_busy),.w_ready(conv_weight_ready),.w_first(w_first),.w_last(w_last),
        .w_weights(w_weights),.w_output_keep(w_output_keep),.w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),
        .w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),.weights_loaded(weights_loaded),
        .c_valid(command&&!write_invalid),.c_ready(conv_ready),.c_base(c_base),.c_height(c_height),.c_width(c_width),.c_channels(c_channels),
        .c_kernel3(c_kernel3),.c_padding(c_padding),.m_valid(code_valid),.m_ready(code_ready),.m_lane(code_lane),.m_last(code_last),
        .m_frame_last(code_final),.m_x(code_x),.m_y(code_y),.m_acc(acc),.m_code(code),.m_error(code_error),
        .busy(conv_busy),.done(conv_done),.error(conv_error),.cache_hits(cache_hits),.cache_misses(cache_misses),
        .m_axi_arid(m_axi_arid),.m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock),.m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready));
    lung_output_writer output_writer(.clk(clk),.rst_n(rst_n),.c_valid(command),.c_ready(write_ready),.c_abort(bad_input),.c_invalid(write_invalid),
        .c_base(c_output_base),.c_height(out_height),.c_width(out_width),.c_blocks(c_output_blocks),.c_block(c_output_block),.c_keep(keep),.c_fill(fill),
        .s_valid(code_valid),.s_ready(code_ready),.s_lane(code_lane),.s_code(code),.s_x(code_x),.s_y(code_y),.s_last(code_last),.s_frame_last(code_final),.s_error(code_error),
        .busy(write_busy),.done(write_done),.error(write_error),.written_words(written_words),
        .m_axi_awid(m_axi_awid),.m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(m_axi_awlock),.m_axi_awcache(m_axi_awcache),.m_axi_awprot(m_axi_awprot),.m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready));
endmodule
