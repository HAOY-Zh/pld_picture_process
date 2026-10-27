`timescale 1ns/1ps
// Collect ascending enabled lanes into one complete HWC8 word per pixel.
// A command writes ONE output channel block; other channel words are untouched.
// On any error, drain the remaining stream but suppress subsequent writes.
module lung_output_writer(
    input wire clk,input wire rst_n,
    input wire c_valid,output wire c_ready,input wire c_abort,output wire c_invalid,
    input wire[31:0]c_base,input wire[9:0]c_height,input wire[9:0]c_width,
    input wire[3:0]c_blocks,input wire[3:0]c_block,input wire[7:0]c_keep,input wire[7:0]c_fill,
    input wire s_valid,output wire s_ready,input wire[2:0]s_lane,input wire[7:0]s_code,
    input wire[8:0]s_x,input wire[8:0]s_y,input wire s_last,input wire s_frame_last,input wire s_error,
    output wire busy,output reg done,output reg error,output reg[31:0]written_words,
    output wire[3:0]m_axi_awid,output wire[31:0]m_axi_awaddr,output wire[7:0]m_axi_awlen,
    output wire[2:0]m_axi_awsize,output wire[1:0]m_axi_awburst,output wire m_axi_awlock,
    output wire[3:0]m_axi_awcache,output wire[2:0]m_axi_awprot,output wire m_axi_awvalid,input wire m_axi_awready,
    output wire[63:0]m_axi_wdata,output wire[7:0]m_axi_wstrb,output wire m_axi_wlast,
    output wire m_axi_wvalid,input wire m_axi_wready,
    input wire[3:0]m_axi_bid,input wire[1:0]m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready
);
    localparam IDLE=0,COLLECT=1,SEND=2,WAIT_B=3,FINISH=4;
    reg[2:0]state;reg[31:0]address;reg[9:0]height,width;reg[3:0]blocks;
    reg[8:0]x,y;reg[7:0]keep,remaining,fill;reg[63:0]word_data;reg final_word;
    function[2:0]first_lane(input[7:0]mask);
        integer i;reg found;
        begin first_lane=0;found=0;for(i=0;i<8;i=i+1)if(mask[i]&&!found)begin first_lane=i;found=1;end end
    endfunction
    wire[2:0]lane=first_lane(remaining);
    wire[7:0]next_remaining=remaining&~(8'b1<<lane);
    wire last_lane=next_remaining==0;
    wire final_pixel=x==width-1'b1&&y==height-1'b1;
    wire bad_beat=s_error||s_lane!=lane||s_x!=x||s_y!=y||s_last!=last_lane||s_frame_last!=(last_lane&&final_pixel);
    wire[23:0]command_words=c_height*c_width*c_blocks;
    wire[32:0]command_end={1'b0,c_base}+{9'b0,command_words};
    assign c_invalid=c_abort||c_height==0||c_height>512||c_width==0||c_width>512||
        c_blocks==0||c_blocks>12||c_block>=c_blocks||c_keep==0||command_end>33'h020000000;
    assign c_ready=rst_n&&state==IDLE;assign busy=state!=IDLE;
    assign s_ready=rst_n&&state==COLLECT;
    wire response_valid,response_error,request_ready;
    lung_axi_write64 writer(.clk(clk),.rst_n(rst_n),.b_valid(state==SEND),.b_ready(request_ready),
        .b_addr(address),.b_data(word_data),.b_strb(8'hff),.q_valid(response_valid),.q_ready(state==WAIT_B),.q_error(response_error),
        .m_axi_awid(m_axi_awid),.m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),.m_axi_awlock(m_axi_awlock),.m_axi_awcache(m_axi_awcache),.m_axi_awprot(m_axi_awprot),
        .m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),.m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready));
    task advance_pixel;
        begin
            address<=address+blocks;remaining<=keep;word_data<={8{fill}};
            if(x==width-1'b1)begin x<=0;y<=y+1'b1;end else x<=x+1'b1;
            state<=COLLECT;
        end
    endtask
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;height<=0;width<=0;blocks<=0;x<=0;y<=0;
            keep<=0;remaining<=0;fill<=0;word_data<=0;final_word<=0;done<=0;error<=0;written_words<=0;
        end else begin
            done<=0;
            case(state)
                IDLE:if(c_valid)begin
                    error<=c_invalid;written_words<=0;x<=0;y<=0;final_word<=0;
                    address<=c_base+c_block;height<=c_height;width<=c_width;blocks<=c_blocks;
                    keep<=c_keep;remaining<=c_keep;fill<=c_fill;word_data<={8{c_fill}};
                    state<=c_invalid?FINISH:COLLECT;
                end
                COLLECT:if(s_valid)begin
                    word_data[8*lane+:8]<=s_code;remaining<=next_remaining;error<=error||bad_beat;
                    if((bad_beat||error)&&s_frame_last)state<=FINISH;
                    else if(last_lane)begin
                        final_word<=final_pixel;
                        if(bad_beat||error)begin
                            if(final_pixel)state<=FINISH;else advance_pixel();
                        end else state<=SEND;
                    end
                end
                SEND:if(request_ready)state<=WAIT_B;
                WAIT_B:if(response_valid)begin
                    error<=error||response_error;
                    if(!response_error)written_words<=written_words+1'b1;
                    if(final_word)state<=FINISH;else advance_pixel();
                end
                FINISH:begin done<=1;state<=IDLE;end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
