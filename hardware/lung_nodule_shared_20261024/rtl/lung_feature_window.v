`timescale 1ns/1ps
// Same HWC8 window order. Registered word-address stepping replaces per-tap
// coordinate multiplications and their PREP/ROW bubbles. No speculative reads:
// padding is checked before issuing. A consumed response may issue the next
// request at the same edge; at most one unconsumed response is outstanding.
module lung_feature_window(
    input wire clk,input wire rst_n,
    input wire c_valid,output wire c_ready,
    input wire[31:0]c_base,input wire[9:0]c_height,input wire[9:0]c_width,
    input wire[6:0]c_channels,input wire c_kernel3,input wire c_padding,
    input wire[7:0]c_input_zero,
    output wire req_valid,input wire req_ready,output wire[31:0]req_addr,
    input wire rsp_valid,output wire rsp_ready,input wire[63:0]rsp_data,input wire rsp_error,
    output wire m_valid,input wire m_ready,output wire[63:0]m_data,
    output wire[7:0]m_keep,output wire m_first,output wire m_last,output wire m_frame_last,
    output wire[8:0]m_x,output wire[8:0]m_y,output wire m_error,
    output reg done,output reg error,output reg config_error
);
    localparam IDLE=0,ORIGIN=1,ISSUE=2,RESPONSE=3,OUTPUT=4,FINISH=5;
    reg[2:0]state;
    reg[63:0]held_data;reg held_error;
    reg[31:0]base;
    reg[9:0]height,width,out_height,out_width;
    reg[6:0]channels;reg[3:0]blocks,cb;
    reg kernel3,padding;reg[7:0]input_zero;
    reg[8:0]x,y;reg[1:0]kx,ky;reg[13:0]row_stride;
    // Two extra bits permit negative padded origins without truncating a legal
    // address at the top of the 32-bit word range. Only in-bounds taps issue.
    reg[33:0]address,pixel_address,kernel_row_address,next_row_address;
    wire[33:0]origin={2'd0,base}-(padding?({20'd0,row_stride}+{30'd0,blocks}):34'd0);
    wire[3:0]command_blocks=({1'b0,c_channels}+7)/8;
    wire[23:0]command_words=c_height*c_width*command_blocks;
    wire[32:0]command_end={1'b0,c_base}+{9'd0,command_words};
    wire invalid=c_height==0||c_height>512||c_width==0||c_width>512||
        c_channels==0||c_channels>96||(!c_kernel3&&c_padding)||
        (c_kernel3&&!c_padding&&(c_height<3||c_width<3))||command_end>33'h100000000;
    wire signed[10:0]sy=$signed({2'b0,y})+$signed({9'd0,ky})-$signed({10'd0,padding});
    wire signed[10:0]sx=$signed({2'b0,x})+$signed({9'd0,kx})-$signed({10'd0,padding});
    wire outside=sy<0||sx<0||sy>=$signed({1'b0,height})||sx>=$signed({1'b0,width});
    wire kernel_last=!kernel3||(kx==2&&ky==2);
    wire pixel_last=cb==blocks-1'b1&&kernel_last;
    wire frame_last=x==out_width-1'b1&&y==out_height-1'b1;
    assign c_ready=rst_n&&state==IDLE;
    // The lookahead address uses exactly the same stepping/order as advance_window.
    wire next_channel=cb!=blocks-1'b1;
    wire next_column=!next_channel&&kernel3&&kx!=2;
    wire next_row=!next_channel&&!next_column&&kernel3&&ky!=2;
    wire next_pixel=!next_channel&&!next_column&&!next_row;
    wire final_word=pixel_last&&frame_last;
    wire [3:0] next_cb=next_channel?cb+1'b1:4'd0;
    wire [1:0] next_kx=next_channel?kx:next_column?kx+1'b1:2'd0;
    wire [1:0] next_ky=(next_channel||next_column)?ky:next_row?ky+1'b1:2'd0;
    wire [8:0] next_x=next_pixel?(x!=out_width-1'b1 ? x+1'b1 : 9'd0):x;
    wire [8:0] next_y=next_pixel&&x==out_width-1'b1 ? y+1'b1 : y;
    wire [33:0] next_address=(next_channel||next_column)?address+1'b1:
        next_row?kernel_row_address+row_stride:
        x!=out_width-1'b1 ? pixel_address+blocks : next_row_address;
    wire signed [10:0] next_sy=$signed({2'b0,next_y})+$signed({9'd0,next_ky})-$signed({10'd0,padding});
    wire signed [10:0] next_sx=$signed({2'b0,next_x})+$signed({9'd0,next_kx})-$signed({10'd0,padding});
    wire next_outside=next_sy<0||next_sx<0||next_sy>=$signed({1'b0,height})||next_sx>=$signed({1'b0,width});
    wire consume_window=(state==OUTPUT||(state==RESPONSE&&rsp_valid))&&m_ready;
    wire chain_request=consume_window&&!final_word&&!next_outside;
    assign req_valid=rst_n&&((state==ISSUE&&!outside)||chain_request);
    assign req_addr=state==ISSUE?address[31:0]:next_address[31:0];
    assign rsp_ready=rst_n&&state==RESPONSE;
    // Consume a synchronous response directly when the MAC is ready. The
    // response is always accepted into the holding register if it is stalled.
    wire response_output=state==RESPONSE&&rsp_valid;
    assign m_valid=rst_n&&(state==OUTPUT||response_output);
    assign m_data=response_output?(rsp_error?{8{input_zero}}:rsp_data):held_data;
    assign m_error=response_output?rsp_error:held_error;
    assign m_keep=(cb==blocks-1'b1&&channels[2:0]!=0)?((9'd1<<channels[2:0])-1'b1):8'hff;
    assign m_first=cb==0&&kx==0&&ky==0;
    assign m_last=pixel_last;assign m_frame_last=pixel_last&&frame_last;
    assign m_x=x;assign m_y=y;

    task advance_window;
        begin
            state<=ISSUE;
            if(cb!=blocks-1'b1)begin cb<=cb+1'b1;address<=address+1'b1;end
            else begin
                cb<=0;
                if(kernel3&&kx!=2)begin kx<=kx+1'b1;address<=address+1'b1;end
                else begin
                    kx<=0;
                    if(kernel3&&ky!=2)begin
                        ky<=ky+1'b1;kernel_row_address<=kernel_row_address+row_stride;
                        address<=kernel_row_address+row_stride;
                    end else begin
                        ky<=0;
                        if(frame_last)state<=FINISH;
                        else if(x!=out_width-1'b1)begin
                            x<=x+1'b1;pixel_address<=pixel_address+blocks;
                            kernel_row_address<=pixel_address+blocks;address<=pixel_address+blocks;
                        end else begin
                            x<=0;y<=y+1'b1;pixel_address<=next_row_address;
                            kernel_row_address<=next_row_address;address<=next_row_address;
                            next_row_address<=next_row_address+row_stride;
                        end
                    end
                end
            end
        end
    endtask
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;base<=0;height<=0;width<=0;out_height<=0;out_width<=0;
            channels<=0;blocks<=0;cb<=0;kernel3<=0;padding<=0;input_zero<=0;
            x<=0;y<=0;kx<=0;ky<=0;row_stride<=0;
            address<=0;pixel_address<=0;kernel_row_address<=0;next_row_address<=0;
            held_data<=0;held_error<=0;done<=0;error<=0;config_error<=0;
        end else begin
            done<=0;
            case(state)
            IDLE:if(c_valid)begin
                error<=invalid;config_error<=invalid;cb<=0;kx<=0;ky<=0;x<=0;y<=0;
                if(invalid)state<=FINISH;
                else begin
                    base<=c_base;height<=c_height;width<=c_width;channels<=c_channels;
                    kernel3<=c_kernel3;padding<=c_padding;input_zero<=c_input_zero;blocks<=command_blocks;
                    row_stride<=c_width*command_blocks;
                    out_height<=c_kernel3&&!c_padding?c_height-2:c_height;
                    out_width<=c_kernel3&&!c_padding?c_width-2:c_width;
                    state<=ORIGIN;
                end
            end
            ORIGIN:begin
                address<=origin;pixel_address<=origin;kernel_row_address<=origin;
                next_row_address<=origin+row_stride;state<=ISSUE;
            end
            ISSUE:begin
                held_error<=0;
                if(outside)begin held_data<={8{input_zero}};state<=OUTPUT;end
                else if(req_ready)state<=RESPONSE;
            end
            RESPONSE:if(rsp_valid)begin
                held_data<=rsp_error?{8{input_zero}}:rsp_data;held_error<=rsp_error;
                error<=error||rsp_error;
                if(m_ready)advance_window();else state<=OUTPUT;
            end
            OUTPUT:if(m_ready)advance_window();
            FINISH:begin done<=1;state<=IDLE;end
            default:state<=IDLE;
            endcase
            // Retain the existing holding-register path whenever downstream is
            // stalled. No geometry can advance before that word is accepted.
            if(consume_window&&!final_word)begin
                if(next_outside)begin held_data<={8{input_zero}};held_error<=0;state<=OUTPUT;end
                else if(chain_request&&req_ready)state<=RESPONSE;
            end
        end
    end
endmodule
