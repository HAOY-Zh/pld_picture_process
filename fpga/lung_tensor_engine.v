`timescale 1ns/1ps
// HWC8 DDR tensor operations: 0 pool2, 1 nearest2, 2 requantized concat,
// 3 nearest2(A) + requantized concat(A,B). All addresses here are BYTE bases.
// One shared reset with memory masters; error invalidates the entire command.
module lung_tensor_engine(
    input wire clk,input wire rst_n,input wire c_valid,output wire c_ready,
    input wire[1:0]c_op,input wire[31:0]c_a,c_b,c_dst,
    input wire[9:0]c_height,c_width,input wire[6:0]c_ca,c_cb,
    input wire[7:0]c_za,c_zb,c_zo,input wire[30:0]c_ma,c_mb,input wire[5:0]c_sa,c_sb,
    output wire busy,output reg done,output reg error,
    output reg[31:0]written_words,read_words,cycles,
    output wire rd_valid,input wire rd_ready,output wire[31:0]rd_addr,
    input wire r_valid,output wire r_ready,input wire[63:0]r_data,input wire r_error,
    output wire wr_valid,input wire wr_ready,output wire[31:0]wr_addr,output wire[63:0]wr_data,
    input wire b_valid,output wire b_ready,input wire b_error
);
    localparam IDLE=0,SHAPE=1,SIZE=2,WORDS=3,CHECK=4,WORD_START=5,BYTE_PREP=6,
        MAP_PIXEL=7,MAP_WORD=8,READ_CHECK=9,READ_REQ=10,READ_WAIT=11,
        RQ_SEND=12,RQ_WAIT=13,WRITE_REQ=14,WRITE_WAIT=15,ADVANCE=16,FINISH=17,RQ_PREP=18;
    reg[4:0]state;
    reg[1:0]op,slot;
    reg[31:0]a_base,b_base,d_base,address,output_address;
    reg[9:0]height,width,oh,ow,x,y;
    reg[6:0]ca,cb;reg[7:0]total;
    reg[3:0]ablocks,bblocks,oblocks,block_index;
    reg[7:0]za,zb,zo,input_code;
    reg[30:0]ma,mb;reg[5:0]sa,sb;
    reg[19:0]apixels,opixels,source_pixel,pixel_index;
    reg[23:0]awords,bwords,owords;
    reg[2:0]lane,source_lane;
    reg[3:0]source_block;
    reg branch_b,cache_valid;
    reg[31:0]cache_address;
    reg[63:0]cache_data,pool_max,output_word;
    wire concatenate=op[1];
    wire[7:0]channel={1'b0,block_index,3'b000}+lane;
    wire[7:0]branch_channel=channel-{1'b0,ca};
    wire[9:0]source_y=op==0?((y<<1)+slot[1]):((op==1||(op==3&&!branch_b))?(y>>1):y);
    wire[9:0]source_x=op==0?((x<<1)+slot[0]):((op==1||(op==3&&!branch_b))?(x>>1):x);
    wire[9:0]source_width=branch_b?ow:width;
    wire[3:0]source_blocks=branch_b?bblocks:ablocks;
    wire[31:0]source_base=branch_b?b_base:a_base;
    wire[32:0]a_end={1'b0,a_base}+awords,b_end={1'b0,b_base}+bwords,d_end={1'b0,d_base}+owords;
    wire overlap_a={1'b0,a_base}<d_end&&{1'b0,d_base}<a_end;
    wire overlap_b={1'b0,b_base}<d_end&&{1'b0,d_base}<b_end;
    function[63:0]maximum(input[63:0]a,input[63:0]b);
        integer i;begin for(i=0;i<8;i=i+1)maximum[8*i+:8]=a[8*i+:8]>b[8*i+:8]?a[8*i+:8]:b[8*i+:8];end
    endfunction
    function[63:0]masked(input[63:0]value,input[6:0]channels,input[3:0]block,input[7:0]zero);
        integer i;begin for(i=0;i<8;i=i+1)masked[8*i+:8]=(block*8+i<channels)?value[8*i+:8]:zero;end
    endfunction
    wire[63:0]next_max=slot==0?r_data:maximum(pool_max,r_data);
    wire rq_ready,rq_valid,rq_error;wire[7:0]rq_code;
    reg signed[8:0]centered_reg;reg[30:0]multiplier_reg;reg[5:0]shift_reg;
    lung_requant requant(.clk(clk),.rst_n(rst_n),.s_valid(state==RQ_SEND),.s_ready(rq_ready),
        .s_acc({{23{centered_reg[8]}},centered_reg}),.s_multiplier($signed({1'b0,multiplier_reg})),.s_shift(shift_reg),
        .s_zero_point(zo),.s_relu(1'b0),.m_valid(rq_valid),.m_ready(state==RQ_WAIT),.m_code(rq_code),.m_error(rq_error));
    assign c_ready=rst_n&&state==IDLE&&rd_ready&&wr_ready;
    assign busy=state!=IDLE;
    assign rd_valid=rst_n&&state==READ_REQ;assign rd_addr=address;
    assign r_ready=rst_n&&state==READ_WAIT;
    assign wr_valid=rst_n&&state==WRITE_REQ;assign wr_addr=output_address;assign wr_data=output_word;
    assign b_ready=rst_n&&state==WRITE_WAIT;
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;op<=0;slot<=0;a_base<=0;b_base<=0;d_base<=0;address<=0;output_address<=0;
            height<=0;width<=0;oh<=0;ow<=0;x<=0;y<=0;ca<=0;cb<=0;total<=0;
            ablocks<=0;bblocks<=0;oblocks<=0;block_index<=0;za<=0;zb<=0;zo<=0;ma<=0;mb<=0;sa<=0;sb<=0;
            apixels<=0;opixels<=0;source_pixel<=0;pixel_index<=0;awords<=0;bwords<=0;owords<=0;
            lane<=0;source_lane<=0;source_block<=0;branch_b<=0;cache_valid<=0;cache_address<=0;
            cache_data<=0;pool_max<=0;output_word<=0;input_code<=0;done<=0;error<=0;
            written_words<=0;read_words<=0;cycles<=0;centered_reg<=0;multiplier_reg<=0;shift_reg<=0;
        end else begin
            done<=0;
            if(busy&&cycles!=32'hffffffff)cycles<=cycles+1'b1;
            case(state)
            IDLE:if(c_valid&&c_ready)begin
                op<=c_op;a_base<={3'd0,c_a[31:3]};b_base<={3'd0,c_b[31:3]};d_base<={3'd0,c_dst[31:3]};
                height<=c_height;width<=c_width;ca<=c_ca;cb<=c_cb;
                za<=c_za;zb<=c_zb;zo<=c_zo;ma<=c_ma;mb<=c_mb;sa<=c_sa;sb<=c_sb;
                error<=c_a[2:0]!=0||c_dst[2:0]!=0||(c_op[1]&&c_b[2:0]!=0);
                written_words<=0;read_words<=0;cycles<=0;cache_valid<=0;
                x<=0;y<=0;pixel_index<=0;block_index<=0;state<=SHAPE;
            end
            SHAPE:begin
                total<=op[1]?({1'b0,ca}+cb):{1'b0,ca};
                ablocks<=({1'b0,ca}+7)>>3;bblocks<=({1'b0,cb}+7)>>3;
                oblocks<=op[1]?(({1'b0,ca}+cb+7)>>3):(({1'b0,ca}+7)>>3);
                oh<=op==0?(height>>1):((op==1||op==3)?(height<<1):height);
                ow<=op==0?(width>>1):((op==1||op==3)?(width<<1):width);
                error<=error||height==0||width==0||height>512||width>512||ca==0||ca>96||
                    (op[1]?(cb==0||cb>96||({1'b0,ca}+cb)>96):(cb!=0||zo!=za))||
                    (op==0&&(height[0]||width[0]))||((op==1||op==3)&&(height>256||width>256))||
                    (op[1]&&(ma==0||mb==0||sa>62||sb>62));
                state<=SIZE;
            end
            SIZE:begin apixels<=height*width;opixels<=oh*ow;state<=WORDS;end
            WORDS:begin awords<=apixels*ablocks;bwords<=opixels*bblocks;owords<=opixels*oblocks;state<=CHECK;end
            CHECK:begin
                if(error||a_end>33'h020000000||d_end>33'h020000000||overlap_a||
                    (op[1]&&(b_end>33'h020000000||overlap_b)))begin error<=1;state<=FINISH;end
                else state<=WORD_START;
            end
            WORD_START:begin
                lane<=0;slot<=0;pool_max<=0;output_word<={8{zo}};
                branch_b<=0;source_block<=block_index;
                output_address<=d_base+pixel_index*oblocks+block_index;
                state<=concatenate?BYTE_PREP:MAP_PIXEL;
            end
            BYTE_PREP:begin
                if(channel>=total)begin
                    output_word[8*lane+:8]<=zo;
                    if(lane==7)state<=WRITE_REQ;else lane<=lane+1'b1;
                end else begin
                    branch_b<=channel>=ca;
                    source_block<=channel>=ca?branch_channel[6:3]:channel[6:3];
                    source_lane<=channel>=ca?branch_channel[2:0]:channel[2:0];state<=MAP_PIXEL;
                end
            end
            MAP_PIXEL:begin source_pixel<=source_y*source_width+source_x;state<=MAP_WORD;end
            MAP_WORD:begin address<=source_base+source_pixel*source_blocks+source_block;state<=READ_CHECK;end
            READ_CHECK:begin
                if(concatenate&&cache_valid&&cache_address==address)begin
                    input_code<=cache_data[8*source_lane+:8];state<=RQ_PREP;
                end else state<=READ_REQ;
            end
            READ_REQ:if(rd_ready)state<=READ_WAIT;
            READ_WAIT:if(r_valid)begin
                read_words<=read_words+1'b1;
                if(r_error)begin error<=1;state<=FINISH;end
                else begin
                    cache_valid<=1;cache_address<=address;cache_data<=r_data;
                    if(concatenate)begin input_code<=r_data[8*source_lane+:8];state<=RQ_PREP;end
                    else if(op==0)begin
                        pool_max<=next_max;
                        if(slot==3)begin output_word<=masked(next_max,ca,block_index,zo);state<=WRITE_REQ;end
                        else begin slot<=slot+1'b1;state<=MAP_PIXEL;end
                    end else begin output_word<=masked(r_data,ca,block_index,zo);state<=WRITE_REQ;end
                end
            end
            RQ_PREP:begin
                centered_reg<=$signed({1'b0,input_code})-$signed({1'b0,(branch_b?zb:za)});
                multiplier_reg<=branch_b?mb:ma;shift_reg<=branch_b?sb:sa;state<=RQ_SEND;
            end
            RQ_SEND:if(rq_ready)state<=RQ_WAIT;
            RQ_WAIT:if(rq_valid)begin
                if(rq_error)begin error<=1;state<=FINISH;end
                else begin
                    output_word[8*lane+:8]<=rq_code;
                    if(lane==7)state<=WRITE_REQ;else begin lane<=lane+1'b1;state<=BYTE_PREP;end
                end
            end
            WRITE_REQ:if(wr_ready)state<=WRITE_WAIT;
            WRITE_WAIT:if(b_valid)begin
                if(b_error)begin error<=1;state<=FINISH;end
                else begin written_words<=written_words+1'b1;state<=ADVANCE;end
            end
            ADVANCE:begin
                if(block_index==oblocks-1'b1)begin
                    block_index<=0;pixel_index<=pixel_index+1'b1;
                    if(x==ow-1'b1)begin x<=0;y<=y+1'b1;end else x<=x+1'b1;
                    state<=pixel_index==opixels-1'b1 ? FINISH : WORD_START;
                end else begin block_index<=block_index+1'b1;state<=WORD_START;end
            end
            // A malformed read can still be draining. Never report completion
            // or accept new work while either memory master is outstanding.
            FINISH:if(rd_ready&&wr_ready)begin done<=1;state<=IDLE;end
            default:begin error<=1;state<=FINISH;end
            endcase
        end
    end
endmodule
