// Generated from frozen word engine: unchanged Q31/RNE, mapping, masks, geometry checks.
`timescale 1ns/1ps
// ABI-compatible tensor engine. Aligned concat handles a complete HWC8 word
// with eight parallel, EXACT original Q31/RNE lanes. Unaligned ca uses the
// original byte path; pool/nearest and shared-memory drain rules unchanged.
module lung_tensor_burst_engine(
    input wire clk,input wire rst_n,input wire c_valid,output wire c_ready,
    input wire[1:0]c_op,input wire[31:0]c_a,c_b,c_dst,
    input wire[9:0]c_height,c_width,input wire[6:0]c_ca,c_cb,
    input wire[7:0]c_za,c_zb,c_zo,input wire[30:0]c_ma,c_mb,input wire[5:0]c_sa,c_sb,
    output wire busy,output reg done,output reg error,
    output reg[31:0]written_words,read_words,cycles,
    output wire rd_valid,input wire rd_ready,output wire[31:0]rd_addr,output wire[7:0]rd_beats,
    input wire r_valid,output wire r_ready,input wire[63:0]r_data,input wire r_error,input wire r_last,
    output wire wr_valid,input wire wr_ready,output wire[31:0]wr_addr,output wire[63:0]wr_data,
    output wire[7:0]wr_beats,output wire stream_valid,input wire stream_ready,output wire stream_last,
    input wire b_valid,output wire b_ready,input wire b_error
);
    localparam IDLE=0,SHAPE=1,SIZE=2,WORDS=3,CHECK=4,WORD_START=5,BYTE_PREP=6,
        MAP_PIXEL=7,MAP_WORD=8,READ_CHECK=9,READ_REQ=10,READ_WAIT=11,
        RQ_SEND=12,RQ_WAIT=13,WRITE_REQ=14,WRITE_WAIT=15,ADVANCE=16,FINISH=17,RQ_PREP=18,
        V_PREP=19,V_SEND=20,V_WAIT=21,LINE_REQ=22,LINE_WAIT=23,STREAM=24,WAIT_B=25,LINE_CLIP=26,LINE_LENGTH=27,LINE_SETUP=28,TABLE_PREP=29;
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
    reg[63:0]cache_data,pool_max,output_word,vector_input;
    reg table_loading;reg[8:0]table_index;
    wire concatenate=op[1];
    wire vector_concat=concatenate&&ca[2:0]==0;
    wire[7:0]channel={1'b0,block_index,3'b000}+lane;
    wire[7:0]branch_channel=channel-{1'b0,ca};
    wire[9:0]source_y=op==0?((y<<1)+slot[1]):((op==1||(op==3&&!branch_b))?(y>>1):y);
    wire[9:0]source_x=op==0?((x<<1)+slot[0]):((op==1||(op==3&&!branch_b))?(x>>1):x);
    wire[9:0]source_width=branch_b?ow:width;
    wire[3:0]source_blocks=branch_b?bblocks:ablocks;
    wire[31:0]source_base=branch_b?b_base:a_base;
    
    // Tiny separate A/B line caches: 2 x 16 x 64 bits, invalidated per command.
    reg[63:0]line_a[0:15],line_b[0:15],output_bank[0:15];
    reg valid_a,valid_b;reg[27:0]tag_a,tag_b;
    reg[31:0]line_base,burst_base;reg[4:0]line_length,line_index,buffer_count,buffer_limit,stream_index;
    reg line_error;reg[63:0]wanted_word,fetched_word;
    // In pool mode both banks describe A, so either matching bank may hit.
    // Prefer the row-parity bank only on a miss. This avoids duplicate fills
    // when two short rows share one aligned line.
    wire pool_hit_a=valid_a&&tag_a==address[31:4];
    wire pool_hit_b=valid_b&&tag_b==address[31:4];
    wire cache_bank_b=op==0?(pool_hit_a?1'b0:(pool_hit_b?1'b1:slot[1])):branch_b;
    wire line_hit=cache_bank_b?(valid_b&&tag_b==address[31:4]):(valid_a&&tag_a==address[31:4]);
    wire[32:0]aligned_lo={1'b0,address[31:4],4'd0};
    reg[32:0]branch_end,line_end,clipped_lo,clipped_hi,clipped_length;
    wire expected_line_last=line_index==line_length-1'b1;
    wire fill_error=line_error||r_error||(r_last!=expected_line_last);
    wire[9:0]page_left=10'd512-{1'b0,output_address[8:0]};
    wire[4:0]first_limit=page_left<16?page_left[4:0]:5'd16;
    wire last_output=block_index==oblocks-1'b1&&pixel_index==opixels-1'b1;
    assign rd_beats={3'd0,line_length};assign wr_beats={3'd0,buffer_count};
    assign stream_valid=rst_n&&state==STREAM;assign stream_last=stream_index==buffer_count-1'b1;
    wire[32:0]a_end={1'b0,a_base}+awords,b_end={1'b0,b_base}+bwords,d_end={1'b0,d_base}+owords;
    wire overlap_a={1'b0,a_base}<d_end&&{1'b0,d_base}<a_end;
    wire overlap_b={1'b0,b_base}<d_end&&{1'b0,d_base}<b_end;
    function[63:0]maximum(input[63:0]a,input[63:0]b);
        integer i;begin for(i=0;i<8;i=i+1)maximum[8*i+:8]=a[8*i+:8]>b[8*i+:8]?a[8*i+:8]:b[8*i+:8];end
    endfunction
    function[63:0]masked(input[63:0]value,input[6:0]channels,input[3:0]block,input[7:0]zero);
        integer i;begin for(i=0;i<8;i=i+1)masked[8*i+:8]=(block*8+i<channels)?value[8*i+:8]:zero;end
    endfunction
    wire[63:0]next_max=slot==0?fetched_word:maximum(pool_max,fetched_word);
    wire rq_ready,rq_valid,rq_error;wire[7:0]rq_code;
    reg signed[8:0]centered_reg;reg[30:0]multiplier_reg;reg[5:0]shift_reg;
    lung_requant requant(.clk(clk),.rst_n(rst_n),.s_valid(state==RQ_SEND),.s_ready(rq_ready),
        .s_acc({{23{centered_reg[8]}},centered_reg}),.s_multiplier($signed({1'b0,multiplier_reg})),.s_shift(shift_reg),
        .s_zero_point(zo),.s_relu(1'b0),.m_valid(rq_valid),.m_ready(state==RQ_WAIT),.m_code(rq_code),.m_error(rq_error));
    // Each lane has an independent read port. The original scalar requantizer
    // builds BOTH complete 256-code tables for every aligned concat command.
    // No host ABI, floating approximation, or cached quantization parameters.
    wire[63:0]lookup_codes;
    genvar n;
    generate for(n=0;n<8;n=n+1)begin: lookup_lanes
        (* ram_style="distributed" *) reg[7:0]codes[0:511];
        always @(posedge clk)
            if(rst_n&&table_loading&&state==RQ_WAIT&&rq_valid&&!rq_error)
                codes[table_index]<=rq_code;
        assign lookup_codes[8*n+:8]=codes[{branch_b,vector_input[8*n+:8]}];
    end endgenerate
    assign c_ready=rst_n&&state==IDLE&&rd_ready&&wr_ready;
    assign busy=state!=IDLE;
    assign rd_valid=rst_n&&state==LINE_REQ;assign rd_addr=line_base;
    assign r_ready=rst_n&&state==LINE_WAIT;
    assign wr_valid=rst_n&&state==WRITE_WAIT;assign wr_addr=burst_base;assign wr_data=output_bank[stream_index[3:0]];
    assign b_ready=rst_n&&state==WAIT_B;
    integer i;
    always @(posedge clk)begin
        if(!rst_n)begin
            valid_a<=0;valid_b<=0;tag_a<=0;tag_b<=0;line_base<=0;burst_base<=0;line_length<=0;line_index<=0;
            branch_end<=0;line_end<=0;clipped_lo<=0;clipped_hi<=0;clipped_length<=0;
            buffer_count<=0;buffer_limit<=0;stream_index<=0;line_error<=0;wanted_word<=0;fetched_word<=0;
            state<=IDLE;op<=0;slot<=0;a_base<=0;b_base<=0;d_base<=0;address<=0;output_address<=0;
            height<=0;width<=0;oh<=0;ow<=0;x<=0;y<=0;ca<=0;cb<=0;total<=0;
            ablocks<=0;bblocks<=0;oblocks<=0;block_index<=0;za<=0;zb<=0;zo<=0;ma<=0;mb<=0;sa<=0;sb<=0;
            apixels<=0;opixels<=0;source_pixel<=0;pixel_index<=0;awords<=0;bwords<=0;owords<=0;
            lane<=0;source_lane<=0;source_block<=0;branch_b<=0;cache_valid<=0;cache_address<=0;
            cache_data<=0;pool_max<=0;output_word<=0;input_code<=0;done<=0;error<=0;
            written_words<=0;read_words<=0;cycles<=0;centered_reg<=0;multiplier_reg<=0;shift_reg<=0;vector_input<=0;
            table_loading<=0;table_index<=0;
        end else begin
            done<=0;
            if(busy&&cycles!=32'hffffffff)cycles<=cycles+1'b1;
            case(state)
            IDLE:if(c_valid&&c_ready)begin
                op<=c_op;a_base<={3'd0,c_a[31:3]};b_base<={3'd0,c_b[31:3]};d_base<={3'd0,c_dst[31:3]};
                height<=c_height;width<=c_width;ca<=c_ca;cb<=c_cb;
                za<=c_za;zb<=c_zb;zo<=c_zo;ma<=c_ma;mb<=c_mb;sa<=c_sa;sb<=c_sb;
                error<=c_a[2:0]!=0||c_dst[2:0]!=0||(c_op[1]&&c_b[2:0]!=0);
                written_words<=0;read_words<=0;cycles<=0;cache_valid<=0;valid_a<=0;valid_b<=0;buffer_count<=0;
                table_loading<=0;table_index<=0;
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
                else if(vector_concat)begin table_loading<=1;table_index<=0;state<=TABLE_PREP;end
                else state<=WORD_START;
            end
            WORD_START:begin
                lane<=0;slot<=0;pool_max<=0;output_word<={8{zo}};
                output_address<=d_base+pixel_index*oblocks+block_index;
                if(vector_concat)begin
                    branch_b<=block_index>=ablocks;
                    source_block<=block_index>=ablocks?block_index-ablocks:block_index;
                    state<=MAP_PIXEL;
                end else begin branch_b<=0;source_block<=block_index;state<=concatenate?BYTE_PREP:MAP_PIXEL;end
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
                    if(vector_concat)begin vector_input<=cache_data;state<=V_PREP;end
                    else begin input_code<=cache_data[8*source_lane+:8];state<=RQ_PREP;end
                end else state<=READ_REQ;
            end
            READ_REQ:begin
                if(line_hit)begin fetched_word<=cache_bank_b?line_b[address[3:0]]:line_a[address[3:0]];state<=READ_WAIT;end
                else begin
                    branch_end<={1'b0,source_base}+(branch_b?bwords:awords);line_end<=aligned_lo+16;state<=LINE_CLIP;
                end
            end
            LINE_CLIP:begin
                clipped_lo<=aligned_lo<{1'b0,source_base}?{1'b0,source_base}:aligned_lo;
                clipped_hi<=line_end>branch_end?branch_end:line_end;state<=LINE_LENGTH;
            end
            LINE_LENGTH:begin clipped_length<=clipped_hi-clipped_lo;state<=LINE_SETUP;end
            LINE_SETUP:begin
                if(clipped_lo>address||clipped_hi<=address||clipped_length==0||clipped_length>16)begin error<=1;state<=FINISH;end
                else begin
                    line_base<=clipped_lo[31:0];line_length<=clipped_length[4:0];line_index<=0;line_error<=0;wanted_word<=0;
                    if(cache_bank_b)valid_b<=0;else valid_a<=0;
                    state<=LINE_REQ;
                end
            end
            LINE_REQ:if(rd_ready)state<=LINE_WAIT;
            LINE_WAIT:if(r_valid)begin
                if(cache_bank_b)line_b[(line_base[3:0]+line_index[3:0])&4'hf]<=r_data;
                else line_a[(line_base[3:0]+line_index[3:0])&4'hf]<=r_data;
                if(line_base+line_index==address)wanted_word<=r_data;
                line_error<=fill_error;
                if(r_last||expected_line_last)begin
                    if(fill_error)begin error<=1;state<=FINISH;end
                    else begin
                        if(cache_bank_b)begin valid_b<=1;tag_b<=address[31:4];end else begin valid_a<=1;tag_a<=address[31:4];end
                        fetched_word<=line_base+line_index==address?r_data:wanted_word;state<=READ_WAIT;
                    end
                end else line_index<=line_index+1'b1;
            end
            READ_WAIT:begin
                // ABI read_words counts logical delivered words, NOT prefetched AXI beats.
                read_words<=read_words+1'b1;cache_valid<=1;cache_address<=address;cache_data<=fetched_word;
                if(vector_concat)begin vector_input<=fetched_word;state<=V_PREP;end
                else if(concatenate)begin input_code<=fetched_word[8*source_lane+:8];state<=RQ_PREP;end
                else if(op==0)begin
                    pool_max<=next_max;
                    if(slot==3)begin output_word<=masked(next_max,ca,block_index,zo);state<=WRITE_REQ;end
                    else begin slot<=slot+1'b1;state<=MAP_PIXEL;end
                end else begin output_word<=masked(fetched_word,ca,block_index,zo);state<=WRITE_REQ;end
            end
            TABLE_PREP:begin
                centered_reg<=$signed({1'b0,table_index[7:0]})-$signed({1'b0,(table_index[8]?zb:za)});
                multiplier_reg<=table_index[8]?mb:ma;shift_reg<=table_index[8]?sb:sa;state<=RQ_SEND;
            end
            RQ_PREP:begin
                centered_reg<=$signed({1'b0,input_code})-$signed({1'b0,(branch_b?zb:za)});
                multiplier_reg<=branch_b?mb:ma;shift_reg<=branch_b?sb:sa;state<=RQ_SEND;
            end
            RQ_SEND:if(rq_ready)state<=RQ_WAIT;
            RQ_WAIT:if(rq_valid)begin
                if(rq_error)begin error<=1;state<=FINISH;end
                else if(table_loading)begin
                    if(table_index==511)begin table_loading<=0;state<=WORD_START;end
                    else begin table_index<=table_index+1'b1;state<=TABLE_PREP;end
                end else begin
                    output_word[8*lane+:8]<=rq_code;
                    if(lane==7)state<=WRITE_REQ;else begin lane<=lane+1'b1;state<=BYTE_PREP;end
                end
            end
            V_PREP:begin
                output_word<=masked(lookup_codes,total[6:0],block_index,zo);state<=WRITE_REQ;
            end
            WRITE_REQ:begin
                output_bank[buffer_count[3:0]]<=output_word;
                if(buffer_count==0)begin burst_base<=output_address;buffer_limit<=first_limit;end
                buffer_count<=buffer_count+1'b1;
                if((buffer_count==0&&first_limit==1)||(buffer_count!=0&&buffer_count+1'b1==buffer_limit)||last_output)state<=WRITE_WAIT;
                else state<=ADVANCE;
            end
            WRITE_WAIT:if(wr_ready)begin stream_index<=0;state<=STREAM;end
            STREAM:if(stream_ready)begin
                if(stream_last)state<=WAIT_B;else stream_index<=stream_index+1'b1;
            end
            WAIT_B:if(b_valid)begin
                if(b_error)begin error<=1;buffer_count<=0;state<=FINISH;end
                else begin written_words<=written_words+buffer_count;buffer_count<=0;state<=ADVANCE;end
            end
            ADVANCE:begin
                if(block_index==oblocks-1'b1)begin
                    block_index<=0;pixel_index<=pixel_index+1'b1;
                    if(x==ow-1'b1)begin x<=0;y<=y+1'b1;end else x<=x+1'b1;
                    state<=pixel_index==opixels-1'b1 ? FINISH : WORD_START;
                end else begin block_index<=block_index+1'b1;state<=WORD_START;end
            end
            FINISH:if(rd_ready&&wr_ready)begin done<=1;state<=IDLE;end
            default:begin error<=1;state<=FINISH;end
            endcase
        end
    end
endmodule
