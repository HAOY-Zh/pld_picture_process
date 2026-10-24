`timescale 1ns/1ps
// Same synchronous three-way BRAM cache, replacement and failed-fill policy.
// A hit can handshake directly from CHECK, rather than adding a RETURN cycle.
// If stalled, capture that exact response into RETURN; data/valid stay stable.
module lung_feature_cache #(parameter SETS=256)(
    input wire clk,input wire rst_n,
    input wire cfg_valid,output wire cfg_ready,input wire[31:0]cfg_base,input wire[23:0]cfg_words,
    input wire s_valid,output wire s_ready,input wire[31:0]s_addr,
    output wire m_valid,input wire m_ready,output wire[63:0]m_data,output wire m_error,
    output wire b_valid,input wire b_ready,output wire[31:0]b_addr,output wire[7:0]b_beats,
    input wire q_valid,output wire q_ready,input wire[63:0]q_data,input wire q_error,input wire q_last,
    output reg[31:0]hits,output reg[31:0]misses
);
    localparam SB=$clog2(SETS),AB=SB+4,TB=32-AB;
    localparam IDLE=0,CHECK=1,BURST=2,FILL=3,RETURN=4;
    reg[2:0]state;reg configured;reg[31:0]base,request,fill_base;reg[32:0]frame_end;
    reg[SB-1:0]set_index;reg[TB-1:0]tag;
    reg[1:0]selected;reg[1:0]next_way[0:SETS-1];
    reg[7:0]fill_length,received;reg[AB-1:0]write_index;
    reg fill_error;reg[63:0]wanted_data,held_data;reg held_error;
    wire[2:0]hit;wire[63:0]way_data[0:2];
    wire requested=s_valid&&s_ready;
    wire accepting=q_valid&&q_ready;
    wire fill_end=accepting&&(q_last||received==fill_length-1'b1);
    wire final_error=fill_error||q_error||(q_last!=(received==fill_length-1'b1));
    wire commit=fill_end&&!final_error;
    wire[32:0]aligned_start={1'b0,request[31:4],4'b0000};
    wire[32:0]aligned_end=aligned_start+16;
    wire[32:0]clip_start=aligned_start<{1'b0,base}?{1'b0,base}:aligned_start;
    wire[32:0]clip_end=aligned_end>frame_end?frame_end:aligned_end;
    wire direct_hit=state==CHECK&&(|hit);
    wire[63:0]hit_data=hit[0]?way_data[0]:hit[1]?way_data[1]:way_data[2];
    assign cfg_ready=rst_n&&state==IDLE;
    // A consumed hit/held response frees the synchronous read port immediately.
    // Miss/refill remains single outstanding; no speculative DDR transaction.
    assign s_ready=rst_n&&configured&&!cfg_valid&&
        (state==IDLE||((direct_hit||state==RETURN)&&m_ready));
    assign m_valid=rst_n&&(state==RETURN||direct_hit);
    assign m_data=direct_hit?hit_data:held_data;
    assign m_error=direct_hit?1'b0:held_error;
    assign b_valid=rst_n&&state==BURST;
    assign b_addr=fill_base;assign b_beats=fill_length;
    assign q_ready=rst_n&&state==FILL;
    initial begin if(SETS<2||(SETS&(SETS-1))!=0||SETS>512)$error("SETS must be power of two, 2..512");end
    genvar way;
    generate for(way=0;way<3;way=way+1)begin: bank
        (* ram_style="block" *)reg[63:0]data[0:SETS*16-1];
        (* ram_style="block" *)reg[TB-1:0]tags[0:SETS-1];
        reg[SETS-1:0]valid;
        reg[TB-1:0]tag_q;reg[63:0]data_q;
        assign hit[way]=valid[set_index]&&tag_q==tag;
        assign way_data[way]=data_q;
        always @(posedge clk)begin
            if(requested)begin
                tag_q<=tags[s_addr[AB-1:4]];
                data_q<=data[s_addr[AB-1:0]];
            end
            if(accepting&&selected==way)data[write_index]<=q_data;
            if(commit&&selected==way)tags[set_index]<=tag;
        end
        always @(posedge clk)begin
            if(!rst_n||(cfg_valid&&cfg_ready))valid<=0;
            else begin
                if(state==BURST&&selected==way)valid[set_index]<=0;
                if(commit&&selected==way)valid[set_index]<=1;
            end
        end
    end endgenerate
    integer i;
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;configured<=0;base<=0;frame_end<=0;request<=0;set_index<=0;tag<=0;
            selected<=0;fill_base<=0;fill_length<=0;received<=0;write_index<=0;fill_error<=0;wanted_data<=0;
            held_data<=0;held_error<=0;hits<=0;misses<=0;
            for(i=0;i<SETS;i=i+1)next_way[i]<=0;
        end else begin
            case(state)
            IDLE:begin
                if(cfg_valid)begin
                    base<=cfg_base;frame_end<={1'b0,cfg_base}+{9'd0,cfg_words};
                    configured<=cfg_words!=0&&({1'b0,cfg_base}+{9'd0,cfg_words})<=33'h100000000;
                    hits<=0;misses<=0;for(i=0;i<SETS;i=i+1)next_way[i]<=0;
                end
            end
            CHECK:begin
                if(|hit)begin
                    hits<=hits+1'b1;held_data<=hit_data;held_error<=0;
                    state<=m_ready?IDLE:RETURN;
                end else begin
                    selected<=next_way[set_index];misses<=misses+1'b1;
                    fill_base<=clip_start[31:0];fill_length<=clip_end-clip_start;
                    write_index<={set_index,clip_start[3:0]};received<=0;fill_error<=0;wanted_data<=0;
                    state<=BURST;
                end
            end
            BURST:if(b_ready)state<=FILL;
            FILL:if(accepting)begin
                fill_error<=final_error;
                if(fill_base+received==request)wanted_data<=q_data;
                if(fill_end)begin
                    held_error<=final_error;
                    held_data<=final_error?64'd0:(fill_base+received==request?q_data:wanted_data);
                    if(!final_error)next_way[set_index]<=selected==2?0:selected+1'b1;
                    state<=RETURN;
                end else begin received<=received+1'b1;write_index<=write_index+1'b1;end
            end
            RETURN:if(m_ready)state<=IDLE;
            default:state<=IDLE;
            endcase
            // Last assignment wins only on an actual next-request handshake.
            // At this edge the old response has already been accepted.
            if(requested)begin
                request<=s_addr;set_index<=s_addr[AB-1:4];tag<=s_addr[31:AB];
                if(s_addr<base||{1'b0,s_addr}>=frame_end)begin held_data<=0;held_error<=1;state<=RETURN;end
                else state<=CHECK;
            end
        end
    end
endmodule
