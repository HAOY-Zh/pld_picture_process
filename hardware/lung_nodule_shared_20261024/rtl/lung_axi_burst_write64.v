`timescale 1ns/1ps
// One outstanding INCR write, 1..16 x 64 bits. Descriptor then elastic data.
// AW/W independent; ACK only after final W AND AW AND physical B response.
module lung_axi_burst_write64(
    input wire clk,input wire rst_n,
    input wire b_valid,output wire b_ready,input wire[31:0]b_addr,input wire[7:0]b_beats,
    input wire s_valid,output wire s_ready,input wire[63:0]s_data,input wire s_last,
    output wire q_valid,input wire q_ready,output wire q_error,
    output wire[3:0]m_axi_awid,output wire[31:0]m_axi_awaddr,output wire[7:0]m_axi_awlen,
    output wire[2:0]m_axi_awsize,output wire[1:0]m_axi_awburst,output wire m_axi_awlock,
    output wire[3:0]m_axi_awcache,output wire[2:0]m_axi_awprot,output wire m_axi_awvalid,input wire m_axi_awready,
    output wire[63:0]m_axi_wdata,output wire[7:0]m_axi_wstrb,output wire m_axi_wlast,
    output wire m_axi_wvalid,input wire m_axi_wready,
    input wire[3:0]m_axi_bid,input wire[1:0]m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready
);
    localparam IDLE=0,SEND=1,RESPONSE=2,ACK=3;
    reg[1:0]state;reg[31:0]address;reg[7:0]length,index;
    reg aw_sent,w_sent,failed;
    wire invalid=b_beats==0||b_beats>16||b_addr[31:29]!=0||
        ({1'b0,b_addr[8:0]}+{2'b0,b_beats})>512;
    wire final_beat=index==length;
    wire accepted=s_valid&&s_ready;
    assign b_ready=rst_n&&state==IDLE;
    assign m_axi_awid=0;assign m_axi_awaddr=address;assign m_axi_awlen=length;
    assign m_axi_awsize=3;assign m_axi_awburst=1;assign m_axi_awlock=0;
    assign m_axi_awcache=3;assign m_axi_awprot=0;
    assign m_axi_awvalid=rst_n&&state==SEND&&!aw_sent;
    assign m_axi_wdata=s_data;assign m_axi_wstrb=8'hff;assign m_axi_wlast=final_beat;
    assign m_axi_wvalid=rst_n&&state==SEND&&!w_sent&&s_valid;
    assign s_ready=rst_n&&state==SEND&&!w_sent&&m_axi_wready;
    assign m_axi_bready=rst_n&&state==RESPONSE;
    assign q_valid=rst_n&&state==ACK;assign q_error=failed;
    always @(posedge clk)begin
        if(!rst_n)begin state<=IDLE;address<=0;length<=0;index<=0;aw_sent<=0;w_sent<=0;failed<=0;end
        else case(state)
        IDLE:if(b_valid)begin
            address<={b_addr[28:0],3'd0};length<=b_beats-1'b1;index<=0;
            aw_sent<=0;w_sent<=0;failed<=invalid;state<=invalid?ACK:SEND;
        end
        SEND:begin
            if(m_axi_awvalid&&m_axi_awready)aw_sent<=1;
            if(accepted)begin
                failed<=failed||(s_last!=final_beat);
                if(final_beat)w_sent<=1;else index<=index+1'b1;
            end
            if((aw_sent||(m_axi_awvalid&&m_axi_awready))&&(w_sent||(accepted&&final_beat)))state<=RESPONSE;
        end
        RESPONSE:if(m_axi_bvalid)begin failed<=failed||m_axi_bid!=0||m_axi_bresp[1];state<=ACK;end
        ACK:if(q_ready)state<=IDLE;
        endcase
    end
endmodule
