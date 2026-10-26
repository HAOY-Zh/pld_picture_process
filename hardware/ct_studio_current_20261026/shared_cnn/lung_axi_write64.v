`timescale 1ns/1ps
// One outstanding single-beat AXI4 write. WORD address -> 32-bit byte address.
// AW and W handshakes are independent; completion waits for B and consumer ACK.
module lung_axi_write64(
    input wire clk,input wire rst_n,
    input wire b_valid,output wire b_ready,input wire[31:0]b_addr,
    input wire[63:0]b_data,input wire[7:0]b_strb,
    output wire q_valid,input wire q_ready,output wire q_error,
    output wire[3:0]m_axi_awid,output wire[31:0]m_axi_awaddr,output wire[7:0]m_axi_awlen,
    output wire[2:0]m_axi_awsize,output wire[1:0]m_axi_awburst,output wire m_axi_awlock,
    output wire[3:0]m_axi_awcache,output wire[2:0]m_axi_awprot,
    output wire m_axi_awvalid,input wire m_axi_awready,
    output wire[63:0]m_axi_wdata,output wire[7:0]m_axi_wstrb,output wire m_axi_wlast,
    output wire m_axi_wvalid,input wire m_axi_wready,
    input wire[3:0]m_axi_bid,input wire[1:0]m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready
);
    localparam IDLE=0,SEND=1,RESPONSE=2,ACK=3;
    reg[1:0]state;reg[31:0]address;reg[63:0]data;reg[7:0]strb;
    reg aw_sent,w_sent,failed;
    assign b_ready=rst_n&&state==IDLE;
    assign m_axi_awid=0;assign m_axi_awaddr=address;assign m_axi_awlen=0;
    assign m_axi_awsize=3;assign m_axi_awburst=1;assign m_axi_awlock=0;
    assign m_axi_awcache=4'b0011;assign m_axi_awprot=0;
    assign m_axi_awvalid=rst_n&&state==SEND&&!aw_sent;
    assign m_axi_wdata=data;assign m_axi_wstrb=strb;assign m_axi_wlast=1;
    assign m_axi_wvalid=rst_n&&state==SEND&&!w_sent;
    assign m_axi_bready=rst_n&&state==RESPONSE;
    assign q_valid=rst_n&&state==ACK;assign q_error=failed;
    always @(posedge clk)begin
        if(!rst_n)begin state<=IDLE;address<=0;data<=0;strb<=0;aw_sent<=0;w_sent<=0;failed<=0;end
        else case(state)
            IDLE:if(b_valid)begin
                address<={b_addr[28:0],3'b000};data<=b_data;strb<=b_strb;aw_sent<=0;w_sent<=0;
                failed<=b_addr[31:29]!=0||b_strb==0;
                state<=(b_addr[31:29]!=0||b_strb==0)?ACK:SEND;
            end
            SEND:begin
                if(m_axi_awready)aw_sent<=1;
                if(m_axi_wready)w_sent<=1;
                if((aw_sent||m_axi_awready)&&(w_sent||m_axi_wready))state<=RESPONSE;
            end
            RESPONSE:if(m_axi_bvalid)begin failed<=m_axi_bid!=0||m_axi_bresp[1];state<=ACK;end
            ACK:if(q_ready)state<=IDLE;
            default:state<=IDLE;
        endcase
    end
endmodule
