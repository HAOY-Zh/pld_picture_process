`timescale 1ns/1ps
// One outstanding AXI4 read burst, 64-bit beats, ID 0. No AXI write/control plane.
// Internal addresses are WORD indices; physical AXI byte addresses are 32 bits.
module lung_axi_read64(
    input wire clk,input wire rst_n,
    input wire b_valid,output wire b_ready,input wire[31:0]b_addr,input wire[7:0]b_beats,
    output wire q_valid,input wire q_ready,output wire[63:0]q_data,output wire q_error,output wire q_last,
    output wire[3:0]m_axi_arid,output wire[31:0]m_axi_araddr,output wire[7:0]m_axi_arlen,
    output wire[2:0]m_axi_arsize,output wire[1:0]m_axi_arburst,
    output wire m_axi_arlock,output wire[3:0]m_axi_arcache,output wire[2:0]m_axi_arprot,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire[3:0]m_axi_rid,input wire[63:0]m_axi_rdata,input wire[1:0]m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready
);
    localparam IDLE=0,AR=1,READ=2,DRAIN=3,ERROR=4;
    reg[2:0]state;reg[31:0]address;reg[7:0]length,index;
    wire invalid=b_beats==0||b_beats>16||b_addr[31:29]!=0||
        ({1'b0,b_addr[8:0]}+{2'b0,b_beats})>512;
    wire expected_last=index==length;
    assign b_ready=rst_n&&state==IDLE;
    assign m_axi_arid=0;assign m_axi_araddr=address;assign m_axi_arlen=length;
    assign m_axi_arsize=3;assign m_axi_arburst=1;assign m_axi_arlock=0;
    assign m_axi_arcache=4'b0011;assign m_axi_arprot=0;
    assign m_axi_arvalid=rst_n&&state==AR;
    assign q_valid=rst_n&&(state==ERROR||(state==READ&&m_axi_rvalid));
    assign q_data=state==ERROR?64'd0:m_axi_rdata;
    assign q_error=state==ERROR||m_axi_rresp[1]||m_axi_rid!=0||(m_axi_rlast!=expected_last);
    assign q_last=state==ERROR||m_axi_rlast||expected_last;
    assign m_axi_rready=rst_n&&((state==READ&&q_ready)||state==DRAIN);
    always @(posedge clk)begin
        if(!rst_n)begin state<=IDLE;address<=0;length<=0;index<=0;end
        else case(state)
            IDLE:if(b_valid)begin
                address<={b_addr[28:0],3'b000};length<=b_beats-1'b1;index<=0;
                state<=invalid?ERROR:AR;
            end
            AR:if(m_axi_arready)state<=READ;
            READ:if(m_axi_rvalid&&q_ready)begin
                if(m_axi_rlast)state<=IDLE;
                else if(expected_last)state<=DRAIN;
                else index<=index+1'b1;
            end
            // Missing LAST at expected length: report error, drain remaining
            // beats until actual LAST before accepting a new AR transaction.
            DRAIN:if(m_axi_rvalid&&m_axi_rlast)state<=IDLE;
            ERROR:if(q_ready)state<=IDLE;
            default:state<=IDLE;
        endcase
    end
endmodule
