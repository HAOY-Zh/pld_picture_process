`timescale 1ns/1ps
// Simulation-only independent burst slave. Buffer W before AW as AXI allows.
module lung_test_burst_memory(
    input wire clk,input wire rst_n,input integer mode,input wire block_b,
    input wire[1:0]response,input wire[3:0]response_id,
    input wire[3:0]awid,input wire[31:0]awaddr,input wire[7:0]awlen,input wire[2:0]awsize,
    input wire[1:0]awburst,input wire awlock,input wire[3:0]awcache,input wire[2:0]awprot,
    input wire awvalid,output wire awready,input wire[63:0]wdata,input wire[7:0]wstrb,
    input wire wlast,input wire wvalid,output wire wready,
    output reg[3:0]bid,output reg[1:0]bresp,output reg bvalid,input wire bready,
    output reg commit,output reg[31:0]address,output reg[63:0]data,output reg[7:0]strb
);
    reg has_aw=0,pending=0;reg[31:0]saved_address;reg[7:0]length;
    reg[63:0]words[0:15];reg[7:0]masks[0:15];reg lasts[0:15];
    integer received=0,committed=0,cycle=0,delay_count=0;
    reg aw_stall=0,w_stall=0;reg[60:0]held_aw;reg[72:0]held_w;
    assign awready=rst_n&&!has_aw&&!pending&&!bvalid&&(mode!=1||received!=0)&&cycle%11<7;
    assign wready=rst_n&&received<16&&!pending&&!bvalid&&(mode!=2||has_aw)&&cycle%13<8;
    always @(posedge clk)begin
        cycle<=cycle+1;
        if(!rst_n)begin
            has_aw<=0;pending<=0;bvalid<=0;commit<=0;bid<=0;bresp<=0;
            aw_stall<=0;w_stall<=0;received<=0;committed<=0;
        end else begin
            if(aw_stall&&(!awvalid||{awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot}!==held_aw))$fatal(1,"Burst AW unstable");
            if(w_stall&&(!wvalid||{wdata,wstrb,wlast}!==held_w))$fatal(1,"Burst W unstable");
            aw_stall<=awvalid&&!awready;held_aw<={awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot};
            w_stall<=wvalid&&!wready;held_w<={wdata,wstrb,wlast};commit<=0;
            if(awvalid&&awready)begin
                if($isunknown({awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot})||awid||awaddr[2:0]!=0||
                    awlen>15||awsize!=3||awburst!=1||awlock||awprot||awcache!=3||
                    ({1'b0,awaddr[11:0]}+(awlen+1)*8)>4096)$fatal(1,"Illegal burst AW");
                saved_address<=awaddr;length<=awlen;has_aw<=1;
            end
            if(wvalid&&wready)begin
                if($isunknown({wdata,wstrb,wlast})||wstrb!=255)$fatal(1,"Illegal burst W");
                words[received]<=wdata;masks[received]<=wstrb;lasts[received]<=wlast;received<=received+1;
            end
            if(has_aw&&committed<received&&!pending&&!bvalid)begin
                if(lasts[committed]!==(committed==length)||committed>length)$fatal(1,"Burst WLAST/count mismatch");
                commit<=1;address<=saved_address+committed*8;data<=words[committed];strb<=masks[committed];
                committed<=committed+1;
                if(committed==length)begin
                    pending<=1;delay_count<=cycle%7+2;bid<=response_id;bresp<=response;
                end
            end
            if(pending&&!bvalid&&!block_b)begin
                if(delay_count==0)bvalid<=1;else delay_count<=delay_count-1;
            end
            if(bvalid&&bready)begin
                if(received!=length+1||committed!=length+1)$fatal(1,"B before complete burst");
                bvalid<=0;pending<=0;has_aw<=0;received<=0;committed<=0;
            end
        end
    end
endmodule
