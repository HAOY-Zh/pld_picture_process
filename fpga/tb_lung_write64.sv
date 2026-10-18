`timescale 1ns/1ps
// Shared simulation-only AXI write slave. AW/W are accepted independently.
module lung_test_write_memory(
    input wire clk,input wire rst_n,input integer mode,input wire block_b,
    input wire[1:0]response,input wire[3:0]response_id,
    input wire[3:0]awid,input wire[31:0]awaddr,input wire[7:0]awlen,input wire[2:0]awsize,
    input wire[1:0]awburst,input wire awlock,input wire[3:0]awcache,input wire[2:0]awprot,
    input wire awvalid,output wire awready,input wire[63:0]wdata,input wire[7:0]wstrb,
    input wire wlast,input wire wvalid,output wire wready,
    output reg[3:0]bid,output reg[1:0]bresp,output reg bvalid,input wire bready,
    output reg commit,output reg[31:0]address,output reg[63:0]data,output reg[7:0]strb
);
    reg has_aw=0,has_w=0,pending=0;reg[31:0]saved_address;reg[63:0]saved_data;reg[7:0]saved_strb;
    reg aw_stall=0,w_stall=0;reg[60:0]held_aw;reg[72:0]held_w;
    integer cycle=0,delay_count=0;
    assign awready=rst_n&&!has_aw&&!pending&&!bvalid&&(mode!=1||has_w)&&(cycle%11<7);
    assign wready=rst_n&&!has_w&&!pending&&!bvalid&&(mode!=2||has_aw)&&(cycle%13<8);
    always @(posedge clk)begin
        cycle<=cycle+1;
        if(!rst_n)begin has_aw<=0;has_w<=0;pending<=0;bvalid<=0;commit<=0;bid<=0;bresp<=0;aw_stall<=0;w_stall<=0;end
        else begin
            if(aw_stall&&(!awvalid||{awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot}!==held_aw))$fatal(1,"AW unstable under backpressure");
            if(w_stall&&(!wvalid||{wdata,wstrb,wlast}!==held_w))$fatal(1,"W unstable under backpressure");
            aw_stall<=awvalid&&!awready;held_aw<={awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot};
            w_stall<=wvalid&&!wready;held_w<={wdata,wstrb,wlast};commit<=0;
            if(awvalid&&awready)begin
                if($isunknown({awid,awaddr,awlen,awsize,awburst,awlock,awcache,awprot})||
                    awid!=0||awaddr[2:0]!=0||awlen!=0||awsize!=3||awburst!=1||awlock||awprot!=0||awcache!=3)$fatal(1,"Illegal single-beat AW");
                saved_address<=awaddr;has_aw<=1;
            end
            if(wvalid&&wready)begin
                if($isunknown({wdata,wstrb,wlast})||!wlast||wstrb==0)$fatal(1,"Illegal W payload");
                saved_data<=wdata;saved_strb<=wstrb;has_w<=1;
            end
            if(has_aw&&has_w&&!pending&&!bvalid)begin
                commit<=1;address<=saved_address;data<=saved_data;strb<=saved_strb;
                pending<=1;delay_count<=cycle%7+2;bid<=response_id;bresp<=response;
            end
            if(pending&&!bvalid&&!block_b)begin
                if(delay_count==0)bvalid<=1;else delay_count<=delay_count-1;
            end
            if(bvalid&&bready)begin bvalid<=0;pending<=0;has_aw<=0;has_w<=0;end
        end
    end
endmodule

module tb_lung_axi_write64;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,b_valid=0,q_ready=0,block_b=0;integer mode=0,tests=0,commits=0,prior;
    reg[31:0]addr=0;reg[63:0]payload=0;reg[7:0]mask=0;reg[1:0]response=0;reg[3:0]response_id=0;
    wire b_ready,q_valid,q_error,awvalid,awready,wvalid,wready,wlast,bvalid,bready;
    wire[31:0]awaddr,address;wire[63:0]wdata,data;wire[7:0]awlen,wstrb,strb;wire[3:0]awid,awcache,bid;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;wire awlock,commit;
    lung_axi_write64 dut(.clk(clk),.rst_n(rst_n),.b_valid(b_valid),.b_ready(b_ready),.b_addr(addr),.b_data(payload),.b_strb(mask),
        .q_valid(q_valid),.q_ready(q_ready),.q_error(q_error),.m_axi_awid(awid),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),
        .m_axi_awburst(awburst),.m_axi_awlock(awlock),.m_axi_awcache(awcache),.m_axi_awprot(awprot),.m_axi_awvalid(awvalid),.m_axi_awready(awready),
        .m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
        .m_axi_bid(bid),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready));
    lung_test_write_memory mem(.*);
    always @(posedge clk)if(rst_n&&commit)begin
        if(address!=={addr[28:0],3'b0}||data!==payload||strb!==mask)$fatal(1,"Write payload/address mismatch");commits=commits+1;
    end
    task reset_core;
        begin @(negedge clk);rst_n=0;b_valid=0;q_ready=0;repeat(3)@(negedge clk);rst_n=1;end
    endtask
    task launch(input reg[31:0]a,input reg[7:0]s);
        begin @(negedge clk);addr=a;mask=s;payload=64'h8070605040302010;b_valid=1;@(posedge clk);while(!b_ready)@(posedge clk);@(negedge clk);b_valid=0;end
    endtask
    task transaction(input reg[31:0]a,input reg[7:0]s,input integer ordering,input reg[1:0]r,input reg[3:0]id);
        reg bad;
        begin
            mode=ordering;response=r;response_id=id;prior=commits;bad=a>=32'h20000000||s==0||r[1]||id!=0;
            launch(a,s);while(!q_valid)@(negedge clk);
            repeat(4)begin @(negedge clk);if(!q_valid||q_error!==bad||awvalid||wvalid||bready||b_ready)$fatal(1,"Write ACK unstable/early-ready");end
            if(commits-prior!==((a>=32'h20000000||s==0)?0:1))$fatal(1,"Invalid write reached bus");
            q_ready=1;@(posedge clk);@(negedge clk);q_ready=0;if(!b_ready)$fatal(1,"Write did not retire");tests=tests+1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        transaction(511,8'hff,0,0,0);transaction(512,8'h81,1,0,0);transaction(32'h1fffffff,1,2,0,0);
        transaction(0,8'hff,0,2,0);transaction(1,8'hff,1,3,0);transaction(2,8'hff,2,0,7);
        transaction(32'h20000000,8'hff,0,0,0);transaction(32'hffffffff,8'hff,0,0,0);transaction(0,0,0,0,0);
        mode=1;launch(4,8'hff);while(!mem.has_w)@(negedge clk);reset_core();@(negedge clk);
        if(awvalid||wvalid||q_valid||bready||!b_ready)$fatal(1,"Shared reset retained partial write");
        transaction(7,8'hff,2,0,0);
        $display("Lung AXI write64 protocol RTL: PASS (%0d cases; independent AW/W, ACK stalls, WSTRB, 4K endpoints, physical bounds, BRESP/BID, shared reset)",tests);$finish;
    end
    initial begin #2000000;$fatal(1,"Write64 protocol timeout");end
endmodule
