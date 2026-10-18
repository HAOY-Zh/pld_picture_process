`timescale 1ns/1ps
module tb_lung_axi_burst_write64;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,descriptor=0,s_valid=0,s_last=0,q_ready=0,block_b=0;
    reg[31:0]base=0;reg[7:0]beats=0;reg[63:0]s_data=0;
    wire descriptor_ready,s_ready,q_valid,q_error;
    wire[3:0]awid,awcache,bid;wire[31:0]awaddr,address;wire[7:0]awlen,wstrb,strb;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;wire awlock,awvalid,awready,wvalid,wready,wlast,bvalid,bready,commit;
    wire[63:0]wdata,data;
    reg[1:0]response=0;reg[3:0]response_id=0;integer mode=0,checks=0,writes=0,retired=0;
    lung_axi_burst_write64 dut(.clk(clk),.rst_n(rst_n),.b_valid(descriptor),.b_ready(descriptor_ready),.b_addr(base),.b_beats(beats),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_last(s_last),.q_valid(q_valid),.q_ready(q_ready),.q_error(q_error),
        .m_axi_awid(awid),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),
        .m_axi_awcache(awcache),.m_axi_awprot(awprot),.m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_wdata(wdata),
        .m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
        .m_axi_bid(bid),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready));
    lung_test_burst_memory mem(.*);
    function[63:0]payload(input integer i);payload=64'hbbcc001122330000+i;endfunction
    always @(posedge clk)if(rst_n)begin
        if(commit)begin
            if(address!==(base+retired)*8||data!==payload(retired)||strb!==255)$fatal(1,"Burst word/address mismatch");
            retired=retired+1;writes=writes+1;
        end
    end
    task transaction(input reg[31:0]a,input reg[7:0]n,input integer order,input reg[1:0]resp,input reg[3:0]id,input reg bad_last);
        integer i,prior;reg invalid,bad;
        begin
            mode=order;response=resp;response_id=id;prior=writes;retired=0;
            invalid=n==0||n>16||a>=32'h20000000||((a%512)+n)>512;bad=invalid||resp[1]||id!=0||bad_last;
            @(negedge clk);base=a;beats=n;descriptor=1;
            @(posedge clk);while(!descriptor_ready)@(posedge clk);@(negedge clk);descriptor=0;
            if(!invalid)begin
                block_b=1;
                for(i=0;i<n;i=i+1)begin
                    repeat(i%3)@(negedge clk);
                    s_valid=1;s_data=payload(i);s_last=(i==n-1)^bad_last;
                    @(posedge clk);while(!s_ready)@(posedge clk);@(negedge clk);s_valid=0;
                end
                while(retired!=n)@(negedge clk);
                repeat(4)begin @(negedge clk);if(q_valid||descriptor_ready)$fatal(1,"Burst ACK before final B");end
                block_b=0;
            end
            while(!q_valid)@(negedge clk);
            repeat(3)begin @(negedge clk);if(!q_valid||q_error!==bad||descriptor_ready)$fatal(1,"Burst ACK did not hold/error mismatch");end
            if(writes-prior!=(invalid?0:n))$fatal(1,"Invalid burst issued or incomplete words");
            q_ready=1;@(posedge clk);@(negedge clk);q_ready=0;if(!descriptor_ready)$fatal(1,"Burst did not retire");checks=checks+1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        transaction(0,16,0,0,0,0);transaction(496,16,1,0,0,0);transaction(510,2,2,0,0,0);
        transaction(511,1,0,0,0,0);transaction(32'h1ffffff0,16,1,0,0,0);
        transaction(512,15,2,2,0,0);transaction(37,3,0,0,7,0);transaction(100,2,1,0,0,1);
        transaction(0,0,0,0,0,0);transaction(0,17,0,0,0,0);transaction(511,2,0,0,0,0);
        transaction(32'h20000000,1,0,0,0,0);
        @(negedge clk);base=0;beats=16;descriptor=1;@(posedge clk);while(!descriptor_ready)@(posedge clk);
        @(negedge clk);descriptor=0;s_valid=1;s_last=0;s_data=payload(0);@(posedge clk);while(!s_ready)@(posedge clk);
        @(negedge clk);rst_n=0;s_valid=0;repeat(3)@(negedge clk);rst_n=1;@(negedge clk);
        if(awvalid||wvalid||bready||q_valid||!descriptor_ready)$fatal(1,"Burst shared reset retained partial transaction");
        transaction(800,16,2,0,0,0);
        $display("Lung AXI burst write64 RTL: PASS (%0d cases; AW/W independence, W stalls, ACK stalls, 4K, physical limit, BRESP/BID, bad source LAST, partial shared reset)",checks);$finish;
    end
    initial begin #2000000;$fatal(1,"Burst writer protocol timeout");end
endmodule
