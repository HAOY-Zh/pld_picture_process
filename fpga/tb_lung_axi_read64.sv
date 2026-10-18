`timescale 1ns/1ps
module tb_lung_axi_read64;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,b_valid=0,q_ready=0,arready=0,rvalid=0,rlast=0;
    reg[31:0]addr=0;reg[7:0]beats=0;reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;
    wire b_ready,q_valid,q_error,q_last,arvalid,rready;
    wire[63:0]q_data;wire[3:0]arid,arcache;wire[31:0]araddr;wire[7:0]arlen;wire[2:0]arsize,arprot;wire[1:0]arburst;wire arlock;
    integer i,tests=0;reg[63:0]held;
    lung_axi_read64 dut(.clk(clk),.rst_n(rst_n),.b_valid(b_valid),.b_ready(b_ready),.b_addr(addr),.b_beats(beats),
        .q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),.m_axi_arcache(arcache),.m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready));
    task reset_core;
        begin @(negedge clk);rst_n=0;b_valid=0;rvalid=0;q_ready=0;arready=0;repeat(3)@(negedge clk);rst_n=1;end
    endtask
    task launch(input reg[31:0]a,input reg[7:0]n);
        begin @(negedge clk);b_valid=1;addr=a;beats=n;@(posedge clk);while(!b_ready)@(posedge clk);@(negedge clk);b_valid=0;end
    endtask
    task bad_burst(input reg[31:0]a,input reg[7:0]n);
        begin
            launch(a,n);while(!q_valid)@(negedge clk);
            repeat(4)begin @(negedge clk);if(arvalid||!q_valid||!q_error||!q_last||q_data!==0)$fatal(1,"Invalid AXI request not rejected");end
            q_ready=1;@(posedge clk);@(negedge clk);q_ready=0;if(!b_ready)$fatal(1,"Error transaction did not finish");tests=tests+1;
        end
    endtask
    task good_burst(input integer mode);
        integer count;reg bad,last;
        begin
            launch(509,3);while(!arvalid)@(negedge clk);
            repeat(4)begin @(negedge clk);if(!arvalid||araddr!==32'd4072||arlen!==2||arsize!==3||arburst!==1||arid!==0)$fatal(1,"AR byte conversion/stall mismatch");end
            arready=1;@(posedge clk);@(negedge clk);arready=0;
            count=mode==3?2:mode==4?4:3;
            for(i=0;i<count;i=i+1)begin
                @(negedge clk);rvalid=1;rdata=64'd100+i;rid=(mode==2&&i==1)?1:0;rresp=(mode==1&&i==1)?2'b10:0;rlast=i==count-1;
                bad=(mode==1&&i==1)||(mode==2&&i==1)||(mode==3&&i==1)||(mode==4&&i==2);
                last=i==count-1||i==2;
                if(i<3)begin
                    repeat(3)begin @(negedge clk);if(!q_valid||rready||q_data!==64'd100+i||q_error!==bad||q_last!==last)$fatal(1,"R beat/hold mismatch mode %0d beat %0d",mode,i);end
                    q_ready=1;@(posedge clk);@(negedge clk);q_ready=0;
                end else begin
                    if(q_valid||b_ready||!rready)$fatal(1,"Late LAST not drained in isolation");@(posedge clk);@(negedge clk);
                end
                rvalid=0;
            end
            repeat(2)@(negedge clk);if(!b_ready||q_valid||arvalid)$fatal(1,"AXI burst left a live transaction");tests=tests+1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        good_burst(0);good_burst(1);good_burst(2);good_burst(3);good_burst(4);
        bad_burst(511,2);bad_burst(32'h20000000,1);bad_burst(0,0);bad_burst(0,17);
        launch(0,4);while(!arvalid)@(negedge clk);reset_core();@(negedge clk);if(arvalid||q_valid||!b_ready)$fatal(1,"Reset kept stalled AR");
        good_burst(0);
        $display("Lung AXI read64 protocol RTL: PASS (%0d cases; byte conversion, AR/R stalls, 4K, invalid length/address, RRESP/RID, early/late LAST, drain, shared reset)",tests);$finish;
    end
    initial begin #1000000;$fatal(1,"AXI read protocol timeout");end
endmodule
