`timescale 1ns/1ps
// Direct cache interface oracle: real synchronous RAM fills, no fake hit data.
module tb_lung_cache_return;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,cfg_valid=0,s_valid=0,m_ready=0,q_valid=0,q_error=0,q_last=0;
    reg[31:0]s_addr=0;reg[63:0]q_data=0;
    wire cfg_ready,s_ready,m_valid,m_error,b_valid,q_ready;
    wire[63:0]m_data;wire[31:0]b_addr,hits,misses;wire[7:0]b_beats;
    integer checks=0;
    lung_feature_cache #(.SETS(2)) dut(.clk(clk),.rst_n(rst_n),.cfg_valid(cfg_valid),.cfg_ready(cfg_ready),
        .cfg_base(32'd0),.cfg_words(24'd32),.s_valid(s_valid),.s_ready(s_ready),.s_addr(s_addr),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_error(m_error),
        .b_valid(b_valid),.b_ready(1'b1),.b_addr(b_addr),.b_beats(b_beats),
        .q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),.hits(hits),.misses(misses));
    function[63:0]value(input integer a);value=64'h1234567800000000|a;endfunction
    task request(input integer a);
        begin @(negedge clk);s_addr=a;s_valid=1;@(posedge clk);while(!s_ready)@(posedge clk);@(negedge clk);s_valid=0;s_addr=31;end
    endtask
    task configure;
        begin @(negedge clk);cfg_valid=1;@(posedge clk);while(!cfg_ready)@(posedge clk);@(negedge clk);cfg_valid=0;end
    endtask
    task hit_check(input integer a,input integer stalls);
        begin
            m_ready=0;request(a);
            // Immediately after request's sampling edge, CHECK must be valid.
            if(!m_valid||m_data!==value(a)||m_error||b_valid)$fatal(1,"Missing one-cycle synchronous hit response");
            repeat(stalls)begin @(negedge clk);if(!m_valid||m_data!==value(a)||m_error||s_ready||cfg_ready)$fatal(1,"Stalled direct response changed");end
            m_ready=1;@(posedge clk);@(negedge clk);m_ready=0;
            if(m_valid||!s_ready)$fatal(1,"Hit duplicated or not retired");checks=checks+1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;configure();request(7);
        while(!b_valid)@(negedge clk);
        if(b_addr!=0||b_beats!=16)$fatal(1,"Unexpected miss fill");
        @(negedge clk);
        for(integer j=0;j<16;j=j+1)begin
            q_valid=1;q_data=value(j);q_last=j==15;
            @(posedge clk);while(!q_ready)@(posedge clk);@(negedge clk);
        end
        q_valid=0;q_last=0;
        if(!m_valid||m_data!==value(7)||m_error)$fatal(1,"Miss return differs");
        m_ready=1;@(posedge clk);@(negedge clk);m_ready=0;
        for(integer j=0;j<16;j=j+1)hit_check(j,j%5);
        if(hits!=16||misses!=1)$fatal(1,"Hit counters counted stalls/duplicates");
        // Reset while a direct hit is stalled, then flush proves no stale valid.
        request(2);if(!m_valid)$fatal(1,"Expected direct hit before reset");
        rst_n=0;repeat(3)@(negedge clk);if(m_valid||b_valid||s_ready)$fatal(1,"Reset response leakage");
        rst_n=1;configure();if(hits||misses)$fatal(1,"Reconfigure counters");
        $display("Lung direct cache return RTL: PASS (16 hits, ready and stalled handshakes, synchronous fill, captured address, exact counters, reset/flush)");$finish;
    end
    initial begin #100000;$fatal(1,"Direct cache return timeout");end
endmodule
