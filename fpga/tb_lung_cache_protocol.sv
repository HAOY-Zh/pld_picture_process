`timescale 1ns/1ps
// Small geometry forces FIFO conflicts; bridge BFM injects bad AXI responses.
module tb_lung_cache_protocol;
    parameter CAPACITY_TEST=0;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,cfg_valid=0,s_valid=0,m_ready=0;reg[31:0]base=0,address=0;reg[23:0]words=0;
    wire cfg_ready,s_ready,m_valid,m_error,b_valid,b_ready,q_valid,q_ready,q_error,q_last;
    wire[63:0]m_data,q_data;wire[31:0]b_addr,hits,misses;wire[7:0]b_beats;
    wire[3:0]arid,arcache;wire[31:0]araddr;wire[7:0]arlen;wire[2:0]arsize,arprot;
    wire[1:0]arburst;wire arlock,arvalid,arready,rready;
    reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;reg rlast=0,rvalid=0,active=0;
    integer cycle=0,latency=0,index=0,total=0,mode=0,active_mode=0,version=0,active_version=0,ar_count=0,checks=0;
    reg[31:0]read_addr;reg[63:0]held;
    function automatic[63:0]value(input reg[31:0]a,input integer v);
        value=64'hc001000000000000|((64'(v))<<32)|a;
    endfunction
    lung_feature_cache #(.SETS(CAPACITY_TEST?256:2))cache(.clk(clk),.rst_n(rst_n),.cfg_valid(cfg_valid),.cfg_ready(cfg_ready),.cfg_base(base),.cfg_words(words),
        .s_valid(s_valid),.s_ready(s_ready),.s_addr(address),.m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_error(m_error),
        .b_valid(b_valid),.b_ready(b_ready),.b_addr(b_addr),.b_beats(b_beats),.q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),.hits(hits),.misses(misses));
    lung_axi_read64 reader(.clk(clk),.rst_n(rst_n),.b_valid(b_valid),.b_ready(b_ready),.b_addr(b_addr),.b_beats(b_beats),
        .q_valid(q_valid),.q_ready(q_ready),.q_data(q_data),.q_error(q_error),.q_last(q_last),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),.m_axi_arcache(arcache),.m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready));
    assign arready=rst_n&&!active&&!rvalid&&cycle%7<5;
    always @(negedge clk)cycle=cycle+1;
    always @(posedge clk)begin
        if(!rst_n)begin active<=0;rvalid<=0;latency<=0;end
        else begin
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||arlen>15||araddr[2:0]!=0||({1'b0,araddr[11:0]}+(arlen+1)*8)>4096)$fatal(1,"Protocol BFM received illegal AXI AR");
                ar_count=ar_count+1;active<=1;read_addr<=araddr>>3;index<=0;latency<=2;active_mode<=mode;active_version<=version;
                total<=mode==4?2:mode==5?arlen+2:arlen+1;
            end
            if(active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin rvalid<=1;rdata<=value(read_addr,active_version);rresp<=active_mode==2?2'b10:0;rid<=active_mode==3?1:0;rlast<=index==total-1;end
            end
            if(rvalid&&rready)begin
                rvalid<=0;
                if(rlast)active<=0;
                else begin index<=index+1;read_addr<=read_addr+1'b1;latency<=cycle%3;end
            end
        end
    end
    task reset_core;
        begin @(negedge clk);rst_n=0;cfg_valid=0;s_valid=0;m_ready=0;repeat(3)@(negedge clk);if(m_valid||arvalid)$fatal(1,"Cache protocol reset failed");rst_n=1;end
    endtask
    task configure(input reg[31:0]b,input reg[23:0]n);
        begin @(negedge clk);cfg_valid=1;base=b;words=n;@(posedge clk);while(!cfg_ready)@(posedge clk);@(negedge clk);cfg_valid=0;end
    endtask
    task read_check(input reg[31:0]a,input reg bad,input integer expected_ar);
        integer prior_ar_count;
        begin
            prior_ar_count=ar_count;@(negedge clk);s_valid=1;address=a;@(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
            if(m_error!==bad||m_data!==(bad?64'd0:value(a,version)))$fatal(1,"Cache protocol read %h bad=%b got=%h error=%b",a,bad,m_data,m_error);
            held=m_data;repeat(4)begin @(negedge clk);if(!m_valid||m_data!==held||m_error!==bad)$fatal(1,"Cache response changed under stall");end
            if(ar_count-prior_ar_count!=expected_ar)$fatal(1,"Wrong miss count for %h: %0d/%0d",a,ar_count-prior_ar_count,expected_ar);
            m_ready=1;@(posedge clk);@(negedge clk);m_ready=0;checks=checks+1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        if(CAPACITY_TEST)begin
            // Actual default geometry: retain 3x256xceil(96/8)=9216 words.
            // No configuration/flush between passes; every second-pass read hits.
            configure(0,16384);
            for(integer a=0;a<9216;a=a+1)read_check(a,0,a%16==0?1:0);
            for(integer a=0;a<9216;a=a+1)read_check(a,0,0);
            read_check(12288,0,1);read_check(4096,0,0);read_check(8192,0,0);read_check(0,0,1);read_check(4096,0,1);
            $display("Lung feature cache capacity RTL: PASS (%0d reads; default 96KiB, 9216-word working set retained, second pass all hits, same-set fourth-tag eviction)",checks);$finish;
        end
        configure(3,126);
        read_check(3,0,1);read_check(15,0,0);read_check(32,0,1);read_check(64,0,1);
        read_check(3,0,0);read_check(96,0,1);read_check(32,0,0);read_check(3,0,1);read_check(32,0,1);
        read_check(128,0,1);read_check(2,1,0);read_check(129,1,0);
        version=1;configure(3,126);read_check(3,0,1);read_check(3,0,0);
        // Failed fills never become valid hits; retry with good memory must miss.
        configure(0,32);mode=2;read_check(10,1,1);mode=0;read_check(10,0,1);read_check(10,0,0);
        configure(0,32);mode=3;read_check(10,1,1);mode=0;read_check(10,0,1);
        configure(0,32);mode=4;read_check(10,1,1);mode=0;read_check(10,0,1);
        configure(0,32);mode=5;read_check(10,1,1);mode=0;read_check(10,0,1);
        configure(505,20);read_check(511,0,1);read_check(512,0,1);read_check(524,0,0);
        configure(32'h20000000,16);read_check(32'h20000000,1,0);
        configure(0,32);@(negedge clk);s_valid=1;address=10;@(posedge clk);while(!s_ready)@(posedge clk);
        @(negedge clk);s_valid=0;while(!active)@(negedge clk);reset_core();configure(0,32);read_check(10,0,1);
        $display("Lung feature cache protocol RTL: PASS (%0d reads; FIFO conflict, clipped span, frame flush, failed-fill retry, RRESP/RID, early/late RLAST, 4K, address limit, shared reset)",checks);$finish;
    end
    initial begin #10000000;$fatal(1,"Cache protocol timeout");end
endmodule

module tb_lung_cache_capacity;
    tb_lung_cache_protocol #(.CAPACITY_TEST(1)) test();
endmodule
