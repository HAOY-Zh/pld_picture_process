`timescale 1ns/1ps
// Independent exhaustive signed ties-to-even oracle, with repeated commands.
module tb_lung_concat_lut;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0;wire c_ready,busy,done,error;
    reg[7:0]za,zb,zo;reg[30:0]ma,mb;reg[5:0]sa,sb;
    wire[31:0]written_words,read_words,cycles,rd_addr,wr_addr;
    wire[7:0]rd_beats,wr_beats;wire[63:0]wr_data;
    wire rd_valid,r_ready,wr_valid,stream_valid,stream_last,b_ready;
    lung_tensor_burst_engine dut(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready),
        .c_op(2'd2),.c_a(32'h1000),.c_b(32'h2000),.c_dst(32'h3000),
        .c_height(10'd1),.c_width(10'd1),.c_ca(7'd8),.c_cb(7'd8),
        .c_za(za),.c_zb(zb),.c_zo(zo),.c_ma(ma),.c_mb(mb),.c_sa(sa),.c_sb(sb),
        .busy(busy),.done(done),.error(error),.written_words(written_words),.read_words(read_words),.cycles(cycles),
        .rd_valid(rd_valid),.rd_ready(1'b1),.rd_addr(rd_addr),.rd_beats(rd_beats),
        .r_valid(1'b0),.r_ready(r_ready),.r_data(64'd0),.r_error(1'b0),.r_last(1'b0),
        .wr_valid(wr_valid),.wr_ready(1'b1),.wr_addr(wr_addr),.wr_data(wr_data),.wr_beats(wr_beats),
        .stream_valid(stream_valid),.stream_ready(1'b1),.stream_last(stream_last),
        .b_valid(1'b0),.b_ready(b_ready),.b_error(1'b0));
    function[7:0]gold(input integer code,z,m,s,outzero);
        reg signed[63:0]p,q,r,den,v;
        begin
            p=(64'(code)-64'(z))*64'(m);den=64'd1<<s;
            q=p/den;r=p%den;
            if(r<0)begin q=q-1;r=r+den;end
            if(2*r>den||(2*r==den&&q[0]))q=q+1;
            v=q+outzero;gold=v<0?0:(v>255?255:v[7:0]);
        end
    endfunction
    integer test_id,index;reg[7:0]expected;
    task check_table;
        begin
            @(negedge clk);c_valid=1;
            while(!c_ready)@(negedge clk);
            @(negedge clk);c_valid=0;
            wait(dut.table_loading);wait(!dut.table_loading);@(negedge clk);
            if(error)$fatal(1,"Unexpected LUT initialization error");
            for(index=0;index<512;index=index+1)begin
                expected=gold(index%256,index<256?za:zb,index<256?ma:mb,index<256?sa:sb,zo);
                if(dut.lookup_lanes[0].codes[index]!==expected||dut.lookup_lanes[7].codes[index]!==expected)
                    $fatal(1,"LUT oracle mismatch test=%0d index=%0d expected=%0d",test_id,index,expected);
            end
            // Reset while a read request may be pending: tables must rebuild.
            rst_n=0;repeat(3)@(negedge clk);rst_n=1;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        for(test_id=0;test_id<8;test_id=test_id+1)begin
            case(test_id)
            0:begin za=128;zb=127;zo=128;ma=1;mb=3;sa=1;sb=1;end
            1:begin za=255;zb=0;zo=0;ma=2147483647;mb=2147483647;sa=0;sb=0;end
            2:begin za=0;zb=255;zo=255;ma=2147483647;mb=2147483647;sa=62;sb=62;end
            3:begin za=131;zb=45;zo=82;ma=123456789;mb=987654321;sa=27;sb=30;end
            4:begin za=0;zb=255;zo=128;ma=3;mb=5;sa=2;sb=3;end
            5:begin za=255;zb=0;zo=1;ma=1;mb=1;sa=0;sb=0;end
            6:begin za=33;zb=211;zo=79;ma=1073741824;mb=2147483646;sa=31;sb=32;end
            7:begin za=128;zb=128;zo=128;ma=2147483647;mb=1073741825;sa=33;sb=34;end
            endcase
            check_table();
        end
        $display("Lung exhaustive concat LUT RTL: PASS (4096 codes, signed RNE ties, clamps, shifts 0/62, reset/rebuild)");$finish;
    end
    initial begin #2000000;$fatal(1,"LUT timeout");end
endmodule
