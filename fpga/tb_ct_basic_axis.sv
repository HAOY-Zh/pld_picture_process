`timescale 1ns/1ps
module tb_ct_basic_axis;
    parameter W=16,H=16;
    localparam N=W*H;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0;
    reg [2:0] mode=7;
    reg [31:0] wp=0,tp=0,s_data=0;
    reg s_last=0,s_valid=0,m_ready=0;
    wire s_ready,m_valid,m_last;
    wire [31:0] m_data;
    wire [3:0] m_keep;
    reg [31:0] inputs[0:N-1],expected[0:N-1];
    reg held=0,held_last;
    reg [31:0] held_data;
    integer frames,f,frame,op,i,rc,received=0,cycles=0,checks=0;
    ct_basic_axis #(.IMAGE_WIDTH(W),.IMAGE_HEIGHT(H)) dut (
        .clk(clk),.rst_n(rst_n),.mode(mode),.window_params(wp),.thresholds(tp),
        .s_axis_tdata(s_data),.s_axis_tkeep(4'hf),.s_axis_tlast(s_last),
        .s_axis_tvalid(s_valid),.s_axis_tready(s_ready),
        .m_axis_tdata(m_data),.m_axis_tkeep(m_keep),.m_axis_tlast(m_last),
        .m_axis_tvalid(m_valid),.m_axis_tready(m_ready));
    always @(negedge clk) begin
        cycles=cycles+1;
        m_ready=rst_n && (cycles%19>=5);
    end
    always @(posedge clk) begin
        if (rst_n) begin
            if (held && (!m_valid || m_data!==held_data || m_last!==held_last || m_keep!==4'hf))
                $fatal(1,"Output changed while stalled");
            held=m_valid&&!m_ready;held_data=m_data;held_last=m_last;
            if(m_valid&&m_ready) begin
                if(received>=N || m_data!==expected[received])
                    $fatal(1,"frame=%0d mode=%0d pixel=%0d actual=%h expected=%h",frame,mode,received,m_data,expected[received]);
                if(m_last!==(received==N-1)||m_keep!==4'hf) $fatal(1,"AXIS sideband mismatch");
                received=received+1;checks=checks+1;
            end
        end else held=0;
    end
    initial begin
        f=$fopen("vectors.txt","r");if(!f)$fatal(1,"Missing vectors");
        rc=$fscanf(f,"%d\n",frames);
        repeat(4)@(negedge clk);rst_n=1;
        for(frame=0;frame<frames;frame=frame+1) begin
            @(negedge clk);rc=$fscanf(f,"%d %h %h\n",op,wp,tp);mode=op;
            for(i=0;i<N;i=i+1) rc=$fscanf(f,"%h %h\n",inputs[i],expected[i]);
            received=0;repeat(3)@(negedge clk);
            for(i=0;i<N;i=i+1) begin
                @(negedge clk);s_data=inputs[i];s_last=(i==N-1);s_valid=1;
                @(posedge clk);while(!s_ready)@(posedge clk);
                @(negedge clk);s_valid=0;s_last=0;
                if(i%31==0)repeat(3)@(negedge clk);
            end
            wait(received==N);repeat(4)@(negedge clk);
        end
        $display("CT BASIC RTL PASS frames=%0d pixels=%0d cycles=%0d",frames,checks,cycles);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"Timeout");end
endmodule
