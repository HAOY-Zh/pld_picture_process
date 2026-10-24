`timescale 1ns/1ps
module tb_roi_statistics;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] geometry_in=0,bounds_x_in=0,bounds_y_in=0,thresholds_in=0,modes_in=0;
    wire [31:0] status_out,count_out,identity_out;
    wire signed [63:0] hu_sum_out;wire [63:0] hu_sumsq_out;wire [31:0] hu_minmax_out;wire statistics_valid_out;
    wire [31:0] statistics_identity_out;
    reg [31:0] s_axis_tdata=0;reg [3:0] s_axis_tkeep=15;reg s_axis_tlast=0,s_axis_tvalid=0;
    wire s_axis_tready;wire [31:0] m_axis_tdata;wire [3:0] m_axis_tkeep;wire m_axis_tlast,m_axis_tvalid;
    reg m_axis_tready=0;reg [31:0] lfsr=32'hcb3571a9;
    pynq_z2_roi_statistics dut(.*);
    reg [31:0] inputs[0:810000];reg masks[0:810000],lasts[0:810000];
    integer observed=0,frames=0,cycles=0,fp;
    reg held=0;reg [31:0] held_data;reg held_last;
    always @(negedge clk)begin
        lfsr={lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
        m_axis_tready=lfsr[0]||lfsr[2];
    end
    always @(posedge clk)begin
        cycles=cycles+1;
        if(cycles>4000000)$fatal(1,"timeout");
        if(rst_n)begin
            if(held&&(!m_axis_tvalid||m_axis_tdata!==held_data||m_axis_tlast!==held_last))$fatal(1,"Output changed during backpressure");
            held=m_axis_tvalid&&!m_axis_tready;held_data=m_axis_tdata;held_last=m_axis_tlast;
            if(m_axis_tvalid&&m_axis_tready)begin
                if(m_axis_tdata!=={31'd0,masks[observed]}||m_axis_tlast!==lasts[observed]||m_axis_tkeep!==4'hf)$fatal(1,"mask/last mismatch at %0d",observed);
                observed=observed+1;
            end
        end
    end
    task run_case(input integer id,offset,n,input [31:0] geom,bx,by,thresholds,modes,
        input integer expected_count,input [63:0] expected_sum,expected_sumsq,input [31:0] expected_minmax,input integer bad_keep);
        integer i;
        begin
            @(negedge clk);geometry_in=geom;bounds_x_in=bx;bounds_y_in=by;thresholds_in=thresholds;modes_in=modes;
            for(i=0;i<n;i=i+1)begin
                if(lfsr[3:0]==0)begin s_axis_tvalid=0;@(negedge clk);end
                s_axis_tdata=inputs[offset+i];s_axis_tlast=(i==n-1);s_axis_tvalid=1;
                s_axis_tkeep=(bad_keep&&i==0)?4'h7:4'hf;
                @(posedge clk);while(!s_axis_tready)@(posedge clk);
                @(negedge clk);
            end
            s_axis_tvalid=0;s_axis_tlast=0;s_axis_tkeep=15;
            wait(status_out[1]);#1;
            if(count_out!==expected_count||hu_sum_out!==expected_sum||hu_sumsq_out!==expected_sumsq||hu_minmax_out!==expected_minmax||statistics_valid_out!==(bad_keep==0))
                $fatal(1,"statistics mismatch case=%0d count=%0d sum=%0d sumsq=%0d minmax=%h valid=%b",id,count_out,hu_sum_out,hu_sumsq_out,hu_minmax_out,statistics_valid_out);
            if(status_out!==(bad_keep?32'h0100000a:32'h01000002))$fatal(1,"status mismatch");
            if(statistics_identity_out!==32'h52535431)$fatal(1,"statistics ABI mismatch");
            $fdisplay(fp,"%0d,%0d,%0d,%0d,%h,%b",id,count_out,hu_sum_out,hu_sumsq_out,hu_minmax_out,statistics_valid_out);
            frames=frames+1;
            $display("ROI_STATS_CASE_PASS %0d",id);
        end
    endtask
    initial begin
        $readmemh("../../vectors/statistics_input.mem",inputs);
        $readmemh("../../vectors/statistics_mask.mem",masks);
        $readmemh("../../vectors/statistics_last.mem",lasts);
        fp=$fopen("../../results/fpga_statistics/actual_statistics.csv","w");
        $fdisplay(fp,"case_id,count,sum,sumsq,minmax_hex,valid");
        repeat(5)@(negedge clk);rst_n=1;
        `include "statistics_cases.vh"
        $fclose(fp);$display("ROI_STATISTICS_XSIM_ALL_PASS frames=%0d pixels=%0d cycles=%0d",frames,observed,cycles);$finish;
    end
endmodule
