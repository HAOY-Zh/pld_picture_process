`timescale 1ns/1ps
// Independent 1x1 coordinate/data oracle with synchronous request responses.
// Existing stepped-window regression separately covers 3x3, padding, 512 axes.
module tb_lung_window_flow;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0;integer cycle=0,phase=0;
    wire[1:0]c_ready,req_valid,rsp_ready,m_valid,m_first,m_last,m_final,m_error,done,error,cfg_error;
    wire[31:0]req_addr[0:1];wire[63:0]data[0:1];wire[7:0]keep[0:1];wire[8:0]x[0:1],y[0:1];
    reg[1:0]rsp_valid=0;reg[63:0]rsp_data[0:1];reg[1:0]rsp_error=0;
    wire ready=phase==0||cycle%11<6;
    integer requests[0:1],outputs[0:1],completed[0:1];
    reg[63:0]held[0:1];reg[1:0]stalled=0;
    function[63:0]value(input integer address);value=64'h1234000000000000|address;endfunction
    lung_feature_window_baseline old_window(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready[0]),
        .c_base(32'd4096),.c_height(10'd1),.c_width(10'd64),.c_channels(7'd8),.c_kernel3(1'b0),.c_padding(1'b0),.c_input_zero(8'd128),
        .req_valid(req_valid[0]),.req_ready(!rsp_valid[0]),.req_addr(req_addr[0]),
        .rsp_valid(rsp_valid[0]),.rsp_ready(rsp_ready[0]),.rsp_data(rsp_data[0]),.rsp_error(rsp_error[0]),
        .m_valid(m_valid[0]),.m_ready(ready),.m_data(data[0]),.m_keep(keep[0]),.m_first(m_first[0]),.m_last(m_last[0]),
        .m_frame_last(m_final[0]),.m_x(x[0]),.m_y(y[0]),.m_error(m_error[0]),.done(done[0]),.error(error[0]),.config_error(cfg_error[0]));
    lung_feature_window_flow new_window(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready[1]),
        .c_base(32'd4096),.c_height(10'd1),.c_width(10'd64),.c_channels(7'd8),.c_kernel3(1'b0),.c_padding(1'b0),.c_input_zero(8'd128),
        .req_valid(req_valid[1]),.req_ready(!rsp_valid[1]),.req_addr(req_addr[1]),
        .rsp_valid(rsp_valid[1]),.rsp_ready(rsp_ready[1]),.rsp_data(rsp_data[1]),.rsp_error(rsp_error[1]),
        .m_valid(m_valid[1]),.m_ready(ready),.m_data(data[1]),.m_keep(keep[1]),.m_first(m_first[1]),.m_last(m_last[1]),
        .m_frame_last(m_final[1]),.m_x(x[1]),.m_y(y[1]),.m_error(m_error[1]),.done(done[1]),.error(error[1]),.config_error(cfg_error[1]));
    always @(negedge clk)cycle=cycle+1;
    integer j;
    always @(posedge clk)begin
        for(j=0;j<2;j=j+1)begin
            if(!rst_n)begin rsp_valid[j]<=0;stalled[j]<=0;end
            else begin
                if(stalled[j]&&(!m_valid[j]||data[j]!==held[j]))$fatal(1,"Bypass stalled data changed");
                stalled[j]<=m_valid[j]&&!ready;held[j]<=data[j];
                if(rsp_valid[j]&&rsp_ready[j])rsp_valid[j]<=0;
                if(req_valid[j]&&!rsp_valid[j])begin
                    if(req_addr[j]!==4096+requests[j])$fatal(1,"Bypass request duplicated/skipped");
                    rsp_valid[j]<=1;rsp_data[j]<=value(req_addr[j]);
                    rsp_error[j]<=phase==2&&requests[j]==17;requests[j]=requests[j]+1;
                end
                if(m_valid[j]&&ready)begin
                    if(data[j]!==((phase==2&&outputs[j]==17)?64'h8080808080808080:value(4096+outputs[j]))||
                       m_error[j]!==(phase==2&&outputs[j]==17)||keep[j]!=255||x[j]!=outputs[j]||y[j]!=0||
                       !m_first[j]||!m_last[j]||m_final[j]!==(outputs[j]==63))$fatal(1,"Bypass output oracle mismatch");
                    outputs[j]=outputs[j]+1;
                end
                if(done[j])begin
                    if(completed[j]!=0||requests[j]!=64||outputs[j]!=64||rsp_valid[j]||cfg_error[j]||error[j]!==(phase==2))$fatal(1,"Bypass incomplete/duplicate DONE");
                    completed[j]=cycle;
                end
            end
        end
    end
    initial begin
        for(phase=0;phase<3;phase=phase+1)begin
            @(negedge clk);rst_n=0;c_valid=0;
            for(integer n=0;n<2;n=n+1)begin requests[n]=0;outputs[n]=0;completed[n]=0;end
            repeat(3)@(negedge clk);rst_n=1;
            @(negedge clk);if(c_ready!=3)$fatal(1,"Bypass not idle");c_valid=1;
            @(negedge clk);c_valid=0;
            while(completed[0]==0||completed[1]==0)@(negedge clk);
            if(phase==0)begin
                if(completed[1]>=completed[0])$fatal(1,"No response bypass throughput improvement");
                $display("FLOW_WINDOW_CYCLES baseline_done=%0d flow_done=%0d saved=%0d",completed[0],completed[1],completed[0]-completed[1]);
            end
            repeat(4)@(negedge clk);
        end
        $display("Lung response bypass RTL: PASS (192 exact outputs per variant; backpressure, response error, sequential commands and cycle gain)");$finish;
    end
    initial begin #200000;$fatal(1,"Response bypass timeout");end
endmodule
