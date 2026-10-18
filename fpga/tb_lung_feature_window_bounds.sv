`timescale 1ns/1ps
// Dimension/address edge tests without a huge initialized frame RAM.
module tb_lung_feature_window_bounds;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0;reg[31:0]base=0;reg[9:0]height=1,width=1;
    reg[6:0]channels=1;wire c_ready,req_valid,rsp_ready,m_valid,m_first,m_last,m_final,m_error,done,error,cfg_error;
    wire[31:0]req_addr;wire[63:0]m_data;wire[7:0]m_keep;wire[8:0]m_x,m_y;
    reg rsp_valid=0;reg pending=0;integer cycle=0,wait_cycles=0;
    wire req_ready=rst_n&&!pending&&!rsp_valid&&cycle%7<5;
    wire m_ready=cycle%13<9;
    integer requests=0,outputs=0,groups=0,blocks=0,target=0,expected_x,expected_y,expected_cb;
    reg checking=0;reg[31:0]test_base=0;
    lung_feature_window dut(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready),
        .c_base(base),.c_height(height),.c_width(width),.c_channels(channels),.c_kernel3(1'b0),.c_padding(1'b0),.c_input_zero(8'd128),
        .req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(64'h0102030405060708),.rsp_error(1'b0),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_keep(m_keep),.m_first(m_first),.m_last(m_last),
        .m_frame_last(m_final),.m_x(m_x),.m_y(m_y),.m_error(m_error),.done(done),.error(error),.config_error(cfg_error));
    always @(negedge clk)cycle=cycle+1;
    always @(posedge clk)begin
        if(!rst_n)begin pending<=0;rsp_valid<=0;wait_cycles<=0;end
        else begin
            if(rsp_valid&&rsp_ready)rsp_valid<=0;
            if(req_valid&&req_ready)begin pending<=1;wait_cycles<=2;end
            if(pending)begin
                if(wait_cycles!=0)wait_cycles<=wait_cycles-1;
                else begin pending<=0;rsp_valid<=1;end
            end
        end
    end
    always @(posedge clk)if(rst_n&&checking)begin
        if(req_valid&&req_ready)begin
            if(req_addr!==(test_base+requests))$fatal(1,"Bounds word address %0d: %h/%h",requests,req_addr,test_base+requests);
            requests=requests+1;
        end
        if(m_valid&&m_ready)begin
            expected_cb=outputs%blocks;expected_x=(outputs/blocks)%width;expected_y=(outputs/blocks)/width;
            if(m_error||m_data!==64'h0102030405060708||m_x!==expected_x[8:0]||m_y!==expected_y[8:0]||
                m_first!==(expected_cb==0)||m_last!==(expected_cb==blocks-1)||
                m_keep!==(channels==1?8'h01:8'hff)||m_final!==(outputs==height*width*blocks-1))$fatal(1,"Bounds window/tag failed at %0d",outputs);
            outputs=outputs+1;
        end
    end
    task reset_core;
        begin @(negedge clk);rst_n=0;c_valid=0;repeat(3)@(negedge clk);rst_n=1;end
    endtask
    task start;
        begin @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);@(negedge clk);c_valid=0;end
    endtask
    task run(input reg[31:0]b,input reg[9:0]h,input reg[9:0]w,input reg[6:0]c,input integer count,input reg complete);
        begin
            @(negedge clk);base=b;test_base=b;height=h;width=w;channels=c;blocks=(c+7)/8;requests=0;outputs=0;target=count;checking=1;
            start();while(outputs<target)@(negedge clk);
            if(complete)begin while(!done)@(negedge clk);if(error||requests!=target)$fatal(1,"Bounds frame did not finish cleanly");end
            checking=0;groups=groups+1;reset_core();
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        run(0,512,1,96,6144,1);run(37,1,512,96,6144,1);
        run(32'hffffffff,1,1,1,1,1);
        // Largest configured frame occupies exactly the remaining word range.
        // Exercise row boundary plus next row prefix; do not claim all 3M words.
        run(32'hffd00000,512,512,96,6146,0);
        base=32'hffd00001;height=512;width=512;channels=96;start();
        while(!done)begin @(negedge clk);if(req_valid||m_valid)$fatal(1,"Overflowing frame issued memory read");end
        if(!error||!cfg_error)$fatal(1,"Address end overflow not rejected");
        $display("Lung feature window bounds RTL: PASS (4 configurations; H512/W512/C96, row transition, max word address, address overflow, shared reset)");$finish;
    end
    initial begin #100000000;$fatal(1,"Bounds timeout");end
endmodule
