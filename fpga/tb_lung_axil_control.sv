`timescale 1ns/1ps
// Simulation-only AXI-Lite host; AW-first/W-first/both, held B/R responses.
module lung_test_axil_master(
    input wire clk,input wire rst_n,
    output reg[9:0]s_axi_awaddr,output wire[2:0]s_axi_awprot,output reg s_axi_awvalid,input wire s_axi_awready,
    output reg[31:0]s_axi_wdata,output reg[3:0]s_axi_wstrb,output reg s_axi_wvalid,input wire s_axi_wready,
    input wire[1:0]s_axi_bresp,input wire s_axi_bvalid,output reg s_axi_bready,
    output reg[9:0]s_axi_araddr,output wire[2:0]s_axi_arprot,output reg s_axi_arvalid,input wire s_axi_arready,
    input wire[31:0]s_axi_rdata,input wire[1:0]s_axi_rresp,input wire s_axi_rvalid,output reg s_axi_rready
);
    integer writes=0,reads=0;assign s_axi_awprot=0;assign s_axi_arprot=0;
    initial begin s_axi_awaddr=0;s_axi_awvalid=0;s_axi_wdata=0;s_axi_wstrb=0;s_axi_wvalid=0;
        s_axi_bready=0;s_axi_araddr=0;s_axi_arvalid=0;s_axi_rready=0;end
    task write_reg(input reg[9:0]address,input reg[31:0]value,input reg[3:0]strobes,input integer ordering,input reg[1:0]response);
        begin
            fork
                begin
                    if(ordering==2)repeat(4)@(negedge clk);
                    @(negedge clk);s_axi_awaddr=address;s_axi_awvalid=1;
                    @(posedge clk);while(!s_axi_awready)@(posedge clk);@(negedge clk);s_axi_awvalid=0;
                end
                begin
                    if(ordering==1)repeat(4)@(negedge clk);
                    @(negedge clk);s_axi_wdata=value;s_axi_wstrb=strobes;s_axi_wvalid=1;
                    @(posedge clk);while(!s_axi_wready)@(posedge clk);@(negedge clk);s_axi_wvalid=0;
                end
            join
            while(!s_axi_bvalid)@(negedge clk);
            repeat(2)begin @(negedge clk);if(!s_axi_bvalid||s_axi_bresp!==response)$fatal(1,"AXI-Lite B mismatch addr=%h got=%b expected=%b",address,s_axi_bresp,response);end
            s_axi_bready=1;@(posedge clk);@(negedge clk);s_axi_bready=0;writes=writes+1;
        end
    endtask
    task read_reg(input reg[9:0]address,input reg[1:0]response,output reg[31:0]value);
        reg[31:0]held;
        begin
            @(negedge clk);s_axi_araddr=address;s_axi_arvalid=1;@(posedge clk);while(!s_axi_arready)@(posedge clk);
            @(negedge clk);s_axi_arvalid=0;while(!s_axi_rvalid)@(negedge clk);held=s_axi_rdata;
            repeat(2)begin @(negedge clk);if(!s_axi_rvalid||s_axi_rresp!==response||s_axi_rdata!==held)$fatal(1,"AXI-Lite R unstable/response mismatch");end
            value=held;s_axi_rready=1;@(posedge clk);@(negedge clk);s_axi_rready=0;reads=reads+1;
        end
    endtask
    always @(negedge clk)if(!rst_n)begin s_axi_awvalid=0;s_axi_wvalid=0;s_axi_bready=0;s_axi_arvalid=0;s_axi_rready=0;end
endmodule

module tb_lung_axil_control;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    wire[9:0]s_axi_awaddr,s_axi_araddr;wire[2:0]s_axi_awprot,s_axi_arprot;
    wire[31:0]s_axi_wdata,s_axi_rdata;wire[3:0]s_axi_wstrb;wire[1:0]s_axi_bresp,s_axi_rresp;
    wire s_axi_awvalid,s_axi_awready,s_axi_wvalid,s_axi_wready,s_axi_bvalid,s_axi_bready,s_axi_arvalid,s_axi_arready,s_axi_rvalid,s_axi_rready;
    wire w_valid,w_first,w_last,w_relu,c_valid,c_kernel3,c_padding;
    reg w_ready=0,c_ready=0,weights_loaded=0,core_busy=0,core_done=0,core_error=0;
    wire[511:0]w_weights;wire[7:0]w_output_keep,w_input_zero_point,w_output_zero_point;
    wire[255:0]w_biases,w_multipliers;wire[47:0]w_shifts;
    wire[31:0]c_base,c_output_base;wire[9:0]c_height,c_width;wire[6:0]c_channels;wire[3:0]c_output_blocks,c_output_block;
    reg[31:0]written_words=7,cache_hits=13,cache_misses=2;
    reg[31:0]value;integer i,weight_accepts=0,command_accepts=0,checks=0;
    reg held_weight=0,held_command=0;reg[1098:0]previous_weight;reg[100:0]previous_command;
    reg[100:0]last_config;integer stable_config_cycles=0;
    lung_test_axil_master host(.*);lung_axil_control dut(.*);
    always @(posedge clk)begin
        if(!rst_n)begin weights_loaded<=0;held_weight=0;held_command=0;stable_config_cycles=0;last_config=0;end
        else begin
            if({c_base,c_height,c_width,c_channels,c_kernel3,c_padding,c_output_base,c_output_blocks,c_output_block}!==last_config)stable_config_cycles=0;
            else stable_config_cycles=stable_config_cycles+1;
            last_config={c_base,c_height,c_width,c_channels,c_kernel3,c_padding,c_output_base,c_output_blocks,c_output_block};
            if(c_valid&&stable_config_cycles<4)$fatal(1,"Configuration VALID before four stable cycles");
            if(held_weight&&(!w_valid||{w_weights,w_output_keep,w_input_zero_point,w_output_zero_point,w_biases,w_multipliers,w_shifts,w_relu,w_first,w_last}!==previous_weight))$fatal(1,"Weight command changed under backpressure");
            if(held_command&&(!c_valid||{c_base,c_height,c_width,c_channels,c_kernel3,c_padding,c_output_base,c_output_blocks,c_output_block}!==previous_command))$fatal(1,"Start command changed under backpressure");
            held_weight=w_valid&&!w_ready;previous_weight={w_weights,w_output_keep,w_input_zero_point,w_output_zero_point,w_biases,w_multipliers,w_shifts,w_relu,w_first,w_last};
            held_command=c_valid&&!c_ready;previous_command={c_base,c_height,c_width,c_channels,c_kernel3,c_padding,c_output_base,c_output_blocks,c_output_block};
            if(w_valid&&w_ready)begin weight_accepts=weight_accepts+1;weights_loaded<=w_last;end
            if(c_valid&&c_ready)command_accepts=command_accepts+1;
        end
    end
    task expect_reg(input reg[9:0]address,input reg[31:0]wanted);
        begin host.read_reg(address,0,value);if(value!==wanted)$fatal(1,"Register mismatch at %h: %h/%h",address,value,wanted);checks=checks+1;end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        expect_reg(0,32'h4c554e47);expect_reg(12,32'h10000);expect_reg(8,0);
        host.read_reg(1,2,value);host.read_reg(10'h3fc,2,value);
        host.write_reg(0,1,15,1,2);host.write_reg(10'h3fc,1,15,2,2);host.write_reg(1,1,15,0,2);
        host.write_reg(4,1,15,1,2); // cannot START before weights
        host.write_reg(4,2,15,0,0);host.write_reg(10'h3fc,32'hffffffff,0,2,0);
        host.write_reg(16,32'h12345678,15,1,0);host.write_reg(16,32'haabbccdd,5,2,0);expect_reg(16,32'h12bb56dd);
        host.write_reg(10'h018,32'h00010001,15,0,0);host.write_reg(10'h018,32'hffffffff,15,1,2);expect_reg(10'h018,32'h00010001);
        host.write_reg(10'h100,32'h55667788,15,2,0);host.write_reg(10'h100,32'haabbccdd,5,1,0);expect_reg(10'h100,32'h55bb77dd);
        host.write_reg(10'h180,31,15,0,0);host.write_reg(10'h180,0,2,1,0);expect_reg(10'h180,31);
        host.write_reg(10'h180,63,15,2,2);expect_reg(10'h180,31);
        host.write_reg(10'h01c,32'h01010001,15,0,0);host.write_reg(10'h01c,32'h80000000,15,0,2);expect_reg(10'h01c,32'h01010001);
        host.write_reg(10'h020,32'h01808001,15,1,0);host.write_reg(10'h020,32'h80000000,15,1,2);expect_reg(10'h020,32'h01808001);
        host.write_reg(4,4,15,2,2);host.write_reg(4,12,15,1,0);
        if(!w_valid||!w_first||w_last||w_weights[31:0]!==32'h55bb77dd||w_shifts[5:0]!==31)$fatal(1,"Incorrect staged first weight");
        host.write_reg(10'h100,0,15,2,2);host.write_reg(10'h140,0,15,1,2);host.write_reg(4,1,15,0,2);
        @(negedge clk);w_ready=1;while(w_valid)@(negedge clk);
        host.write_reg(10'h018,0,15,1,2);host.write_reg(10'h100,7,15,2,0);host.write_reg(4,12,15,0,2);
        host.write_reg(4,20,15,1,0);while(w_valid)@(negedge clk);expect_reg(10'h038,2);expect_reg(10'h040,0);
        if(!weights_loaded)$fatal(1,"Weight LAST did not finish loading");
        host.write_reg(4,20,15,0,2); // LAST without FIRST is invalid for a new block
        host.write_reg(16,4097,15,0,0);host.write_reg(20,8192,15,0,0);host.write_reg(4,1,15,0,2);
        host.write_reg(16,4096,15,1,0);host.write_reg(10'h01c,32'h00010001,15,2,0);host.write_reg(4,2,15,0,0);
        host.write_reg(4,1,15,1,0);while(!c_valid)@(negedge clk);if(c_base!==512||c_output_base!==1024||c_height!==1||c_width!==1)$fatal(1,"Start word-address conversion failed");
        host.write_reg(16,0,15,2,2);host.write_reg(10'h100,0,15,0,2);host.write_reg(4,1,15,1,2);
        @(negedge clk);c_ready=1;@(posedge clk);@(negedge clk);c_ready=0;core_busy=1;
        host.read_reg(8,0,value);if(!value[0])$fatal(1,"Active command not busy");
        host.write_reg(4,2,15,0,0);
        fork
            host.write_reg(4,2,15,0,0);
            begin while(!(dut.aw_hold&&dut.w_hold))@(negedge clk);core_busy=0;core_done=1;@(posedge clk);@(negedge clk);core_done=0;end
        join
        host.read_reg(8,0,value);if(!value[1]||value[0]||value[2])$fatal(1,"Completion lost against simultaneous W1C");
        expect_reg(10'h034,1);expect_reg(10'h024,7);expect_reg(10'h028,13);expect_reg(10'h02c,2);
        host.read_reg(10'h030,0,value);if(value==0)$fatal(1,"Cycle counter did not run");
        // Fill the exact maximum weight block; reject an unending 128th beat.
        host.write_reg(4,12,15,0,0);
        for(i=1;i<127;i=i+1)host.write_reg(4,4,15,i%3,0);
        host.write_reg(4,4,15,0,2);expect_reg(10'h040,127);
        host.write_reg(4,20,15,1,0);while(w_valid)@(negedge clk);expect_reg(10'h038,130);expect_reg(10'h040,0);
        host.write_reg(4,2,15,0,0);host.write_reg(4,1,15,2,0);while(!c_valid)@(negedge clk);@(negedge clk);c_ready=1;@(posedge clk);@(negedge clk);c_ready=0;core_busy=1;
        repeat(3)@(negedge clk);core_busy=0;core_error=1;core_done=1;@(posedge clk);@(negedge clk);core_done=0;
        host.read_reg(8,0,value);if(!value[1]||!value[2]||value[0])$fatal(1,"Compute error not sticky");expect_reg(10'h034,2);
        host.read_reg(8,0,value);if(!value[1])$fatal(1,"Status read cleared DONE");
        host.write_reg(4,2,15,1,0);host.read_reg(8,0,value);if(value[2:1]!=0||value[8])$fatal(1,"W1C did not clear sticky status");
        // Pending AW without W is cancelled only by shared reset.
        @(negedge clk);host.s_axi_awaddr=16;host.s_axi_awvalid=1;@(posedge clk);while(!s_axi_awready)@(posedge clk);
        @(negedge clk);host.s_axi_awvalid=0;rst_n=0;repeat(3)@(negedge clk);rst_n=1;@(negedge clk);
        if(c_valid||w_valid||s_axi_rvalid||s_axi_bvalid)$fatal(1,"Reset left a live control transaction");
        expect_reg(8,0);expect_reg(16,0);expect_reg(10'h038,0);
        if(weight_accepts!=130||command_accepts!=2||host.writes<150||host.reads<20)$fatal(1,"Insufficient AXI-Lite protocol coverage");
        $display("Lung AXI-Lite control RTL: PASS (%0d register checks, %0d writes, %0d reads, %0d weights, %0d commands; AW/W order, held B/R, WSTRB, locks, W1C race, limits, shared reset)",checks,host.writes,host.reads,weight_accepts,command_accepts);$finish;
    end
    initial begin #5000000;$fatal(1,"AXI-Lite protocol timeout");end
endmodule
