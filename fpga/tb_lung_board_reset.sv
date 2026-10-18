`timescale 1ns/1ps
// Exercise the actual vendor reset IP, including the original bad auxiliary tie.
module tb_lung_board_reset;
    reg clk=0;
    always #5 clk=~clk;
    reg external_n=0, locked=1;
    wire fixed_n, broken_n;
    lung_reset_check fixed_reset(
        .slowest_sync_clk(clk), .ext_reset_in(external_n), .aux_reset_in(1'b1),
        .mb_debug_sys_rst(1'b0), .dcm_locked(locked), .peripheral_aresetn(fixed_n));
    lung_reset_check broken_reset(
        .slowest_sync_clk(clk), .ext_reset_in(external_n), .aux_reset_in(1'b0),
        .mb_debug_sys_rst(1'b0), .dcm_locked(locked), .peripheral_aresetn(broken_n));

    reg arvalid=0, rready=0;
    wire arready, rvalid;
    wire [31:0] rdata;
    wire [1:0] rresp;
    lung_axil_control control(
        .clk(clk), .rst_n(fixed_n),
        .s_axi_awaddr(10'd0), .s_axi_awprot(3'd0), .s_axi_awvalid(1'b0),
        .s_axi_wdata(32'd0), .s_axi_wstrb(4'd0), .s_axi_wvalid(1'b0), .s_axi_bready(1'b1),
        .s_axi_araddr(10'd0), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .w_ready(1'b0), .weights_loaded(1'b0), .c_ready(1'b0),
        .core_busy(1'b0), .core_done(1'b0), .core_error(1'b0),
        .written_words(32'd0), .cache_hits(32'd0), .cache_misses(32'd0));

    task read_id;
        begin
            @(negedge clk); arvalid=1;
            @(posedge clk); while(!arready) @(posedge clk);
            @(negedge clk); arvalid=0;
            while(!rvalid) @(negedge clk);
            if(rdata!==32'h4c554e47 || rresp!==0) $fatal(1,"Wrong ID after board reset release");
            // A stalled CPU response must remain stable.
            repeat(4) begin
                @(negedge clk);
                if(!rvalid || rdata!==32'h4c554e47) $fatal(1,"ID response lost under backpressure");
            end
            rready=1;
            @(negedge clk); rready=0;
        end
    endtask

    initial begin
        repeat(16) @(negedge clk);
        external_n=1;
        repeat(1024) @(negedge clk);
        if(fixed_n!==1 || broken_n!==0) $fatal(1,"Auxiliary polarity reproduction failed");
        read_id();
        external_n=0;
        repeat(32) @(negedge clk);
        if(fixed_n!==0) $fatal(1,"External reset not asserted");
        external_n=1;
        repeat(1024) @(negedge clk);
        if(fixed_n!==1 || broken_n!==0) $fatal(1,"External reset not released");
        read_id();
        locked=0;
        repeat(32) @(negedge clk);
        if(fixed_n!==0) $fatal(1,"Clock-lock loss not resetting");
        locked=1;
        repeat(1024) @(negedge clk);
        if(fixed_n!==1 || broken_n!==0) $fatal(1,"Clock-lock recovery not releasing reset");
        read_id();
        $display("Board vendor reset RTL: PASS (original tie stays reset; corrected tie releases; external reset and lock recovery; 3 AXI-Lite ID reads)");
        $finish;
    end
    initial begin #200000; $fatal(1,"Board reset simulation timeout"); end
endmodule
