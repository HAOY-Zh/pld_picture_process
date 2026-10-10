`timescale 1ns/1ps

module tb_erosion_3x3;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    reg in_valid = 0;
    reg [8:0] window_in = 0;
    wire out_valid;
    wire [7:0] mask_out;
    integer pattern;
    integer checked = 0;

    erosion_3x3 dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .window_in(window_in), .out_valid(out_valid), .mask_out(mask_out)
    );

    initial begin
        repeat (2) @(negedge clk);
        if (out_valid !== 0 || mask_out !== 0)
            $fatal(1, "Reset failed");
        rst_n = 1;
        // Exhaust all 512 possible binary neighborhoods, including each
        // single-black-pixel position and the all-white neighborhood.
        for (pattern = 0; pattern < 512; pattern = pattern + 1) begin
            @(negedge clk);
            in_valid = 1;
            window_in = pattern;
            @(posedge clk); #1;
            if (out_valid !== 1 ||
                mask_out !== ((pattern == 511) ? 8'd255 : 8'd0))
                $fatal(1, "Window %0d failed: mask=%0d", pattern, mask_out);
            checked = checked + 1;
        end
        @(negedge clk);
        in_valid = 0;
        window_in = 9'h1ff;
        @(posedge clk); #1;
        if (out_valid !== 0 || mask_out !== 0)
            $fatal(1, "Invalid cycle failed");
        @(negedge clk);
        rst_n = 0;
        @(posedge clk); #1;
        if (out_valid !== 0 || mask_out !== 0)
            $fatal(1, "Reset after data failed");
        $display("Erosion core test: PASS (%0d windows)", checked);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "Simulation timeout");
    end
endmodule
