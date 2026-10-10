`timescale 1ns/1ps

// Real CT 32x32 data testbench.
// The input and output files use one integer per line.
// MATLAB exports pixels in row-major order, so this testbench reads file order directly.

module tb_window_level_image;
    localparam integer IMAGE_SIZE = 32;
    localparam integer PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE;

    reg clk;
    reg rst_n;
    reg in_valid;
    reg signed [15:0] hu_in;
    reg signed [15:0] wl_in;
    reg        [15:0] ww_in;
    wire out_valid;
    wire [7:0] gray_out;

    integer input_fd;
    integer output_fd;
    integer read_count;
    integer output_count;
    integer scan_result;
    integer hu_value;

    window_level uut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .hu_in(hu_in),
        .wl_in(wl_in),
        .ww_in(ww_in),
        .out_valid(out_valid),
        .gray_out(gray_out)
    );

    // 10 ns clock.
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // Record data only when out_valid is high.
    // This prevents invalid cycles from becoming output pixels.
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            if (output_count >= PIXEL_COUNT)
                $fatal(1, "RTL output count exceeds %0d", PIXEL_COUNT);
            $fwrite(output_fd, "%0d\n", gray_out);
            output_count = output_count + 1;
        end
    end

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        hu_in = 16'sd0;
        wl_in = 16'sd0;
        ww_in = 16'd0;
        read_count = 0;
        output_count = 0;

        // Keep the input and output files in the existing matlab directory.
        input_fd = $fopen("matlab/hu_input.txt", "r");
        if (input_fd == 0)
            $fatal(1, "Cannot open matlab/hu_input.txt");

        output_fd = $fopen("matlab/rtl_output.txt", "w");
        if (output_fd == 0)
            $fatal(1, "Cannot create matlab/rtl_output.txt");

        // Synchronous reset for two clock cycles.
        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        // Send exactly 1024 pixels. Each input pixel is sent once.
        // WL=40 and WW=400 are applied to every input pixel.
        while (read_count < PIXEL_COUNT) begin
            scan_result = $fscanf(input_fd, "%d\n", hu_value);
            if (scan_result != 1)
                $fatal(1, "Input file ended early at pixel %0d", read_count);

            // Prepare one input on the falling edge for the next rising edge.
            @(negedge clk);
            hu_in = hu_value;
            wl_in = 16'sd40;
            ww_in = 16'd400;
            in_valid = 1'b1;
            read_count = read_count + 1;
        end

        // Stop input after the final pixel has been sampled.
        @(negedge clk);
        in_valid = 1'b0;

        // Wait until the final one-cycle-latency output is recorded.
        while (output_count < PIXEL_COUNT)
            @(posedge clk);
        #1;

        $fclose(input_fd);
        $fclose(output_fd);

        $display("Input pixels read: %0d", read_count);
        $display("RTL pixels written: %0d", output_count);

        if (read_count != PIXEL_COUNT)
            $fatal(1, "Input count error: expected %0d, got %0d", PIXEL_COUNT, read_count);
        if (output_count != PIXEL_COUNT)
            $fatal(1, "Output count error: expected %0d, got %0d", PIXEL_COUNT, output_count);

        $display("Real CT 32x32 RTL test passed");
        $finish;
    end
endmodule
