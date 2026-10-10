`timescale 1ns/1ps

// window_level 测试平台。

// 验证边界截断、窗口内映射、负窗位和 1 拍输出延迟。

module tb_window_level;
    reg clk;
    reg rst_n;
    reg in_valid;
    reg signed [15:0] hu_in;
    reg signed [15:0] wl_in;
    reg        [15:0] ww_in;
    wire out_valid;
    wire [7:0] gray_out;

    integer pass_count;
    integer fail_count;

    window_level uut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .hu_in(hu_in), .wl_in(wl_in), .ww_in(ww_in),
        .out_valid(out_valid), .gray_out(gray_out)
    );

    // 产生 10 ns 时钟。

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 软件参考模型，使用整数运算得到期望灰度值。

    function [7:0] expected_gray;
        input integer hu_value;
        input integer wl_value;
        input integer ww_value;
        integer low_value;
        integer high_value;
        integer delta_value;
        begin
            low_value = wl_value - ww_value / 2;
            high_value = wl_value + ww_value / 2;
            if (ww_value <= 0)
                expected_gray = 8'd0;
            else if (hu_value <= low_value)
                expected_gray = 8'd0;
            else if (hu_value >= high_value)
                expected_gray = 8'd255;
            else begin
                delta_value = hu_value - low_value;
                expected_gray = (delta_value * 255) / ww_value;
            end
        end
    endfunction

    // 发送一个像素，并在输入后的第二个上升沿检查输出。

    task send_and_check;
        input integer hu_value;
        input integer wl_value;
        input integer ww_value;
        reg [7:0] expected_value;
        begin
            expected_value = expected_gray(hu_value, wl_value, ww_value);

            // 下降沿准备输入，下一个上升沿完成输入锁存。

            @(negedge clk);
            hu_in = hu_value;
            wl_in = wl_value;
            ww_in = ww_value;
            in_valid = 1'b1;
            @(posedge clk);

            // 再下一个上升沿输出，并检查 out_valid 与结果是否对齐。

            @(posedge clk);
            #1;
            if (!out_valid) begin
                $display("FAIL: HU=%0d out_valid not asserted", hu_value);
                fail_count = fail_count + 1;
            end
            else if (gray_out !== expected_value) begin
                $display("FAIL: HU=%0d WL=%0d WW=%0d expected=%0d got=%0d",
                         hu_value, wl_value, ww_value,
                         expected_value, gray_out);
                fail_count = fail_count + 1;
            end
            else begin
                $display("PASS: HU=%0d WL=%0d WW=%0d gray=%0d",
                         hu_value, wl_value, ww_value, gray_out);
                pass_count = pass_count + 1;
            end

            // 拉低 valid，避免重复发送同一个输入。

            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        hu_in = 16'sd0;
        wl_in = 16'sd0;
        ww_in = 16'd0;
        pass_count = 0;
        fail_count = 0;

        // 同步复位保持两个时钟周期。

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        // 第一组：WL=40，WW=400，low=-160，high=240。

        send_and_check(-1000, 40, 400);
        send_and_check(-160,  40, 400);
        send_and_check(-100,  40, 400);
        send_and_check(0,     40, 400);
        send_and_check(40,    40, 400);
        send_and_check(100,   40, 400);
        send_and_check(240,   40, 400);
        send_and_check(1000,  40, 400);

        // 第二组：WL=-600，WW=1500，low=-1350，high=150。

        send_and_check(-2000, -600, 1500);
        send_and_check(-1350, -600, 1500);
        send_and_check(-600,  -600, 1500);
        send_and_check(0,     -600, 1500);
        send_and_check(150,   -600, 1500);
        send_and_check(1000,  -600, 1500);

        // 无输入时，out_valid 应回到 0。

        repeat (2) @(posedge clk);
        #1;
        if (out_valid) begin
            $display("FAIL: out_valid remains asserted without input");
            fail_count = fail_count + 1;
        end

        $display("Simulation finished: PASS=%0d FAIL=%0d", pass_count, fail_count);
        if (fail_count == 0)
            $finish;
        else
            $fatal(1, "window_level test failed");
    end
endmodule
