`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// window_level_axis 测试平台
// -----------------------------------------------------------------------------
// 目的：
//   在连接 AXI DMA 之前，先验证 AXI-Stream 包装模块本身：
//   1. AXI valid/ready 握手是否正确；
//   2. HU 数据是否正确送入 window_level 核心；
//   3. 输出灰度值是否与基础模块一致；
//   4. TLAST 是否与最后一个像素正确对齐；
//   5. 输出是否能够在 m_axis_tready=1 时被完整接收。
//
// 测试参数：
//   WL = 40，WW = 400，对应 low=-160，high=240。
// -----------------------------------------------------------------------------
module tb_window_level_axis;

    reg                   clk;
    reg                   rst_n;

    reg  signed [15:0]    wl_in;
    reg         [15:0]    ww_in;

    reg         [31:0]    s_axis_tdata;
    reg         [3:0]     s_axis_tkeep;
    reg                   s_axis_tlast;
    reg                   s_axis_tvalid;
    wire                  s_axis_tready;

    wire        [31:0]    m_axis_tdata;
    wire        [3:0]     m_axis_tkeep;
    wire                  m_axis_tlast;
    wire                  m_axis_tvalid;
    reg                   m_axis_tready;

    integer send_index;
    integer recv_index;
    integer pass_count;
    integer fail_count;

    reg signed [15:0] hu_test [0:7];
    reg        [7:0]  expected_gray [0:7];

    // 100 MHz 测试时钟。
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    window_level_axis dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .wl_in          (wl_in),
        .ww_in          (ww_in),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tkeep   (s_axis_tkeep),
        .s_axis_tlast   (s_axis_tlast),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tkeep   (m_axis_tkeep),
        .m_axis_tlast   (m_axis_tlast),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (m_axis_tready)
    );

    // 统计输出握手结果。
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (m_axis_tdata[7:0] === expected_gray[recv_index]) begin
                pass_count = pass_count + 1;
                $display("PASS pixel=%0d HU=%0d gray=%0d last=%0d",
                         recv_index,
                         hu_test[recv_index],
                         m_axis_tdata[7:0],
                         m_axis_tlast);
            end
            else begin
                fail_count = fail_count + 1;
                $display("FAIL pixel=%0d HU=%0d expected=%0d actual=%0d",
                         recv_index,
                         hu_test[recv_index],
                         expected_gray[recv_index],
                         m_axis_tdata[7:0]);
            end

            // 最后一个输出必须带有 TLAST。
            if ((recv_index == 7) && !m_axis_tlast) begin
                fail_count = fail_count + 1;
                $display("FAIL TLAST was not asserted on the final pixel");
            end

            recv_index = recv_index + 1;
        end
    end

    // 向 AXI-Stream 发送一个像素，并保持 valid，直到 ready 握手成功。
    task send_pixel;
        input integer index;
        reg   handshake_done;
        begin
            @(negedge clk);
            s_axis_tdata  = {16'd0, hu_test[index]};
            s_axis_tkeep  = 4'b1111;
            s_axis_tlast  = (index == 7);
            s_axis_tvalid = 1'b1;
            handshake_done = 1'b0;

            // 等待输入握手。测试平台不会因为 ready=0 而改变数据。
            while (!handshake_done) begin
                @(posedge clk);
                if (s_axis_tready)
                    handshake_done = 1'b1;
            end

            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
        end
    endtask

    initial begin
        // 测试输入。
        hu_test[0] = -16'sd1000;
        hu_test[1] = -16'sd160;
        hu_test[2] = -16'sd100;
        hu_test[3] =  16'sd0;
        hu_test[4] =  16'sd40;
        hu_test[5] =  16'sd100;
        hu_test[6] =  16'sd240;
        hu_test[7] =  16'sd1000;

        // 与基础 window_level.v 相同的整数除法期望值。
        expected_gray[0] = 8'd0;
        expected_gray[1] = 8'd0;
        expected_gray[2] = 8'd38;
        expected_gray[3] = 8'd102;
        expected_gray[4] = 8'd127;
        expected_gray[5] = 8'd165;
        expected_gray[6] = 8'd255;
        expected_gray[7] = 8'd255;

        wl_in         = 16'sd40;
        ww_in         = 16'd400;
        s_axis_tdata  = 32'd0;
        s_axis_tkeep  = 4'd0;
        s_axis_tlast  = 1'b0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;
        send_index    = 0;
        recv_index    = 0;
        pass_count    = 0;
        fail_count    = 0;

        rst_n = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // 依次发送 8 个像素。send_pixel 会自动等待输入 ready。
        for (send_index = 0; send_index < 8; send_index = send_index + 1)
            send_pixel(send_index);

        // 等待最后一个 AXI 输出握手完成。
        wait (recv_index == 8);
        repeat (2) @(posedge clk);

        $display("----------------------------------------");
        $display("AXI test finished: PASS=%0d FAIL=%0d",
                 pass_count, fail_count);
        $display("----------------------------------------");

        $finish;
    end

endmodule
