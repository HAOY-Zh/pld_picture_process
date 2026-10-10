`timescale 1ns/1ps

module tb_threshold_range_axis;

    reg         clk;
    reg         rst_n;
    reg  [31:0] thresholds_in;
    reg  [31:0] s_axis_tdata;
    reg  [3:0]  s_axis_tkeep;
    reg         s_axis_tlast;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    wire [31:0] m_axis_tdata;
    wire [3:0]  m_axis_tkeep;
    wire        m_axis_tlast;
    wire        m_axis_tvalid;
    reg         m_axis_tready;

    integer pass_count;
    integer fail_count;
    integer index;
    reg signed [15:0] test_hu [0:5];
    reg [7:0] expected [0:5];

    threshold_range_axis dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .thresholds_in(thresholds_in),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tkeep (s_axis_tkeep),
        .s_axis_tlast (s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata (m_axis_tdata),
        .m_axis_tkeep (m_axis_tkeep),
        .m_axis_tlast (m_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    always #5 clk = ~clk;

    task send_pixel;
        input signed [15:0] value;
        input              last;
        begin
            @(negedge clk);
            s_axis_tdata  = {16'd0, value[15:0]};
            s_axis_tlast  = last;
            s_axis_tvalid = 1'b1;
            while (!s_axis_tready) @(negedge clk);
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (m_axis_tdata[7:0] == expected[index]) begin
                $display("PASS pixel=%0d HU=%0d mask=%0d last=%0d",
                         index, test_hu[index], m_axis_tdata[7:0], m_axis_tlast);
                pass_count = pass_count + 1;
            end
            else begin
                $display("FAIL pixel=%0d HU=%0d expected=%0d got=%0d",
                         index, test_hu[index], expected[index], m_axis_tdata[7:0]);
                fail_count = fail_count + 1;
            end
            index = index + 1;
        end
    end

    initial begin
        clk           = 1'b0;
        rst_n         = 1'b0;
        thresholds_in = {16'sd300, -16'sd500};
        s_axis_tdata  = 32'd0;
        s_axis_tkeep  = 4'b1111;
        s_axis_tlast  = 1'b0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;
        pass_count    = 0;
        fail_count    = 0;
        index         = 0;

        test_hu[0] = -1000; expected[0] = 8'd0;
        test_hu[1] = -500;  expected[1] = 8'd255;
        test_hu[2] = 0;     expected[2] = 8'd255;
        test_hu[3] = 300;   expected[3] = 8'd255;
        test_hu[4] = 301;   expected[4] = 8'd0;
        test_hu[5] = 1000;  expected[5] = 8'd0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        send_pixel(test_hu[0], 1'b0);
        send_pixel(test_hu[1], 1'b0);
        send_pixel(test_hu[2], 1'b0);
        send_pixel(test_hu[3], 1'b0);
        send_pixel(test_hu[4], 1'b0);
        send_pixel(test_hu[5], 1'b1);

        wait (index == 6);
        if (fail_count == 0)
            $display("Threshold range AXI test finished: PASS=%0d FAIL=0", pass_count);
        else
            $display("Threshold range AXI test finished: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $finish;
    end

endmodule
