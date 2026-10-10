`timescale 1ns/1ps

module tb_threshold_axis;

    reg                   clk;
    reg                   rst_n;
    reg  signed [15:0]    threshold_in;

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

    reg signed [15:0] hu_test [0:5];
    reg        [7:0] expected_mask [0:5];

    threshold_axis dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .threshold_in   (threshold_in),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (m_axis_tdata[7:0] === expected_mask[recv_index]) begin
                pass_count = pass_count + 1;
                $display("PASS pixel=%0d HU=%0d mask=%0d last=%0d",
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
                         expected_mask[recv_index],
                         m_axis_tdata[7:0]);
            end

            if ((recv_index == 5) && !m_axis_tlast) begin
                fail_count = fail_count + 1;
                $display("FAIL TLAST was not asserted on final pixel");
            end
            recv_index = recv_index + 1;
        end
    end

    task send_pixel;
        input integer index;
        reg handshake_done;
        begin
            @(negedge clk);
            s_axis_tdata  = {16'd0, hu_test[index]};
            s_axis_tkeep  = 4'b1111;
            s_axis_tlast  = (index == 5);
            s_axis_tvalid = 1'b1;
            handshake_done = 1'b0;

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
        hu_test[0] = -16'sd1000;
        hu_test[1] = -16'sd501;
        hu_test[2] = -16'sd500;
        hu_test[3] = -16'sd499;
        hu_test[4] =  16'sd0;
        hu_test[5] =  16'sd1000;

        expected_mask[0] = 8'd0;
        expected_mask[1] = 8'd0;
        expected_mask[2] = 8'd255;
        expected_mask[3] = 8'd255;
        expected_mask[4] = 8'd255;
        expected_mask[5] = 8'd255;

        threshold_in = -16'sd500;
        s_axis_tdata  = 32'd0;
        s_axis_tkeep  = 4'd0;
        s_axis_tlast  = 1'b0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;
        send_index = 0;
        recv_index = 0;
        pass_count = 0;
        fail_count = 0;

        rst_n = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        for (send_index = 0; send_index < 6; send_index = send_index + 1)
            send_pixel(send_index);

        wait (recv_index == 6);
        repeat (2) @(posedge clk);

        $display("----------------------------------------");
        $display("Threshold AXI test finished: PASS=%0d FAIL=%0d",
                 pass_count, fail_count);
        $display("----------------------------------------");
        $finish;
    end

endmodule
