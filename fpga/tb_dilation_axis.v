`timescale 1ns/1ps

module tb_dilation_axis;
    parameter integer W = 5;
    parameter integer H = 4;
    localparam integer N = W*H;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    reg [31:0] s_data = 0;
    reg s_last = 0;
    reg s_valid = 0;
    wire s_ready;
    wire [31:0] m_data;
    wire [3:0] m_keep;
    wire m_last, m_valid;
    reg m_ready = 0;
    reg image [0:N-1];
    reg expected [0:N-1];
    integer received = 0;
    integer checks = 0;
    integer frame, i, row, col, dy, dx;
    integer cycles = 0;
    reg held = 0;
    reg [31:0] held_data;
    reg held_last;

    dilation_axis #(.IMAGE_WIDTH(W), .IMAGE_HEIGHT(H)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(s_data), .s_axis_tkeep(4'hf),
        .s_axis_tlast(s_last), .s_axis_tvalid(s_valid), .s_axis_tready(s_ready),
        .m_axis_tdata(m_data), .m_axis_tkeep(m_keep),
        .m_axis_tlast(m_last), .m_axis_tvalid(m_valid), .m_axis_tready(m_ready)
    );

    // Deterministic burst stalls, including across row/frame boundaries.
    always @(negedge clk) begin
        cycles = cycles + 1;
        m_ready = rst_n && ((cycles % 17) >= 7);
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (held && (m_valid !== 1 || m_data !== held_data ||
                         m_last !== held_last || m_keep !== 4'hf))
                $fatal(1, "Output changed under backpressure");
            held = m_valid && !m_ready;
            held_data = m_data;
            held_last = m_last;
            if (m_valid && m_ready) begin
                if (received >= N)
                    $fatal(1, "Too many output pixels");
                if (m_data !== (expected[received] ? 32'd255 : 32'd0))
                    $fatal(1, "Frame %0d pixel %0d: expected %0d, got %0d",
                           frame, received, expected[received], m_data);
                if (m_last !== (received == N-1) || m_keep !== 4'hf)
                    $fatal(1, "TLAST/TKEEP mismatch at pixel %0d", received);
                received = received + 1;
                checks = checks + 1;
            end
        end else held = 0;
    end

    task send_pixel;
        input integer idx;
        begin
            @(negedge clk);
            s_data = image[idx] ? 32'd255 : 32'd0;
            s_last = (idx == N-1);
            s_valid = 1;
            @(posedge clk);
            while (!s_ready) @(posedge clk);
            @(negedge clk);
            s_valid = 0;
            s_last = 0;
            if (idx % 3 == 0) repeat (2) @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        // Reuse the same hardware without resetting between frames.
        for (frame = 0; frame < 5; frame = frame + 1) begin
            received = 0;
            for (i = 0; i < N; i = i + 1) begin
                case (frame)
                    0: image[i] = 1; // all-white image stays white
                    1: image[i] = 0; // previous frame's line data must not leak
                    2: image[i] = (i == N/2); // isolated foreground
                    3: image[i] = (i != N/2); // hole in white region
                    4: image[i] = ((i * 7 + i/3) % 11 != 0);
                endcase
            end
            for (row = 0; row < H; row = row + 1)
                for (col = 0; col < W; col = col + 1) begin
                    expected[row*W+col] = 0;
                    for (dy = -1; dy <= 1; dy = dy + 1)
                        for (dx = -1; dx <= 1; dx = dx + 1)
                            if (row+dy >= 0 && row+dy < H && col+dx >= 0 && col+dx < W)
                                if (image[(row+dy)*W+col+dx])
                                    expected[row*W+col] = 1;
                end
            for (i = 0; i < N; i = i + 1) send_pixel(i);
            wait (received == N);
            repeat (4) @(negedge clk);
        end
        $display("Dilation AXIS test: PASS W=%0d H=%0d frames=5 pixels=%0d", W, H, checks);
        $finish;
    end
    initial begin
        #200000000;
        $fatal(1, "Timeout");
    end
endmodule
