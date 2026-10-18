`timescale 1ns/1ps

// Centered 3x3 binary dilation, raster order, zero padding, same image size.
// One 32-bit stream word per pixel. Low byte != 0 means foreground.
// Geometry defines the frame; input TLAST must be on its final pixel.
// Output TLAST is regenerated after flushing the bottom/right padding.
// Two 1-bit line stores; contents need no reset because startup rows are gated.
module dilation_axis #(
    parameter integer IMAGE_WIDTH = 512,
    parameter integer IMAGE_HEIGHT = 512
) (
    input wire clk,
    input wire rst_n,
    input wire [31:0] s_axis_tdata,
    input wire [3:0] s_axis_tkeep,
    input wire s_axis_tlast,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    output wire [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tlast,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);
    localparam integer X_BITS = $clog2(IMAGE_WIDTH + 1);
    localparam integer Y_BITS = $clog2(IMAGE_HEIGHT + 1);
    reg [X_BITS-1:0] x;
    reg [Y_BITS-1:0] y;
    reg line_recent [0:IMAGE_WIDTH];
    reg line_older [0:IMAGE_WIDTH];
    reg top_left2, top_left1;
    reg mid_left2, mid_left1;
    reg bot_left2, bot_left1;
    reg pending;
    reg pending_last;
    reg [31:0] out_data;
    reg out_valid;
    reg out_last;

    wire padding = (x == IMAGE_WIDTH) || (y == IMAGE_HEIGHT);
    wire can_step = rst_n && !pending && (!out_valid || m_axis_tready);
    assign s_axis_tready = can_step && !padding;
    wire step = can_step && (padding || s_axis_tvalid);
    wire pixel = !padding && (|s_axis_tdata[7:0]);
    wire top_pixel = (y >= 2) ? line_older[x] : 1'b0;
    wire mid_pixel = (y >= 1) ? line_recent[x] : 1'b0;
    wire has_output = (x >= 1) && (y >= 1);
    wire core_valid;
    wire [7:0] core_mask;
    wire [8:0] window_bits = {
        top_left2, top_left1, top_pixel,
        mid_left2, mid_left1, mid_pixel,
        bot_left2, bot_left1, pixel
    };

    dilation_3x3 core (
        .clk(clk), .rst_n(rst_n), .in_valid(step && has_output),
        .window_in(window_bits), .out_valid(core_valid), .mask_out(core_mask)
    );

    assign m_axis_tdata = out_data;
    assign m_axis_tkeep = 4'b1111;
    assign m_axis_tlast = out_last;
    assign m_axis_tvalid = out_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            x <= 0;
            y <= 0;
            top_left2 <= 0; top_left1 <= 0;
            mid_left2 <= 0; mid_left1 <= 0;
            bot_left2 <= 0; bot_left1 <= 0;
            pending <= 0;
            pending_last <= 0;
            out_data <= 0;
            out_valid <= 0;
            out_last <= 0;
        end else begin
            if (out_valid && m_axis_tready) begin
                out_valid <= 0;
                out_last <= 0;
            end
            if (step) begin
                line_older[x] <= mid_pixel;
                line_recent[x] <= pixel;
                // Left padding is zero; don't carry taps across rows.
                top_left2 <= (x == 0) ? 1'b0 : top_left1;
                mid_left2 <= (x == 0) ? 1'b0 : mid_left1;
                bot_left2 <= (x == 0) ? 1'b0 : bot_left1;
                top_left1 <= top_pixel;
                mid_left1 <= mid_pixel;
                bot_left1 <= pixel;
                if (has_output) begin
                    pending <= 1;
                    pending_last <= (y == IMAGE_HEIGHT) && (x == IMAGE_WIDTH);
                end
                if (x == IMAGE_WIDTH) begin
                    x <= 0;
                    y <= (y == IMAGE_HEIGHT) ? 0 : y + 1'b1;
                end else begin
                    x <= x + 1'b1;
                end
            end
            if (core_valid) begin
                pending <= 0;
                out_data <= {24'd0, core_mask};
                out_valid <= 1;
                out_last <= pending_last;
            end
        end
    end
endmodule
