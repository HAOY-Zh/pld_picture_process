`timescale 1ns/1ps

// One bit per mask pixel: nonzero mask pixels are represented as 1.
// window_in contains all nine pixels of a centered 3x3 neighborhood.
// The future image-stream wrapper supplies the window and zero padding.
module erosion_3x3 (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [8:0] window_in,
    output reg out_valid,
    output reg [7:0] mask_out
);
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            mask_out <= 8'd0;
        end else begin
            out_valid <= in_valid;
            mask_out <= (in_valid && (&window_in)) ? 8'd255 : 8'd0;
        end
    end
endmodule
