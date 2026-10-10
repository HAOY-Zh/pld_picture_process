`timescale 1ns/1ps

// CT HU range-segmentation core.
// A pixel is foreground when low_threshold <= HU <= high_threshold.
module threshold_range (
    input                   clk,
    input                   rst_n,
    input                   in_valid,
    input  signed [15:0]    hu_in,
    input  signed [15:0]    low_threshold_in,
    input  signed [15:0]    high_threshold_in,
    output reg              out_valid,
    output reg        [7:0] mask_out
);

    reg signed [15:0] hu_reg;
    reg signed [15:0] low_reg;
    reg signed [15:0] high_reg;
    reg               valid_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            hu_reg        <= 16'sd0;
            low_reg       <= 16'sd0;
            high_reg      <= 16'sd0;
            valid_reg     <= 1'b0;
            out_valid     <= 1'b0;
            mask_out      <= 8'd0;
        end
        else begin
            out_valid <= valid_reg;

            if (valid_reg && (hu_reg >= low_reg) && (hu_reg <= high_reg))
                mask_out <= 8'd255;
            else
                mask_out <= 8'd0;

            valid_reg <= in_valid;
            if (in_valid) begin
                hu_reg   <= hu_in;
                low_reg  <= low_threshold_in;
                high_reg <= high_threshold_in;
            end
        end
    end

endmodule
