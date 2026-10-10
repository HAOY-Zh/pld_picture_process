`timescale 1ns/1ps

// CT HU 阈值分割核心。
// HU 大于等于 threshold 时输出 255，否则输出 0。
// 输入有效后延迟 1 个时钟输出结果。
module threshold (
    input                   clk,
    input                   rst_n,
    input                   in_valid,
    input  signed [15:0]    hu_in,
    input  signed [15:0]    threshold_in,
    output reg              out_valid,
    output reg        [7:0] mask_out
);

    reg signed [15:0] hu_reg;
    reg signed [15:0] threshold_reg;
    reg               valid_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            hu_reg        <= 16'sd0;
            threshold_reg <= 16'sd0;
            valid_reg     <= 1'b0;
            out_valid     <= 1'b0;
            mask_out      <= 8'd0;
        end
        else begin
            out_valid <= valid_reg;

            if (valid_reg) begin
                if (hu_reg >= threshold_reg)
                    mask_out <= 8'd255;
                else
                    mask_out <= 8'd0;
            end
            else begin
                mask_out <= 8'd0;
            end

            valid_reg <= in_valid;
            if (in_valid) begin
                hu_reg        <= hu_in;
                threshold_reg <= threshold_in;
            end
        end
    end

endmodule
