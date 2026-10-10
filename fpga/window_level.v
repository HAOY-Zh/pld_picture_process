`timescale 1ns/1ps

// CT 窗宽窗位处理模块

// 连续输入一个 HU 像素，经过 1 个时钟周期后输出 8 bit 灰度。

// 算法为 low=WL-WW/2，high=WL+WW/2。

// HU<=low 输出 0，HU>=high 输出 255，中间区域输出 (HU-low)*255/WW。

module window_level (
    input                   clk,
    input                   rst_n,
    input                   in_valid,
    input  signed [15:0]    hu_in,
    input  signed [15:0]    wl_in,
    input         [15:0]    ww_in,
    output reg              out_valid,
    output reg        [7:0] gray_out
);

// 输入寄存器，在 in_valid=1 的上升沿锁存像素和对应参数。

reg signed [15:0] hu_reg;
reg signed [15:0] wl_reg;
reg        [15:0] ww_reg;
reg               valid_reg;

// hu/wl 为 signed，ww 为 unsigned，先显式扩展后再参与计算。

// 使用 18 bit 可覆盖完整输入范围，例如 low 可能小于 -65536。

wire signed [17:0] hu_ext  = {{2{hu_reg[15]}}, hu_reg};
wire signed [17:0] wl_ext  = {{2{wl_reg[15]}}, wl_reg};
wire signed [17:0] ww_ext  = {2'b00, ww_reg};
wire signed [17:0] ww_half = ww_ext >>> 1;

// low 和 high 使用 signed 运算，避免负窗位被错误解释为无符号数。

wire signed [17:0] low  = wl_ext - ww_half;
wire signed [17:0] high = wl_ext + ww_half;

// 中间差值为正数，乘以 255 后使用 25 bit 保存乘法结果。

wire signed [17:0] delta = hu_ext - low;
wire        [24:0] product = delta * 25'd255;

// valid_reg 表示上一拍已经锁存了有效输入。

// 因此 out_valid 和 gray_out 相对于输入 valid 延迟 1 个时钟周期。

always @(posedge clk) begin
    if (!rst_n) begin
        hu_reg    <= 16'sd0;
        wl_reg    <= 16'sd0;
        ww_reg    <= 16'd0;
        valid_reg <= 1'b0;
        out_valid <= 1'b0;
        gray_out  <= 8'd0;
    end
    else begin
        out_valid <= valid_reg;

        if (valid_reg) begin
            // WW 应为正数，此处保护除零异常输入。

            if (ww_reg == 16'd0)
                gray_out <= 8'd0;
            else if (hu_ext <= low)
                gray_out <= 8'd0;
            else if (hu_ext >= high)
                gray_out <= 8'd255;
            else begin
                // 整数除法向下取整，结果为 0 到 255。

                gray_out <= product / ww_reg;
            end
        end
        else begin
            gray_out <= 8'd0;
        end

        // 锁存当前输入，供下一拍计算。

        valid_reg <= in_valid;
        if (in_valid) begin
            hu_reg <= hu_in;
            wl_reg <= wl_in;
            ww_reg <= ww_in;
        end
    end
end

endmodule
