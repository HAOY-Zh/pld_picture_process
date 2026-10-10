`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// window_level_axis
// -----------------------------------------------------------------------------
// 功能：
//   将已经验证过的 window_level 核心模块包装成 AXI4-Stream 接口，便于后续
//   连接 AXI DMA。
//
// 数据格式：
//   1. 输入 AXI-Stream 为 32 bit，每个 beat 的低 16 bit 保存一个有符号 HU 值；
//      高 16 bit 暂不使用。
//   2. 输出 AXI-Stream 也为 32 bit，低 8 bit 保存窗宽窗位处理后的灰度值；
//      高 24 bit 固定为 0。
//   3. 使用 TLAST 标记一帧数据的最后一个像素。对于当前 32x32 测试，最后
//      一个像素的 TLAST 应为 1。
//
// 时序：
//   window_level 核心本身为 1 拍延迟；本包装模块再使用一个输出寄存器，
//   因此从输入握手成功到输出握手数据有效，整体延迟为 2 个时钟周期。
//
// 握手策略：
//   为了保证不丢数据，当前版本只允许一个像素处于核心计算过程中，并且
//   只允许一个像素停留在输出寄存器中。因此吞吐率不是最终优化版本的最高
//   吞吐率，但控制简单，适合第一版功能验证。
// -----------------------------------------------------------------------------
module window_level_axis (
    input                   clk,
    input                   rst_n,

    // 窗宽窗位参数。参数会随每个有效输入像素送入核心模块。
    input  signed [15:0]    wl_in,
    input         [15:0]    ww_in,

    // AXI4-Stream 输入：低 16 bit 为有符号 HU 像素。
    input         [31:0]    s_axis_tdata,
    input         [3:0]     s_axis_tkeep,
    input                   s_axis_tlast,
    input                   s_axis_tvalid,
    output                  s_axis_tready,

    // AXI4-Stream 输出：低 8 bit 为 0~255 灰度值。
    output        [31:0]    m_axis_tdata,
    output        [3:0]     m_axis_tkeep,
    output                  m_axis_tlast,
    output                  m_axis_tvalid,
    input                   m_axis_tready
);

    // 现有窗宽窗位核心的接口信号。
    wire                  core_in_valid;
    wire                  core_out_valid;
    wire [7:0]            core_gray_out;

    // core_pending 表示已有一个输入像素送入核心，正在等待核心输出。
    reg                   core_pending;

    // 记录送入核心的那个像素是否为一帧的最后一个像素。
    reg                   last_pending;

    // AXI 输出寄存器。输出有效但下游未 ready 时，数据必须保持不变。
    reg [31:0]            out_data_reg;
    reg [3:0]             out_keep_reg;
    reg                   out_last_reg;
    reg                   out_valid_reg;

    // 当前版本采用保守握手：
    //   1. 核心没有未完成计算；
    //   2. 输出寄存器为空，或者本拍输出可以被下游接收。
    // 这样可以避免核心结果覆盖输出寄存器。
    assign s_axis_tready = (!core_pending) &&
                           ((!out_valid_reg) || m_axis_tready);

    assign core_in_valid = s_axis_tvalid && s_axis_tready;

    // AXI 输出端口直接连接输出寄存器。
    assign m_axis_tdata  = out_data_reg;
    assign m_axis_tkeep  = out_keep_reg;
    assign m_axis_tlast  = out_last_reg;
    assign m_axis_tvalid = out_valid_reg;

    // 调用已经通过基础数值验证的窗宽窗位核心。
    window_level u_window_level (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (core_in_valid),
        .hu_in     ($signed(s_axis_tdata[15:0])),
        .wl_in     (wl_in),
        .ww_in     (ww_in),
        .out_valid (core_out_valid),
        .gray_out  (core_gray_out)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            core_pending <= 1'b0;
            last_pending <= 1'b0;

            out_data_reg  <= 32'd0;
            out_keep_reg  <= 4'd0;
            out_last_reg  <= 1'b0;
            out_valid_reg <= 1'b0;
        end
        else begin
            // 下游完成一次输出握手后，清除输出有效标志。
            if (out_valid_reg && m_axis_tready) begin
                out_valid_reg <= 1'b0;
                out_last_reg  <= 1'b0;
            end

            // 保存输入像素对应的 TLAST。核心输出有效时，将它与灰度结果
            // 一起送到 AXI 输出端，保证帧尾标志和像素严格对齐。
            if (core_in_valid) begin
                core_pending <= 1'b1;
                last_pending <= s_axis_tlast;
            end

            // window_level 核心输出一个有效结果后，写入 AXI 输出寄存器。
            if (core_out_valid) begin
                core_pending <= 1'b0;

                out_data_reg  <= {24'd0, core_gray_out};
                out_keep_reg  <= 4'b1111;
                out_last_reg  <= last_pending;
                out_valid_reg <= 1'b1;
            end
        end
    end

endmodule
