`timescale 1ns/1ps

// AXI4-Stream wrapper for range thresholding.
// thresholds_in[15:0]  = signed lower HU threshold
// thresholds_in[31:16] = signed upper HU threshold
module threshold_range_axis (
    input                   clk,
    input                   rst_n,

    input         [31:0]    thresholds_in,

    input         [31:0]    s_axis_tdata,
    input         [3:0]     s_axis_tkeep,
    input                   s_axis_tlast,
    input                   s_axis_tvalid,
    output                  s_axis_tready,

    output        [31:0]    m_axis_tdata,
    output        [3:0]     m_axis_tkeep,
    output                  m_axis_tlast,
    output                  m_axis_tvalid,
    input                   m_axis_tready
);

    wire                  core_in_valid;
    wire                  core_out_valid;
    wire [7:0]            core_mask_out;
    reg                   core_pending;
    reg                   last_pending;

    reg [31:0]            out_data_reg;
    reg [3:0]             out_keep_reg;
    reg                   out_last_reg;
    reg                   out_valid_reg;

    assign s_axis_tready = (!core_pending) &&
                           ((!out_valid_reg) || m_axis_tready);
    assign core_in_valid = s_axis_tvalid && s_axis_tready;

    assign m_axis_tdata  = out_data_reg;
    assign m_axis_tkeep  = out_keep_reg;
    assign m_axis_tlast  = out_last_reg;
    assign m_axis_tvalid = out_valid_reg;

    threshold_range u_threshold_range (
        .clk               (clk),
        .rst_n             (rst_n),
        .in_valid          (core_in_valid),
        .hu_in             ($signed(s_axis_tdata[15:0])),
        .low_threshold_in  ($signed(thresholds_in[15:0])),
        .high_threshold_in ($signed(thresholds_in[31:16])),
        .out_valid         (core_out_valid),
        .mask_out          (core_mask_out)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            core_pending  <= 1'b0;
            last_pending  <= 1'b0;
            out_data_reg  <= 32'd0;
            out_keep_reg  <= 4'd0;
            out_last_reg  <= 1'b0;
            out_valid_reg <= 1'b0;
        end
        else begin
            if (out_valid_reg && m_axis_tready) begin
                out_valid_reg <= 1'b0;
                out_last_reg  <= 1'b0;
            end

            if (core_in_valid) begin
                core_pending <= 1'b1;
                last_pending <= s_axis_tlast;
            end

            if (core_out_valid) begin
                core_pending  <= 1'b0;
                out_data_reg  <= {24'd0, core_mask_out};
                out_keep_reg  <= 4'b1111;
                out_last_reg  <= last_pending;
                out_valid_reg <= 1'b1;
            end
        end
    end

endmodule
