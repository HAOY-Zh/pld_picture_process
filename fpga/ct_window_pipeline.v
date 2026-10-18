`timescale 1ns/1ps
// Exact integer window mapping, one pixel/cycle when unstalled. Eight quotient
// bits are enough because only the interior 0..254 range enters the divider.
module ct_window_pipeline (
    input wire clk, rst_n,
    input wire signed [15:0] wl_in,
    input wire [15:0] ww_in,
    input wire [31:0] s_axis_tdata,
    input wire [3:0] s_axis_tkeep,
    input wire s_axis_tlast, s_axis_tvalid,
    output wire s_axis_tready,
    output wire [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tlast, m_axis_tvalid,
    input wire m_axis_tready
);
    wire signed [17:0] hu = {{2{s_axis_tdata[15]}},s_axis_tdata[15:0]};
    wire signed [17:0] wl = {{2{wl_in[15]}},wl_in};
    wire signed [17:0] half_width = {3'b0,ww_in[15:1]};
    wire signed [17:0] low = wl-half_width;
    wire signed [17:0] high = wl+half_width;
    wire signed [17:0] delta = hu-low;
    reg [24:0] remainder [0:8];
    reg [15:0] divisor [0:8];
    reg [7:0] quotient [0:8];
    reg [7:0] saturated [0:8];
    reg [8:0] valid=0, last=0, clamp=0;
    wire advance = !valid[8] || m_axis_tready;
    assign s_axis_tready = rst_n && advance;
    assign m_axis_tdata = {24'd0,clamp[8] ? saturated[8] : quotient[8]};
    assign m_axis_tkeep = 4'hf;
    assign m_axis_tlast = last[8];
    assign m_axis_tvalid = valid[8];
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid<=0;last<=0;clamp<=0;
            for(i=0;i<=8;i=i+1) begin
                remainder[i]<=0;divisor[i]<=1;quotient[i]<=0;saturated[i]<=0;
            end
        end else if (advance) begin
            valid[0]<=s_axis_tvalid;last[0]<=s_axis_tlast;
            remainder[0]<=delta*25'd255;divisor[0]<=ww_in;quotient[0]<=0;
            clamp[0]<=(ww_in==0 || hu<=low || hu>=high);
            saturated[0]<=(ww_in==0 || hu<=low) ? 0 : 255;
            for(i=1;i<=8;i=i+1) begin
                valid[i]<=valid[i-1];last[i]<=last[i-1];clamp[i]<=clamp[i-1];
                divisor[i]<=divisor[i-1];saturated[i]<=saturated[i-1];
                if (remainder[i-1] >= ({9'd0,divisor[i-1]} << (8-i))) begin
                    remainder[i]<=remainder[i-1]-({9'd0,divisor[i-1]} << (8-i));
                    quotient[i]<=quotient[i-1] | (8'b1 << (8-i));
                end else begin
                    remainder[i]<=remainder[i-1];quotient[i]<=quotient[i-1];
                end
            end
        end
    end
endmodule
