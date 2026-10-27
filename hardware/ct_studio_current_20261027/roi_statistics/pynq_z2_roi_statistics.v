`timescale 1ns/1ps
// Optional ROI statistics wrapper for xc7z020clg400-1, Vivado 2018.2.
// Compile alongside pynq_z2_roi.v (Verilog, not HLS).
// Existing mask/count ABI is retained; extra ports need owner-managed AXI GPIO.
// Sums are exact integers. CPU computes mean=sum/count, var=sumsq/count-mean^2.
module pynq_z2_roi_statistics (
    (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET rst_n, FREQ_HZ 100000000" *)
    input wire clk,
    (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 rst_n RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
    input wire rst_n,
    input wire [31:0] geometry_in,bounds_x_in,bounds_y_in,thresholds_in,modes_in,
    output wire [31:0] status_out,count_out,identity_out,
    output reg signed [63:0] hu_sum_out,
    output reg [63:0] hu_sumsq_out,
    output reg [31:0] hu_minmax_out, // min[15:0],max[31:16], signed int16; zero for empty
    output reg statistics_valid_out,
    output wire [31:0] statistics_identity_out,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TDATA" *) input wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TKEEP" *) input wire [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TLAST" *) input wire s_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TVALID" *) input wire s_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TREADY" *) output wire s_axis_tready,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TDATA" *) output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TKEEP" *) output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TLAST" *) output wire m_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TVALID" *) output wire m_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TREADY" *) input wire m_axis_tready
);
    pynq_z2_roi roi(
        .clk(clk),.rst_n(rst_n),.geometry_in(geometry_in),.bounds_x_in(bounds_x_in),
        .bounds_y_in(bounds_y_in),.thresholds_in(thresholds_in),.modes_in(modes_in),
        .status_out(status_out),.count_out(count_out),.identity_out(identity_out),
        .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),.s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid),.s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),.m_axis_tkeep(m_axis_tkeep),.m_axis_tlast(m_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),.m_axis_tready(m_axis_tready)
    );
    // The base core has exactly one elastic output register. Capture matching HU
    // only on ingress acceptance; hold during backpressure. Old HU is consumed
    // before the nonblocking assignment on simultaneous egress/ingress.
    reg signed [15:0] aligned_hu;
    reg packet_open,has_value;
    reg signed [63:0] sum_work;
    reg [63:0] sumsq_work;
    reg signed [15:0] min_work,max_work;
    wire take_in=s_axis_tvalid&&s_axis_tready;
    wire take_out=m_axis_tvalid&&m_axis_tready;
    wire selected=m_axis_tdata[0];
    wire signed [31:0] square=aligned_hu*aligned_hu;
    wire signed [63:0] next_sum=sum_work+(selected?{{48{aligned_hu[15]}},aligned_hu}:64'd0);
    wire [63:0] next_sumsq=sumsq_work+(selected?{32'd0,square}:64'd0);
    wire signed [15:0] next_min=selected&&(!has_value||aligned_hu<min_work)?aligned_hu:min_work;
    wire signed [15:0] next_max=selected&&(!has_value||aligned_hu>max_work)?aligned_hu:max_work;
    assign statistics_identity_out=32'h52535431; // ASCII RST1, optional statistics ABI
    always @(posedge clk) begin
        if(!rst_n) begin
            aligned_hu<=0;packet_open<=0;has_value<=0;sum_work<=0;sumsq_work<=0;
            min_work<=0;max_work<=0;hu_sum_out<=0;hu_sumsq_out<=0;hu_minmax_out<=0;statistics_valid_out<=0;
        end else begin
            if(take_in) begin
                aligned_hu<=s_axis_tdata[15:0];
                if(!packet_open) begin
                    packet_open<=1;has_value<=0;sum_work<=0;sumsq_work<=0;min_work<=0;max_work<=0;
                    statistics_valid_out<=0;
                end
            end
            if(take_out) begin
                sum_work<=next_sum;sumsq_work<=next_sumsq;min_work<=next_min;max_work<=next_max;
                if(selected)has_value<=1;
                if(m_axis_tlast)begin
                    packet_open<=0;hu_sum_out<=next_sum;hu_sumsq_out<=next_sumsq;
                    hu_minmax_out<=(has_value||selected)?{next_max,next_min}:32'd0;
                    statistics_valid_out<=!(|status_out[5:2]);
                end
            end
        end
    end
endmodule
