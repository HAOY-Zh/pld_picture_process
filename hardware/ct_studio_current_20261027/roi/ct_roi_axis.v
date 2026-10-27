`timescale 1ns/1ps
// ABI ROI1. 32-bit AXI Stream, one signed-HU pixel and lung bit per beat.
// Input [15:0]=int16 HU, [16]=lung 0/1, [31:17]=0.
// Output [0]=filtered mask, [31:1]=0; TKEEP=F; TLAST ends the frame.
// Config is captured at FIRST input handshake, immutable for that packet.
// One elastic output register: accepts one pixel/cycle when downstream ready.
module ct_roi_axis #(
    parameter integer MAX_WIDTH=512,
    parameter integer MAX_HEIGHT=512
) (
    (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET rst_n, FREQ_HZ 100000000" *)
    input wire clk,
    (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 rst_n RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
    input wire rst_n,
    input wire [31:0] geometry_in,     // width[15:0], height[31:16]
    input wire [31:0] bounds_x_in,     // x0[15:0], x1[31:16], half-open
    input wire [31:0] bounds_y_in,     // y0[15:0], y1[31:16], half-open
    input wire [31:0] thresholds_in,   // low/high signed int16, inclusive
    input wire [31:0] modes_in,        // bit0 use_hu, bit1 use_lung; rest zero
    output wire [31:0] status_out,
    output reg [31:0] count_out,
    output wire [31:0] identity_out,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TDATA" *) input wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TKEEP" *) input wire [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TLAST" *) input wire s_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TVALID" *) input wire s_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TREADY" *) output wire s_axis_tready,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TDATA" *) output reg [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TKEEP" *) output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TLAST" *) output reg m_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TVALID" *) output reg m_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TREADY" *) input wire m_axis_tready
);
    reg active,end_pending,done;
    reg [31:0] geometry,bounds_x,bounds_y,thresholds,modes;
    reg [15:0] x,y;
    reg [31:0] working_count;
    reg cfg_error,keep_error,last_error,reserved_error;
    wire [31:0] g=active?geometry:geometry_in;
    wire [31:0] bx=active?bounds_x:bounds_x_in;
    wire [31:0] by=active?bounds_y:bounds_y_in;
    wire [31:0] th=active?thresholds:thresholds_in;
    wire [31:0] mo=active?modes:modes_in;
    wire [15:0] cx=active?x:16'd0, cy=active?y:16'd0;
    wire signed [15:0] hu=$signed(s_axis_tdata[15:0]);
    wire signed [15:0] low=$signed(th[15:0]),high=$signed(th[31:16]);
    wire config_ok=(g[15:0]>=1 && g[15:0]<=MAX_WIDTH && g[31:16]>=1 && g[31:16]<=MAX_HEIGHT)
        && (bx[15:0]<=bx[31:16] && bx[31:16]<=g[15:0])
        && (by[15:0]<=by[31:16] && by[31:16]<=g[31:16])
        && (!mo[0] || low<=high) && (mo[31:2]==0);
    wire inside_roi=(cx>=bx[15:0] && cx<bx[31:16] && cy>=by[15:0] && cy<by[31:16]);
    // ROI is bounded by geometry. Thus inside_roi already implies in-frame;
    // no width*height multiplier or global pixel counter is needed per pixel.
    wire selected=config_ok && inside_roi
        && (!mo[0] || (hu>=low && hu<=high)) && (!mo[1] || s_axis_tdata[16]);
    wire expected_last=config_ok && cx==g[15:0]-1 && cy==g[31:16]-1;
    wire [31:0] next_count=(active?working_count:32'd0)+(selected?32'd1:32'd0);
    assign s_axis_tready=(!end_pending)&&(!m_axis_tvalid||m_axis_tready);
    assign m_axis_tkeep=4'hf;
    assign identity_out=32'h524f4931; // ASCII ROI1
    // [31:24]=ABI major=1; bit0 busy, bit1 done; bits2..5 sticky errors.
    assign status_out={8'h01,18'd0,reserved_error,last_error,keep_error,cfg_error,done,active};
    always @(posedge clk) begin
        if(!rst_n) begin
            active<=0; end_pending<=0; done<=0;
            geometry<=0; bounds_x<=0; bounds_y<=0; thresholds<=0; modes<=0;
            x<=0; y<=0; working_count<=0; count_out<=0;
            cfg_error<=0; keep_error<=0; last_error<=0; reserved_error<=0;
            m_axis_tdata<=0; m_axis_tlast<=0; m_axis_tvalid<=0;
        end else begin
            if(m_axis_tvalid&&m_axis_tready) begin
                m_axis_tvalid<=0;
                if(m_axis_tlast) begin
                    active<=0; end_pending<=0; done<=1;
                    count_out<=working_count;
                end
            end
            if(s_axis_tvalid&&s_axis_tready) begin
                if(!active) begin
                    geometry<=geometry_in; bounds_x<=bounds_x_in; bounds_y<=bounds_y_in;
                    thresholds<=thresholds_in; modes<=modes_in;
                    active<=1; done<=0;
                    cfg_error<=!config_ok;
                    keep_error<=(s_axis_tkeep!=4'hf);
                    last_error<=(s_axis_tlast!=expected_last);
                    reserved_error<=(s_axis_tdata[31:17]!=0);
                end else begin
                    keep_error<=keep_error||(s_axis_tkeep!=4'hf);
                    last_error<=last_error||(s_axis_tlast!=expected_last);
                    reserved_error<=reserved_error||(s_axis_tdata[31:17]!=0);
                end
                working_count<=next_count;
                if(cx+1>=g[15:0]) begin x<=0; y<=cy+1; end
                else begin x<=cx+1; y<=cy; end
                m_axis_tdata<={31'd0,selected};
                m_axis_tlast<=s_axis_tlast; m_axis_tvalid<=1;
                if(s_axis_tlast) end_pending<=1;
            end
        end
    end
endmodule
