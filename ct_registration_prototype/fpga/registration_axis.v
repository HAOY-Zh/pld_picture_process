`timescale 1ns/1ps
// Sequential frame-buffered warp + overlap SSD evaluator. Source RTL prototype.
// Not a one-pixel/clock architecture; no synthesis/board claims.
// Input N words: [7:0]=fixed, [15:8]=moving, upper bits ignored.
// Output N words: [7:0]=warped, [15:8]=fixed, [16]=valid source overlap.
// Then 4 trailer words: SSD low32, SSD high32, valid count, protocol flags.
// Input TLAST on N-1; output TLAST on N+3. TKEEP must be 4'hf.
// Coefficients signed Q16.16, latched on first accepted input word per frame:
// sx = a*x+b*y+c; sy = d*x+e*y+f. They map OUTPUT/FIXED to INPUT/MOVING.
// Caller must prevent signed 32-bit coordinate accumulator overflow.
module registration_axis #(
    parameter integer IMAGE_WIDTH = 128,
    parameter integer IMAGE_HEIGHT = 128
) (
    input wire clk, input wire rst_n,
    input wire signed [31:0] a_in, b_in, c_in, d_in, e_in, f_in,
    input wire [31:0] s_axis_tdata,
    input wire [3:0] s_axis_tkeep,
    input wire s_axis_tlast, s_axis_tvalid,
    output wire s_axis_tready,
    output reg [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output reg m_axis_tlast, m_axis_tvalid,
    input wire m_axis_tready
);
    localparam integer N = IMAGE_WIDTH*IMAGE_HEIGHT;
    localparam signed [31:0] MAX_X_Q = (IMAGE_WIDTH-1)*65536;
    localparam signed [31:0] MAX_Y_Q = (IMAGE_HEIGHT-1)*65536;
    localparam LOAD=0, INIT=1, ADDRESS=2, READ01=3, READ10=4,
               READ11=5, INTERPOLATE=6, SCORE=7, EMIT_PIXEL=8,
               PREP_TRAILER=9, EMIT_TRAILER=10, CAPTURE11=11;
    reg [3:0] state;
    // One sequential read per memory per cycle. RAM mapping must be checked
    // in the actual Vivado synthesis report. Do not reset the arrays.
    (* ram_style = "block" *) reg [7:0] fixed_mem [0:N-1];
    (* ram_style = "block" *) reg [7:0] moving_mem [0:N-1];
    reg [31:0] load_index, pixel_index, px, py;
    reg signed [31:0] a,b,c,d,e,f, source_x, source_y, row_x, row_y;
    reg [31:0] addr01, addr10, addr11;
    integer ix,iy,jx,jy;
    reg [7:0] p00,p01,p10,p11,fraction_x,fraction_y,reference_pixel,warped_pixel;
    reg [7:0] moving_read;
    reg pixel_valid;
    reg [63:0] ssd;
    reg [31:0] valid_count, flags;
    reg [1:0] trailer_index;
    wire [8:0] wx0=9'd256-{1'b0,fraction_x};
    wire [8:0] wy0=9'd256-{1'b0,fraction_y};
    wire [16:0] horizontal0=p00*wx0+p01*{1'b0,fraction_x};
    wire [16:0] horizontal1=p10*wx0+p11*{1'b0,fraction_x};
    wire [31:0] bilinear=horizontal0*wy0+horizontal1*{1'b0,fraction_y}+32'd32768;
    wire signed [8:0] difference=$signed({1'b0,reference_pixel})-$signed({1'b0,warped_pixel});
    wire [17:0] square=difference*difference;
    wire source_inside=(source_x>=0 && source_y>=0 && source_x<=MAX_X_Q && source_y<=MAX_Y_Q);
    wire [31:0] first_address=source_inside ? (($unsigned(source_y)>>16)*IMAGE_WIDTH+($unsigned(source_x)>>16)) : 32'd0;
    wire [31:0] moving_read_address=(state==ADDRESS)?first_address:
        (state==READ01)?addr01:(state==READ10)?addr10:(state==READ11)?addr11:32'd0;
    assign s_axis_tready=(state==LOAD);
    assign m_axis_tkeep=4'hf;

    // Explicit single synchronous read port, shared by all four samples.
    // No reset on the RAM output register, preserving block-RAM inference.
    always @(posedge clk) moving_read<=moving_mem[moving_read_address];

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=LOAD; load_index<=0; pixel_index<=0; px<=0; py<=0;
            a<=0; b<=0; c<=0; d<=0; e<=0; f<=0;
            source_x<=0; source_y<=0; row_x<=0; row_y<=0;
            addr01<=0; addr10<=0; addr11<=0;
            p00<=0; p01<=0; p10<=0; p11<=0;
            fraction_x<=0; fraction_y<=0; reference_pixel<=0;
            warped_pixel<=0; pixel_valid<=0;
            ssd<=0; valid_count<=0; flags<=0; trailer_index<=0;
            m_axis_tdata<=0; m_axis_tlast<=0; m_axis_tvalid<=0;
        end else begin
            case (state)
                LOAD: if (s_axis_tvalid && s_axis_tready) begin
                    fixed_mem[load_index]<=s_axis_tdata[7:0];
                    moving_mem[load_index]<=s_axis_tdata[15:8];
                    if (load_index==0) begin
                        a<=a_in; b<=b_in; c<=c_in; d<=d_in; e<=e_in; f<=f_in;
                        flags<={30'd0,(s_axis_tlast!=(N==1)),(s_axis_tkeep!=4'hf)};
                    end else begin
                        flags<=flags | ((s_axis_tkeep!=4'hf)?32'd1:32'd0)
                                     | ((s_axis_tlast!=(load_index==N-1))?32'd2:32'd0);
                    end
                    if (load_index==N-1) begin load_index<=0; state<=INIT; end
                    else load_index<=load_index+1;
                end
                INIT: begin
                    px<=0; py<=0; pixel_index<=0;
                    source_x<=c; source_y<=f; row_x<=c; row_y<=f;
                    ssd<=0; valid_count<=0; trailer_index<=0; state<=ADDRESS;
                end
                ADDRESS: begin
                    reference_pixel<=fixed_mem[pixel_index];
                    if (!source_inside) begin
                        pixel_valid<=0; warped_pixel<=0; state<=SCORE;
                    end else begin
                        pixel_valid<=1;
                        ix=$signed(source_x)>>>16; iy=$signed(source_y)>>>16;
                        jx=(ix+1<IMAGE_WIDTH)?ix+1:ix;
                        jy=(iy+1<IMAGE_HEIGHT)?iy+1:iy;
                        fraction_x<=source_x[15:8]; fraction_y<=source_y[15:8];
                        addr01<=iy*IMAGE_WIDTH+jx;
                        addr10<=jy*IMAGE_WIDTH+ix;
                        addr11<=jy*IMAGE_WIDTH+jx;
                        state<=READ01;
                    end
                end
                READ01: begin p00<=moving_read; state<=READ10; end
                READ10: begin p01<=moving_read; state<=READ11; end
                READ11: begin p10<=moving_read; state<=CAPTURE11; end
                CAPTURE11: begin p11<=moving_read; state<=INTERPOLATE; end
                INTERPOLATE: begin warped_pixel<=bilinear[23:16]; state<=SCORE; end
                SCORE: begin
                    if (pixel_valid) begin
                        ssd<=ssd+{{46{1'b0}},square}; valid_count<=valid_count+1;
                    end
                    m_axis_tdata<={15'd0,pixel_valid,reference_pixel,warped_pixel};
                    m_axis_tlast<=0; m_axis_tvalid<=1; state<=EMIT_PIXEL;
                end
                EMIT_PIXEL: if (m_axis_tvalid && m_axis_tready) begin
                    m_axis_tvalid<=0;
                    if (pixel_index==N-1) state<=PREP_TRAILER;
                    else begin
                        pixel_index<=pixel_index+1;
                        if (px==IMAGE_WIDTH-1) begin
                            px<=0; py<=py+1;
                            row_x<=row_x+b; row_y<=row_y+e;
                            source_x<=row_x+b; source_y<=row_y+e;
                        end else begin
                            px<=px+1; source_x<=source_x+a; source_y<=source_y+d;
                        end
                        state<=ADDRESS;
                    end
                end
                PREP_TRAILER: begin
                    case (trailer_index)
                        0: m_axis_tdata<=ssd[31:0];
                        1: m_axis_tdata<=ssd[63:32];
                        2: m_axis_tdata<=valid_count;
                        3: m_axis_tdata<=flags;
                    endcase
                    m_axis_tlast<=(trailer_index==3);
                    m_axis_tvalid<=1; state<=EMIT_TRAILER;
                end
                EMIT_TRAILER: if (m_axis_tvalid && m_axis_tready) begin
                    m_axis_tvalid<=0; m_axis_tlast<=0;
                    if (trailer_index==3) state<=LOAD;
                    else begin trailer_index<=trailer_index+1; state<=PREP_TRAILER; end
                end
                default: state<=LOAD;
            endcase
        end
    end
endmodule
