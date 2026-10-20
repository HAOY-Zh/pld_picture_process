`timescale 1ns/1ps
// Tile-based signed-HU accelerator. PS supplies overlapping real image tiles.
// cfg[3:0]: 0 bypass,1 Gaussian5,2 bilateral5,3 joint7/guide5,
// 4 joint7/guide3,5 self-guided3,6 frozen integer CNN,7 bilateral7.
// cfg[10:4]: active tile width 1..64; cfg[17:11]: height 1..64.
// cfg[31] reset. Parameters latch at first input handshake. TLAST=tile end.
// Optional CNN compact protocol: cfg[26]=1; [19:18],[21:20],[23:22],
// [25:24] are left/right/top/bottom trims, each 0 at true frame edges or 3
// at internal tile edges. Return only the untrimmed core and omit unused
// intermediate border computations. cfg[26]=0 preserves the original protocol.
// Status: bit0 TKEEP,bit1 TLAST,bit2 mode,bit3 dimensions,bit4 divide-zero.
module ct_denoise_tile_axis(
    (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET rst_n" *)
    (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK" *) input clk,
    (* X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 rst_n RST" *) input rst_n,
    input [31:0] cfg, output reg [31:0] status,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TDATA" *) input [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TKEEP" *) input [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TLAST" *) input s_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TVALID" *) input s_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 s_axis TREADY" *) output s_axis_tready,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TDATA" *) output [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TKEEP" *) output [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TLAST" *) output m_axis_tlast,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TVALID" *) output m_axis_tvalid,
    (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 m_axis TREADY" *) input m_axis_tready
);
    `include "ct_denoise_roms.vh"
    localparam LOAD=0,BEGIN_MODE=1,PIXEL_BEGIN=2,TAP_ADDR=3,TAP_WAIT=4,TAP_CAPTURE=5,
        GAUSS_ACC=6,GAUSS_WRITE=7,PIXEL_NEXT=8,CENTER_ADDR=9,CENTER_WAIT=10,CENTER_CAPTURE=11,
        RANGE_PRODUCT=12,RANGE_ACC=13,DIV_LAUNCH=14,DIV_WAIT=15,RANGE_WRITE=16,
        GF_SQUARE=17,GF_ACC=18,GF_VARIANCE=19,GF_DIV_PREP=20,GF_COEFF_WRITE=21,
        GF_APPLY_ACC=22,GF_CENTER_ADDR=23,GF_CENTER_WAIT=24,GF_APPLY_PREP=25,GF_OUTPUT_WRITE=26,
        CNN_BEGIN=27,CNN_RUN=28,CNN_WRITE=29,OUT_ADDR=30,OUT_WAIT=31,OUT_CAPTURE=32,OUT_SEND=33,
        CNN_QUANT=34,GF_VARIANCE_SUB=35,GF_DENOM=36,GF_SCALE=37,GF_FINAL_ADD=38,
        GF_ABS=39,GF_DIV_NUM=40,GF_COEFF_PREP=41,GF_COEFF_MULT=42;
    reg [5:0] state;
    reg [3:0] mode;
    reg [6:0] width,height;
    reg compact_cnn;
    reg [1:0] trim_l,trim_r,trim_t,trim_b;
    reg [5:0] load_x,load_y,px,py;
    reg [12:0] load_count;
    reg [11:0] read_addr;
    reg [5:0] tap;
    reg [2:0] radius;
    reg [1:0] pass;
    reg signed [15:0] output_hu;
    wire reset_active=!rst_n || cfg[31];
    assign s_axis_tready=state==LOAD && !reset_active;
    assign m_axis_tvalid=state==OUT_SEND && !reset_active;
    assign m_axis_tdata={16'b0,output_hu};
    assign m_axis_tkeep=4'hf;
    assign m_axis_tlast=px==width-1-trim_r && py==height-1-trim_b;
    wire [11:0] center_addr={py,px};
    wire first_input=load_count==0;
    wire [6:0] cfg_width=(cfg[10:4]>=1 && cfg[10:4]<=64) ? cfg[10:4] : 7'd1;
    wire [6:0] cfg_height=(cfg[17:11]>=1 && cfg[17:11]<=64) ? cfg[17:11] : 7'd1;
    // Coordinate end detection avoids an external GPIO -> clamp -> 7x7
    // multiplier -> counter comparator path on the first input handshake.
    wire incoming_last=first_input ? cfg_width==1 && cfg_height==1 :
        load_x==width-1 && load_y==height-1;
    wire incoming_crop_valid=cfg[3:0]==6 &&
        (cfg[19:18]==0 || cfg[19:18]==3) && (cfg[21:20]==0 || cfg[21:20]==3) &&
        (cfg[23:22]==0 || cfg[23:22]==3) && (cfg[25:24]==0 || cfg[25:24]==3) &&
        cfg_width>({1'b0,cfg[19:18]}+{1'b0,cfg[21:20]}) &&
        cfg_height>({1'b0,cfg[23:22]}+{1'b0,cfg[25:24]});

    function [11:0] address;
        input [5:0] x,y; input integer dx,dy; integer xx,yy;
        begin
            xx=x+dx;yy=y+dy;
            if(xx<0)xx=0;else if(xx>=width)xx=width-1;
            if(yy<0)yy=0;else if(yy>=height)yy=height-1;
            address=yy*64+xx;
        end
    endfunction
    function integer tap_dx;
        input [5:0] t;input [2:0] r;
        begin case(r) 1:tap_dx=t%3-1;2:tap_dx=t%5-2;default:tap_dx=t%7-3;endcase end
    endfunction
    function integer tap_dy;
        input [5:0] t;input [2:0] r;
        begin case(r) 1:tap_dy=t/3-1;2:tap_dy=t/5-2;default:tap_dy=t/7-3;endcase end
    endfunction
    function [3:0] binomial;
        input integer pos;input [2:0] r;
        begin
            if(r==1)case(pos)0:binomial=1;1:binomial=2;default:binomial=1;endcase
            else case(pos)0:binomial=1;1:binomial=4;2:binomial=6;3:binomial=4;default:binomial=1;endcase
        end
    endfunction
    function signed [15:0] saturate64;
        input signed [63:0] value;
        begin if(value>32767)saturate64=32767;else if(value< -32768)saturate64=-32768;else saturate64=value[15:0];end
    endfunction
    function signed [15:0] cnn_quantize;
        input signed [31:0] value; input [1:0] layer;
        reg signed [32:0] widened,rounded,q;
        begin
            // Signed rounding: adding half-1 for negative values implements
            // ties away from zero without two long 64-bit negation chains.
            widened={value[31],value};
            case(layer)
                0:begin rounded=widened+((33'sd1<<(CNN_SHIFT_0-1))-(value[31] ? 33'sd1 : 33'sd0));q=rounded>>>CNN_SHIFT_0;end
                1:begin rounded=widened+((33'sd1<<(CNN_SHIFT_1-1))-(value[31] ? 33'sd1 : 33'sd0));q=rounded>>>CNN_SHIFT_1;end
                default:begin rounded=widened+((33'sd1<<(CNN_SHIFT_2-1))-(value[31] ? 33'sd1 : 33'sd0));q=rounded>>>CNN_SHIFT_2;end
            endcase
            if(layer<2 && q<0)q=0;
            if(q>32767)cnn_quantize=32767;
            else if(q< -32768)cnn_quantize=-32768;
            else cnn_quantize=q[15:0];
        end
    endfunction

    (* ram_style="block" *) reg [15:0] source_mem[0:4095];
    (* ram_style="block" *) reg [15:0] guide_mem[0:4095];
    (* ram_style="block" *) reg [15:0] result_mem[0:4095];
    (* ram_style="block" *) reg [15:0] coeff_mem[0:4095];
    (* ram_style="block" *) reg signed [39:0] intercept_mem[0:4095];
    reg signed [15:0] source_rd,guide_rd;
    reg [15:0] result_rd,coeff_rd;
    reg signed [39:0] intercept_rd;
    reg write_guide,write_result,write_coeff;
    reg [15:0] write_value,write_a;
    reg signed [39:0] write_b;
    always @(posedge clk) begin
        if(s_axis_tvalid && s_axis_tready)source_mem[{load_y,load_x}]<=s_axis_tdata[15:0];
        if(write_guide)guide_mem[center_addr]<=write_value;
        if(write_result)result_mem[center_addr]<=write_value;
        if(write_coeff)begin coeff_mem[center_addr]<=write_a;intercept_mem[center_addr]<=write_b;end
        source_rd<=source_mem[read_addr];guide_rd<=guide_mem[read_addr];result_rd<=result_mem[read_addr];
        coeff_rd<=coeff_mem[read_addr];intercept_rd<=intercept_mem[read_addr];
    end

    reg [63:0] numerator,denominator,div_n,div_d;
    reg div_start;
    wire div_busy,div_done;wire [63:0] div_q;
    reg [5:0] div_return;
    ct_unsigned_divider divider(.clk(clk),.rst_n(!reset_active),.start(div_start),
        .numerator(div_n),.denominator(div_d),.busy(div_busy),.done(div_done),.quotient(div_q));
    reg [15:0] center_guide,pixel_u;
    reg [12:0] range_reg,spatial_reg,weight_reg;
    reg [6:0] gauss_weight;
    reg [31:0] square_reg;
    reg signed [31:0] sum_i;
    wire signed [63:0] sum_i_extended={{32{sum_i[31]}},sum_i};
    reg [63:0] sum_sq,variance;
    reg [63:0] gf_square_sum,gf_sum9,gf_denom_reg,gf_abs_reg;
    reg signed [39:0] gf_pixel_a,gf_intercept;
    reg signed [43:0] gf_scaled;
    reg signed [63:0] gf_final_reg;
    reg signed [17:0] gf_inverse_coeff;
    reg [31:0] sum_a;
    reg signed [63:0] sum_b,final_n;
    reg negative_division;
    wire [15:0] neighbor_guide=(mode==3 || mode==4) ? guide_rd : source_rd;
    wire [15:0] difference=$signed(neighbor_guide)>=$signed(center_guide) ?
        neighbor_guide-center_guide : center_guide-neighbor_guide;
    wire [15:0] range_index=difference >> ((mode==3 || mode==4) ? 2 : 4);
    wire [25:0] spatial_range=spatial_reg*range_reg;
    wire [63:0] range_pixel_product=pixel_u*weight_reg;
    wire signed [63:0] gf_quotient_signed=negative_division ? -$signed(div_q) : $signed(div_q);
    wire [5:0] tap_last=radius==1 ? 6'd8 : radius==2 ? 6'd24 : 6'd48;

    // Eight independent RAM banks; 64 parallel int16 x int8 multipliers in layer 1.
    reg [1:0] cnn_layer;
    wire [2:0] cnn_margin={1'b0,cnn_layer}+3'd1;
    wire [5:0] cnn_x_start=trim_l!=0 ? cnn_margin : 0;
    wire [5:0] cnn_y_start=trim_t!=0 ? cnn_margin : 0;
    wire [6:0] cnn_x_end=width-1-(trim_r!=0 ? cnn_margin : 0);
    wire [6:0] cnn_y_end=height-1-(trim_b!=0 ? cnn_margin : 0);
    reg [4:0] cnn_cycle;
    reg [3:0] tap_d0,tap_d1;
    reg valid0,valid1,valid2,valid3;
    reg signed [31:0] accumulator[0:7];
    wire signed [15:0] feature_a_rd[0:7],feature_b_rd[0:7];
    reg signed [15:0] cnn_value[0:7];
    wire signed [16:0] cnn_residual=$signed({source_rd[15],source_rd})-$signed({cnn_value[0][15],cnn_value[0]});
    wire signed [31:0] dot_sum[0:7];
    wire feature_a_write=state==CNN_WRITE && cnn_layer==0 && !reset_active;
    wire feature_b_write=state==CNN_WRITE && cnn_layer==1 && !reset_active;
    genvar o,i;
    generate for(o=0;o<8;o=o+1)begin:bank
        (* ram_style="block" *) reg signed [15:0] feature_a[0:4095];
        (* ram_style="block" *) reg signed [15:0] feature_b[0:4095];
        reg signed [15:0] qa,qb;
        assign feature_a_rd[o]=qa;assign feature_b_rd[o]=qb;
        always @(posedge clk)begin
            qa<=feature_a[read_addr];qb<=feature_b[read_addr];
            if(feature_a_write)feature_a[center_addr]<=cnn_value[o];
            if(feature_b_write)feature_b[center_addr]<=cnn_value[o];
        end
        wire signed [31:0] product_ext[0:7];
        for(i=0;i<8;i=i+1)begin:mac
            wire signed [15:0] a=cnn_layer==0 ? (i==0 ? source_rd : 16'sd0) :
                cnn_layer==1 ? feature_a_rd[i] : feature_b_rd[i];
            wire signed [7:0] w=cnn_layer==0 ? (i==0 ? cnn_w0(o*9+tap_d1) : 8'sd0) :
                cnn_layer==1 ? cnn_w1(o*72+i*9+tap_d1) : (o==0 ? cnn_w2(i*9+tap_d1) : 8'sd0);
            (* use_dsp="yes" *) reg signed [23:0] product;
            always @(posedge clk)if(valid1)product<=a*w;
            assign product_ext[i]={{8{product[23]}},product};
        end
        reg signed [31:0] dot;
        always @(posedge clk)if(valid2)dot<=product_ext[0]+product_ext[1]+product_ext[2]+product_ext[3]+
            product_ext[4]+product_ext[5]+product_ext[6]+product_ext[7];
        assign dot_sum[o]=dot;
    end endgenerate
    integer c;

    always @*begin
        write_guide=0;write_result=0;write_coeff=0;write_value=0;write_a=0;write_b=0;
        if(!reset_active)begin
            if(state==GAUSS_WRITE)begin
                write_value=((numerator+(radius==1 ? 64'd8 : 64'd128))>>(radius==1 ? 4 : 8))^16'h8000;
                if(mode==1)write_result=1;else write_guide=1;
            end
            if(state==RANGE_WRITE)begin write_result=1;write_value=div_q[15:0]^16'h8000;end
            if(state==GF_COEFF_WRITE)begin
                write_coeff=1;write_a=div_q[15:0];write_b=gf_intercept;
            end
            if(state==GF_OUTPUT_WRITE)begin write_result=1;write_value=saturate64(gf_quotient_signed);end
            if(state==CNN_WRITE && cnn_layer==2)begin
                write_result=1;
                if(cnn_residual>32767)write_value=32767;
                else if(cnn_residual< -32768)write_value=-32768;
                else write_value=cnn_residual[15:0];
            end
        end
    end

    always @(posedge clk)begin
        if(reset_active)begin
            state<=LOAD;status<=0;mode<=0;width<=1;height<=1;load_x<=0;load_y<=0;load_count<=0;
            compact_cnn<=0;trim_l<=0;trim_r<=0;trim_t<=0;trim_b<=0;
            px<=0;py<=0;read_addr<=0;tap<=0;radius<=1;pass<=0;output_hu<=0;
            numerator<=0;denominator<=0;div_n<=0;div_d<=1;div_start<=0;div_return<=LOAD;
            center_guide<=0;pixel_u<=0;range_reg<=0;spatial_reg<=0;weight_reg<=0;gauss_weight<=0;
            square_reg<=0;sum_i<=0;sum_sq<=0;variance<=0;sum_a<=0;sum_b<=0;final_n<=0;negative_division<=0;
            gf_square_sum<=0;gf_sum9<=0;gf_denom_reg<=0;gf_abs_reg<=0;gf_pixel_a<=0;
            gf_scaled<=0;gf_final_reg<=0;gf_inverse_coeff<=0;gf_intercept<=0;
            cnn_layer<=0;cnn_cycle<=0;tap_d0<=0;tap_d1<=0;valid0<=0;valid1<=0;valid2<=0;valid3<=0;
            for(c=0;c<8;c=c+1)begin accumulator[c]<=0;cnn_value[c]<=0;end
        end else begin
            div_start<=0;
            case(state)
            LOAD:if(s_axis_tvalid && s_axis_tready)begin
                if(first_input)begin
                    mode<=cfg[3:0]<=7 ? cfg[3:0] : 0;width<=cfg_width;height<=cfg_height;
                    compact_cnn<=cfg[26] && incoming_crop_valid;
                    trim_l<=cfg[26] && incoming_crop_valid ? cfg[19:18] : 0;
                    trim_r<=cfg[26] && incoming_crop_valid ? cfg[21:20] : 0;
                    trim_t<=cfg[26] && incoming_crop_valid ? cfg[23:22] : 0;
                    trim_b<=cfg[26] && incoming_crop_valid ? cfg[25:24] : 0;
                    if(cfg[26] && !incoming_crop_valid)status[3]<=1;
                    if(cfg[3:0]>7)status[2]<=1;
                    if(cfg[10:4]<1 || cfg[10:4]>64 || cfg[17:11]<1 || cfg[17:11]>64)status[3]<=1;
                end
                if(s_axis_tkeep!=15)status[0]<=1;
                if(s_axis_tlast!=incoming_last)status[1]<=1;
                if(incoming_last)begin px<=0;py<=0;state<=BEGIN_MODE;end
                else begin
                    load_count<=load_count+1'b1;
                    if(load_x==(first_input ? cfg_width : width)-1)begin load_x<=0;load_y<=load_y+1'b1;end
                    else load_x<=load_x+1'b1;
                end
            end
            BEGIN_MODE:begin
                pass<=0;cnn_layer<=0;
                if(mode==0)state<=OUT_ADDR;
                else if(mode==6)begin
                    px<=trim_l!=0 ? 1 : 0;py<=trim_t!=0 ? 1 : 0;state<=CNN_BEGIN;
                end
                else begin radius<=mode==7 ? 3 : (mode==1 || mode==3 || mode==2) ? 2 : 1;state<=PIXEL_BEGIN;end
            end
            PIXEL_BEGIN:begin
                tap<=0;numerator<=0;denominator<=0;sum_i<=0;sum_sq<=0;sum_a<=0;sum_b<=0;
                if(mode==2 || mode==7 || ((mode==3 || mode==4) && pass==1))state<=CENTER_ADDR;
                else state<=TAP_ADDR;
            end
            CENTER_ADDR:begin read_addr<=center_addr;state<=CENTER_WAIT;end
            CENTER_WAIT:state<=CENTER_CAPTURE;
            CENTER_CAPTURE:begin
                center_guide<=((mode==3 || mode==4) ? guide_rd : source_rd);state<=TAP_ADDR;
            end
            TAP_ADDR:begin read_addr<=address(px,py,tap_dx(tap,radius),tap_dy(tap,radius));state<=TAP_WAIT;end
            TAP_WAIT:state<=TAP_CAPTURE;
            TAP_CAPTURE:begin
                if(mode==5)begin
                    if(pass==0)begin square_reg<=$signed(source_rd)*$signed(source_rd);state<=GF_ACC;end
                    else state<=GF_APPLY_ACC;
                end else if(mode==1 || ((mode==3 || mode==4) && pass==0))begin
                    pixel_u<=source_rd^16'h8000;
                    gauss_weight<=binomial(tap_dx(tap,radius)+radius,radius)*binomial(tap_dy(tap,radius)+radius,radius);
                    state<=GAUSS_ACC;
                end else begin
                    pixel_u<=source_rd^16'h8000;
                    range_reg<=range_weight(range_index);
                    spatial_reg<=mode==2 ? spatial_2(tap) : mode==7 ? spatial_7(tap) : spatial_3(tap);
                    state<=RANGE_PRODUCT;
                end
            end
            GAUSS_ACC:begin
                numerator<=numerator+pixel_u*gauss_weight;
                if(tap==tap_last)state<=GAUSS_WRITE;else begin tap<=tap+1'b1;state<=TAP_ADDR;end
            end
            GAUSS_WRITE:state<=PIXEL_NEXT;
            RANGE_PRODUCT:begin weight_reg<=(spatial_range+26'd2048)>>12;state<=RANGE_ACC;end
            RANGE_ACC:begin
                numerator<=numerator+range_pixel_product;denominator<=denominator+weight_reg;
                if(tap==tap_last)begin state<=DIV_LAUNCH;div_return<=RANGE_WRITE;end
                else begin tap<=tap+1'b1;state<=TAP_ADDR;end
            end
            DIV_LAUNCH:begin
                if(div_return==RANGE_WRITE)begin div_n<=numerator+(denominator>>1);div_d<=denominator;end
                if(div_d==0 && div_return!=RANGE_WRITE)status[4]<=1;
                if(denominator==0 && div_return==RANGE_WRITE)status[4]<=1;
                div_start<=1;state<=DIV_WAIT;
            end
            DIV_WAIT:if(div_done)state<=div_return==GF_COEFF_WRITE ? GF_COEFF_PREP : div_return;
            RANGE_WRITE:state<=PIXEL_NEXT;
            GF_ACC:begin
                sum_i<=sum_i+$signed(source_rd);sum_sq<=sum_sq+square_reg;
                if(tap==8)state<=GF_VARIANCE;else begin tap<=tap+1'b1;state<=TAP_ADDR;end
            end
            GF_VARIANCE:begin
                gf_sum9<=(sum_sq<<3)+sum_sq;gf_square_sum<=sum_i_extended*sum_i_extended;
                state<=GF_VARIANCE_SUB;
            end
            GF_VARIANCE_SUB:begin variance<=gf_sum9-gf_square_sum;state<=GF_DENOM;end
            GF_DENOM:begin gf_denom_reg<=variance+64'd84934656;state<=GF_DIV_PREP;end
            GF_DIV_PREP:begin
                div_n<=(variance<<16)+(gf_denom_reg>>1);div_d<=gf_denom_reg;
                div_return<=GF_COEFF_WRITE;state<=DIV_LAUNCH;
            end
            GF_COEFF_PREP:begin gf_inverse_coeff<=18'sd65536-$signed({1'b0,div_q[16:0]});state<=GF_COEFF_MULT;end
            // sum_i is the sum of nine int16 values, so signed 20 bits suffice.
            GF_COEFF_MULT:begin gf_intercept<=$signed(sum_i[19:0])*gf_inverse_coeff;state<=GF_COEFF_WRITE;end
            GF_COEFF_WRITE:state<=PIXEL_NEXT;
            GF_APPLY_ACC:begin
                sum_a<=sum_a+coeff_rd;sum_b<=sum_b+$signed(intercept_rd);
                if(tap==8)state<=GF_CENTER_ADDR;else begin tap<=tap+1'b1;state<=TAP_ADDR;end
            end
            GF_CENTER_ADDR:begin read_addr<=center_addr;state<=GF_CENTER_WAIT;end
            GF_CENTER_WAIT:state<=GF_APPLY_PREP;
            GF_APPLY_PREP:begin
                // sum_a <= 9*65535 < 2^20; bound the multiplier explicitly.
                gf_pixel_a<=$signed(source_rd)*$signed({1'b0,sum_a[19:0]});state<=GF_SCALE;
            end
            GF_SCALE:begin gf_scaled<=gf_pixel_a*44'sd9;state<=GF_FINAL_ADD;end
            GF_FINAL_ADD:begin gf_final_reg<=gf_scaled+sum_b;state<=GF_ABS;end
            GF_ABS:begin
                negative_division<=gf_final_reg<0;
                gf_abs_reg<=gf_final_reg<0 ? -gf_final_reg : gf_final_reg;state<=GF_DIV_NUM;
            end
            GF_DIV_NUM:begin
                div_n<=gf_abs_reg+64'd2654208;div_d<=64'd5308416;
                div_return<=GF_OUTPUT_WRITE;state<=DIV_LAUNCH;
            end
            GF_OUTPUT_WRITE:state<=PIXEL_NEXT;
            PIXEL_NEXT:begin
                if(px==width-1)begin
                    px<=0;
                    if(py==height-1)begin
                        py<=0;
                        if((mode==3 || mode==4 || mode==5) && pass==0)begin
                            pass<=1;radius<=mode==5 ? 1 : 3;state<=PIXEL_BEGIN;
                        end else state<=OUT_ADDR;
                    end else begin py<=py+1'b1;state<=PIXEL_BEGIN;end
                end else begin px<=px+1'b1;state<=PIXEL_BEGIN;end
            end
            CNN_BEGIN:begin
                cnn_cycle<=0;valid0<=0;valid1<=0;valid2<=0;valid3<=0;
                for(c=0;c<8;c=c+1)accumulator[c]<=cnn_layer==0 ? cnn_b0(c) : cnn_layer==1 ? cnn_b1(c) : cnn_b2(c);
                state<=CNN_RUN;
            end
            CNN_RUN:begin
                valid0<=cnn_cycle<9;valid1<=valid0;valid2<=valid1;valid3<=valid2;
                tap_d0<=cnn_cycle;tap_d1<=tap_d0;
                read_addr<=cnn_cycle<9 ? address(px,py,(cnn_cycle%3)-1,(cnn_cycle/3)-1) : center_addr;
                if(valid3)for(c=0;c<8;c=c+1)accumulator[c]<=accumulator[c]+dot_sum[c];
                if(cnn_cycle==12)state<=CNN_QUANT;else cnn_cycle<=cnn_cycle+1'b1;
            end
            CNN_QUANT:begin
                for(c=0;c<8;c=c+1)cnn_value[c]<=cnn_quantize(accumulator[c],cnn_layer);
                state<=CNN_WRITE;
            end
            CNN_WRITE:begin
                valid0<=0;valid1<=0;valid2<=0;valid3<=0;
                if(px==cnn_x_end)begin
                    px<=cnn_x_start;
                    if(py==cnn_y_end)begin
                        if(cnn_layer==2)begin px<=trim_l;py<=trim_t;state<=OUT_ADDR;end
                        else begin
                            cnn_layer<=cnn_layer+1'b1;
                            px<=trim_l!=0 ? cnn_margin+1'b1 : 0;
                            py<=trim_t!=0 ? cnn_margin+1'b1 : 0;
                            state<=CNN_BEGIN;
                        end
                    end else begin py<=py+1'b1;state<=CNN_BEGIN;end
                end else begin px<=px+1'b1;state<=CNN_BEGIN;end
            end
            OUT_ADDR:begin read_addr<=center_addr;state<=OUT_WAIT;end
            OUT_WAIT:state<=OUT_CAPTURE;
            OUT_CAPTURE:begin output_hu<=mode==0 ? source_rd : result_rd;state<=OUT_SEND;end
            OUT_SEND:if(m_axis_tready)begin
                if(px==width-1-trim_r)begin
                    px<=trim_l;
                    if(py==height-1-trim_b)begin py<=0;load_x<=0;load_y<=0;load_count<=0;state<=LOAD;end
                    else begin py<=py+1'b1;state<=OUT_ADDR;end
                end else begin px<=px+1'b1;state<=OUT_ADDR;end
            end
            default:state<=LOAD;
            endcase
        end
    end
endmodule
