`timescale 1ns/1ps
// 8 input lanes x 8 output lanes = 64 signed MACs/accepted input beat.
// Weight byte index is output_lane*8 + input_lane; LSB byte is lane 0.
// Partial accumulators are per input lane. Bias added ONCE after reduction.
// One shared lung_requant, output tags preserved by an 8-entry metadata FIFO.
// A group is not overlapped with the next group until its LAST output drains.
module lung_mac_tile8 (
    input wire clk, input wire rst_n,
    input wire s_valid, output wire s_ready, input wire s_first, input wire s_last,
    input wire [63:0] s_activations, input wire [511:0] s_weights,
    input wire [7:0] s_input_keep, input wire [7:0] s_output_keep,
    input wire [7:0] s_input_zero_point, input wire [7:0] s_output_zero_point,
    input wire [255:0] s_biases, input wire [255:0] s_multipliers,
    input wire [47:0] s_shifts, input wire s_relu,
    output wire m_valid, input wire m_ready,
    output wire [2:0] m_lane, output wire m_last,
    output wire signed [31:0] m_acc, output wire [7:0] m_code,
    output wire m_error
);
    reg input_active, flushing, emitting;
    reg p_valid, p_first, p_last, p_marker_error;
    reg [7:0] input_zero, output_zero, output_keep, remaining;
    reg [255:0] biases, multipliers;
    reg [47:0] shifts;
    reg relu, group_error;
    wire starts_group = s_first || !input_active;
    wire [7:0] selected_input_zero = starts_group ? s_input_zero_point : input_zero;
    wire [7:0] selected_output_keep = starts_group ? (s_output_keep == 0 ? 8'h01 : s_output_keep) : output_keep;
    assign s_ready = rst_n && !flushing && !emitting;
    wire accepted = s_valid && s_ready;
    reg reduce_start, r1_valid, r2_valid, r3_valid, r4_valid;
    reg signed [32:0] pairs[0:31];
    reg signed [33:0] quarters[0:15];
    reg signed [34:0] totals[0:7];
    reg signed [31:0] held_acc[0:7];
    reg held_error[0:7];
    wire [63:0] partial_errors;
    genvar o,i;
    generate for(o=0;o<8;o=o+1)begin: output_lane
        for(i=0;i<8;i=i+1)begin: input_lane
            wire enabled = s_input_keep[i] && selected_output_keep[o];
            wire signed [8:0] centered = enabled ?
                ($signed({1'b0,s_activations[8*i+:8]}) - $signed({1'b0,selected_input_zero})) : 9'sd0;
            wire signed [7:0] weight = s_weights[8*(o*8+i)+:8];
            (* use_dsp = "yes" *) wire signed [16:0] product = centered * weight;
            reg signed [16:0] product_reg;
            reg signed [31:0] accumulator;
            reg accumulator_error;
            wire signed [32:0] extended_product = {{16{product_reg[16]}},product_reg};
            wire signed [32:0] partial_sum = p_first ? extended_product :
                ($signed({accumulator[31],accumulator}) + extended_product);
            assign partial_errors[o*8+i] = accumulator_error;
            always @(posedge clk)begin
                if(!rst_n)begin product_reg<=0;accumulator<=0;accumulator_error<=0;end
                else begin
                    if(accepted)product_reg<=product;
                    if(p_valid)begin
                        accumulator<=partial_sum[31:0];
                        accumulator_error<=(p_first?1'b0:accumulator_error)||(partial_sum[32]!=partial_sum[31]);
                    end
                end
            end
        end
        for(i=0;i<4;i=i+1)begin: pair_reduce
            wire signed [31:0] a = input_lane[2*i].accumulator;
            wire signed [31:0] b = input_lane[2*i+1].accumulator;
            always @(posedge clk)begin
                if(!rst_n)pairs[o*4+i]<=0;
                else if(reduce_start)pairs[o*4+i]<=$signed({a[31],a})+$signed({b[31],b});
            end
        end
        for(i=0;i<2;i=i+1)begin: quarter_reduce
            wire signed [32:0] a = pairs[o*4+2*i];
            wire signed [32:0] b = pairs[o*4+2*i+1];
            always @(posedge clk)begin
                if(!rst_n)quarters[o*2+i]<=0;
                else if(r1_valid)quarters[o*2+i]<=$signed({a[32],a})+$signed({b[32],b});
            end
        end
        wire signed [33:0] a = quarters[o*2];
        wire signed [33:0] b = quarters[o*2+1];
        wire signed [31:0] bias = biases[32*o+:32];
        wire signed [35:0] final_sum = $signed({totals[o][34],totals[o]})+$signed({{4{bias[31]}},bias});
        always @(posedge clk)begin
            if(!rst_n)begin totals[o]<=0;held_acc[o]<=0;held_error[o]<=0;end
            else begin
                if(r2_valid)totals[o]<=$signed({a[33],a})+$signed({b[33],b});
                if(r3_valid)begin
                    held_acc[o]<=final_sum[31:0];
                    held_error[o]<=group_error || (|partial_errors[8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});
                end
            end
        end
    end endgenerate

    reg [2:0] selected;
    integer pick;
    always @*begin
        selected=0;
        for(pick=7;pick>=0;pick=pick-1)if(remaining[pick])selected=pick;
    end
    wire [7:0] after_selected = remaining & ~(8'h01 << selected);
    wire req_ready,rq_valid,rq_error;
    wire [7:0] rq_code;
    reg [2:0] fifo_lane[0:7];reg fifo_last[0:7],fifo_error[0:7];
    reg signed [31:0] fifo_acc[0:7];
    reg [2:0] write_pointer,read_pointer;reg [3:0] fifo_count;
    reg dispatch_valid,dispatch_last,dispatch_error;
    reg [2:0] dispatch_lane;
    reg signed [31:0] dispatch_acc,dispatch_multiplier;
    reg [5:0] dispatch_shift;
    assign m_valid=rq_valid && (fifo_count!=0);
    assign m_lane=fifo_lane[read_pointer];assign m_last=fifo_last[read_pointer];
    assign m_acc=fifo_acc[read_pointer];assign m_code=rq_code;
    assign m_error=rq_error || fifo_error[read_pointer];
    wire popped=m_valid && m_ready;
    wire fifo_space=(fifo_count<8)||popped;
    wire req_valid=dispatch_valid && fifo_space;
    wire pushed=req_valid && req_ready;
    wire dispatch_ready=!dispatch_valid || (req_ready && fifo_space);
    wire selected_valid=emitting && (remaining!=0);
    wire selected_fire=selected_valid && dispatch_ready;
    lung_requant shared_quant(
        .clk(clk),.rst_n(rst_n),.s_valid(req_valid),.s_ready(req_ready),
        .s_acc(dispatch_acc),.s_multiplier(dispatch_multiplier),
        .s_shift(dispatch_shift),.s_zero_point(output_zero),.s_relu(relu),
        .m_valid(rq_valid),.m_ready(m_ready && fifo_count!=0),.m_code(rq_code),.m_error(rq_error)
    );
    integer f;
    always @(posedge clk)begin
        if(!rst_n)begin
            input_active<=0;flushing<=0;emitting<=0;p_valid<=0;p_first<=0;p_last<=0;p_marker_error<=0;
            input_zero<=0;output_zero<=0;output_keep<=0;remaining<=0;biases<=0;multipliers<=0;shifts<=0;relu<=0;group_error<=0;
            reduce_start<=0;r1_valid<=0;r2_valid<=0;r3_valid<=0;r4_valid<=0;
            write_pointer<=0;read_pointer<=0;fifo_count<=0;
            dispatch_valid<=0;dispatch_lane<=0;dispatch_last<=0;dispatch_error<=0;
            dispatch_acc<=0;dispatch_multiplier<=0;dispatch_shift<=0;
            for(f=0;f<8;f=f+1)begin fifo_lane[f]<=0;fifo_last[f]<=0;fifo_error[f]<=0;fifo_acc[f]<=0;end
        end else begin
            p_valid<=accepted;
            if(accepted)begin
                p_first<=starts_group;p_last<=s_last;
                p_marker_error<=(s_first!=!input_active) || (s_first && s_output_keep==0);
                input_active<=!s_last;
                if(starts_group)begin
                    input_zero<=s_input_zero_point;output_zero<=s_output_zero_point;
                    output_keep<=s_output_keep==0?8'h01:s_output_keep;
                    biases<=s_biases;multipliers<=s_multipliers;shifts<=s_shifts;relu<=s_relu;
                end
                if(s_last)flushing<=1;
            end
            if(p_valid)group_error<=(p_first?1'b0:group_error)||p_marker_error;
            reduce_start<=p_valid && p_last;
            r1_valid<=reduce_start;r2_valid<=r1_valid;r3_valid<=r2_valid;r4_valid<=r3_valid;
            if(r4_valid)begin emitting<=1;remaining<=output_keep;end
            // Break priority-select/mux -> wide-multiplier critical path.
            // This elastic dispatch register keeps metadata paired with data.
            if(dispatch_ready)begin
                dispatch_valid<=selected_valid;
                if(selected_valid)begin
                    dispatch_acc<=held_acc[selected];dispatch_multiplier<=multipliers[32*selected+:32];
                    dispatch_shift<=shifts[6*selected+:6];dispatch_lane<=selected;
                    dispatch_last<=after_selected==0;dispatch_error<=held_error[selected];
                end
            end
            if(selected_fire)remaining<=after_selected;
            if(pushed)begin
                fifo_lane[write_pointer]<=dispatch_lane;fifo_last[write_pointer]<=dispatch_last;
                fifo_error[write_pointer]<=dispatch_error;fifo_acc[write_pointer]<=dispatch_acc;
                write_pointer<=write_pointer+1'b1;
            end
            if(popped)begin
                read_pointer<=read_pointer+1'b1;
                if(m_last)begin flushing<=0;emitting<=0;end
            end
            case({pushed,popped})
                2'b10:fifo_count<=fifo_count+1'b1;
                2'b01:fifo_count<=fifo_count-1'b1;
                default:fifo_count<=fifo_count;
            endcase
        end
    end
endmodule
