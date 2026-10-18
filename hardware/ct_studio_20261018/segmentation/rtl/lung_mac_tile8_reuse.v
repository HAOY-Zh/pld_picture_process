`timescale 1ns/1ps
// Persistent weights for ONE output-channel tile. Not AXI or a feature window generator.
// Load a complete block on w_*, then reuse it for any number of s_* pixel groups.
// Configuration lives with weights; reload cannot overlap an active pixel/result.
module lung_mac_tile8_reuse #(parameter DEPTH=128)(
    input wire clk,input wire rst_n,
    input wire w_valid,output wire w_ready,input wire w_first,input wire w_last,
    input wire [511:0] w_weights,input wire [7:0] w_output_keep,
    input wire [7:0] w_input_zero_point,input wire [7:0] w_output_zero_point,
    input wire [255:0] w_biases,input wire [255:0] w_multipliers,
    input wire [47:0] w_shifts,input wire w_relu,
    output wire weights_loaded,
    input wire s_valid,output wire s_ready,input wire s_first,input wire s_last,
    input wire [63:0] s_activations,input wire [7:0] s_input_keep,
    output wire m_valid,input wire m_ready,output wire [2:0] m_lane,output wire m_last,
    output wire signed [31:0] m_acc,output wire [7:0] m_code,output wire m_error
);
    (* ram_style="block" *)reg [511:0] weight_bank[0:DEPTH-1];
    reg loaded,loading,active,flushing,load_error,group_error;
    reg [8:0] load_count,weight_count,beat_index;
    reg [7:0] output_keep,input_zero,output_zero;
    reg [255:0] biases,multipliers;reg [47:0] shifts;reg relu;
    reg q_valid,q_first,q_last;reg [63:0] q_activations;
    reg [7:0] q_keep;reg [511:0] q_weights;
    wire tile_ready,tile_error;
    wire slot_ready=!q_valid || tile_ready;
    assign weights_loaded=rst_n && loaded && !loading;
    assign w_ready=rst_n && !active && !q_valid;
    // An idle loader has priority. A pending reload MUST NOT prevent an active
    // pixel finishing its remaining inputs (independent producers, no deadlock).
    assign s_ready=weights_loaded && !(w_valid && w_ready) && !flushing && slot_ready;
    wire writing=w_valid && w_ready;
    wire accepted=s_valid && s_ready;
    wire [8:0] write_index=loading?load_count:9'd0;
    assign m_error=tile_error || load_error || group_error;
    initial begin
        if(DEPTH<1 || DEPTH>256)$error("Weight cache DEPTH must be 1..256");
    end
    lung_mac_tile8 tile(
        .clk(clk),.rst_n(rst_n),.s_valid(q_valid),.s_ready(tile_ready),
        .s_first(q_first),.s_last(q_last),.s_activations(q_activations),
        .s_weights(q_weights),.s_input_keep(q_keep),.s_output_keep(output_keep),
        .s_input_zero_point(input_zero),.s_output_zero_point(output_zero),
        .s_biases(biases),.s_multipliers(multipliers),.s_shifts(shifts),.s_relu(relu),
        .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),
        .m_acc(m_acc),.m_code(m_code),.m_error(tile_error)
    );
    // Synchronous write/read, no reset of RAM contents. Bounds gate every read.
    always @(posedge clk)begin
        if(writing && write_index<DEPTH)weight_bank[write_index]<=w_weights;
        if(accepted)begin
            if(beat_index<weight_count)q_weights<=weight_bank[beat_index];
            else q_weights<=0;
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            loaded<=0;loading<=0;active<=0;flushing<=0;load_error<=0;group_error<=0;
            load_count<=0;weight_count<=0;beat_index<=0;q_valid<=0;q_first<=0;q_last<=0;
            q_activations<=0;q_keep<=0;
            output_keep<=0;input_zero<=0;output_zero<=0;biases<=0;multipliers<=0;shifts<=0;relu<=0;
        end else begin
            if(writing)begin
                loaded<=0;loading<=!w_last;
                load_error<=(loading?load_error:1'b0) || (w_first!=!loading) || write_index>=DEPTH;
                if(!loading)begin
                    output_keep<=w_output_keep;input_zero<=w_input_zero_point;output_zero<=w_output_zero_point;
                    biases<=w_biases;multipliers<=w_multipliers;shifts<=w_shifts;relu<=w_relu;
                end
                if(write_index<DEPTH)load_count<=write_index+1'b1;
                if(w_last)begin
                    loaded<=1;weight_count<=write_index<DEPTH?write_index+1'b1:DEPTH;
                    load_count<=0;beat_index<=0;group_error<=0;
                end
            end
            if(slot_ready)q_valid<=accepted;
            if(accepted)begin
                active<=1;q_first<=beat_index==0;q_last<=s_last;
                q_activations<=s_activations;q_keep<=s_input_keep;
                group_error<=group_error || (s_first!=(beat_index==0)) ||
                    (s_last!=(beat_index==weight_count-1'b1)) || beat_index>=weight_count;
                // Saturate, do not wrap and read valid-looking stale weights.
                if(beat_index<256)beat_index<=beat_index+1'b1;
                if(s_last)flushing<=1;
            end
            if(m_valid && m_ready && m_last)begin
                active<=0;flushing<=0;beat_index<=0;group_error<=0;q_valid<=0;
            end
        end
    end
endmodule
