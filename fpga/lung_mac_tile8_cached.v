`timescale 1ns/1ps
// One bounded input-packet group cache, depth 128 = >= max 108 beats in model.
// 72-bit activations/keep + 512-bit weights in synchronous block RAM banks.
// NOT a whole-image line cache, persistent weight cache, or AXI DMA controller.
module lung_mac_tile8_cached #(parameter DEPTH=128)(
    input wire clk,input wire rst_n,
    input wire s_valid,output wire s_ready,input wire s_first,input wire s_last,
    input wire [63:0] s_activations,input wire [511:0] s_weights,
    input wire [7:0] s_input_keep,input wire [7:0] s_output_keep,
    input wire [7:0] s_input_zero_point,input wire [7:0] s_output_zero_point,
    input wire [255:0] s_biases,input wire [255:0] s_multipliers,input wire [47:0] s_shifts,input wire s_relu,
    output wire m_valid,input wire m_ready,output wire [2:0] m_lane,output wire m_last,
    output wire signed [31:0] m_acc,output wire [7:0] m_code,output wire m_error
);
    (* ram_style="block" *)reg [71:0] activation_bank[0:DEPTH-1];
    (* ram_style="block" *)reg [511:0] weight_bank[0:DEPTH-1];
    reg [8:0] count,read_index;
    reg complete,loader_error,q_valid,q_first,q_last;
    reg [71:0] q_activations;reg [511:0] q_weights;
    reg [7:0] output_keep,input_zero,output_zero;reg [255:0] biases,multipliers;reg [47:0] shifts;reg relu;
    wire tile_ready,tile_error;
    assign s_ready=rst_n && !complete;
    wire loaded=s_valid && s_ready;
    wire slot_ready=!q_valid || tile_ready;
    wire issue=complete && read_index<count && slot_ready;
    assign m_error=tile_error || loader_error;
    initial begin
        if(DEPTH<1 || DEPTH>256)$error("Packet cache DEPTH must be 1..256");
    end
    lung_mac_tile8 tile(
        .clk(clk),.rst_n(rst_n),.s_valid(q_valid),.s_ready(tile_ready),.s_first(q_first),.s_last(q_last),
        .s_activations(q_activations[63:0]),.s_weights(q_weights),.s_input_keep(q_activations[71:64]),.s_output_keep(output_keep),
        .s_input_zero_point(input_zero),.s_output_zero_point(output_zero),.s_biases(biases),.s_multipliers(multipliers),
        .s_shifts(shifts),.s_relu(relu),.m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),
        .m_acc(m_acc),.m_code(m_code),.m_error(tile_error)
    );
    // RAM is deliberately NOT reset. Only valid/count/pointers are reset.
    always @(posedge clk)begin
        if(loaded && count<DEPTH)begin
            activation_bank[count]<={s_input_keep,s_activations};weight_bank[count]<=s_weights;
        end
        if(issue)begin q_activations<=activation_bank[read_index];q_weights<=weight_bank[read_index];end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            count<=0;read_index<=0;complete<=0;loader_error<=0;q_valid<=0;q_first<=0;q_last<=0;
            output_keep<=0;input_zero<=0;output_zero<=0;biases<=0;multipliers<=0;shifts<=0;relu<=0;
        end else begin
            if(loaded)begin
                loader_error<=loader_error || (s_first!=(count==0)) || count>=DEPTH;
                if(count==0)begin
                    output_keep<=s_output_keep;input_zero<=s_input_zero_point;output_zero<=s_output_zero_point;
                    biases<=s_biases;multipliers<=s_multipliers;shifts<=s_shifts;relu<=s_relu;
                end
                if(count<DEPTH)count<=count+1'b1;
                if(s_last)complete<=1;
            end
            if(slot_ready)begin
                q_valid<=issue;
                if(issue)begin
                    q_first<=read_index==0;q_last<=read_index==count-1'b1;read_index<=read_index+1'b1;
                end
            end
            if(m_valid && m_ready && m_last)begin
                count<=0;read_index<=0;complete<=0;loader_error<=0;q_valid<=0;
            end
        end
    end
endmodule
