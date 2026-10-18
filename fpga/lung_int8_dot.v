`timescale 1ns/1ps
// One reusable convolution dot-product lane. Controller supplies Cin*K*K beats,
// including padding beats encoded with input zero point. Not AXI DMA or whole NN.
// Parameters/bias captured on first beat; output stable until valid && ready.
module lung_int8_dot (
    input wire clk, input wire rst_n,
    input wire s_valid, output wire s_ready,
    input wire s_first, input wire s_last,
    input wire [7:0] s_activation, input wire [7:0] s_input_zero_point,
    input wire signed [7:0] s_weight, input wire signed [31:0] s_bias,
    input wire signed [31:0] s_multiplier, input wire [5:0] s_shift,
    input wire [7:0] s_output_zero_point, input wire s_relu,
    output wire m_valid, input wire m_ready,
    output wire [7:0] m_code, output wire signed [31:0] m_acc,
    output wire m_error
);
    reg active, waiting, pending;
    reg signed [31:0] accumulator, held_acc, multiplier_reg;
    reg [5:0] shift_reg;
    reg [7:0] zero_reg, input_zero_reg;
    reg relu_reg, error_acc, held_error;
    wire req_ready, quant_error;
    wire [7:0] selected_input_zero = s_first ? s_input_zero_point : input_zero_reg;
    wire signed [8:0] centered = $signed({1'b0,s_activation}) - $signed({1'b0,selected_input_zero});
    (* use_dsp = "yes" *) wire signed [16:0] product = centered * s_weight;
    wire signed [31:0] base = s_first ? s_bias : accumulator;
    wire signed [32:0] sum = $signed({base[31],base}) + $signed({{16{product[16]}},product});
    wire overflow = sum[32] != sum[31];
    wire marker_error = (s_first != !active);
    assign s_ready = rst_n && !waiting && !pending;
    assign m_acc = held_acc;
    assign m_error = quant_error || held_error;
    lung_requant quant (
        .clk(clk), .rst_n(rst_n), .s_valid(pending), .s_ready(req_ready),
        .s_acc(held_acc), .s_multiplier(multiplier_reg), .s_shift(shift_reg),
        .s_zero_point(zero_reg), .s_relu(relu_reg),
        .m_valid(m_valid), .m_ready(m_ready), .m_code(m_code), .m_error(quant_error)
    );
    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 0; waiting <= 0; pending <= 0;
            accumulator <= 0; held_acc <= 0; multiplier_reg <= 0;
            shift_reg <= 0; zero_reg <= 0; input_zero_reg <= 0; relu_reg <= 0;
            error_acc <= 0; held_error <= 0;
        end else begin
            if (pending && req_ready) pending <= 0;
            if (m_valid && m_ready) waiting <= 0;
            if (s_valid && s_ready) begin
                accumulator <= sum[31:0];
                active <= !s_last;
                error_acc <= (s_first ? 1'b0 : error_acc) || overflow || marker_error;
                if (s_first) begin
                    input_zero_reg <= s_input_zero_point;
                    multiplier_reg <= s_multiplier; shift_reg <= s_shift;
                    zero_reg <= s_output_zero_point; relu_reg <= s_relu;
                end
                if (s_last) begin
                    held_acc <= sum[31:0];
                    held_error <= (s_first ? 1'b0 : error_acc) || overflow || marker_error;
                    waiting <= 1; pending <= 1;
                end
            end
        end
    end
endmodule
