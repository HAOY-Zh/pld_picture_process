`timescale 1ns/1ps
// Elastic INT32 * positive Q31 -> signed RNE -> UINT8, contract v1.
// Exactly ONE rounding. Invalid shift/multiplier emits error, not silent wrap.
module lung_requant (
    input wire clk, input wire rst_n,
    input wire s_valid, output wire s_ready,
    input wire signed [31:0] s_acc,
    input wire signed [31:0] s_multiplier,
    input wire [5:0] s_shift,
    input wire [7:0] s_zero_point,
    input wire s_relu,
    output reg m_valid, input wire m_ready,
    output reg [7:0] m_code, output reg m_error
);
    reg product_valid, quotient_valid, biased_valid;
    reg signed [63:0] product_reg;
    reg [5:0] shift_reg;
    reg [7:0] zero_reg;
    reg relu_reg, error_reg;
    reg signed [63:0] quotient_reg, biased_reg;
    reg increment_reg;
    reg [7:0] quotient_zero, biased_zero;
    reg quotient_relu, quotient_error, biased_relu, biased_error;
    wire output_ready = !m_valid || m_ready;
    wire biased_ready = !biased_valid || output_ready;
    wire quotient_ready = !quotient_valid || biased_ready;
    wire product_ready = !product_valid || quotient_ready;
    assign s_ready = rst_n && product_ready;
    (* use_dsp = "yes" *) wire signed [63:0] product = s_acc * s_multiplier;
    wire signed [63:0] quotient = product_reg >>> shift_reg;
    wire guard_bit = (shift_reg == 0) ? 1'b0 : product_reg[shift_reg-1'b1];
    wire [63:0] sticky_bits;
    genvar bit_index;
    generate for(bit_index=0;bit_index<64;bit_index=bit_index+1)begin: sticky
        assign sticky_bits[bit_index] = ({1'b0,shift_reg} > (bit_index+1)) && product_reg[bit_index];
    end endgenerate
    wire increment = guard_bit && ((|sticky_bits) || quotient[0]);
    wire [8:0] adjustment = {1'b0,quotient_zero} + increment_reg;
    reg [7:0] clipped;
    always @* begin
        if (biased_error) clipped = 8'd0;
        else if (biased_reg[63]) clipped = biased_relu ? biased_zero : 8'd0;
        else if (|biased_reg[62:8]) clipped = 8'd255;
        else if (biased_relu && biased_reg[7:0] < biased_zero) clipped = biased_zero;
        else clipped = biased_reg[7:0];
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            product_valid <= 0; quotient_valid <= 0; biased_valid <= 0;
            m_valid <= 0; product_reg <= 0;
            shift_reg <= 0; zero_reg <= 0; relu_reg <= 0; error_reg <= 0;
            m_code <= 0; m_error <= 0;
            quotient_reg <= 0; biased_reg <= 0; increment_reg <= 0;
            quotient_zero <= 0; biased_zero <= 0; quotient_relu <= 0;
            quotient_error <= 0; biased_relu <= 0; biased_error <= 0;
        end else begin
            if (output_ready) begin
                m_valid <= biased_valid;
                if (biased_valid) begin m_code <= clipped; m_error <= biased_error; end
            end
            if (biased_ready) begin
                biased_valid <= quotient_valid;
                if (quotient_valid) begin
                    biased_reg <= quotient_reg + $signed({55'd0,adjustment});
                    biased_zero <= quotient_zero; biased_relu <= quotient_relu;
                    biased_error <= quotient_error;
                end
            end
            if (quotient_ready) begin
                quotient_valid <= product_valid;
                if (product_valid) begin
                    quotient_reg <= quotient; increment_reg <= increment;
                    quotient_zero <= zero_reg; quotient_relu <= relu_reg;
                    quotient_error <= error_reg;
                end
            end
            if (product_ready) begin
                product_valid <= s_valid;
                if (s_valid) begin
                    product_reg <= product; shift_reg <= s_shift;
                    zero_reg <= s_zero_point; relu_reg <= s_relu;
                    error_reg <= (s_multiplier <= 0) || (s_shift > 62);
                end
            end
        end
    end
endmodule
