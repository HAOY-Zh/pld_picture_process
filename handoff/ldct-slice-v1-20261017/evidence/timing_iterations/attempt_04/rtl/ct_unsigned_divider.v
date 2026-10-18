`timescale 1ns/1ps
// Unsigned restoring divider. Fixed 64 iteration latency, no combinational /.
module ct_unsigned_divider(
    input clk, input rst_n, input start,
    input [63:0] numerator, input [63:0] denominator,
    output reg busy, output reg done, output reg [63:0] quotient
);
    reg [63:0] dividend, divisor;
    reg [64:0] remainder;
    reg [6:0] count;
    wire [64:0] trial={remainder[63:0],dividend[63]};
    wire subtract=trial>={1'b0,divisor};
    wire [63:0] next_value={dividend[62:0],subtract};
    always @(posedge clk) begin
        if(!rst_n) begin
            busy<=0;done<=0;quotient<=0;dividend<=0;divisor<=1;remainder<=0;count<=0;
        end else begin
            done<=0;
            if(start && !busy) begin
                dividend<=numerator;divisor<=denominator;remainder<=0;count<=0;busy<=1;
            end else if(busy) begin
                remainder<=subtract ? trial-{1'b0,divisor} : trial;
                dividend<=next_value;
                if(count==63) begin quotient<=next_value;busy<=0;done<=1;end
                else count<=count+1'b1;
            end
        end
    end
endmodule
