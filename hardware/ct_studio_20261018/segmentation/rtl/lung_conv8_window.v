`timescale 1ns/1ps
// One convolution output-channel block over a whole feature frame.
// Generated windows feed the unchanged resident-weight 64-MAC core.
module lung_conv8_window(
    input wire clk,input wire rst_n,
    input wire w_valid,output wire w_ready,input wire w_first,input wire w_last,
    input wire [511:0] w_weights,input wire [7:0] w_output_keep,
    input wire [7:0] w_input_zero_point,input wire [7:0] w_output_zero_point,
    input wire [255:0] w_biases,input wire [255:0] w_multipliers,
    input wire [47:0] w_shifts,input wire w_relu,output wire weights_loaded,
    input wire c_valid,output wire c_ready,input wire [31:0] c_base,
    input wire [9:0] c_height,input wire [9:0] c_width,input wire [6:0] c_channels,
    input wire c_kernel3,input wire c_padding,
    output wire req_valid,input wire req_ready,output wire [31:0] req_addr,
    input wire rsp_valid,output wire rsp_ready,input wire [63:0] rsp_data,input wire rsp_error,
    output wire m_valid,input wire m_ready,output wire [2:0] m_lane,output wire m_last,
    output wire m_frame_last,output wire [8:0] m_x,output wire [8:0] m_y,
    output wire signed [31:0] m_acc,output wire [7:0] m_code,output wire m_error,
    output reg busy,output reg done,output reg error
);
    wire reuse_w_ready,window_c_ready,win_valid,win_ready,win_first,win_last,win_frame_last,win_error;
    wire window_done,window_error,window_config_error,tile_error;
    wire [63:0]win_data;wire [7:0]win_keep;wire [8:0]win_x,win_y;
    reg [7:0] resident_input_zero;reg [8:0] pixel_x,pixel_y;reg pixel_final,pixel_error;
    reg loading_weights;
    assign w_ready=reuse_w_ready&&!busy;
    assign c_ready=rst_n&&!busy&&!w_valid&&weights_loaded&&window_c_ready;
    wire command=c_valid&&c_ready;
    assign m_x=pixel_x;assign m_y=pixel_y;
    assign m_frame_last=pixel_final&&m_last;
    assign m_error=tile_error||pixel_error;
    lung_feature_window window(
        .clk(clk),.rst_n(rst_n),.c_valid(command),.c_ready(window_c_ready),.c_base(c_base),
        .c_height(c_height),.c_width(c_width),.c_channels(c_channels),.c_kernel3(c_kernel3),
        .c_padding(c_padding),.c_input_zero(resident_input_zero),
        .req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(rsp_data),.rsp_error(rsp_error),
        .m_valid(win_valid),.m_ready(win_ready),.m_data(win_data),.m_keep(win_keep),
        .m_first(win_first),.m_last(win_last),.m_frame_last(win_frame_last),
        .m_x(win_x),.m_y(win_y),.m_error(win_error),.done(window_done),
        .error(window_error),.config_error(window_config_error));
    lung_mac_tile8_reuse core(
        .clk(clk),.rst_n(rst_n),.w_valid(w_valid&&!busy),.w_ready(reuse_w_ready),
        .w_first(w_first),.w_last(w_last),.w_weights(w_weights),.w_output_keep(w_output_keep),
        .w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),
        .w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),
        .weights_loaded(weights_loaded),.s_valid(win_valid),.s_ready(win_ready),
        .s_first(win_first),.s_last(win_last),.s_activations(win_data),.s_input_keep(win_keep),
        .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),
        .m_acc(m_acc),.m_code(m_code),.m_error(tile_error));
    always @(posedge clk)begin
        if(!rst_n)begin
            busy<=0;done<=0;error<=0;resident_input_zero<=0;loading_weights<=0;
            pixel_x<=0;pixel_y<=0;pixel_final<=0;pixel_error<=0;
        end else begin
            done<=0;
            if(w_valid&&w_ready)begin
                if(!loading_weights)resident_input_zero<=w_input_zero_point;
                loading_weights<=!w_last;
            end
            if(command)begin busy<=1;error<=0;pixel_error<=0;end
            if(win_valid&&win_ready)begin
                if(win_first)begin pixel_x<=win_x;pixel_y<=win_y;pixel_error<=win_error;pixel_final<=0;end
                else pixel_error<=pixel_error||win_error;
                if(win_last)pixel_final<=win_frame_last;
                error<=error||win_error;
            end
            if(m_valid&&m_ready)begin
                error<=error||m_error;
                if(m_frame_last)begin busy<=0;done<=1;end
            end
            if(window_done&&window_config_error)begin busy<=0;done<=1;error<=1;end
        end
    end
endmodule
