`timescale 1ns/1ps
// AXI4-Lite 32-bit, 1 KiB aperture, same clock/reset as the compute/data masters.
// Independent AW/W buffering, one B and one R outstanding. No isolated soft reset.
module lung_axil_control(
    input wire clk,input wire rst_n,
    input wire[9:0]s_axi_awaddr,input wire[2:0]s_axi_awprot,input wire s_axi_awvalid,output wire s_axi_awready,
    input wire[31:0]s_axi_wdata,input wire[3:0]s_axi_wstrb,input wire s_axi_wvalid,output wire s_axi_wready,
    output reg[1:0]s_axi_bresp,output reg s_axi_bvalid,input wire s_axi_bready,
    input wire[9:0]s_axi_araddr,input wire[2:0]s_axi_arprot,input wire s_axi_arvalid,output wire s_axi_arready,
    output reg[31:0]s_axi_rdata,output reg[1:0]s_axi_rresp,output reg s_axi_rvalid,input wire s_axi_rready,
    output reg w_valid,input wire w_ready,output reg w_first,output reg w_last,
    output wire[511:0]w_weights,output wire[7:0]w_output_keep,
    output wire[7:0]w_input_zero_point,output wire[7:0]w_output_zero_point,
    output wire[255:0]w_biases,output wire[255:0]w_multipliers,output wire[47:0]w_shifts,output wire w_relu,
    input wire weights_loaded,output reg c_valid,input wire c_ready,
    output wire[31:0]c_base,output wire[9:0]c_height,output wire[9:0]c_width,output wire[6:0]c_channels,
    output wire c_kernel3,output wire c_padding,output wire[31:0]c_output_base,
    output wire[3:0]c_output_blocks,output wire[3:0]c_output_block,
    input wire core_busy,input wire core_done,input wire core_error,input wire[31:0]written_words,
    input wire[31:0]cache_hits,input wire[31:0]cache_misses
);
    reg aw_hold,w_hold;reg[9:0]aw_address;reg[31:0]write_data;reg[3:0]write_strb;
    reg[31:0]input_address,output_address,shape,geometry,quant;
    reg[31:0]weight_words[0:15],bias_words[0:7],mult_words[0:7];reg[5:0]shift_words[0:7];
    reg active,loading,done_sticky,error_sticky,control_error,start_pending;
    reg[2:0]start_guard;
    reg[7:0]load_count;reg[31:0]cycles,completed,weight_beats,rejected;
    wire busy=active||c_valid||start_pending||w_valid||core_busy;
    wire config_locked=busy||loading;
    wire[31:0]status={23'd0,control_error,(c_valid||start_pending),w_valid,loading,
        (!busy&&!loading&&weights_loaded),weights_loaded,error_sticky,done_sticky,busy};
    function[31:0]merge_bytes(input[31:0]old_value,input[31:0]new_value,input[3:0]strobes);
        integer b;begin merge_bytes=old_value;for(b=0;b<4;b=b+1)if(strobes[b])merge_bytes[8*b+:8]=new_value[8*b+:8];end
    endfunction
    wire[31:0]command_bits=merge_bytes(0,write_data,write_strb);
    wire[31:0]merged_shape=merge_bytes(shape,write_data,write_strb);
    wire[31:0]merged_geometry=merge_bytes(geometry,write_data,write_strb);
    wire[31:0]merged_quant=merge_bytes(quant,write_data,write_strb);
    wire[31:0]merged_shift=merge_bytes({26'd0,shift_words[aw_address[4:2]]},write_data,write_strb);
    reg shifts_valid;integer i,j;
    always @*begin shifts_valid=1;for(j=0;j<8;j=j+1)if(shift_words[j]>62)shifts_valid=0;end
    assign s_axi_awready=rst_n&&!aw_hold&&!s_axi_bvalid;
    assign s_axi_wready=rst_n&&!w_hold&&!s_axi_bvalid;
    assign s_axi_arready=rst_n&&!s_axi_rvalid;
    assign c_base={3'd0,input_address[31:3]};assign c_output_base={3'd0,output_address[31:3]};
    assign c_height=shape[9:0];assign c_width=shape[25:16];assign c_channels=geometry[6:0];
    assign c_kernel3=geometry[8];assign c_padding=geometry[9];
    assign c_output_blocks=geometry[19:16];assign c_output_block=geometry[27:24];
    assign w_output_keep=quant[7:0];assign w_input_zero_point=quant[15:8];assign w_output_zero_point=quant[23:16];assign w_relu=quant[24];
    genvar lane;
    generate for(lane=0;lane<16;lane=lane+1)begin:weight_map assign w_weights[32*lane+:32]=weight_words[lane];end
    for(lane=0;lane<8;lane=lane+1)begin:parameter_map
        assign w_biases[32*lane+:32]=bias_words[lane];assign w_multipliers[32*lane+:32]=mult_words[lane];assign w_shifts[6*lane+:6]=shift_words[lane];
    end endgenerate
    task reject_write;
        begin s_axi_bresp<=2'b10;control_error<=1;rejected<=rejected+1'b1;end
    endtask
    always @(posedge clk)begin
        if(!rst_n)begin
            aw_hold<=0;w_hold<=0;aw_address<=0;write_data<=0;write_strb<=0;
            s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rdata<=0;s_axi_rresp<=0;
            input_address<=0;output_address<=0;shape<=0;geometry<=0;quant<=0;
            active<=0;loading<=0;done_sticky<=0;error_sticky<=0;control_error<=0;load_count<=0;start_pending<=0;start_guard<=0;
            cycles<=0;completed<=0;weight_beats<=0;rejected<=0;c_valid<=0;w_valid<=0;w_first<=0;w_last<=0;
            for(i=0;i<16;i=i+1)weight_words[i]<=0;
            for(i=0;i<8;i=i+1)begin bias_words[i]<=0;mult_words[i]<=0;shift_words[i]<=0;end
        end else begin
            if(s_axi_awvalid&&s_axi_awready)begin aw_hold<=1;aw_address<=s_axi_awaddr;end
            if(s_axi_wvalid&&s_axi_wready)begin w_hold<=1;write_data<=s_axi_wdata;write_strb<=s_axi_wstrb;end
            if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
            if(aw_hold&&w_hold&&!s_axi_bvalid)begin
                aw_hold<=0;w_hold<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
                if(aw_address[1:0]!=0)reject_write();
                else if(write_strb==0)begin end
                else case(aw_address)
                    10'h004:begin
                        if(command_bits==0)begin end
                        else if(command_bits==2)begin done_sticky<=0;error_sticky<=0;control_error<=0;end
                        else if(command_bits==1)begin
                            if(busy||loading||!weights_loaded||input_address[2:0]!=0||output_address[2:0]!=0)reject_write();
                            else begin start_pending<=1;start_guard<=3;done_sticky<=0;error_sticky<=0;control_error<=0;cycles<=0;end
                        end else if((command_bits&32'hffffffe3)==0&&command_bits[2])begin
                            if(busy||command_bits[3]!=!loading||load_count>=128||(load_count==127&&!command_bits[4])||
                                (!loading&&(quant[7:0]==0||!shifts_valid)))reject_write();
                            else begin w_valid<=1;w_first<=command_bits[3];w_last<=command_bits[4];end
                        end else reject_write();
                    end
                    10'h010:if(config_locked)reject_write();else input_address<=merge_bytes(input_address,write_data,write_strb);
                    10'h014:if(config_locked)reject_write();else output_address<=merge_bytes(output_address,write_data,write_strb);
                    10'h018:if(config_locked||(merged_shape&32'hfc00fc00)!=0)reject_write();else shape<=merged_shape;
                    10'h01c:if(config_locked||(merged_geometry&32'hf0f0fc80)!=0)reject_write();else geometry<=merged_geometry;
                    10'h020:if(config_locked||(merged_quant&32'hfe000000)!=0)reject_write();else quant<=merged_quant;
                    default:begin
                        if(aw_address>=10'h100&&aw_address<=10'h13c)begin
                            if(busy)reject_write();else weight_words[aw_address[5:2]]<=merge_bytes(weight_words[aw_address[5:2]],write_data,write_strb);
                        end else if(aw_address>=10'h140&&aw_address<=10'h15c)begin
                            if(config_locked)reject_write();else bias_words[aw_address[4:2]]<=merge_bytes(bias_words[aw_address[4:2]],write_data,write_strb);
                        end else if(aw_address>=10'h160&&aw_address<=10'h17c)begin
                            if(config_locked)reject_write();else mult_words[aw_address[4:2]]<=merge_bytes(mult_words[aw_address[4:2]],write_data,write_strb);
                        end else if(aw_address>=10'h180&&aw_address<=10'h19c)begin
                            if(config_locked||(merged_shift&32'hffffffc0)!=0||merged_shift>62)reject_write();
                            else shift_words[aw_address[4:2]]<=merged_shift[5:0];
                        end else reject_write();
                    end
                endcase
            end
            if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
            if(s_axi_arvalid&&s_axi_arready)begin
                s_axi_rvalid<=1;s_axi_rresp<=0;s_axi_rdata<=0;
                if(s_axi_araddr[1:0]!=0)s_axi_rresp<=2'b10;
                else case(s_axi_araddr)
                    10'h000:s_axi_rdata<=32'h4c554e47;
                    10'h004:s_axi_rdata<=0;
                    10'h008:s_axi_rdata<=status;
                    10'h00c:s_axi_rdata<=32'h00010000;
                    10'h010:s_axi_rdata<=input_address;10'h014:s_axi_rdata<=output_address;
                    10'h018:s_axi_rdata<=shape;10'h01c:s_axi_rdata<=geometry;10'h020:s_axi_rdata<=quant;
                    10'h024:s_axi_rdata<=written_words;10'h028:s_axi_rdata<=cache_hits;10'h02c:s_axi_rdata<=cache_misses;
                    10'h030:s_axi_rdata<=cycles;10'h034:s_axi_rdata<=completed;
                    10'h038:s_axi_rdata<=weight_beats;10'h03c:s_axi_rdata<=rejected;10'h040:s_axi_rdata<={24'd0,load_count};
                    default:begin
                        if(s_axi_araddr>=10'h100&&s_axi_araddr<=10'h13c)s_axi_rdata<=weight_words[s_axi_araddr[5:2]];
                        else if(s_axi_araddr>=10'h140&&s_axi_araddr<=10'h15c)s_axi_rdata<=bias_words[s_axi_araddr[4:2]];
                        else if(s_axi_araddr>=10'h160&&s_axi_araddr<=10'h17c)s_axi_rdata<=mult_words[s_axi_araddr[4:2]];
                        else if(s_axi_araddr>=10'h180&&s_axi_araddr<=10'h19c)s_axi_rdata<={26'd0,shift_words[s_axi_araddr[4:2]]};
                        else s_axi_rresp<=2'b10;
                    end
                endcase
            end
            if(w_valid&&w_ready)begin
                w_valid<=0;loading<=!w_last;load_count<=w_last?0:load_count+1'b1;weight_beats<=weight_beats+1'b1;
            end
            // Explicit immutable configuration window, including before VALID.
            // The scoped four-cycle configuration constraints rely on this.
            if(start_pending)begin
                if(start_guard!=0)start_guard<=start_guard-1'b1;
                else begin start_pending<=0;c_valid<=1;end
            end
            if(c_valid&&c_ready)begin c_valid<=0;active<=1;cycles<=0;end
            if(active&&cycles!=32'hffffffff)cycles<=cycles+1'b1;
            // Completion wins over simultaneous W1C: never lose a done event.
            if(core_done&&active)begin active<=0;done_sticky<=1;error_sticky<=core_error;completed<=completed+1'b1;end
        end
    end
endmodule
