`timescale 1ns/1ps
// Independent AXI-Lite AW/W holding, byte strobes, sticky result. No soft reset.
module lung_tensor_control(
    input wire clk,input wire rst_n,
    input wire[9:0]s_axi_awaddr,input wire[2:0]s_axi_awprot,input wire s_axi_awvalid,output wire s_axi_awready,
    input wire[31:0]s_axi_wdata,input wire[3:0]s_axi_wstrb,input wire s_axi_wvalid,output wire s_axi_wready,
    output reg[1:0]s_axi_bresp,output reg s_axi_bvalid,input wire s_axi_bready,
    input wire[9:0]s_axi_araddr,input wire[2:0]s_axi_arprot,input wire s_axi_arvalid,output wire s_axi_arready,
    output reg[31:0]s_axi_rdata,output reg[1:0]s_axi_rresp,output reg s_axi_rvalid,input wire s_axi_rready,
    output reg c_valid,input wire c_ready,
    output wire[1:0]c_op,output wire[31:0]c_a,c_b,c_dst,
    output wire[9:0]c_height,c_width,output wire[6:0]c_ca,c_cb,
    output wire[7:0]c_za,c_zb,c_zo,output wire[30:0]c_ma,c_mb,output wire[5:0]c_sa,c_sb,
    input wire core_busy,core_done,core_error,input wire[31:0]written_words,read_words,cycles
);
    reg[31:0]config_words[0:9];
    reg aw_hold,w_hold,active,done_sticky,error_sticky,control_error;
    reg[9:0]aw_address;reg[31:0]write_data;reg[3:0]write_strobe;
    reg[31:0]completed,rejected;
    wire busy=active||c_valid||core_busy;
    wire[31:0]status={23'd0,control_error,5'd0,error_sticky,done_sticky,busy};
    assign c_a=config_words[0];assign c_b=config_words[1];assign c_dst=config_words[2];
    assign c_height=config_words[3][9:0];assign c_width=config_words[3][25:16];
    assign c_op=config_words[4][1:0];assign c_ca=config_words[4][14:8];assign c_cb=config_words[4][22:16];
    assign c_za=config_words[5][7:0];assign c_zb=config_words[5][15:8];assign c_zo=config_words[5][23:16];
    assign c_ma=config_words[6][30:0];assign c_sa=config_words[7][5:0];
    assign c_mb=config_words[8][30:0];assign c_sb=config_words[9][5:0];
    assign s_axi_awready=rst_n&&!aw_hold&&!s_axi_bvalid;
    assign s_axi_wready=rst_n&&!w_hold&&!s_axi_bvalid;
    assign s_axi_arready=rst_n&&!s_axi_rvalid;
    function[31:0]merge(input[31:0]old_value,new_value,input[3:0]strb);
        integer j;begin merge=old_value;for(j=0;j<4;j=j+1)if(strb[j])merge[8*j+:8]=new_value[8*j+:8];end
    endfunction
    wire[3:0]config_index=aw_address[5:2]-4;
    wire[31:0]merged=merge(config_words[config_index],write_data,write_strobe);
    wire[31:0]command=merge(0,write_data,write_strobe);
    wire bad_reserved=(aw_address==10'h01c&&(merged&32'hfc00fc00)!=0)||
        (aw_address==10'h020&&(merged&32'hff8080fc)!=0)||
        (aw_address==10'h024&&(merged&32'hff000000)!=0)||
        ((aw_address==10'h028||aw_address==10'h030)&&merged[31])||
        ((aw_address==10'h02c||aw_address==10'h034)&&(merged&32'hffffffc0)!=0);
    task reject_write;
        begin s_axi_bresp<=2'b10;control_error<=1;rejected<=rejected+1'b1;end
    endtask
    integer i;
    always @(posedge clk)begin
        if(!rst_n)begin
            aw_hold<=0;w_hold<=0;aw_address<=0;write_data<=0;write_strobe<=0;
            s_axi_bresp<=0;s_axi_bvalid<=0;s_axi_rdata<=0;s_axi_rresp<=0;s_axi_rvalid<=0;
            active<=0;c_valid<=0;done_sticky<=0;error_sticky<=0;control_error<=0;completed<=0;rejected<=0;
            for(i=0;i<10;i=i+1)config_words[i]<=0;
        end else begin
            if(s_axi_awvalid&&s_axi_awready)begin aw_hold<=1;aw_address<=s_axi_awaddr;end
            if(s_axi_wvalid&&s_axi_wready)begin w_hold<=1;write_data<=s_axi_wdata;write_strobe<=s_axi_wstrb;end
            if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
            if(aw_hold&&w_hold&&!s_axi_bvalid)begin
                aw_hold<=0;w_hold<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
                if(aw_address[1:0]!=0)reject_write();
                else if(write_strobe==0)begin end
                else if(aw_address==10'h004)begin
                    if(command==0)begin end
                    else if(command==2)begin done_sticky<=0;error_sticky<=0;control_error<=0;end
                    else if(command==1)begin
                        if(busy)reject_write();
                        else begin c_valid<=1;active<=1;done_sticky<=0;error_sticky<=0;control_error<=0;end
                    end else reject_write();
                end else if(aw_address>=10'h010&&aw_address<=10'h034)begin
                    if(busy||bad_reserved)reject_write();else config_words[config_index]<=merged;
                end else reject_write();
            end
            if(c_valid&&c_ready)c_valid<=0;
            if(core_done&&active)begin
                active<=0;done_sticky<=1;error_sticky<=core_error;completed<=completed+1'b1;
            end
            if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
            if(s_axi_arvalid&&s_axi_arready)begin
                s_axi_rvalid<=1;s_axi_rresp<=0;s_axi_rdata<=0;
                if(s_axi_araddr[1:0]!=0)s_axi_rresp<=2'b10;
                else if(s_axi_araddr>=10'h010&&s_axi_araddr<=10'h034)s_axi_rdata<=config_words[s_axi_araddr[5:2]-4];
                else case(s_axi_araddr)
                    10'h000:s_axi_rdata<=32'h54454e53;
                    10'h004:s_axi_rdata<=0;
                    10'h008:s_axi_rdata<=status;
                    10'h00c:s_axi_rdata<=32'h00010000;
                    10'h038:s_axi_rdata<=written_words;
                    10'h03c:s_axi_rdata<=read_words;
                    10'h040:s_axi_rdata<=cycles;
                    10'h044:s_axi_rdata<=completed;
                    10'h048:s_axi_rdata<=rejected;
                    default:s_axi_rresp<=2'b10;
                endcase
            end
        end
    end
endmodule
