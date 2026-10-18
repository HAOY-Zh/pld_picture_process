`timescale 1ns/1ps
module tb_lung_axil_accel;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    wire[9:0]s_axi_awaddr,s_axi_araddr;wire[2:0]s_axi_awprot,s_axi_arprot;
    wire[31:0]s_axi_wdata,s_axi_rdata;wire[3:0]s_axi_wstrb;wire[1:0]s_axi_bresp,s_axi_rresp;
    wire s_axi_awvalid,s_axi_awready,s_axi_wvalid,s_axi_wready,s_axi_bvalid,s_axi_bready,s_axi_arvalid,s_axi_arready,s_axi_rvalid,s_axi_rready;
    wire[3:0]arid,arcache;wire[31:0]araddr;wire[7:0]arlen;wire[2:0]arsize,arprot;
    wire[1:0]arburst;wire arlock,arvalid,arready,rready;
    reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;reg rlast=0,rvalid=0;
    reg read_active=0,read_fault=0;integer latency=0,remaining=0,cycle=0;reg[31:0]read_address;
    wire awvalid,awready,wvalid,wready,wlast,bvalid,bready,awlock,commit;
    wire[31:0]awaddr,address;wire[63:0]wdata,data;wire[7:0]awlen,wstrb,strb;wire[3:0]awid,awcache,bid;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;
    integer mode=0;reg[1:0]response=0;reg[3:0]response_id=0;reg hold_final=1,force_block=0,checking=0;
    integer frame_commits=0,frame_pixels=0,writes=0,acks=0,bursts=0,read_beats=0,frame_start;
    wire block_b=force_block||(checking&&hold_final&&frame_commits==frame_pixels);
    reg[31:0]header[0:8],frames[0:7999],traffic[0:1999],write_header[0:2],descriptors[0:1999],expected_addresses[0:19999];
    reg[63:0]features[0:99999],expected_words[0:19999],memory[0:19999];reg[511:0]weights[0:59999];
    reg[31:0]biases[0:15999],multipliers[0:15999];reg[5:0]shifts[0:15999];
    integer f,t,p,k,n,offset,param_base,fh,index,a,expected_index,loads=0,prior_ar,prior_acks,errors=0;
    reg[31:0]status,value;
    reg[100:0]last_config;integer stable_config_cycles=0;
    lung_test_axil_master host(.*);lung_test_write_memory mem(.*);
    lung_conv8_axil dut(.clk(clk),.rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),.s_axi_awprot(s_axi_awprot),.s_axi_awvalid(s_axi_awvalid),.s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),.s_axi_wstrb(s_axi_wstrb),.s_axi_wvalid(s_axi_wvalid),.s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),.s_axi_bvalid(s_axi_bvalid),.s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),.s_axi_arprot(s_axi_arprot),.s_axi_arvalid(s_axi_arvalid),.s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),.s_axi_rresp(s_axi_rresp),.s_axi_rvalid(s_axi_rvalid),.s_axi_rready(s_axi_rready),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),.m_axi_arcache(arcache),.m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
        .m_axi_awid(awid),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),.m_axi_awcache(awcache),.m_axi_awprot(awprot),
        .m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
        .m_axi_bid(bid),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready));
    assign arready=rst_n&&!read_active&&!rvalid&&cycle%11<8;
    always @(negedge clk)cycle=cycle+1;
    always @(posedge clk)begin
        if(!rst_n)begin read_active<=0;rvalid<=0;latency<=0;remaining<=0;stable_config_cycles=0;last_config=0;end
        else begin
            if({dut.c_base,dut.c_height,dut.c_width,dut.c_channels,dut.c_kernel3,dut.c_padding,dut.c_output_base,dut.c_output_blocks,dut.c_output_block}!==last_config)stable_config_cycles=0;
            else stable_config_cycles=stable_config_cycles+1;
            last_config={dut.c_base,dut.c_height,dut.c_width,dut.c_channels,dut.c_kernel3,dut.c_padding,dut.c_output_base,dut.c_output_blocks,dut.c_output_block};
            if(dut.c_valid&&stable_config_cycles<4)$fatal(1,"Real engine configuration not stable for multicycle capture");
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||araddr[2:0]!=0||arlen>15||({1'b0,araddr[11:0]}+((arlen+1)*8))>4096)$fatal(1,"Bad read AXI command");
                read_active<=1;read_address<=araddr>>3;remaining<=arlen+1;latency<=cycle%5+1;bursts=bursts+1;
            end
            if(read_active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin if(read_address>=header[1])$fatal(1,"Read outside fixture span");rvalid<=1;rid<=0;rdata<=features[read_address];rresp<=read_fault?2:0;rlast<=remaining==1;end
            end
            if(rvalid&&rready)begin rvalid<=0;read_beats=read_beats+1;
                if(rlast)read_active<=0;else begin remaining<=remaining-1;read_address<=read_address+1'b1;latency<=cycle%3;end
            end
            if(bvalid&&bready)acks=acks+1;
            if(checking&&commit)begin
                if(writes>=write_header[2]||frame_commits>=frame_pixels||address!==expected_addresses[writes]*8||data!==expected_words[writes]||strb!==255)$fatal(1,"AXI-Lite initiated output mismatch %0d",writes);
                index=(address>>3)-65536;if(index<0||index>=20000)$fatal(1,"Output allocation exceeded");
                memory[index]=data;writes=writes+1;frame_commits=frame_commits+1;
            end
        end
    end
    task put(input reg[9:0]address,input reg[31:0]word);
        begin host.write_reg(address,word,15,host.writes%3,0);end
    endtask
    task geometry(input reg[31:0]input_word,input integer h,input integer w,input integer cin,input integer k3,input integer pad,
                  input reg[31:0]output_word,input integer blocks,input integer block);
        begin put(16,input_word*8);put(20,output_word*8);put(24,h|(w<<16));put(28,cin|(k3<<8)|(pad<<9)|(blocks<<16)|(block<<24));end
    endtask
    task await_completion(input reg failed);
        begin
            host.read_reg(8,0,status);while(!status[1]||status[0])host.read_reg(8,0,status);
            if(status[2]!==failed||status[8])$fatal(1,"Bad command status: %h",status);
        end
    endtask
    initial begin
        fh=$fopen("../lung_cache_axi_20261015/vectors/header.mem","r");if(!fh)$fatal(1,"Missing frozen fixtures");$fclose(fh);
        $readmemh("../lung_cache_axi_20261015/vectors/header.mem",header,0,8);
        $readmemh("../lung_memory_axi_20261015/vectors/header.mem",write_header,0,2);
        if(header[0]!==147||header[4]!==68849||write_header[2]!==8866)$fatal(1,"Wrong frozen fixture counts");
        $readmemh("../lung_cache_axi_20261015/vectors/frames.mem",frames,0,header[0]*16-1);
        $readmemh("../lung_cache_axi_20261015/vectors/features.mem",features,0,header[1]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/weights.mem",weights,0,header[2]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/bias.mem",biases,0,header[3]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/multipliers.mem",multipliers,0,header[3]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/shifts.mem",shifts,0,header[3]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/traffic.mem",traffic,0,header[0]*6-1);
        $readmemh("../lung_memory_axi_20261015/vectors/descriptors.mem",descriptors,0,header[0]*7-1);
        $readmemh("../lung_memory_axi_20261015/vectors/addresses.mem",expected_addresses,0,write_header[2]-1);
        $readmemh("../lung_memory_axi_20261015/vectors/words.mem",expected_words,0,write_header[2]-1);
        repeat(3)@(negedge clk);rst_n=1;checking=1;
        host.read_reg(0,0,value);if(value!==32'h4c554e47)$fatal(1,"Wrong hardware ID");
        for(f=0;f<header[0];f=f+1)begin
            t=f*16;offset=frames[t+10];n=frames[t+11];param_base=frames[t+12];
            geometry(frames[t],frames[t+1],frames[t+2],frames[t+3],frames[t+4]==3,frames[t+5],descriptors[f*7],descriptors[f*7+3],descriptors[f*7+4]);
            put(32,frames[t+8]|(frames[t+6]<<8)|(frames[t+7]<<16)|(frames[t+9]<<24));
            for(k=0;k<8;k=k+1)begin put(10'h140+4*k,biases[param_base+k]);put(10'h160+4*k,multipliers[param_base+k]);put(10'h180+4*k,shifts[param_base+k]);end
            for(p=0;p<n;p=p+1)begin
                for(k=0;k<16;k=k+1)put(10'h100+4*k,weights[offset+p][32*k+:32]);
                put(4,4|((p==0)<<3)|((p==n-1)<<4));loads=loads+1;
            end
            host.read_reg(8,0,status);if(status[0]||!status[3]||status[5]||status[8])$fatal(1,"Weights not loaded through registers");
            host.read_reg(10'h038,0,value);if(value!==loads)$fatal(1,"Weight handshake count differs");
            frame_start=descriptors[f*7+5];frame_pixels=descriptors[f*7+6];frame_commits=0;hold_final=1;mode=f%3;
            for(a=0;a<20000;a=a+1)memory[a]=64'hdeadbeefcafef00d;
            put(4,1);while(frame_commits!=frame_pixels)@(negedge clk);
            host.read_reg(8,0,status);if(!status[0]||status[1])$fatal(1,"DONE before final B response");
            if(f==0)begin host.write_reg(16,0,15,1,2);host.write_reg(4,1,15,2,2);put(4,2);end
            hold_final=0;await_completion(0);
            host.read_reg(10'h024,0,value);if(value!==frame_pixels||acks!=writes)$fatal(1,"Write completion counter mismatch");
            host.read_reg(10'h028,0,value);if(value!==traffic[f*6+1])$fatal(1,"Cache hits changed");
            host.read_reg(10'h02c,0,value);if(value!==traffic[f*6+2])$fatal(1,"Cache misses changed");
            host.read_reg(10'h034,0,value);if(value!==f+1)$fatal(1,"Completed command counter differs");
            host.read_reg(10'h030,0,value);if(value==0)$fatal(1,"Missing cycles");
            expected_index=frame_start;
            for(a=0;a<20000;a=a+1)begin
                if(expected_index<frame_start+frame_pixels&&a==expected_addresses[expected_index]-65536)begin
                    if(memory[a]!==expected_words[expected_index])$fatal(1,"Stored HWC8 word differs");expected_index=expected_index+1;
                end else if(memory[a]!==64'hdeadbeefcafef00d)$fatal(1,"Canary changed outside assigned block");
            end
            if(expected_index!=frame_start+frame_pixels)$fatal(1,"Missing output allocation coverage");
            if(f%15==0)$display("AXI-Lite accelerator %0d/%0d complete (%0d memory words)",f+1,header[0],writes);
        end
        if(writes!=8866||acks!=writes||loads!=header[2]||bursts!=header[7]||read_beats!=header[8])$fatal(1,"Incomplete register-driven integration");
        checking=0;hold_final=0;prior_ar=bursts;prior_acks=acks;
        for(p=0;p<3;p=p+1)begin
            geometry(0,p==0?0:1,1,1,0,0,p==1?0:65536,1,p==2?1:0);put(4,1);await_completion(1);errors=errors+1;
            if(bursts!=prior_ar||acks!=prior_acks)$fatal(1,"Invalid register config reached AXI memory");
        end
        geometry(0,1,1,1,0,0,65536,1,0);put(32,32'h00808001);
        for(k=0;k<8;k=k+1)begin put(10'h140+4*k,0);put(10'h160+4*k,32'h40000000);put(10'h180+4*k,31);end
        for(k=0;k<16;k=k+1)put(10'h100+4*k,k==0?1:0);put(4,28);
        read_fault=1;put(4,1);await_completion(1);errors=errors+1;host.read_reg(10'h024,0,value);
        if(value!=0||acks!=prior_acks)$fatal(1,"Read error written to output");read_fault=0;
        response=2;put(4,1);await_completion(1);errors=errors+1;host.read_reg(10'h024,0,value);
        if(value!=0||acks!=prior_acks+1)$fatal(1,"Write error not visible in control status");response=0;
        force_block=1;put(4,1);while(!commit)@(negedge clk);host.read_reg(8,0,status);if(!status[0]||status[1])$fatal(1,"Expected in-flight B before shared reset");
        @(negedge clk);rst_n=0;repeat(3)@(negedge clk);rst_n=1;@(negedge clk);
        host.read_reg(8,0,status);if(status!=0||awvalid||wvalid||arvalid||rready||bready)$fatal(1,"Shared reset did not clear accelerator");
        $display("Lung AXI-Lite accelerator RTL: PASS (%0d frames, %0d words, %0d weight packets, %0d memory read bursts, %0d memory read beats, %0d error cases; register-only launch, locks, counters, final-B wait, canaries, shared reset)",header[0],writes,header[2],header[7],header[8],errors);$finish;
    end
    initial begin #2000000000;$fatal(1,"AXI-Lite accelerator timeout");end
endmodule
