`timescale 1ns/1ps
module tb_lung_memory_axi;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,w_valid=0,w_first=0,w_last=0,c_valid=0;
    reg[511:0]w_weights=0;reg[7:0]w_output_keep=0,w_input_zero_point=0,w_output_zero_point=0;
    reg[255:0]w_biases=0,w_multipliers=0;reg[47:0]w_shifts=0;reg w_relu=0;
    reg[31:0]c_base=0,c_output_base=65536;reg[9:0]c_height=0,c_width=0;
    reg[6:0]c_channels=0;reg c_kernel3=0,c_padding=0;reg[3:0]c_output_blocks=1,c_output_block=0;
    wire w_ready,c_ready,weights_loaded,busy,done,error;wire[31:0]written_words,cache_hits,cache_misses;
    wire[3:0]arid,arcache;wire[31:0]araddr;wire[7:0]arlen;wire[2:0]arsize,arprot;
    wire[1:0]arburst;wire arlock,arvalid,arready,rready;
    reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;reg rlast=0,rvalid=0;
    reg read_active=0,read_fault=0;integer latency=0,remaining=0,cycle=0;reg[31:0]read_address;
    wire awvalid,awready,wvalid,wready,wlast,bvalid,bready,awlock,commit;
    wire[31:0]awaddr,address;wire[63:0]wdata,data;wire[7:0]awlen,wstrb,strb;wire[3:0]awid,awcache,bid;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;
    integer mode=0;reg[1:0]response=0;reg[3:0]response_id=0;reg hold_final=1,force_block=0;
    wire block_b;
    reg[31:0]header[0:8],frames[0:7999],write_header[0:2],descriptors[0:1999],expected_addresses[0:19999];
    reg[63:0]features[0:99999],expected_words[0:19999],memory[0:19999];reg[511:0]weight_bank[0:59999];
    reg signed[31:0]bias_bank[0:15999],mult_bank[0:15999],expected_acc[0:159999];
    reg[5:0]shift_bank[0:15999];reg[7:0]expected_code[0:159999];reg[2:0]lanes[0:159999];
    reg lasts[0:159999];reg[20:0]coords[0:159999];
    integer received=0,writes=0,read_beats=0,bursts=0,acks=0,frame_commits=0,frame_pixels=0,frame_start=0;
    integer f,t,p,n,offset,pbase,lane,fh,index,a,expected_index,prior_ar,prior_writes,extra_cases=0;
    reg checking=0;
    assign block_b=force_block||(checking&&hold_final&&frame_commits==frame_pixels);
    lung_conv8_memory_axi dut(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),.w_first(w_first),.w_last(w_last),
        .w_weights(w_weights),.w_output_keep(w_output_keep),.w_input_zero_point(w_input_zero_point),.w_output_zero_point(w_output_zero_point),
        .w_biases(w_biases),.w_multipliers(w_multipliers),.w_shifts(w_shifts),.w_relu(w_relu),.weights_loaded(weights_loaded),
        .c_valid(c_valid),.c_ready(c_ready),.c_base(c_base),.c_height(c_height),.c_width(c_width),.c_channels(c_channels),
        .c_kernel3(c_kernel3),.c_padding(c_padding),.c_output_base(c_output_base),.c_output_blocks(c_output_blocks),.c_output_block(c_output_block),
        .busy(busy),.done(done),.error(error),.written_words(written_words),.cache_hits(cache_hits),.cache_misses(cache_misses),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),.m_axi_arcache(arcache),.m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
        .m_axi_awid(awid),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),.m_axi_awcache(awcache),.m_axi_awprot(awprot),
        .m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
        .m_axi_bid(bid),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready));
    lung_test_write_memory mem(.*);
    assign arready=rst_n&&!read_active&&!rvalid&&(cycle%11<8);
    always @(negedge clk)cycle=cycle+1;
    always @(posedge clk)begin
        if(!rst_n)begin read_active<=0;rvalid<=0;latency<=0;remaining<=0;end
        else begin
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||araddr[2:0]!=0||arlen>15||
                    ({1'b0,araddr[11:0]}+((arlen+1)*8))>4096)$fatal(1,"Illegal AXI read burst");
                read_active<=1;read_address<=araddr>>3;remaining<=arlen+1;latency<=cycle%5+1;
            end
            if(read_active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin
                    if(read_address>=header[1])$fatal(1,"Read outside frozen feature bank");
                    rvalid<=1;rid<=0;rdata<=features[read_address];rresp<=read_fault?2'b10:0;rlast<=remaining==1;
                end
            end
            if(rvalid&&rready)begin
                rvalid<=0;
                if(rlast)read_active<=0;else begin remaining<=remaining-1;read_address<=read_address+1'b1;latency<=cycle%3;end
            end
        end
    end
    always @(posedge clk)if(rst_n)begin
        if(arvalid&&arready)bursts=bursts+1;
        if(rvalid&&rready)read_beats=read_beats+1;
        if(bvalid&&bready)acks=acks+1;
        if(checking)begin
            if(dut.code_valid&&dut.code_ready)begin
                if(received>=header[4]||dut.code_error||$isunknown({dut.acc,dut.code,dut.code_x,dut.code_y,dut.code_lane,dut.code_last,dut.code_final})||
                    dut.acc!==expected_acc[received]||dut.code!==expected_code[received]||dut.code_lane!==lanes[received]||dut.code_last!==lasts[received]||
                    {dut.code_final,1'b0,dut.code_y,1'b0,dut.code_x}!==coords[received])$fatal(1,"Memory convolution code mismatch %0d",received);
                received=received+1;
            end
            if(commit)begin
                if(writes>=write_header[2]||frame_commits>=frame_pixels||address!==expected_addresses[writes]*8||data!==expected_words[writes]||strb!==255)$fatal(1,"AXI write mismatch %0d addr=%h/%h data=%h/%h",writes,address,expected_addresses[writes]*8,data,expected_words[writes]);
                index=(address>>3)-65536;if(index<0||index>=20000)$fatal(1,"Write outside simulation allocation");
                memory[index]=data;writes=writes+1;frame_commits=frame_commits+1;
            end
            if(done&&(error||busy||frame_commits!=frame_pixels||written_words!==frame_pixels||acks!=writes||mem.pending||read_active||rvalid))$fatal(1,"Frame done before all read/write responses");
        end
    end
    task send_weight(input reg[511:0]word,input reg first,input reg last);
        begin @(negedge clk);w_valid=1;w_first=first;w_last=last;w_weights=word;@(posedge clk);while(!w_ready)@(posedge clk);end
    endtask
    task start_frame;
        begin @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);@(negedge clk);c_valid=0;end
    endtask
    task simple_config;
        begin c_base=0;c_height=1;c_width=1;c_channels=1;c_kernel3=0;c_padding=0;c_output_base=65536;c_output_blocks=1;c_output_block=0;end
    endtask
    task wait_fault;
        begin while(!done)@(negedge clk);if(!error||busy)$fatal(1,"Invalid/error frame accepted as valid");extra_cases=extra_cases+1;@(negedge clk);end
    endtask
    initial begin
        fh=$fopen("../lung_cache_axi_20261015/vectors/header.mem","r");if(!fh)$fatal(1,"Missing frozen inputs");$fclose(fh);
        $readmemh("../lung_cache_axi_20261015/vectors/header.mem",header,0,8);$readmemh("vectors/header.mem",write_header,0,2);
        if($isunknown({header[0],header[1],header[2],header[3],header[4],write_header[2]})||header[0]==0||header[4]==0||
            header[0]>285||header[1]>100000||header[2]>60000||header[3]>16000||header[4]>160000||write_header[0]!=header[0]||write_header[1]!=header[4]||write_header[2]==0||write_header[2]>20000)$fatal(1,"Bad input fixture counts");
        $readmemh("../lung_cache_axi_20261015/vectors/frames.mem",frames,0,header[0]*16-1);$readmemh("../lung_cache_axi_20261015/vectors/features.mem",features,0,header[1]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/weights.mem",weight_bank,0,header[2]-1);$readmemh("../lung_cache_axi_20261015/vectors/bias.mem",bias_bank,0,header[3]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/multipliers.mem",mult_bank,0,header[3]-1);$readmemh("../lung_cache_axi_20261015/vectors/shifts.mem",shift_bank,0,header[3]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/accumulators.mem",expected_acc,0,header[4]-1);$readmemh("../lung_cache_axi_20261015/vectors/codes.mem",expected_code,0,header[4]-1);
        $readmemh("../lung_cache_axi_20261015/vectors/lanes.mem",lanes,0,header[4]-1);$readmemh("../lung_cache_axi_20261015/vectors/lasts.mem",lasts,0,header[4]-1);$readmemh("../lung_cache_axi_20261015/vectors/coords.mem",coords,0,header[4]-1);
        $readmemh("vectors/descriptors.mem",descriptors,0,header[0]*7-1);$readmemh("vectors/addresses.mem",expected_addresses,0,write_header[2]-1);$readmemh("vectors/words.mem",expected_words,0,write_header[2]-1);
        repeat(3)@(negedge clk);rst_n=1;checking=1;
        for(f=0;f<header[0];f=f+1)begin
            t=f*16;offset=frames[t+10];n=frames[t+11];pbase=frames[t+12];
            @(negedge clk);w_output_keep=frames[t+8][7:0];w_input_zero_point=frames[t+6][7:0];w_output_zero_point=frames[t+7][7:0];w_relu=frames[t+9][0];
            for(lane=0;lane<8;lane=lane+1)begin w_biases[32*lane+:32]=bias_bank[pbase+lane];w_multipliers[32*lane+:32]=mult_bank[pbase+lane];w_shifts[6*lane+:6]=shift_bank[pbase+lane];end
            for(p=0;p<n;p=p+1)send_weight(weight_bank[offset+p],p==0,p==n-1);
            @(negedge clk);w_valid=0;c_base=frames[t];c_height=frames[t+1][9:0];c_width=frames[t+2][9:0];c_channels=frames[t+3][6:0];c_kernel3=frames[t+4]==3;c_padding=frames[t+5][0];
            c_output_base=descriptors[f*7];c_output_blocks=descriptors[f*7+3];c_output_block=descriptors[f*7+4];
            frame_start=descriptors[f*7+5];frame_pixels=descriptors[f*7+6];frame_commits=0;hold_final=1;mode=f%3;
            if(frame_start!=writes)$fatal(1,"Fixture write offsets differ");
            for(a=0;a<20000;a=a+1)memory[a]=64'hdeadbeefcafef00d;
            start_frame();c_base=32'hffffffff;c_height=0;c_width=0;c_channels=0;c_output_base=32'hffffffff;c_output_blocks=0;c_output_block=15;
            while(!dut.conv_done)@(negedge clk);
            while(frame_commits!=frame_pixels)@(negedge clk);
            repeat(5)begin @(negedge clk);if(done||!busy||c_ready||w_ready)$fatal(1,"Last B not included in busy/done");end
            hold_final=0;while(!done)@(negedge clk);
            if(error||busy||written_words!==frame_pixels||frame_commits!=frame_pixels||acks!=writes)$fatal(1,"Incomplete frame writeback");
            expected_index=frame_start;
            for(a=0;a<20000;a=a+1)begin
                if(expected_index<frame_start+frame_pixels&&a==expected_addresses[expected_index]-65536)begin
                    if(memory[a]!==expected_words[expected_index])$fatal(1,"Final memory wrong");expected_index=expected_index+1;
                end else if(memory[a]!==64'hdeadbeefcafef00d)$fatal(1,"Canary or other channel block overwritten");
            end
            if(expected_index!=frame_start+frame_pixels)$fatal(1,"Incomplete canary coverage");
            if(f%15==0)$display("Memory AXI frame %0d/%0d complete (%0d codes, %0d writes)",f+1,header[0],received,writes);
        end
        if(received!=header[4]||writes!=write_header[2]||acks!=writes||bursts!=header[7]||read_beats!=header[8])$fatal(1,"Incomplete memory integration counts");
        checking=0;hold_final=0;prior_ar=bursts;prior_writes=acks;
        for(p=0;p<12;p=p+1)begin
            simple_config();case(p)
                0:c_height=0;1:c_width=513;2:c_channels=97;3:c_padding=1;
                4:begin c_kernel3=1;c_padding=0;end
                5:c_base=32'h20000000;6:c_output_base=0;7:c_output_base=32'h20000000;
                8:c_output_blocks=0;9:c_output_blocks=13;10:c_output_block=1;
                11:begin c_base=32'h1fffffff;c_width=2;end
            endcase
            start_frame();wait_fault();if(bursts!=prior_ar||acks!=prior_writes||awvalid||arvalid||wvalid)$fatal(1,"Invalid config issued memory transaction");
        end
        @(negedge clk);w_output_keep=1;w_input_zero_point=128;w_output_zero_point=128;w_biases=0;w_multipliers={8{32'd1073741824}};w_shifts={8{6'd31}};w_relu=0;
        send_weight(512'd1,1,1);@(negedge clk);w_valid=0;simple_config();read_fault=1;start_frame();wait_fault();
        if(written_words!=0||acks!=prior_writes)$fatal(1,"Read-error pixel written to memory");read_fault=0;
        response=2;simple_config();start_frame();wait_fault();if(written_words!=0||acks!=prior_writes+1)$fatal(1,"BRESP not propagated to frame");response=0;
        force_block=1;simple_config();start_frame();while(!commit)@(negedge clk);repeat(4)@(negedge clk);
        if(done||!busy)$fatal(1,"Missing pending write before reset");rst_n=0;repeat(3)@(negedge clk);rst_n=1;@(negedge clk);
        if(busy||done||awvalid||wvalid||arvalid||rready||bready||weights_loaded)$fatal(1,"Integrated shared reset failed");
        $display("Lung memory AXI convolution RTL: PASS (%0d frames, %0d codes, %0d writes, %0d read bursts, %0d read beats, %0d rejection/error cases; HWC8, canaries, captured commands, delayed final B, shared reset)",header[0],received,writes,header[7],header[8],extra_cases);$finish;
    end
    initial begin #1500000000;$fatal(1,"Memory AXI integration timeout");end
endmodule
