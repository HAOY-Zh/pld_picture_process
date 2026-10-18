`timescale 1ns/1ps
module tb_lung_tensor_axil;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    wire[9:0]s_axi_awaddr,s_axi_araddr;wire[2:0]s_axi_awprot,s_axi_arprot;
    wire[31:0]s_axi_wdata,s_axi_rdata;wire[3:0]s_axi_wstrb;wire[1:0]s_axi_bresp,s_axi_rresp;
    wire s_axi_awvalid,s_axi_awready,s_axi_wvalid,s_axi_wready,s_axi_bvalid,s_axi_bready,s_axi_arvalid,s_axi_arready,s_axi_rvalid,s_axi_rready;
    wire[3:0]arid,arcache;wire[31:0]araddr;wire[7:0]arlen;wire[2:0]arsize,arprot;
    wire[1:0]arburst;wire arlock,arvalid,arready,rready;
    reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;reg rlast=0,rvalid=0;
    reg read_active=0;integer read_fault=0,read_mode=0,latency=0,remaining=0,cycle=0;reg[31:0]read_address;
    wire awvalid,awready,wvalid,wready,wlast,bvalid,bready,awlock,commit;
    wire[31:0]awaddr,address;wire[63:0]wdata,data;wire[7:0]awlen,wstrb,strb;wire[3:0]awid,awcache,bid;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;
    integer mode=0;reg[1:0]response=0;reg[3:0]response_id=0;reg hold_final=1,force_block=0,checking=0;
    integer frame_commits=0,frame_pixels=0,writes=0,acks=0,bursts=0,read_beats=0;
    wire block_b=force_block||(checking&&hold_final&&frame_commits==frame_pixels);
    reg[31:0]header[0:4],commands[0:1999];
    reg[63:0]a_bank[0:99999],b_bank[0:99999],gold[0:99999],memory[0:19999];
    integer f,t,k,p,fh,a,a_offset=0,a_length=0,b_offset=0,b_length=0,gold_offset=0;
    integer prior_ar,prior_acks,prior_writes,errors=0,normal_writes=0,normal_reads=0;
    reg[31:0]status,value;
    reg held_ar=0;reg[56:0]previous_ar;
    lung_test_axil_master host(.*);lung_test_write_memory mem(.*);
    lung_tensor_axil dut(.clk(clk),.rst_n(rst_n),
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
        if(!rst_n)begin read_active<=0;rvalid<=0;latency<=0;remaining<=0;held_ar<=0;end
        else begin
            if(held_ar&&(!arvalid||{arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot}!==previous_ar))$fatal(1,"AR changed under backpressure");
            held_ar<=arvalid&&!arready;previous_ar<={arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot};
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||araddr[2:0]!=0||arlen!=0||arlock||arprot!=0||arcache!=3)$fatal(1,"Bad tensor AR");
                read_active<=1;read_address<=araddr>>3;remaining<=read_fault==3?2:1;
                read_mode<=read_fault;latency<=cycle%5+1;bursts=bursts+1;
            end
            if(read_active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin
                    if(read_address>=4096&&read_address<4096+a_length)rdata<=a_bank[a_offset+read_address-4096];
                    else if(read_address>=32768&&read_address<32768+b_length)rdata<=b_bank[b_offset+read_address-32768];
                    else $fatal(1,"Read outside logical tensor allocation addr=%h",read_address);
                    rvalid<=1;rid<=read_mode==2?7:0;rresp<=read_mode==1?2:0;rlast<=remaining==1;
                end
            end
            if(rvalid&&rready)begin
                rvalid<=0;read_beats=read_beats+1;
                if(rlast)read_active<=0;else begin remaining<=remaining-1;latency<=5;end
            end
            if(bvalid&&bready)acks=acks+1;
            if(commit)begin
                writes=writes+1;
                if(checking)begin
                    if(frame_commits>=frame_pixels||address!==(65536+frame_commits)*8||data!==gold[gold_offset+frame_commits]||strb!==255)
                        $fatal(1,"Tensor mismatch frame=%0d word=%0d address=%h data=%h expected=%h",f,frame_commits,address,data,gold[gold_offset+frame_commits]);
                    memory[frame_commits]=data;frame_commits=frame_commits+1;
                end
            end
            if(dut.core_done&&(read_active||rvalid||mem.pending||bvalid))$fatal(1,"DONE before shared memory transactions drained");
        end
    end
    task put(input reg[9:0]address,input reg[31:0]word);
        begin host.write_reg(address,word,15,host.writes%3,0);end
    endtask
    task setup(input integer op,input integer h,input integer w,input integer ca,input integer cb);
        begin
            put(16,4096*8);put(20,32768*8);put(24,65536*8);put(28,h|(w<<16));
            put(32,op|(ca<<8)|(cb<<16));put(36,128|(64<<8)|((op<2?128:99)<<16));
            put(40,32'h40000000);put(44,31);put(48,32'h60000000);put(52,30);
        end
    endtask
    task await_completion(input reg failed);
        begin
            host.read_reg(8,0,status);while(!status[1]||status[0])host.read_reg(8,0,status);
            if(status[2]!==failed||status[8])$fatal(1,"Bad tensor status: %h",status);
        end
    endtask
    initial begin
        fh=$fopen("vectors/header.mem","r");if(!fh)$fatal(1,"Missing tensor fixtures");$fclose(fh);
        $readmemh("vectors/header.mem",header,0,4);
        if(header[0]!==26||header[4]!==18||header[1]>100000||header[2]>100000||header[3]>100000)$fatal(1,"Wrong tensor fixture dimensions");
        $readmemh("vectors/commands.mem",commands,0,header[0]*18-1);
        $readmemh("vectors/a.mem",a_bank,0,header[1]-1);
        $readmemh("vectors/b.mem",b_bank,0,header[2]-1);
        $readmemh("vectors/gold.mem",gold,0,header[3]-1);
        repeat(3)@(negedge clk);rst_n=1;
        host.read_reg(0,0,value);if(value!==32'h54454e53)$fatal(1,"Wrong tensor ID");
        host.read_reg(12,0,value);if(value!==32'h10000)$fatal(1,"Wrong tensor ABI");
        host.read_reg(1,2,value);host.read_reg(10'h3fc,2,value);
        host.write_reg(0,1,15,1,2);host.write_reg(10'h3fc,1,15,2,2);host.write_reg(1,1,15,0,2);
        host.write_reg(10'h3fc,32'hffffffff,0,2,0);
        put(16,32'h12345678);host.write_reg(16,32'haabbccdd,5,2,0);
        host.read_reg(16,0,value);if(value!==32'h12bb56dd)$fatal(1,"WSTRB merge failed");
        host.write_reg(28,32'hffffffff,15,1,2);host.write_reg(32,32'h80000000,15,0,2);
        host.write_reg(36,32'h80000000,15,1,2);host.write_reg(40,32'h80000000,15,2,2);
        host.write_reg(44,64,15,0,2);put(4,2);
        checking=1;
        for(f=0;f<header[0];f=f+1)begin
            t=f*18;a_offset=commands[t+12];a_length=commands[t+13];
            b_offset=commands[t+14];b_length=commands[t+15];gold_offset=commands[t+16];frame_pixels=commands[t+17];
            setup(commands[t],commands[t+1],commands[t+2],commands[t+3],commands[t+4]);
            put(36,commands[t+5]|(commands[t+6]<<8)|(commands[t+7]<<16));
            put(40,commands[t+8]);put(44,commands[t+9]);put(48,commands[t+10]);put(52,commands[t+11]);
            frame_commits=0;hold_final=1;mode=f%3;prior_ar=bursts;prior_acks=acks;
            for(a=0;a<20000;a=a+1)memory[a]=64'hdeadbeefcafef00d;
            put(4,1);
            if(f==0)begin host.write_reg(16,0,15,1,2);host.write_reg(4,1,15,2,2);put(4,2);end
            while(frame_commits!=frame_pixels)@(negedge clk);
            host.read_reg(8,0,status);if(!status[0]||status[1])$fatal(1,"Tensor DONE before last B response");
            hold_final=0;await_completion(0);
            host.read_reg(56,0,value);if(value!==frame_pixels||acks-prior_acks!=frame_pixels)$fatal(1,"Tensor write counter mismatch");
            host.read_reg(60,0,value);if(value!==bursts-prior_ar)$fatal(1,"Tensor read counter mismatch");
            host.read_reg(64,0,value);if(value==0)$fatal(1,"Missing tensor cycles");
            host.read_reg(68,0,value);if(value!==f+1)$fatal(1,"Tensor completion counter mismatch");
            for(a=0;a<20000;a=a+1)begin
                if(a<frame_pixels)begin if(memory[a]!==gold[gold_offset+a])$fatal(1,"Stored tensor differs");end
                else if(memory[a]!==64'hdeadbeefcafef00d)$fatal(1,"Tensor canary overwritten");
            end
            $display("Tensor operator frame %0d/%0d PASS (%0d output words)",f+1,header[0],frame_pixels);
        end
        if(writes!=header[3]||acks!=writes)$fatal(1,"Missing tensor output coverage");
        normal_writes=writes;normal_reads=bursts;checking=0;hold_final=0;
        prior_ar=bursts;prior_acks=acks;
        for(p=0;p<14;p=p+1)begin
            setup(p==3?1:((p==4||p==5||p==6||p==7||p==13)?2:0),4,4,8,(p==4||p==5||p==6||p==7||p==13)?8:0);
            case(p)
                0:put(28,4<<16);
                1:put(28,3|(4<<16));
                2:put(32,0);
                3:put(28,257|(4<<16));
                4:put(32,2|(90<<8)|(8<<16));
                5:put(32,2|(8<<8));
                6:put(40,0);
                7:put(44,63);
                8:put(16,4096*8+1);
                9:put(16,32'hfffffff8);
                10:put(24,4096*8);
                11:put(24,32'hfffffff8);
                12:put(36,128|(64<<8)|(99<<16));
                13:put(20,65536*8);
            endcase
            put(4,1);await_completion(1);errors=errors+1;
            if(bursts!=prior_ar||acks!=prior_acks)$fatal(1,"Invalid tensor config reached memory");
        end
        a_offset=0;a_length=4;b_offset=0;b_length=0;setup(0,2,2,1,0);
        for(p=1;p<=3;p=p+1)begin
            read_fault=p;put(4,1);await_completion(1);errors=errors+1;
            host.read_reg(56,0,value);if(value!=0||acks!=prior_acks)$fatal(1,"Read fault wrote tensor output");
        end
        read_fault=0;
        for(p=0;p<2;p=p+1)begin
            response=p==0?2:0;response_id=p==1?7:0;put(4,1);await_completion(1);errors=errors+1;
            host.read_reg(56,0,value);if(value!=0||acks!=prior_acks+p+1)$fatal(1,"Write fault not reported");
        end
        response=0;response_id=0;force_block=1;put(4,1);while(!commit)@(negedge clk);
        host.read_reg(8,0,status);if(!status[0]||status[1])$fatal(1,"Expected in-flight tensor before reset");
        @(negedge clk);rst_n=0;repeat(3)@(negedge clk);rst_n=1;force_block=0;
        host.read_reg(8,0,status);if(status!=0||awvalid||wvalid||arvalid||rready||bready)$fatal(1,"Tensor shared reset did not clear transactions");
        setup(0,2,2,1,0);put(4,1);await_completion(0);
        host.read_reg(56,0,value);if(value!=1)$fatal(1,"Tensor failed after shared reset");
        host.read_reg(68,0,value);if(value!=1)$fatal(1,"Tensor completion counter not reset");
        $display("Lung tensor AXI-Lite RTL: PASS (%0d frames, %0d words, %0d reads, %0d error cases; pool, nearest, concat, fused, tails, locks, held B/R, final-B wait, malformed LAST drain, canaries, shared reset)",
            header[0],normal_writes,normal_reads,errors);$finish;
    end
    initial begin #2000000000;$fatal(1,"Tensor integration timeout");end
endmodule
