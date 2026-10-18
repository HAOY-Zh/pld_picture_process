`timescale 1ns/1ps
module tb_lung_output_writer;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0,c_abort=0,s_valid=0,s_last=0,s_frame_last=0,s_error=0,block_b=0;
    reg[31:0]c_base=511;reg[9:0]c_height=1,c_width=1;reg[3:0]c_blocks=1,c_block=0;
    reg[7:0]c_keep=1,c_fill=128,s_code=0;reg[2:0]s_lane=0;reg[8:0]s_x=0,s_y=0;
    wire c_ready,c_invalid,s_ready,busy,done,error;wire[31:0]written_words;
    wire awvalid,awready,wvalid,wready,wlast,bvalid,bready,awlock,commit;
    wire[31:0]awaddr,address;wire[63:0]wdata,data;wire[7:0]awlen,wstrb,strb;wire[3:0]awid,awcache,bid;
    wire[2:0]awsize,awprot;wire[1:0]awburst,bresp;integer mode=0;
    reg[1:0]response=0;reg[3:0]response_id=0;
    integer tests=0,total_writes=0,frame_writes=0,limit=0,expected_success=0,x,y,l,k,mask,failmode,case_index;
    reg[63:0]expected_data;reg checking=0;
    lung_output_writer dut(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready),.c_abort(c_abort),.c_invalid(c_invalid),
        .c_base(c_base),.c_height(c_height),.c_width(c_width),.c_blocks(c_blocks),.c_block(c_block),.c_keep(c_keep),.c_fill(c_fill),
        .s_valid(s_valid),.s_ready(s_ready),.s_lane(s_lane),.s_code(s_code),.s_x(s_x),.s_y(s_y),.s_last(s_last),.s_frame_last(s_frame_last),.s_error(s_error),
        .busy(busy),.done(done),.error(error),.written_words(written_words),
        .m_axi_awid(awid),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awlock(awlock),.m_axi_awcache(awcache),.m_axi_awprot(awprot),
        .m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
        .m_axi_bid(bid),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready));
    lung_test_write_memory mem(.*);
    function[7:0]code_value(input integer xx,input integer yy,input integer lane);
        begin code_value=19+xx*3+yy*7+lane;end
    endfunction
    always @(posedge clk)if(rst_n&&checking)begin
        if(commit)begin
            if(frame_writes>=limit)$fatal(1,"Output writer wrote after error/outside expected count");
            expected_data={8{c_fill}};
            for(k=0;k<8;k=k+1)if(c_keep[k])expected_data[8*k+:8]=code_value(frame_writes%c_width,frame_writes/c_width,k);
            if(address!==((c_base+frame_writes*c_blocks+c_block)*8)||data!==expected_data||strb!==255)$fatal(1,"Packed HWC8 mismatch");
            frame_writes=frame_writes+1;total_writes=total_writes+1;
        end
        if(done&&(busy||bvalid||mem.pending||frame_writes!=limit||written_words!==expected_success))$fatal(1,"Done before B completion/count mismatch");
    end
    task start_command;
        begin @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);@(negedge clk);c_valid=0;end
    endtask
    task wait_complete(input reg expected_error);
        begin while(!done)@(negedge clk);if(error!==expected_error||busy||frame_writes!=limit||written_words!==expected_success)$fatal(1,"Bad writer completion");tests=tests+1;@(negedge clk);end
    endtask
    task send_frame(input integer fault);
        integer p;
        begin
            for(y=0;y<c_height;y=y+1)for(x=0;x<c_width;x=x+1)for(l=0;l<8;l=l+1)if(c_keep[l])begin
                @(negedge clk);s_valid=1;s_x=x;s_y=y;s_lane=l;s_code=code_value(x,y,l);
                s_last=(c_keep>>(l+1))==0;s_frame_last=s_last&&x==c_width-1&&y==c_height-1;s_error=0;p=y*c_width+x;
                if(p==1)case(fault)
                    1:s_error=1;
                    2:s_lane=7;
                    3:s_x=511;
                    4:s_last=!s_last;
                    5:if(s_last)s_frame_last=0;
                endcase
                @(posedge clk);while(!s_ready)@(posedge clk);
                @(negedge clk);s_valid=0;
                if(fault==6&&p==0)begin
                    while(!commit)@(negedge clk);
                    repeat(10)begin @(negedge clk);if(done||!busy)$fatal(1,"Writer done while B blocked");end
                    block_b=0;
                end
            end
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;checking=1;
        for(case_index=0;case_index<5;case_index=case_index+1)begin
            c_height=case_index==2?512:case_index==3?3:1;
            c_width=case_index==1?512:case_index==3?5:1;c_blocks=case_index==3?12:1;c_block=case_index==3?11:0;
            c_keep=case_index==3?8'h42:case_index==4?8'hff:1;
            c_base=case_index==0?32'h1fffffff:511;c_fill=128;frame_writes=0;limit=c_height*c_width;expected_success=limit;mode=case_index%3;
            block_b=case_index==0;start_command();send_frame(case_index==0?6:0);wait_complete(0);
        end
        c_base=4095;c_height=1;c_width=3;c_blocks=3;c_block=2;c_keep=1;
        for(failmode=1;failmode<=7;failmode=failmode+1)begin
            frame_writes=0;limit=(failmode>=6)?1:1;expected_success=failmode>=6?0:1;
            response=failmode==6?2:0;response_id=failmode==7?3:0;
            // Fault 5 is a missing final marker on the expected final pixel.
            if(failmode==5)c_width=2;else c_width=3;
            start_command();send_frame(failmode<=5?failmode:0);wait_complete(1);
        end
        response=0;response_id=0;c_width=1;
        for(case_index=0;case_index<11;case_index=case_index+1)begin
            c_base=0;c_height=1;c_width=1;c_blocks=1;c_block=0;c_keep=1;c_abort=0;
            case(case_index)
                0:c_height=0;1:c_height=513;2:c_width=0;3:c_width=513;4:c_blocks=0;5:c_blocks=13;
                6:c_block=1;7:c_keep=0;8:c_base=32'h20000000;9:begin c_base=32'h1fffffff;c_width=2;end 10:c_abort=1;
            endcase
            frame_writes=0;limit=0;expected_success=0;start_command();wait_complete(1);
        end
        c_abort=0;c_base=0;c_height=1;c_width=1;c_blocks=1;c_block=0;c_keep=1;block_b=1;
        frame_writes=0;limit=1;expected_success=1;start_command();
        @(negedge clk);s_valid=1;s_x=0;s_y=0;s_lane=0;s_code=19;s_last=1;s_frame_last=1;s_error=0;
        @(posedge clk);while(!s_ready)@(posedge clk);@(negedge clk);s_valid=0;while(!commit)@(negedge clk);
        repeat(4)@(negedge clk);if(done||!busy)$fatal(1,"Expected pending B before reset");
        checking=0;rst_n=0;repeat(3)@(negedge clk);rst_n=1;@(negedge clk);
        if(awvalid||wvalid||bready||busy||done||error||!c_ready)$fatal(1,"Writer shared reset failed");
        $display("Lung output writer RTL: PASS (%0d cases, %0d writes; sparse/tail lanes, 512 axes, physical limits, metadata/read/B errors, delayed B, shared reset)",tests,total_writes);$finish;
    end
    initial begin #20000000;$fatal(1,"Output writer timeout");end
endmodule
