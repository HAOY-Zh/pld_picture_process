`timescale 1ns/1ps
// Port-only arithmetic oracle, including 1-pixel axes and 512-sized axes.
module tb_nodule_window_edges;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0;
    wire c_ready,req_valid,req_ready,rsp_ready,m_valid,m_first,m_last,m_frame_last,m_error,done,error,config_error;
    reg [31:0] base=0;reg [9:0] height=1,width=1;reg [6:0] channels=1;
    reg kernel3=0,padding=0,m_ready=0;
    wire [31:0] req_addr;wire [63:0] m_data;wire [7:0] m_keep;wire [8:0] m_x,m_y;
    reg pending=0;integer delay=0,cycle=0;reg [63:0] response=0;
    wire rsp_valid=pending&&delay==0;
    assign req_ready=!pending||(rsp_valid&&rsp_ready);
    function [63:0] memory_word(input [31:0] a);
        memory_word=64'h0123456789abcdef^{a,a};
    endfunction
    lung_feature_window dut(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready),
        .c_base(base),.c_height(height),.c_width(width),.c_channels(channels),.c_kernel3(kernel3),.c_padding(padding),
        .c_input_zero(8'd137),.req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(response),.rsp_error(1'b0),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_keep(m_keep),.m_first(m_first),.m_last(m_last),
        .m_frame_last(m_frame_last),.m_x(m_x),.m_y(m_y),.m_error(m_error),.done(done),.error(error),.config_error(config_error));
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%13)<9;end
    always @(posedge clk)begin
        if(!rst_n)begin pending<=0;delay<=0;response<=0;end
        else begin
            if(pending&&delay>0)delay<=delay-1;
            if(rsp_valid&&rsp_ready)pending<=0;
            if(req_valid&&req_ready)begin
                if(req_addr<base||{1'b0,req_addr}>={1'b0,base}+height*width*((channels+7)/8))$fatal(1,"Out-of-frame request");
                pending<=1;delay<=cycle%4;response<=memory_word(req_addr);
            end
        end
    end
    integer ex=0,ey=0,ekx=0,eky=0,ecb=0,count=0,frames=0,total=0;
    integer oh,ow,blocks,sy,sx,expected_keep;reg [31:0] address;reg [63:0] expected;
    reg checking=0,stalled=0;reg [93:0] held;
    always @(posedge clk)if(rst_n&&checking)begin
        if(stalled&&(!m_valid||{m_data,m_keep,m_x,m_y,m_first,m_last,m_frame_last,m_error}!==held))
            $fatal(1,"Window output changed under stall");
        stalled=m_valid&&!m_ready;held={m_data,m_keep,m_x,m_y,m_first,m_last,m_frame_last,m_error};
        if(m_valid&&m_ready)begin
            sy=ey+eky-(padding?1:0);sx=ex+ekx-(padding?1:0);
            if(sy<0||sx<0||sy>=height||sx>=width)expected={8{8'd137}};
            else begin address=base+(sy*width+sx)*blocks+ecb;expected=memory_word(address);end
            expected_keep=ecb==blocks-1&&channels%8!=0?(1<<(channels%8))-1:255;
            if(m_error||m_data!==expected||m_keep!==expected_keep[7:0]||m_x!==ex[8:0]||m_y!==ey[8:0]||
                m_first!==(ecb==0&&ekx==0&&eky==0)||
                m_last!==(ecb==blocks-1&&(!kernel3||(ekx==2&&eky==2)))||
                m_frame_last!==(ecb==blocks-1&&(!kernel3||(ekx==2&&eky==2))&&ex==ow-1&&ey==oh-1))
                $fatal(1,"Window edge oracle frame=%0d xy=%0d,%0d tap=%0d,%0d block=%0d",frames,ex,ey,ekx,eky,ecb);
            count=count+1;total=total+1;
            if(ecb!=blocks-1)ecb=ecb+1;
            else begin ecb=0;
                if(kernel3&&ekx!=2)ekx=ekx+1;
                else begin ekx=0;
                    if(kernel3&&eky!=2)eky=eky+1;
                    else begin eky=0;if(ex!=ow-1)ex=ex+1;else begin ex=0;ey=ey+1;end end
                end
            end
        end
    end else stalled=0;
    integer dims[0:5],chs[0:3],hi,wi,ci,mode;
    initial begin
        dims[0]=1;dims[1]=2;dims[2]=3;dims[3]=5;dims[4]=17;dims[5]=512;
        chs[0]=1;chs[1]=9;chs[2]=24;chs[3]=96;
        repeat(3)@(negedge clk);rst_n=1;
        for(hi=0;hi<6;hi=hi+1)for(wi=0;wi<6;wi=wi+1)if(dims[hi]*dims[wi]<=1024)
            for(ci=0;ci<4;ci=ci+1)for(mode=0;mode<3;mode=mode+1)
                if(mode!=1||(dims[hi]>=3&&dims[wi]>=3))begin
                    @(negedge clk);height=dims[hi];width=dims[wi];channels=chs[ci];
                    kernel3=mode!=0;padding=mode==2;base=frames%2?32'hffff8000:32'h12345000;
                    blocks=(channels+7)/8;oh=kernel3&&!padding?height-2:height;ow=kernel3&&!padding?width-2:width;
                    ex=0;ey=0;ekx=0;eky=0;ecb=0;count=0;checking=1;c_valid=1;
                    @(posedge clk);while(!c_ready)@(posedge clk);
                    @(negedge clk);c_valid=0;while(!done)@(negedge clk);
                    if(error||config_error||pending||count!=oh*ow*blocks*(kernel3?9:1))$fatal(1,"Incomplete edge frame");
                    checking=0;frames=frames+1;
                end
        $display("Nodule window edge RTL: PASS (%0d frames, %0d words, 1/2/3/5/17/512 axes, 1/9/24/96 channels, signed coordinate oracle, stalls, simultaneous reads)",frames,total);$finish;
    end
    initial begin #1000000000;$fatal(1,"Window edges timeout");end
endmodule
