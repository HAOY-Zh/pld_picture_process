`timescale 1ns/1ps
// Independent coordinate oracle, not the DUT stepping recurrence.
module tb_lung_window_step;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,c_valid=0,kernel3=0,padding=0;
    reg[31:0]base=0;reg[9:0]height=1,width=1;reg[6:0]channels=1;
    wire c_ready,req_valid,rsp_ready,m_valid,m_first,m_last,m_final,m_error,done,error,cfg_error;
    wire[31:0]req_addr;wire[63:0]m_data;wire[7:0]m_keep;wire[8:0]m_x,m_y;
    reg rsp_valid=0,rsp_error=0,pending=0;integer cycle=0,delay_count=0;
    wire req_ready=rst_n&&!pending&&!rsp_valid&&cycle%7<5;
    wire m_ready=cycle%13<8;
    reg[31:0]expected_req[0:65535];reg[99:0]expected_out[0:65535];
    integer nr=0,no=0,requests=0,outputs=0,cases=0,fail_read=-1;
    integer ox,oy,kx,ky,cb,sx,sy,b,oh,ow,k;reg in_bounds,err;
    reg[99:0]held;reg out_stall=0,req_stall=0;reg[31:0]held_addr;
    lung_feature_window dut(.clk(clk),.rst_n(rst_n),.c_valid(c_valid),.c_ready(c_ready),
        .c_base(base),.c_height(height),.c_width(width),.c_channels(channels),.c_kernel3(kernel3),.c_padding(padding),.c_input_zero(8'd128),
        .req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(64'h0102030405060708),.rsp_error(rsp_error),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_keep(m_keep),.m_first(m_first),.m_last(m_last),
        .m_frame_last(m_final),.m_x(m_x),.m_y(m_y),.m_error(m_error),.done(done),.error(error),.config_error(cfg_error));
    wire[99:0]packed_out={m_error,m_final,m_last,m_first,m_y,m_x,m_keep,6'b0,m_data};
    always @(negedge clk)cycle=cycle+1;
    always @(posedge clk)begin
        if(!rst_n)begin pending<=0;rsp_valid<=0;delay_count<=0;out_stall<=0;req_stall<=0;end
        else begin
            if(out_stall&&(!m_valid||packed_out!==held))$fatal(1,"Output changed under backpressure");
            if(req_stall&&(!req_valid||req_addr!==held_addr))$fatal(1,"Request changed under backpressure");
            held<=packed_out;out_stall<=m_valid&&!m_ready;held_addr<=req_addr;req_stall<=req_valid&&!req_ready;
            if(rsp_valid&&rsp_ready)rsp_valid<=0;
            if(req_valid&&req_ready)begin
                if(requests>=nr||req_addr!==expected_req[requests])$fatal(1,"Stepping address mismatch %0d",requests);
                pending<=1;delay_count<=cycle%4;rsp_error<=requests==fail_read;requests=requests+1;
            end
            if(pending)begin
                if(delay_count!=0)delay_count<=delay_count-1;
                else begin pending<=0;rsp_valid<=1;end
            end
            if(m_valid&&m_ready)begin
                if(outputs>=no||packed_out!==expected_out[outputs])$fatal(1,"Window mismatch %0d got=%h expected=%h",outputs,packed_out,expected_out[outputs]);
                outputs=outputs+1;
            end
        end
    end
    task reset_core;
        begin @(negedge clk);rst_n=0;c_valid=0;repeat(3)@(negedge clk);rst_n=1;end
    endtask
    task start;
        begin @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);@(negedge clk);c_valid=0;end
    endtask
    task run(input reg[31:0]addr,input integer h,input integer w,input integer c,input reg k3,input reg pad,input integer fault);
        reg[63:0]data;reg[7:0]keep;reg first,last,final_tag;
        begin
            reset_core();base=addr;height=h;width=w;channels=c;kernel3=k3;padding=pad;
            nr=0;no=0;requests=0;outputs=0;fail_read=fault;
            b=(c+7)/8;oh=k3&&!pad?h-2:h;ow=k3&&!pad?w-2:w;k=k3?3:1;
            for(oy=0;oy<oh;oy=oy+1)for(ox=0;ox<ow;ox=ox+1)
                for(ky=0;ky<k;ky=ky+1)for(kx=0;kx<k;kx=kx+1)for(cb=0;cb<b;cb=cb+1)begin
                    sy=oy+ky-pad;sx=ox+kx-pad;in_bounds=sy>=0&&sy<h&&sx>=0&&sx<w;
                    err=in_bounds&&nr==fault;
                    if(in_bounds)begin expected_req[nr]=addr+(sy*w+sx)*b+cb;nr=nr+1;end
                    data=in_bounds&&!err?64'h0102030405060708:64'h8080808080808080;
                    keep=cb==b-1&&c%8!=0?(1<<(c%8))-1:255;
                    first=cb==0&&kx==0&&ky==0;last=cb==b-1&&kx==k-1&&ky==k-1;
                    final_tag=last&&ox==ow-1&&oy==oh-1;
                    expected_out[no]={err,final_tag,last,first,oy[8:0],ox[8:0],keep,6'b0,data};no=no+1;
                end
            start();base=32'hdeadbeef;height=0;width=0;channels=0;kernel3=!k3;padding=!pad;
            while(!done)@(negedge clk);
            if(cfg_error||error!==(fault>=0)||outputs!=no||requests!=nr||pending||rsp_valid)$fatal(1,"Stepping frame incomplete");
            cases=cases+1;
        end
    endtask
    initial begin
        run(0,1,1,1,1,1,-1);run(32'hffffffff,1,1,1,1,1,-1);
        run(53,3,4,9,1,1,5);run(32'hffffffdc,3,4,24,1,0,-1);
        run(0,512,1,96,1,1,-1);run(32'hffffe800,1,512,96,1,1,-1);
        run(31,3,3,96,1,0,-1);run(0,2,3,9,0,0,-1);
        reset_core();base=0;height=2;width=2;channels=1;kernel3=1;padding=0;nr=0;no=0;start();
        while(!done)@(negedge clk);if(!cfg_error||!error)$fatal(1,"Invalid kernel accepted");cases=cases+1;
        reset_core();height=1;width=1;kernel3=0;padding=1;start();
        while(!done)@(negedge clk);if(!cfg_error||!error)$fatal(1,"Invalid padding accepted");cases=cases+1;
        $display("Lung stepped window RTL: PASS (10 cases; independent coordinates, padding origins, H512/W512/C96, high word bounds, valid crop, errors, stalls, config capture)");$finish;
    end
    initial begin #100000000;$fatal(1,"Stepped window timeout");end
endmodule
