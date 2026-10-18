`timescale 1ns/1ps
module tb_lung_conv8_window;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,w_valid=0,w_first=0,w_last=0,c_valid=0,m_ready=0;
    reg[511:0]weights=0;reg[7:0]output_keep=0,input_zero=0,output_zero=0;
    reg[255:0]biases=0,multipliers=0;reg[47:0]shifts=0;reg relu=0;
    reg[31:0]base=0;reg[9:0]height=0,width=0;reg[6:0]channels=0;reg kernel3=0,padding=0;
    wire w_ready,c_ready,loaded,req_valid,req_ready,rsp_ready,m_valid,m_last,m_frame_last,m_error,busy,done,error;
    wire[31:0]req_addr;wire[8:0]m_x,m_y;wire[2:0]m_lane;wire signed[31:0]m_acc;wire[7:0]m_code;
    reg rsp_valid=0,rsp_error=0;reg[63:0]rsp_data=0;
    reg pending=0;reg[31:0]pending_addr;integer delay_cycles=0,cycle=0;reg inject_error=0;
    reg[31:0]header[0:6],frames[0:7999],expected_requests[0:799999];
    reg[63:0]features[0:99999];reg[511:0]weight_bank[0:59999];reg[74:0]expected_windows[0:799999];
    reg[19:0]expected_window_coords[0:799999];reg[20:0]expected_coords[0:159999];
    reg signed[31:0]bias_bank[0:15999],mult_bank[0:15999],expected_acc[0:159999];
    reg[5:0]shift_bank[0:15999];reg[7:0]expected_code[0:159999];reg[2:0]expected_lane[0:159999];reg expected_last[0:159999];
    integer received=0,window_count=0,request_count=0,completed=0,loads=0,f,t,p,n,offset,pbase,lane,fh;
    reg checking=0,out_stall=0,req_stall=0,win_stall=0;
    reg[63:0]held_output;reg[31:0]held_request;reg[93:0]held_window;
    lung_conv8_window dut(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),
        .w_first(w_first),.w_last(w_last),.w_weights(weights),.w_output_keep(output_keep),
        .w_input_zero_point(input_zero),.w_output_zero_point(output_zero),.w_biases(biases),
        .w_multipliers(multipliers),.w_shifts(shifts),.w_relu(relu),.weights_loaded(loaded),
        .c_valid(c_valid),.c_ready(c_ready),.c_base(base),.c_height(height),.c_width(width),
        .c_channels(channels),.c_kernel3(kernel3),.c_padding(padding),
        .req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),
        .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(rsp_data),.rsp_error(rsp_error),
        .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),.m_frame_last(m_frame_last),
        .m_x(m_x),.m_y(m_y),.m_acc(m_acc),.m_code(m_code),.m_error(m_error),.busy(busy),.done(done),.error(error));
    assign req_ready=rst_n&&!pending&&!rsp_valid&&(cycle%11<8);
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%23)<15;end
    // External memory model: nonzero variable latency and independent ready.
    // Shares reset with requester: canceled reads MUST NOT reappear after reset.
    always @(posedge clk)begin
        if(!rst_n)begin pending<=0;rsp_valid<=0;rsp_error<=0;delay_cycles<=0;end
        else begin
            if(rsp_valid&&rsp_ready)rsp_valid<=0;
            if(req_valid&&req_ready)begin pending<=1;pending_addr<=req_addr;delay_cycles<=cycle%5+1;end
            if(pending)begin
                if(delay_cycles!=0)delay_cycles<=delay_cycles-1;
                else begin
                    if(pending_addr>=header[1])$fatal(1,"Memory read outside initialized feature bank: %0d",pending_addr);
                    rsp_data<=features[pending_addr];rsp_error<=inject_error;rsp_valid<=1;pending<=0;
                end
            end
        end
    end
    always @(posedge clk)if(rst_n&&checking)begin
        if(out_stall&&(!m_valid||{m_x,m_y,m_frame_last,m_lane,m_last,m_acc,m_code,m_error}!==held_output))$fatal(1,"Output/tag changed under stall");
        if(req_stall&&(!req_valid||req_addr!==held_request))$fatal(1,"Memory request changed under stall");
        if(win_stall&&(!dut.win_valid||{dut.win_x,dut.win_y,dut.win_frame_last,dut.win_last,dut.win_first,dut.win_keep,dut.win_data,dut.win_error}!==held_window))$fatal(1,"Window changed under stall");
        out_stall=m_valid&&!m_ready;held_output={m_x,m_y,m_frame_last,m_lane,m_last,m_acc,m_code,m_error};
        req_stall=req_valid&&!req_ready;held_request=req_addr;
        win_stall=dut.win_valid&&!dut.win_ready;held_window={dut.win_x,dut.win_y,dut.win_frame_last,dut.win_last,dut.win_first,dut.win_keep,dut.win_data,dut.win_error};
        if(w_valid&&w_ready)loads=loads+1;
        if(req_valid&&req_ready)begin
            if(request_count>=header[6]||req_addr!==expected_requests[request_count])$fatal(1,"Wrong memory address %0d: %0d/%0d",request_count,req_addr,expected_requests[request_count]);
            request_count=request_count+1;
        end
        if(dut.win_valid&&dut.win_ready)begin
            if(window_count>=header[5]||dut.win_error||
                {dut.win_frame_last,dut.win_last,dut.win_first,dut.win_keep,dut.win_data}!==expected_windows[window_count]||
                {1'b0,dut.win_y,1'b0,dut.win_x}!==expected_window_coords[window_count])
                $fatal(1,"Window %0d mismatch: XY=%0d,%0d flags=%b%b%b keep=%h",window_count,dut.win_x,dut.win_y,dut.win_frame_last,dut.win_last,dut.win_first,dut.win_keep);
            window_count=window_count+1;
        end
        if(m_valid&&m_ready)begin
            if(received>=header[4]||$isunknown({m_acc,m_code,m_x,m_y,m_lane,m_last,m_frame_last,m_error})||m_error||
                m_acc!==expected_acc[received]||m_code!==expected_code[received]||m_lane!==expected_lane[received]||
                m_last!==expected_last[received]||{m_frame_last,1'b0,m_y,1'b0,m_x}!==expected_coords[received])
                $fatal(1,"Convolution output %0d acc %0d/%0d code %0d/%0d XY %0d,%0d lane %0d error=%b",received,m_acc,expected_acc[received],m_code,expected_code[received],m_x,m_y,m_lane,m_error);
            received=received+1;if(m_frame_last)completed=completed+1;
        end
        if(done&&error)$fatal(1,"Unexpected command/frame error");
    end else begin out_stall=0;req_stall=0;win_stall=0;end
    task reset_core;
        begin
            @(negedge clk);rst_n=0;w_valid=0;c_valid=0;repeat(3)@(negedge clk);
            if(m_valid||req_valid||busy||loaded)$fatal(1,"Reset retained pending computation");rst_n=1;
        end
    endtask
    task send_weight(input reg[511:0]word,input reg first,input reg last);
        begin
            @(negedge clk);w_valid=1;w_first=first;w_last=last;weights=word;
            if(!first&&checking)begin input_zero=37;output_zero=255;biases=0;multipliers=0;shifts=0;end
            @(posedge clk);while(!w_ready)@(posedge clk);
        end
    endtask
    task load_simple;
        begin
            @(negedge clk);output_keep=1;input_zero=128;output_zero=128;biases=0;
            multipliers={8{32'd1073741824}};shifts={8{6'd31}};relu=0;
            send_weight(512'd1,1,1);@(negedge clk);w_valid=0;
        end
    endtask
    task start_frame;
        begin
            @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);
            @(negedge clk);c_valid=0;
        end
    endtask
    task expect_bad_command(input reg[9:0]h,input reg[9:0]w,input reg[6:0]c,input reg k,input reg pad,input reg[31:0]b);
        begin
            height=h;width=w;channels=c;kernel3=k;padding=pad;base=b;start_frame();
            while(!done)begin @(negedge clk);if(req_valid||m_valid)$fatal(1,"Invalid command issued memory/result");end
            if(!error||busy)$fatal(1,"Invalid command did not finish with error");
        end
    endtask
    initial begin
        fh=$fopen("vectors/header.mem","r");if(!fh)$fatal(1,"Missing window vectors");$fclose(fh);
        $readmemh("vectors/header.mem",header,0,6);
        if($isunknown(header[0])||header[0]==0||header[4]==0||header[0]*16>8000||header[1]>100000||header[2]>60000||header[3]>16000||header[4]>160000||header[5]>800000||header[6]>800000)$fatal(1,"Invalid/empty window bank count");
        $readmemh("vectors/frames.mem",frames,0,header[0]*16-1);
        $readmemh("vectors/features.mem",features,0,header[1]-1);
        $readmemh("vectors/weights.mem",weight_bank,0,header[2]-1);
        $readmemh("vectors/bias.mem",bias_bank,0,header[3]-1);$readmemh("vectors/multipliers.mem",mult_bank,0,header[3]-1);
        $readmemh("vectors/shifts.mem",shift_bank,0,header[3]-1);
        $readmemh("vectors/windows.mem",expected_windows,0,header[5]-1);$readmemh("vectors/window_coords.mem",expected_window_coords,0,header[5]-1);
        $readmemh("vectors/requests.mem",expected_requests,0,header[6]-1);
        $readmemh("vectors/accumulators.mem",expected_acc,0,header[4]-1);$readmemh("vectors/codes.mem",expected_code,0,header[4]-1);
        $readmemh("vectors/lanes.mem",expected_lane,0,header[4]-1);$readmemh("vectors/lasts.mem",expected_last,0,header[4]-1);
        $readmemh("vectors/coords.mem",expected_coords,0,header[4]-1);
        repeat(3)@(negedge clk);rst_n=1;if(c_ready)$fatal(1,"Frame accepted without weights");checking=1;
        for(f=0;f<header[0];f=f+1)begin
            t=f*16;offset=frames[t+10];n=frames[t+11];pbase=frames[t+12];
            @(negedge clk);output_keep=frames[t+8][7:0];input_zero=frames[t+6][7:0];output_zero=frames[t+7][7:0];relu=frames[t+9][0];
            for(lane=0;lane<8;lane=lane+1)begin biases[32*lane+:32]=bias_bank[pbase+lane];multipliers[32*lane+:32]=mult_bank[pbase+lane];shifts[6*lane+:6]=shift_bank[pbase+lane];end
            for(p=0;p<n;p=p+1)send_weight(weight_bank[offset+p],p==0,p==n-1);
            @(negedge clk);w_valid=0;base=frames[t];height=frames[t+1][9:0];width=frames[t+2][9:0];channels=frames[t+3][6:0];kernel3=frames[t+4]==3;padding=frames[t+5][0];
            start_frame();
            // Captured command must survive garbage configuration while busy.
            base=32'hffffffff;height=0;width=0;channels=0;kernel3=0;padding=1;
            w_valid=1;c_valid=1;
            repeat(3)begin @(negedge clk);if(w_ready||c_ready)$fatal(1,"Frame allowed concurrent reload/command");end
            w_valid=0;c_valid=0;
            while(!done)@(negedge clk);
            if(error||busy)$fatal(1,"Frame finished incorrectly");
            if(f%15==0)$display("Window frame %0d/%0d complete (%0d outputs)",f+1,header[0],received);
        end
        if(received!==header[4]||window_count!==header[5]||request_count!==header[6]||completed!==header[0]||loads!==header[2])$fatal(1,"Incomplete window/output/request/frame coverage");
        checking=0;reset_core();load_simple();
        expect_bad_command(0,1,1,0,0,0);expect_bad_command(513,1,1,0,0,0);
        expect_bad_command(1,1,0,0,0,0);expect_bad_command(1,1,97,0,0,0);
        expect_bad_command(1,1,1,0,1,0);expect_bad_command(2,2,1,1,0,0);
        expect_bad_command(2,2,1,0,0,32'hfffffffe);
        // Shared reset while a memory request is pending, and while output stalls.
        base=0;height=1;width=1;channels=1;kernel3=0;padding=0;start_frame();
        while(!pending)@(negedge clk);reset_core();load_simple();
        force m_ready=0;start_frame();while(!m_valid)@(negedge clk);repeat(4)@(negedge clk);reset_core();release m_ready;
        // Memory error substitutes real-zero code and flags the whole pixel;
        // final frame completion waits for the downstream LAST handshake.
        load_simple();inject_error=1;force m_ready=0;start_frame();while(!m_valid)@(negedge clk);
        if(!m_error||m_acc!==0||m_code!==128||!m_frame_last||done||!busy)$fatal(1,"Memory error/final handshake mishandled");
        repeat(4)@(negedge clk);if(!m_error||!m_valid||done)$fatal(1,"Error output changed during stall");
        force m_ready=1;@(posedge clk);@(negedge clk);if(!done||busy||!error)$fatal(1,"Failed frame did not finish");
        release m_ready;inject_error=0;reset_core();
        $display("Lung window convolution RTL: PASS (%0d frames, %0d outputs, %0d windows, %0d memory reads, %0d weight loads; padding, tails, latency, stalls, capture, invalid commands, shared reset, memory errors)",completed,received,window_count,request_count,loads);$finish;
    end
    initial begin #1000000000;$fatal(1,"Window test timeout");end
endmodule
