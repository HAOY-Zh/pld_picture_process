`timescale 1ns/1ps
module tb_lung_int8_dot;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,s_valid=0,s_first=0,s_last=0,m_ready=0;
    wire s_ready,m_valid,m_error;wire [7:0] m_code;wire signed[31:0] m_acc;
    reg [7:0] activation=0,iz=0,oz=0;reg signed[7:0] weight=0;
    reg signed[31:0] bias=0,mult=1;reg [5:0] shift=0;reg relu=0;
    reg [31:0] header[0:5],desc[0:4095];
    reg [7:0] inputs[0:499999],weights[0:999999],codes[0:99999];
    reg signed[31:0] biases[0:4095],multipliers[0:4095],expected_acc[0:99999];
    reg [5:0] shifts[0:4095];
    integer cycle=0,received=0,beats=0,c,oc,y,x,ic,ky,kx,sy,sx,t,term,total;
    integer ci,co,h,w,k,pad,input_offset,weight_offset,param_offset;
    reg checking=0,was_stalled=0;reg [40:0] held;
    integer file_handle;
    lung_int8_dot dut(.clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(s_ready),
        .s_first(s_first),.s_last(s_last),.s_activation(activation),.s_input_zero_point(iz),
        .s_weight(weight),.s_bias(bias),.s_multiplier(mult),.s_shift(shift),
        .s_output_zero_point(oz),.s_relu(relu),.m_valid(m_valid),.m_ready(m_ready),
        .m_code(m_code),.m_acc(m_acc),.m_error(m_error));
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%17)<12;end
    always @(posedge clk)if(rst_n&&checking)begin
        if(was_stalled&&(!m_valid||{m_acc,m_code,m_error}!==held))$fatal(1,"Dot backpressure changed output");
        was_stalled=m_valid&&!m_ready;held={m_acc,m_code,m_error};
        if(m_valid&&m_ready)begin
            if($isunknown({m_acc,m_code,m_error})||received>=header[4]||m_error||m_code!==codes[received]||m_acc!==expected_acc[received])
                $fatal(1,"Dot output %0d acc=%0d expected=%0d code=%0d expected=%0d error=%b",received,m_acc,expected_acc[received],m_code,codes[received],m_error);
            received=received+1;
        end
    end else was_stalled=0;
    task send(input reg first,input reg last,input reg[7:0] a,input reg signed[7:0] wt,
              input reg[7:0] in_z,input reg signed[31:0] b,input reg signed[31:0] m,
              input reg[5:0] sh,input reg[7:0] out_z,input reg r);
        begin
            if(beats%31==0)begin @(negedge clk);s_valid=0;end
            @(negedge clk);s_valid=1;s_first=first;s_last=last;activation=a;weight=wt;
            iz=in_z;bias=b;mult=m;shift=sh;oz=out_z;relu=r;
            @(posedge clk);while(!s_ready)@(posedge clk);beats=beats+1;
        end
    endtask
    initial begin
        file_handle=$fopen("vectors/header.mem","r");if(file_handle==0)$fatal(1,"Missing vectors/header.mem");$fclose(file_handle);
        $readmemh("vectors/header.mem",header,0,5);
        if($isunknown(header[0])||$isunknown(header[4])||header[0]==0||header[4]==0||header[0]*16>4096||header[1]>500000||header[2]>1000000||header[3]>4096||header[4]>100000)$fatal(1,"Invalid/empty vector count or capacity exceeded");
        $readmemh("vectors/descriptors.mem",desc,0,header[0]*16-1);
        $readmemh("vectors/inputs.mem",inputs,0,header[1]-1);
        $readmemh("vectors/weights.mem",weights,0,header[2]-1);
        $readmemh("vectors/bias.mem",biases,0,header[3]-1);
        $readmemh("vectors/multipliers.mem",multipliers,0,header[3]-1);
        $readmemh("vectors/shifts.mem",shifts,0,header[3]-1);
        $readmemh("vectors/codes.mem",codes,0,header[4]-1);
        $readmemh("vectors/accumulators.mem",expected_acc,0,header[4]-1);
        repeat(3)@(negedge clk);rst_n=1;
        send(1,0,255,127,128,0,1073741824,31,128,0);
        @(negedge clk);s_valid=0;rst_n=0;repeat(3)@(negedge clk);
        if(m_valid!==0)$fatal(1,"Reset did not flush active dot");rst_n=1;checking=1;beats=0;
        for(c=0;c<header[0];c=c+1)begin
            t=c*16;ci=desc[t];co=desc[t+1];h=desc[t+2];w=desc[t+3];k=desc[t+4];pad=desc[t+5];
            input_offset=desc[t+9];weight_offset=desc[t+10];param_offset=desc[t+11];total=ci*k*k;
            for(oc=0;oc<co;oc=oc+1)for(y=0;y<h;y=y+1)for(x=0;x<w;x=x+1)begin
                term=0;
                for(ic=0;ic<ci;ic=ic+1)for(ky=0;ky<k;ky=ky+1)for(kx=0;kx<k;kx=kx+1)begin
                    sy=y+ky-pad;sx=x+kx-pad;
                    send(term==0,term==total-1,
                        (sy<0||sy>=h||sx<0||sx>=w)?desc[t+6][7:0]:inputs[input_offset+(ic*h+sy)*w+sx],
                        weights[weight_offset+((oc*ci+ic)*k+ky)*k+kx],
                        term==0?desc[t+6][7:0]:8'd37,
                        term==0?biases[param_offset+oc]:32'sd0,
                        term==0?multipliers[param_offset+oc]:32'sd1,
                        term==0?shifts[param_offset+oc]:6'd0,
                        term==0?desc[t+7][7:0]:8'd255,term==0?desc[t+8][0]:1'b0);
                    term=term+1;
                end
            end
            $display("Dot vector case %0d/%0d queued",c+1,header[0]);
        end
        @(negedge clk);s_valid=0;
        while(received<header[4])@(negedge clk);
        checking=0;repeat(5)@(negedge clk);
        // Explicit invalid-overflow and missing-first paths must raise error.
        send(1,1,255,127,0,32'sh7fffffff,1073741824,31,0,0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        if(!m_error)$fatal(1,"ACC32 overflow was not flagged");
        while(!s_ready)@(negedge clk);
        send(0,1,1,1,0,0,1073741824,31,0,0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        if(!m_error)$fatal(1,"Missing-first marker was not flagged");
        $display("Lung convolution-dot RTL: PASS (%0d outputs, %0d cases, padding, zero points, metadata capture, stalls, reset, overflow)",received,header[0]);$finish;
    end
    initial begin #1000000000;$fatal(1,"Dot timeout");end
endmodule
