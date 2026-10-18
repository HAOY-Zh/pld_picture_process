`timescale 1ns/1ps
module tb_lung_requant;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,s_valid=0,m_ready=0;
    wire s_ready,m_valid,m_error; wire [7:0] m_code;
    reg signed [31:0] acc=0,mult=1;
    reg [5:0] shift=0; reg [7:0] zp=0; reg relu=0;
    reg [31:0] header[0:5]; reg [87:0] vectors[0:99999];
    integer cycle=0,sent=0,received=0,index;
    integer file_handle;
    reg checking=0,was_stalled=0;reg [8:0] held;
    lung_requant dut(.clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(s_ready),
        .s_acc(acc),.s_multiplier(mult),.s_shift(shift),.s_zero_point(zp),.s_relu(relu),
        .m_valid(m_valid),.m_ready(m_ready),.m_code(m_code),.m_error(m_error));
    always @(negedge clk) begin cycle=cycle+1;m_ready=(cycle%17)<12;end
    always @(posedge clk) if (rst_n && checking) begin
        if(was_stalled && (!m_valid || {m_code,m_error}!==held))$fatal(1,"Backpressure changed output");
        was_stalled=m_valid&&!m_ready;held={m_code,m_error};
        if(m_valid&&m_ready) begin
            if($isunknown({m_code,m_error}) || received>=header[5] || {m_code,m_error}!==vectors[received][8:0])
                $fatal(1,"Requant vector %0d code=%0d error=%0d expected=%h",received,m_code,m_error,vectors[received][8:0]);
            received=received+1;
        end
    end else was_stalled=0;
    task send(input reg [87:0] word);
        begin
            @(negedge clk);s_valid=1;acc=word[87:56];mult=word[55:24];shift=word[23:18];zp=word[17:10];relu=word[9];
            @(posedge clk);while(!s_ready)@(posedge clk);
            sent=sent+1;
        end
    endtask
    initial begin
        file_handle=$fopen("vectors/header.mem","r");if(file_handle==0)$fatal(1,"Missing vectors/header.mem");$fclose(file_handle);
        $readmemh("vectors/header.mem",header,0,5);
        if($isunknown(header[5])||header[5]==0||header[5]>100000)$fatal(1,"Invalid/empty vector count");
        $readmemh("vectors/requant.mem",vectors,0,header[5]-1);
        repeat(3)@(negedge clk);rst_n=1;
        // Reset with a transaction in the pipeline: it must be discarded.
        send(vectors[0]);@(negedge clk);s_valid=0;rst_n=0;
        repeat(3)@(negedge clk);if(m_valid!==0)$fatal(1,"Reset failed");rst_n=1;sent=0;checking=1;
        for(index=0;index<header[5];index=index+1)begin
            if(index%7==0)begin @(negedge clk);s_valid=0;end
            send(vectors[index]);
        end
        @(negedge clk);s_valid=0;
        while(received<header[5])@(negedge clk);
        repeat(5)@(negedge clk);
        if(m_valid)$fatal(1,"Unexpected extra output");
        $display("Lung requant RTL: PASS (%0d outputs, signed RNE, invalid metadata, bubbles, backpressure, reset)",received);$finish;
    end
    initial begin #10000000;$fatal(1,"Requant timeout");end
endmodule
