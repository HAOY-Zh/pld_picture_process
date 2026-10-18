`timescale 1ns/1ps
module tb_lung_weight_reuse;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,w_valid=0,w_first=0,w_last=0,s_valid=0,s_first=0,s_last=0,m_ready=0;
    wire w_ready,s_ready,loaded,m_valid,m_last,m_error;
    wire[2:0]m_lane;wire signed[31:0]m_acc;wire[7:0]m_code;
    reg[511:0]weights=0;reg[63:0]activations=0;reg[7:0]keep=0;
    reg[7:0]output_keep=0,input_zero=0,output_zero=0;
    reg[255:0]biases=0,multipliers=0;reg[47:0]shifts=0;reg relu=0;
    reg[31:0]header[0:6],desc[0:239999],wdesc[0:5999];
    reg[511:0]weight_bank[0:59999];reg[73:0]act_bank[0:599999];
    reg signed[31:0]bias_bank[0:15999],mult_bank[0:15999],expected_acc[0:99999];
    reg[5:0]shift_bank[0:15999];reg[7:0]expected_code[0:99999];
    reg[2:0]expected_lane[0:99999];reg expected_last[0:99999];
    integer cycle=0,received=0,loads=0,inputs=0,g,t,block,p,n,base,offset,lane,fh;
    reg checking=0,stalled=0;reg[44:0]held;
    lung_mac_tile8_reuse dut(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),
        .w_first(w_first),.w_last(w_last),.w_weights(weights),.w_output_keep(output_keep),
        .w_input_zero_point(input_zero),.w_output_zero_point(output_zero),.w_biases(biases),
        .w_multipliers(multipliers),.w_shifts(shifts),.w_relu(relu),.weights_loaded(loaded),
        .s_valid(s_valid),.s_ready(s_ready),.s_first(s_first),.s_last(s_last),
        .s_activations(activations),.s_input_keep(keep),.m_valid(m_valid),.m_ready(m_ready),
        .m_lane(m_lane),.m_last(m_last),.m_acc(m_acc),.m_code(m_code),.m_error(m_error));
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%23)<15;end
    always @(posedge clk)if(rst_n&&checking)begin
        if(w_valid&&w_ready)loads=loads+1;
        if(s_valid&&s_ready)inputs=inputs+1;
        if(stalled&&(!m_valid||{m_lane,m_last,m_acc,m_code,m_error}!==held))$fatal(1,"Reuse output changed during stall");
        stalled=m_valid&&!m_ready;held={m_lane,m_last,m_acc,m_code,m_error};
        if(m_valid&&m_ready)begin
            if($isunknown({m_lane,m_last,m_acc,m_code,m_error})||received>=header[3]||m_error||
                m_acc!==expected_acc[received]||m_code!==expected_code[received]||
                m_lane!==expected_lane[received]||m_last!==expected_last[received])
                $fatal(1,"Reuse output %0d acc %0d/%0d code %0d/%0d lane %0d/%0d last %0d/%0d error=%b",received,m_acc,expected_acc[received],m_code,expected_code[received],m_lane,expected_lane[received],m_last,expected_last[received],m_error);
            received=received+1;
        end
    end else stalled=0;
    task reset_core;
        begin
            @(negedge clk);rst_n=0;w_valid=0;s_valid=0;repeat(3)@(negedge clk);
            if(m_valid||loaded||s_ready)$fatal(1,"Reset retained old weights/output");rst_n=1;
        end
    endtask
    task send_weight(input reg[511:0]word,input reg first,input reg last);
        begin
            @(negedge clk);w_valid=1;w_first=first;w_last=last;weights=word;
            if(!first&&checking)begin output_keep=0;input_zero=37;output_zero=255;biases=0;multipliers=0;shifts=0;relu=0;end
            @(posedge clk);while(!w_ready)@(posedge clk);
        end
    endtask
    task end_weights;
        begin @(negedge clk);w_valid=0;end
    endtask
    task send_activation(input reg[73:0]word);
        begin
            if(cycle%17==0)begin @(negedge clk);s_valid=0;end
            @(negedge clk);s_valid=1;s_first=word[72];s_last=word[73];keep=word[71:64];activations=word[63:0];
            @(posedge clk);while(!s_ready)@(posedge clk);
        end
    endtask
    task end_activations;
        begin @(negedge clk);s_valid=0;end
    endtask
    task configure_simple(input reg[7:0]mask);
        begin
            @(negedge clk);output_keep=mask;input_zero=0;output_zero=0;
            biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};relu=0;
        end
    endtask
    task expect_flagged;
        begin
            force m_ready=0;while(!m_valid)@(negedge clk);
            if(!m_error)$fatal(1,"Malformed group/load not flagged");
            repeat(4)@(negedge clk);
            if(!m_valid||!m_error)$fatal(1,"Error unstable under backpressure");
            reset_core();release m_ready;
        end
    endtask
    initial begin
        fh=$fopen("vectors/header.mem","r");if(!fh)$fatal(1,"Missing reuse vectors");$fclose(fh);
        $readmemh("vectors/header.mem",header,0,6);
        if($isunknown(header[0])||header[0]==0||header[3]==0||header[0]*12>240000||header[1]>600000||header[2]>16000||header[3]>100000||header[5]*3>6000||header[6]>60000)$fatal(1,"Invalid/empty reuse banks");
        $readmemh("vectors/descriptors.mem",desc,0,header[0]*12-1);
        $readmemh("vectors/weight_descriptors.mem",wdesc,0,header[5]*3-1);
        $readmemh("vectors/weights.mem",weight_bank,0,header[6]-1);
        $readmemh("vectors/activations.mem",act_bank,0,header[1]-1);
        $readmemh("vectors/bias.mem",bias_bank,0,header[2]-1);
        $readmemh("vectors/multipliers.mem",mult_bank,0,header[2]-1);
        $readmemh("vectors/shifts.mem",shift_bank,0,header[2]-1);
        $readmemh("vectors/accumulators.mem",expected_acc,0,header[3]-1);
        $readmemh("vectors/codes.mem",expected_code,0,header[3]-1);
        $readmemh("vectors/lanes.mem",expected_lane,0,header[3]-1);
        $readmemh("vectors/lasts.mem",expected_last,0,header[3]-1);
        repeat(3)@(negedge clk);rst_n=1;
        if(s_ready)$fatal(1,"Accepted activation before weight load");
        configure_simple(1);send_weight(0,1,0);end_weights();reset_core();
        configure_simple(1);send_weight(0,1,1);end_weights();
        send_activation(74'd1<<72);end_activations();reset_core();checking=1;
        for(g=0;g<header[0];g=g+1)begin
            t=g*12;offset=desc[t];n=desc[t+1];block=desc[t+8];
            if(desc[t+9])begin
                // Wait for prior pixel to drain BEFORE changing configuration.
                @(negedge clk);while(!w_ready)@(negedge clk);
                output_keep=desc[t+2][7:0];input_zero=desc[t+3][7:0];output_zero=desc[t+4][7:0];relu=desc[t+5][0];base=desc[t+6];
                for(lane=0;lane<8;lane=lane+1)begin
                    biases[32*lane+:32]=bias_bank[base+lane];multipliers[32*lane+:32]=mult_bank[base+lane];shifts[6*lane+:6]=shift_bank[base+lane];
                end
                for(p=0;p<n;p=p+1)begin
                    send_weight(weight_bank[wdesc[block*3]+p],p==0,p==n-1);
                end
                end_weights();
            end
            for(p=0;p<n;p=p+1)send_activation(act_bank[offset+p]);
            end_activations();
            if(g%1000==0)$display("Reuse group %0d/%0d queued",g+1,header[0]);
        end
        while(received<header[3])@(negedge clk);
        if(loads!==header[6]||inputs!==header[1])$fatal(1,"Unexpected transfer counts: weights=%0d/%0d inputs=%0d/%0d",loads,header[6],inputs,header[1]);
        checking=0;repeat(5)@(negedge clk);reset_core();
        // Sparse lanes, reuse three single-beat pixels, forbid reload during stalled output.
        configure_simple(8'ha5);
        for(lane=0;lane<8;lane=lane+1)biases[32*lane+:32]=lane*128;
        send_weight(512'd1,1,1);end_weights();force m_ready=0;
        for(p=0;p<3;p=p+1)begin
            send_activation((74'd3<<72)|(74'd1<<64)|74'd2);end_activations();
            while(!m_valid)@(negedge clk);
            @(negedge clk);w_valid=1;w_first=1;w_last=1;weights=0;
            held={m_lane,m_last,m_acc,m_code,m_error};
            repeat(4)begin @(negedge clk);if(w_ready||{m_lane,m_last,m_acc,m_code,m_error}!==held)$fatal(1,"Reload overwrote active weights/output");end
            w_valid=0;
            for(lane=0;lane<8;lane=lane+1)if((8'ha5>>lane)&1)begin
                while(!m_valid)@(negedge clk);
                if(m_error||m_lane!==lane[2:0]||m_last!==(lane==7)||m_acc!==(lane*128+(lane==0?2:0))||m_code!==(lane==0?8'd1:lane*64>255?8'd255:lane*64))$fatal(1,"Sparse reusable single-beat result failed");
                force m_ready=1;@(posedge clk);@(negedge clk);force m_ready=0;
            end
        end
        reset_core();release m_ready;
        // Independent weight producer may request next block before a pixel
        // finishes. Keep draining inputs; only accept reload after LAST output.
        configure_simple(1);send_weight(512'd1,1,0);send_weight(512'd3,0,1);end_weights();
        force m_ready=0;send_activation((74'd1<<72)|(74'd1<<64)|74'd2);end_activations();
        w_valid=1;w_first=1;w_last=1;weights=512'd7;
        send_activation((74'd1<<73)|(74'd1<<64)|74'd4);end_activations();
        while(!m_valid)@(negedge clk);
        if(w_ready||m_error||m_acc!==32'd14||m_code!==8'd7||!m_last)$fatal(1,"Pending reload blocked/changed active pixel");
        force m_ready=1;@(posedge clk);@(negedge clk);force m_ready=0;
        @(posedge clk);if(!w_ready)$fatal(1,"Pending reload not accepted after final output");end_weights();
        send_activation((74'd3<<72)|(74'd1<<64)|74'd2);end_activations();
        while(!m_valid)@(negedge clk);
        if(m_error||m_acc!==32'd14||m_code!==8'd7||!m_last)$fatal(1,"Pending replacement weights not used");
        reset_core();release m_ready;
        // Early LAST; late LAST with bounded reads; bad load FIRST; depth overflow.
        configure_simple(1);send_weight(0,1,0);send_weight(0,0,1);end_weights();
        force m_ready=0;send_activation(74'd3<<72);end_activations();expect_flagged();
        configure_simple(1);send_weight(0,1,1);end_weights();force m_ready=0;
        send_activation(74'd1<<72);send_activation(74'd1<<73);end_activations();expect_flagged();
        configure_simple(1);send_weight(0,0,1);end_weights();force m_ready=0;
        send_activation(74'd3<<72);end_activations();expect_flagged();
        configure_simple(1);
        for(p=0;p<129;p=p+1)send_weight(0,p==0,p==128);end_weights();force m_ready=0;
        for(p=0;p<128;p=p+1)send_activation((p==0?(74'd1<<72):0)|(p==127?(74'd1<<73):0));
        end_activations();expect_flagged();
        configure_simple(1);send_weight(0,1,1);end_weights();force m_ready=0;
        send_activation(74'd1<<73);end_activations();expect_flagged();
        $display("Lung weight reuse RTL: PASS (%0d outputs, %0d groups, %0d weight loads, %0d activation beats; reuse, reload exclusion, pending reload progress, sparse, reset, lengths)",received,header[0],loads,inputs);$finish;
    end
    initial begin #1000000000;$fatal(1,"Reuse simulation timeout");end
endmodule
