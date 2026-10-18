`timescale 1ns/1ps
// Same packets exercised through raw tile and synchronous BRAM cache.
module tb_lung_mac_tile8;
    parameter USE_CACHE=0;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,s_valid=0,s_first=0,s_last=0,m_ready=0;
    wire s_ready,m_valid,m_last,m_error;wire [2:0] m_lane;wire signed[31:0] m_acc;wire[7:0] m_code;
    reg [63:0] activations=0;reg [511:0] weights=0;
    reg [7:0] input_keep=0,output_keep=0,input_zero=0,output_zero=0;
    reg [255:0] biases=0,multipliers=0;reg [47:0] shifts=0;reg relu=0;
    reg [31:0] header[0:4],descriptors[0:239999];reg [585:0] packets[0:599999];
    reg signed[31:0] bias_bank[0:159999],mult_bank[0:159999],expected_acc[0:99999];
    reg [5:0] shift_bank[0:159999];reg [7:0] expected_code[0:99999];
    reg [2:0] expected_lane[0:99999];reg expected_last[0:99999];
    integer cycle=0,received=0,beat_count=0,file_handle,g,p,t,n,param_start,offset,lane;
    reg checking=0,was_stalled=0;reg [44:0] held;
    generate if(USE_CACHE)begin: buffered
        lung_mac_tile8_cached dut(.clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(s_ready),.s_first(s_first),.s_last(s_last),
            .s_activations(activations),.s_weights(weights),.s_input_keep(input_keep),.s_output_keep(output_keep),
            .s_input_zero_point(input_zero),.s_output_zero_point(output_zero),.s_biases(biases),.s_multipliers(multipliers),.s_shifts(shifts),.s_relu(relu),
            .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),.m_acc(m_acc),.m_code(m_code),.m_error(m_error));
    end else begin: direct
        lung_mac_tile8 dut(.clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(s_ready),.s_first(s_first),.s_last(s_last),
            .s_activations(activations),.s_weights(weights),.s_input_keep(input_keep),.s_output_keep(output_keep),
            .s_input_zero_point(input_zero),.s_output_zero_point(output_zero),.s_biases(biases),.s_multipliers(multipliers),.s_shifts(shifts),.s_relu(relu),
            .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),.m_acc(m_acc),.m_code(m_code),.m_error(m_error));
    end endgenerate
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%23)<15;end
    always @(posedge clk)if(rst_n&&checking)begin
        if(was_stalled&&(!m_valid||{m_lane,m_last,m_acc,m_code,m_error}!==held))$fatal(1,"Tile output changed during stall");
        was_stalled=m_valid&&!m_ready;held={m_lane,m_last,m_acc,m_code,m_error};
        if(m_valid&&m_ready)begin
            if($isunknown({m_lane,m_last,m_acc,m_code,m_error})||received>=header[3]||m_error||
                m_lane!==expected_lane[received]||m_last!==expected_last[received]||
                m_acc!==expected_acc[received]||m_code!==expected_code[received])
                $fatal(1,"Tile cached=%0d output %0d lane %0d/%0d last %0d/%0d acc %0d/%0d code %0d/%0d err=%b",USE_CACHE,received,m_lane,expected_lane[received],m_last,expected_last[received],m_acc,expected_acc[received],m_code,expected_code[received],m_error);
            received=received+1;
        end
    end else was_stalled=0;
    task send_packet(input reg[585:0] word,input reg corrupt_metadata);
        begin
            if(beat_count%19==0)begin @(negedge clk);s_valid=0;end
            @(negedge clk);s_valid=1;s_first=word[584];s_last=word[585];
            activations=word[63:0];weights=word[575:64];input_keep=word[583:576];
            if(corrupt_metadata)begin output_keep=0;input_zero=37;output_zero=255;biases=0;multipliers={8{32'd1}};shifts=0;relu=0;end
            @(posedge clk);while(!s_ready)@(posedge clk);beat_count=beat_count+1;
        end
    endtask
    task pulse_reset;
        begin
            @(negedge clk);s_valid=0;rst_n=0;repeat(3)@(negedge clk);
            if(m_valid!==0)$fatal(1,"Reset left a live output");rst_n=1;
        end
    endtask
    initial begin
        file_handle=$fopen("vectors/header.mem","r");if(file_handle==0)$fatal(1,"Missing tile vectors");$fclose(file_handle);
        $readmemh("vectors/header.mem",header,0,4);
        if($isunknown(header[0])||header[0]==0||header[3]==0||header[0]*12>240000||header[1]>600000||header[2]>160000||header[3]>100000)$fatal(1,"Invalid/empty tile vector count");
        $readmemh("vectors/descriptors.mem",descriptors,0,header[0]*12-1);
        $readmemh("vectors/packets.mem",packets,0,header[1]-1);
        $readmemh("vectors/bias.mem",bias_bank,0,header[2]-1);$readmemh("vectors/multipliers.mem",mult_bank,0,header[2]-1);
        $readmemh("vectors/shifts.mem",shift_bank,0,header[2]-1);$readmemh("vectors/accumulators.mem",expected_acc,0,header[3]-1);
        $readmemh("vectors/codes.mem",expected_code,0,header[3]-1);$readmemh("vectors/lanes.mem",expected_lane,0,header[3]-1);
        $readmemh("vectors/lasts.mem",expected_last,0,header[3]-1);
        repeat(3)@(negedge clk);rst_n=1;
        // Abandon a partially loaded/active group; RAM is not initialized/reset.
        send_packet(packets[0] & ~(586'd1<<585),0);pulse_reset();beat_count=0;checking=1;
        for(g=0;g<header[0];g=g+1)begin
            t=g*12;offset=descriptors[t];n=descriptors[t+1];param_start=descriptors[t+6];
            for(p=0;p<n;p=p+1)begin
                if(p==0)begin
                    // Never change live first-beat metadata in the same active
                    // region as the previous accept edge (especially 1-beat head).
                    @(negedge clk);s_valid=0;
                    output_keep=descriptors[t+2][7:0];input_zero=descriptors[t+3][7:0];output_zero=descriptors[t+4][7:0];relu=descriptors[t+5][0];
                    for(lane=0;lane<8;lane=lane+1)begin biases[32*lane+:32]=bias_bank[param_start+lane];multipliers[32*lane+:32]=mult_bank[param_start+lane];shifts[6*lane+:6]=shift_bank[param_start+lane];end
                end
                send_packet(packets[offset+p],p!=0);
            end
            if(g%1000==0)$display("Tile cached=%0d group %0d/%0d queued",USE_CACHE,g+1,header[0]);
        end
        @(negedge clk);s_valid=0;while(received<header[3])@(negedge clk);
        checking=0;repeat(5)@(negedge clk);
        // Reset while a completed output is stalled, including metadata FIFO.
        force m_ready=0;
        output_keep=1;input_zero=0;output_zero=0;biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};
        send_packet((586'd3<<584)|(586'hff<<576)|(586'd1<<64)|586'd1,0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        repeat(4)@(negedge clk);pulse_reset();release m_ready;repeat(5)@(negedge clk);
        if(m_valid)$fatal(1,"Reset replayed an old result");
        // Sparse output mask: tag FIFO must skip disabled channels and put
        // LAST on the final enabled lane, not always lane 7 or count-1.
        force m_ready=0;
        output_keep=8'ha5;input_zero=0;output_zero=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};
        for(lane=0;lane<8;lane=lane+1)biases[32*lane+:32]=lane*128;
        send_packet((586'd3<<584)|(586'hff<<576),0);
        @(negedge clk);s_valid=0;
        for(lane=0;lane<8;lane=lane+1)if(output_keep[lane])begin
            while(!m_valid)@(negedge clk);
            if(m_error||m_lane!==lane[2:0]||m_acc!==lane*128||m_code!==(lane*64>255?8'd255:lane*64)||m_last!==(lane==7))
                $fatal(1,"Sparse output mask/tag failed lane %0d",lane);
            force m_ready=1;@(posedge clk);@(negedge clk);force m_ready=0;
        end
        release m_ready;repeat(5)@(negedge clk);pulse_reset();
        // Invalid zero output mask must not hang: emit one flagged result.
        output_keep=0;input_zero=0;output_zero=0;biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};
        send_packet((586'd3<<584)|(586'hff<<576),0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        if(!m_error||!m_last)$fatal(1,"Zero output mask not flagged");pulse_reset();
        // Final bias overflow must be flagged for affected output, not accepted silently.
        output_keep=1;input_zero=0;output_zero=0;biases={8{32'h7fffffff}};multipliers={8{32'd1073741824}};shifts={8{6'd31}};
        send_packet((586'd3<<584)|(586'd1<<576)|(586'd127<<64)|586'd255,0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        if(!m_error)$fatal(1,"Tile ACC32 overflow not flagged");pulse_reset();
        // A missing FIRST after reset must produce an error, not hang.
        output_keep=1;input_zero=0;output_zero=0;biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};
        send_packet((586'd1<<585)|(586'd1<<576),0);
        @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
        if(!m_error||!m_last)$fatal(1,"Missing first marker not flagged");pulse_reset();
        if(USE_CACHE)begin
            // A group beyond physical depth is rejected by error; never read unwritten RAM.
            output_keep=1;input_zero=0;output_zero=0;biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};
            for(p=0;p<129;p=p+1)send_packet((p==0?(586'd1<<584):0)|(p==128?(586'd1<<585):0)|(586'd1<<576),0);
            @(negedge clk);s_valid=0;while(!m_valid)@(negedge clk);
            if(!m_error)$fatal(1,"Cache depth overflow not flagged");pulse_reset();
        end
        $display("Lung tile8 RTL: PASS (cached=%0d, %0d outputs, %0d groups, 64 MACs/shared quant, tails, stalls, reset, errors)",USE_CACHE,received,header[0]);$finish;
    end
    initial begin #1000000000;$fatal(1,"Tile timeout");end
endmodule

module tb_lung_mac_tile8_cached;
    tb_lung_mac_tile8 #(.USE_CACHE(1)) test();
endmodule
