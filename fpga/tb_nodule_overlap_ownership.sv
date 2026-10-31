`timescale 1ns/1ps
// Independent overlap ownership oracle: a malformed group beside two valid
// groups, pending reload, output stalls, then replacement weight verification.
module tb_nodule_overlap_ownership;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,w_valid=0,w_first=1,w_last=1,s_valid=0,s_first=0,s_last=1,m_ready=0;
    reg [511:0] weights=1;reg [63:0] activations=0;
    wire w_ready,s_ready,loaded,m_valid,m_last,m_error;
    wire [2:0] m_lane;wire [31:0] m_acc;wire [7:0] m_code;
    integer received=0,cycle=0,max_pending=0,overlaps=0;
    reg checking=0,stalled=0;reg [44:0] held;
    lung_mac_tile8_reuse dut(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),
        .w_first(w_first),.w_last(w_last),.w_weights(weights),.w_output_keep(8'd1),
        .w_input_zero_point(8'd0),.w_output_zero_point(8'd0),.w_biases(256'd0),
        .w_multipliers({8{32'd1073741824}}),.w_shifts({8{6'd30}}),.w_relu(1'b0),
        .weights_loaded(loaded),.s_valid(s_valid),.s_ready(s_ready),.s_first(s_first),.s_last(s_last),
        .s_activations(activations),.s_input_keep(8'd1),.m_valid(m_valid),.m_ready(m_ready),
        .m_lane(m_lane),.m_last(m_last),.m_acc(m_acc),.m_code(m_code),.m_error(m_error));
    always @(posedge clk)if(rst_n)begin
        cycle=cycle+1;
        if(dut.pending_groups>max_pending)max_pending=dut.pending_groups;
        if(dut.tile.accepted&&!dut.tile.input_active&&dut.tile.inflight!=0)overlaps=overlaps+1;
        if(dut.pending_groups>3||dut.tile.inflight>2)$fatal(1,"Ownership overflow");
        if(stalled&&(!m_valid||{m_lane,m_last,m_acc,m_code,m_error}!==held))$fatal(1,"Stalled output corrupted by another pixel");
        stalled=m_valid&&!m_ready;held={m_lane,m_last,m_acc,m_code,m_error};
        if(checking&&m_valid&&m_ready)begin
            if(received>=3||m_lane!==0||!m_last||m_acc!==(3+2*received)||m_code!==(8'd3+2*received)||m_error!==(received==0))
                $fatal(1,"Cross-group error/result leak index %0d acc=%0d code=%0d error=%b",received,m_acc,m_code,m_error);
            received=received+1;
        end
    end else stalled=0;
    task send(input integer value,input reg first);
        begin
            @(negedge clk);s_valid=1;activations=value;s_first=first;
            @(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);s_valid=0;
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;w_valid=1;
        @(posedge clk);while(!w_ready)@(posedge clk);
        @(negedge clk);w_valid=0;checking=1;
        send(3,0); // Bad FIRST must affect this group only.
        w_valid=1;weights=2; // A different producer requests the next weight block.
        send(5,1);send(7,1);
        while(dut.tile.inflight!=2)@(negedge clk);
        repeat(30)begin @(negedge clk);if(w_ready)$fatal(1,"Reload accepted while results pending");end
        if(max_pending!=3||overlaps<1)$fatal(1,"Did not exercise both slots plus the input holding register");
        m_ready=1;
        while(received<3)@(negedge clk);
        checking=0;
        // Exactly one replacement load, immediately after all prior outputs.
        @(posedge clk);if(!w_ready)$fatal(1,"Pending reload did not make progress");
        @(negedge clk);w_valid=0;m_ready=0;
        send(11,1);
        while(!m_valid)@(negedge clk);
        if(m_error||m_acc!==32'd22||m_code!==8'd22||!m_last)$fatal(1,"Replacement weight/result ownership failed");
        repeat(5)@(negedge clk);rst_n=0;
        repeat(3)@(negedge clk);if(m_valid||s_ready||loaded||dut.pending_groups||dut.tile.inflight)$fatal(1,"Reset leaked ownership");
        rst_n=1;repeat(8)@(negedge clk);if(m_valid)$fatal(1,"Old result replayed after reset");
        $display("Nodule overlap ownership RTL: PASS (bad/valid adjacency, three pending groups, two result slots, pending reload, output stability, reset)");$finish;
    end
    initial begin #1000000;$fatal(1,"Ownership timeout");end
endmodule
