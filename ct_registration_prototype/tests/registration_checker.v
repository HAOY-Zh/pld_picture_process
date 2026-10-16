// Synthesizable self-checking stimulus for Yosys cycle simulation.
// The SystemVerilog testbench in fpga/ is the Vivado/XSim entry point.
module registration_checker(input wire clk, input wire rst,
    output reg failed, output reg done, output reg [31:0] received,
    output reg [31:0] cycles);
    localparam W=5,H=4,N=20,OUT=24,FRAMES=6;
    reg [31:0] configs[0:35];
    reg [31:0] inputs[0:119];
    reg [31:0] expected[0:143];
    initial begin
        $readmemh("tests/vectors/config.mem",configs);
        $readmemh("tests/vectors/input.mem",inputs);
        $readmemh("tests/vectors/expected.mem",expected);
    end
    reg [31:0] frame_id,pixel_id,lfsr;
    reg waiting,stalled;
    reg [31:0] held_data; reg held_last;
    wire [31:0] in_data=inputs[frame_id*N+pixel_id];
    wire in_valid=!done&&!waiting&&(frame_id<FRAMES)&&(cycles[1:0]!=0);
    wire in_last=(pixel_id==N-1)||(frame_id==5&&pixel_id==3);
    wire [3:0] in_keep=(frame_id==5&&pixel_id==3)?4'h7:4'hf;
    wire in_ready;
    wire [31:0] out_data; wire [3:0] out_keep;
    wire out_last,out_valid;
    wire out_ready=lfsr[0]|lfsr[3];
    registration_axis #(.IMAGE_WIDTH(W),.IMAGE_HEIGHT(H)) dut(
        .clk(clk),.rst_n(!rst),
        .a_in(configs[frame_id*6]),.b_in(configs[frame_id*6+1]),
        .c_in(configs[frame_id*6+2]),.d_in(configs[frame_id*6+3]),
        .e_in(configs[frame_id*6+4]),.f_in(configs[frame_id*6+5]),
        .s_axis_tdata(in_data),.s_axis_tkeep(in_keep),.s_axis_tlast(in_last),
        .s_axis_tvalid(in_valid),.s_axis_tready(in_ready),
        .m_axis_tdata(out_data),.m_axis_tkeep(out_keep),.m_axis_tlast(out_last),
        .m_axis_tvalid(out_valid),.m_axis_tready(out_ready));
    always @(posedge clk) begin
        if(rst) begin
            failed<=0; done<=0; received<=0; cycles<=0;
            frame_id<=0; pixel_id<=0; lfsr<=32'h12345678;
            waiting<=0; stalled<=0; held_data<=0; held_last<=0;
        end else begin
            cycles<=cycles+1;
            lfsr<={lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
            if(stalled&&(!out_valid||out_data!=held_data||out_last!=held_last||out_keep!=4'hf)) failed<=1;
            stalled<=out_valid&&!out_ready;
            held_data<=out_data; held_last<=out_last;
            if(in_valid&&in_ready) begin
                if(pixel_id==N-1) waiting<=1;
                else pixel_id<=pixel_id+1;
            end
            if(out_valid&&out_ready) begin
                if(received>=FRAMES*OUT) failed<=1;
                else if(out_data!=expected[received]||out_keep!=4'hf||out_last!=((received%OUT)==OUT-1)) failed<=1;
                received<=received+1;
                if((received%OUT)==OUT-1) begin
                    frame_id<=frame_id+1; pixel_id<=0; waiting<=0;
                end
                if(received==FRAMES*OUT-1) done<=1;
            end
        end
    end
endmodule
