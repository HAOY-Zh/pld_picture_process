`timescale 1ns/1ps
module tb_ct_denoise_tile_axis;
    `include "vectors.vh"
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0;reg [31:0]cfg=0,s_data=0;reg [3:0]s_keep=15;
    reg s_last=0,s_valid=0,m_ready=0;
    wire s_ready,m_valid,m_last;wire [31:0]m_data,status;wire [3:0]m_keep;
    reg [31:0]inputs[0:PIXELS-1],expected[0:OUTPUT_PIXELS-1];
    reg [31:0]settings[0:FRAMES-1],offsets[0:FRAMES-1],sizes[0:FRAMES-1],statuses[0:FRAMES-1],bad[0:FRAMES-1];
    reg [31:0]output_offsets[0:FRAMES-1],output_sizes[0:FRAMES-1];
    integer frame=0,received=0,cycles=0,checks=0,i,start_cycle;
    integer qlayer,qtrial,qshift;
    reg signed [31:0] qinput;
    reg signed [63:0] qwide,qexpected;
    reg checking=0,held=0;reg [31:0]held_data;reg held_last;
    ct_denoise_tile_axis dut(.clk(clk),.rst_n(rst_n),.cfg(cfg),.status(status),
        .s_axis_tdata(s_data),.s_axis_tkeep(s_keep),.s_axis_tlast(s_last),.s_axis_tvalid(s_valid),.s_axis_tready(s_ready),
        .m_axis_tdata(m_data),.m_axis_tkeep(m_keep),.m_axis_tlast(m_last),.m_axis_tvalid(m_valid),.m_axis_tready(m_ready));
    always @(negedge clk)begin cycles=cycles+1;m_ready=rst_n && ((cycles%19)>6);end
    always @(posedge clk)begin
        if(rst_n && !cfg[31] && checking)begin
            if(held && (m_valid!==1 || m_data!==held_data || m_last!==held_last || m_keep!==15))$fatal(1,"Backpressure violation");
            held=m_valid&&!m_ready;held_data=m_data;held_last=m_last;
            if(m_valid&&m_ready)begin
                if(received>=output_sizes[frame])$fatal(1,"Extra output");
                if(m_data!==expected[output_offsets[frame]+received])$fatal(1,"Mismatch frame=%0d mode=%0d pixel=%0d got=%h expected=%h",frame,settings[frame]&15,received,m_data,expected[output_offsets[frame]+received]);
                if(m_last!==(received==output_sizes[frame]-1)||m_keep!==15)$fatal(1,"Output sidebands");
                received=received+1;checks=checks+1;
            end
        end else held=0;
    end
    task send_word(input [31:0]word,input integer last,input integer malformed);
        begin
            @(negedge clk);s_data=word;s_valid=1;s_last=last;s_keep=malformed?3:15;
            if(malformed)s_last=!last;
            @(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);s_valid=0;s_last=0;s_keep=15;
        end
    endtask
    initial begin
        // Audit rounding independently, including int32 extremes and ties.
        for(qlayer=0;qlayer<3;qlayer=qlayer+1)begin
            qshift=qlayer==0 ? MODEL_SHIFT_0 : qlayer==1 ? MODEL_SHIFT_1 : MODEL_SHIFT_2;
            for(qtrial=0;qtrial<1200;qtrial=qtrial+1)begin
                case(qtrial)
                    0:qinput=32'sh80000000;1:qinput=32'sh7fffffff;
                    2:qinput=-(1<<(qshift-1));3:qinput=(1<<(qshift-1));
                    default:qinput=qtrial<600 ? qtrial-300 : $random;
                endcase
                qwide={{32{qinput[31]}},qinput};
                qexpected=((qwide<0 ? -qwide : qwide)+(64'sd1<<(qshift-1)))>>>qshift;
                if(qwide<0)qexpected=-qexpected;
                if(qlayer<2 && qexpected<0)qexpected=0;
                if(qexpected>32767)qexpected=32767;else if(qexpected< -32768)qexpected=-32768;
                if(dut.cnn_quantize(qinput,qlayer)!==qexpected[15:0])$fatal(1,"CNN requantization mismatch layer=%0d input=%0d",qlayer,qinput);
            end
        end
        $readmemh("input.hex",inputs);$readmemh("expected.hex",expected);
        $readmemh("settings.hex",settings);$readmemh("offsets.hex",offsets);$readmemh("sizes.hex",sizes);
        $readmemh("statuses.hex",statuses);$readmemh("bad.hex",bad);
        $readmemh("output_offsets.hex",output_offsets);$readmemh("output_sizes.hex",output_sizes);
        repeat(4)@(negedge clk);rst_n=1;
        // Abort a CNN during computation; software reset must clear history.
        cfg=6|(1<<4)|(1<<11);send_word(32'hffff,1,0);
        repeat(8)@(negedge clk);cfg=32'h80000000;repeat(4)@(negedge clk);cfg=0;
        for(frame=0;frame<FRAMES;frame=frame+1)begin
            checking=0;
            // Compact/legacy CNN tiles alternate with no reset, as in the
            // real driver: dimensions/crops change and BRAM history persists.
            if(frame<39 || frame>=74)begin cfg=32'h80000000;repeat(3)@(negedge clk);end
            else @(negedge clk);
            cfg=settings[frame];received=0;checking=1;start_cycle=cycles;
            for(i=0;i<sizes[frame];i=i+1)begin
                send_word(inputs[offsets[frame]+i],i==sizes[frame]-1,bad[frame]&&i==0);
                if(i==0)cfg=32'h0003f7ff; // Changing config midframe must be ignored.
                if(i%5==0)repeat(2)@(negedge clk);
            end
            wait(received==output_sizes[frame]);repeat(3)@(negedge clk);
            if(status!==statuses[frame])$fatal(1,"Status frame=%0d got=%h expected=%h",frame,status,statuses[frame]);
            $display("TILE_FRAME_PASS frame=%0d mode=%0d pixels=%0d cycles=%0d",frame,settings[frame]&15,received,cycles-start_cycle);
        end
        checking=0;$display("CT_TILE_PASS frames=%0d pixels=%0d cycles=%0d",FRAMES,checks,cycles);$finish;
    end
    initial begin #2000000000;$fatal(1,"Timeout");end
endmodule
