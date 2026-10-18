`timescale 1ns/1ps
module tb_lung_cached_axi;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,w_valid=0,w_first=0,w_last=0,c_valid=0,m_ready=0;
    reg[511:0]weights=0;reg[7:0]output_keep=0,input_zero=0,output_zero=0;
    reg[255:0]biases=0,multipliers=0;reg[47:0]shifts=0;reg relu=0;
    reg[31:0]base=0;reg[9:0]height=0,width=0;reg[6:0]channels=0;reg kernel3=0,padding=0;
    wire w_ready,c_ready,loaded,m_valid,m_last,m_final,m_error,busy,done,error;
    wire[8:0]m_x,m_y;wire[2:0]m_lane;wire signed[31:0]m_acc;wire[7:0]m_code;
    wire[31:0]hits,misses,araddr;wire[7:0]arlen;wire[3:0]arid,arcache;wire[2:0]arsize,arprot;
    wire[1:0]arburst;wire arlock,arvalid,arready,rready;
    reg[3:0]rid=0;reg[63:0]rdata=0;reg[1:0]rresp=0;reg rlast=0,rvalid=0;
    reg read_active=0,inject_error=0;integer latency=0,remaining=0,cycle=0;reg[31:0]read_address;
    reg[31:0]header[0:8],frames[0:7999],traffic[0:2999],expected_requests[0:799999],burst_addresses[0:19999];
    reg[7:0]burst_beats[0:19999];reg[63:0]features[0:99999];reg[511:0]weight_bank[0:59999];
    reg[74:0]expected_windows[0:799999];reg[19:0]window_coords[0:799999];reg[20:0]coords[0:159999];
    reg signed[31:0]bias_bank[0:15999],mult_bank[0:15999],expected_acc[0:159999];
    reg[5:0]shift_bank[0:15999];reg[7:0]expected_code[0:159999];reg[2:0]lanes[0:159999];reg lasts[0:159999];
    integer received=0,windows=0,requests=0,bursts=0,read_beats=0,completed=0,loads=0,f,t,p,n,offset,pbase,lane,fh;
    reg checking=0,out_stall=0,ar_stall=0,r_stall=0;reg[63:0]held_output;
    reg[60:0]held_ar;reg[70:0]held_r;
    lung_conv8_cached_axi dut(.clk(clk),.rst_n(rst_n),.w_valid(w_valid),.w_ready(w_ready),.w_first(w_first),.w_last(w_last),
        .w_weights(weights),.w_output_keep(output_keep),.w_input_zero_point(input_zero),.w_output_zero_point(output_zero),
        .w_biases(biases),.w_multipliers(multipliers),.w_shifts(shifts),.w_relu(relu),.weights_loaded(loaded),
        .c_valid(c_valid),.c_ready(c_ready),.c_base(base),.c_height(height),.c_width(width),.c_channels(channels),.c_kernel3(kernel3),.c_padding(padding),
        .m_valid(m_valid),.m_ready(m_ready),.m_lane(m_lane),.m_last(m_last),.m_frame_last(m_final),.m_x(m_x),.m_y(m_y),.m_acc(m_acc),.m_code(m_code),.m_error(m_error),
        .busy(busy),.done(done),.error(error),.cache_hits(hits),.cache_misses(misses),
        .m_axi_arid(arid),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arlock(arlock),.m_axi_arcache(arcache),.m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rid(rid),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready));
    assign arready=rst_n&&!read_active&&!rvalid&&(cycle%11<8);
    always @(negedge clk)begin cycle=cycle+1;m_ready=(cycle%23)<15;end
    // AXI memory BFM, shared cancellation reset. Independent AR stalls/R gaps.
    always @(posedge clk)begin
        if(!rst_n)begin read_active<=0;rvalid<=0;latency<=0;remaining<=0;end
        else begin
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||araddr[2:0]!=0||arlen>15||
                    ({1'b0,araddr[11:0]}+((arlen+1)*8))>4096)$fatal(1,"Illegal AXI read burst");
                read_active<=1;read_address<=araddr>>3;remaining<=arlen+1;latency<=cycle%5+1;
            end
            if(read_active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin
                    if(read_address>=header[1])$fatal(1,"AXI read outside feature bank");
                    rvalid<=1;rid<=0;rdata<=features[read_address];rresp<=inject_error?2'b10:0;rlast<=remaining==1;
                end
            end
            if(rvalid&&rready)begin
                rvalid<=0;
                if(rlast)read_active<=0;
                else begin remaining<=remaining-1;read_address<=read_address+1'b1;latency<=cycle%3;end
            end
        end
    end
    always @(posedge clk)if(rst_n&&checking)begin
        if(out_stall&&(!m_valid||{m_x,m_y,m_final,m_lane,m_last,m_acc,m_code,m_error}!==held_output))$fatal(1,"Cached output unstable");
        if(ar_stall&&(!arvalid||{arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot}!==held_ar))$fatal(1,"AXI AR unstable");
        if(r_stall&&(!rvalid||{rid,rdata,rresp,rlast}!==held_r))$fatal(1,"AXI R unstable");
        out_stall=m_valid&&!m_ready;held_output={m_x,m_y,m_final,m_lane,m_last,m_acc,m_code,m_error};
        ar_stall=arvalid&&!arready;held_ar={arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot};
        r_stall=rvalid&&!rready;held_r={rid,rdata,rresp,rlast};
        if(w_valid&&w_ready)loads=loads+1;
        if(arvalid&&arready)begin
            if(bursts>=header[7]||(araddr>>3)!==burst_addresses[bursts]||arlen+1!==burst_beats[bursts])$fatal(1,"AXI miss burst mismatch %0d",bursts);
            bursts=bursts+1;
        end
        if(rvalid&&rready)read_beats=read_beats+1;
        if(dut.req_valid&&dut.req_ready)begin
            if(requests>=header[6]||dut.req_addr!==expected_requests[requests])$fatal(1,"Upstream request mismatch %0d",requests);
            requests=requests+1;
        end
        if(dut.conv.win_valid&&dut.conv.win_ready)begin
            if(windows>=header[5]||dut.conv.win_error||
                {dut.conv.win_frame_last,dut.conv.win_last,dut.conv.win_first,dut.conv.win_keep,dut.conv.win_data}!==expected_windows[windows]||
                {1'b0,dut.conv.win_y,1'b0,dut.conv.win_x}!==window_coords[windows])$fatal(1,"Cached window mismatch %0d",windows);
            windows=windows+1;
        end
        if(m_valid&&m_ready)begin
            if(received>=header[4]||m_error||$isunknown({m_acc,m_code,m_x,m_y,m_lane,m_last,m_final,m_error})||
                m_acc!==expected_acc[received]||m_code!==expected_code[received]||m_lane!==lanes[received]||m_last!==lasts[received]||
                {m_final,1'b0,m_y,1'b0,m_x}!==coords[received])$fatal(1,"Cached convolution output %0d acc=%0d/%0d code=%0d/%0d err=%b",received,m_acc,expected_acc[received],m_code,expected_code[received],m_error);
            received=received+1;if(m_final)completed=completed+1;
        end
        if(done&&error)$fatal(1,"Unexpected cached frame error");
    end else begin out_stall=0;ar_stall=0;r_stall=0;end
    task reset_core;
        begin @(negedge clk);rst_n=0;w_valid=0;c_valid=0;repeat(3)@(negedge clk);if(m_valid||arvalid||busy||loaded)$fatal(1,"Cached reset failed");rst_n=1;end
    endtask
    task send_weight(input reg[511:0]word,input reg first,input reg last);
        begin @(negedge clk);w_valid=1;w_first=first;w_last=last;weights=word;@(posedge clk);while(!w_ready)@(posedge clk);end
    endtask
    task start_frame;
        begin @(negedge clk);c_valid=1;@(posedge clk);while(!c_ready)@(posedge clk);@(negedge clk);c_valid=0;end
    endtask
    task load_simple;
        begin
            @(negedge clk);output_keep=1;input_zero=128;output_zero=128;biases=0;multipliers={8{32'd1073741824}};shifts={8{6'd31}};relu=0;
            send_weight(512'd1,1,1);@(negedge clk);w_valid=0;base=0;height=1;width=1;channels=1;kernel3=0;padding=0;
        end
    endtask
    initial begin
        fh=$fopen("vectors/header.mem","r");if(!fh)$fatal(1,"Missing cached AXI vectors");$fclose(fh);
        $readmemh("vectors/header.mem",header,0,8);
        if($isunknown(header[0])||header[0]==0||header[4]==0||header[0]*16>8000||header[1]>100000||header[2]>60000||header[3]>16000||header[4]>160000||header[5]>800000||header[6]>800000||header[7]>20000||header[8]==0)$fatal(1,"Invalid cached vector counts");
        $readmemh("vectors/frames.mem",frames,0,header[0]*16-1);$readmemh("vectors/traffic.mem",traffic,0,header[0]*6-1);
        $readmemh("vectors/features.mem",features,0,header[1]-1);$readmemh("vectors/weights.mem",weight_bank,0,header[2]-1);
        $readmemh("vectors/bias.mem",bias_bank,0,header[3]-1);$readmemh("vectors/multipliers.mem",mult_bank,0,header[3]-1);$readmemh("vectors/shifts.mem",shift_bank,0,header[3]-1);
        $readmemh("vectors/windows.mem",expected_windows,0,header[5]-1);$readmemh("vectors/window_coords.mem",window_coords,0,header[5]-1);
        $readmemh("vectors/requests.mem",expected_requests,0,header[6]-1);$readmemh("vectors/burst_addresses.mem",burst_addresses,0,header[7]-1);$readmemh("vectors/burst_beats.mem",burst_beats,0,header[7]-1);
        $readmemh("vectors/accumulators.mem",expected_acc,0,header[4]-1);$readmemh("vectors/codes.mem",expected_code,0,header[4]-1);
        $readmemh("vectors/lanes.mem",lanes,0,header[4]-1);$readmemh("vectors/lasts.mem",lasts,0,header[4]-1);$readmemh("vectors/coords.mem",coords,0,header[4]-1);
        repeat(3)@(negedge clk);rst_n=1;checking=1;
        for(f=0;f<header[0];f=f+1)begin
            t=f*16;offset=frames[t+10];n=frames[t+11];pbase=frames[t+12];
            @(negedge clk);output_keep=frames[t+8][7:0];input_zero=frames[t+6][7:0];output_zero=frames[t+7][7:0];relu=frames[t+9][0];
            for(lane=0;lane<8;lane=lane+1)begin biases[32*lane+:32]=bias_bank[pbase+lane];multipliers[32*lane+:32]=mult_bank[pbase+lane];shifts[6*lane+:6]=shift_bank[pbase+lane];end
            for(p=0;p<n;p=p+1)send_weight(weight_bank[offset+p],p==0,p==n-1);
            @(negedge clk);w_valid=0;base=frames[t];height=frames[t+1][9:0];width=frames[t+2][9:0];channels=frames[t+3][6:0];kernel3=frames[t+4]==3;padding=frames[t+5][0];
            start_frame();base=32'hffffffff;height=0;width=0;channels=0;
            while(!done)@(negedge clk);
            if(error||busy||hits!==traffic[f*6+1]||misses!==traffic[f*6+2])$fatal(1,"Per-frame cache statistics mismatch %0d hits=%0d/%0d misses=%0d/%0d",f,hits,traffic[f*6+1],misses,traffic[f*6+2]);
            if(f%15==0)$display("Cached AXI frame %0d/%0d complete (%0d outputs)",f+1,header[0],received);
        end
        if(received!==header[4]||windows!==header[5]||requests!==header[6]||bursts!==header[7]||read_beats!==header[8]||completed!==header[0]||loads!==header[2])$fatal(1,"Incomplete cached AXI coverage");
        checking=0;reset_core();load_simple();start_frame();while(!read_active)@(negedge clk);reset_core();
        load_simple();inject_error=1;force m_ready=0;start_frame();while(!m_valid)@(negedge clk);
        if(!m_error||m_acc!==0||m_code!==128||done||!busy)$fatal(1,"AXI error did not propagate to pixel");
        repeat(4)@(negedge clk);if(!m_error||!m_valid||done)$fatal(1,"Error output unstable");
        force m_ready=1;@(posedge clk);@(negedge clk);if(!done||busy||!error)$fatal(1,"Error frame did not finish on handshake");
        release m_ready;inject_error=0;reset_core();
        $display("Lung cached AXI convolution RTL: PASS (%0d frames, %0d outputs, %0d windows, %0d requests, %0d AXI bursts, %0d AXI read beats; clipped spans, hits, stalls, shared reset, AXI errors)",completed,received,windows,requests,bursts,read_beats);$finish;
    end
    initial begin #1000000000;$fatal(1,"Cached AXI timeout");end
endmodule
