"""Actual XSim co-simulation of all 11 nodule nodes in shared DDR memory.

Only the initial lowered input is injected. Each subsequent input is a previous
RTL output. All writes, head tail lanes, DDR canaries, completion ordering and
AXI backpressure are checked against an independent integer oracle.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import load_package,sha
from lung_axil_smoke import feature_words
from lung_fast import prepare_block


def mem(path,values,bits):
    mask=(1<<bits)-1
    Path(path).write_text(''.join(f'{int(v)&mask:0{(bits+3)//4}x}\n' for v in values),encoding='ascii')


def wrapper():
    # Port wiring copied verbatim from the accepted AXI-Lite integration bench.
    prior=(ROOT/'data/lung_flow_20261017/benches/tb_lung_axil_accel.sv').read_text(encoding='utf-8')
    begin=prior.index('    wire[9:0]');end=prior.index('    integer mode=')
    declarations=prior[begin:end]
    start=prior.index('lung_conv8_axil dut(')
    ports=prior[start+len('lung_conv8_axil dut('):prior.index('));',start)+1]
    declarations=declarations.replace('reg hold_final=1','reg hold_final=1')
    return '''`timescale 1ns/1ps
module nodule_sim_engine #(parameter TENSOR=0)(input clk,input rst_n);
'''+declarations+'''
    integer mode=0;reg[1:0]response=0;reg[3:0]response_id=0;
    reg hold_final=1,force_block=0,checking=0;
    integer frame_commits=0,frame_pixels=0,acks=0,bursts=0;
    integer a_base=0,a_words=0,b_base=0,b_words=0,dst_base=0,dst_words=0,blocks=1,block=0;
    wire block_b=force_block||(checking&&hold_final&&frame_commits==frame_pixels);
    reg held_ar=0;reg[56:0]previous_ar;
    lung_test_axil_master host(.*);lung_test_burst_memory mem(.*);
    generate if(TENSOR)begin:ten
        lung_tensor_axil dut('''+ports+''');
    end else begin:con
        lung_conv8_axil dut('''+ports+''');
    end endgenerate
    assign arready=rst_n&&!read_active&&!rvalid&&cycle%11<8;
    always @(negedge clk)cycle=cycle+1;
    integer wi;
    always @(posedge clk)begin
        if(!rst_n)begin read_active<=0;rvalid<=0;latency<=0;remaining<=0;held_ar<=0;end
        else begin
            if(held_ar&&(!arvalid||{arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot}!==previous_ar))$fatal(1,"Nodule AR changed during stall");
            held_ar<=arvalid&&!arready;previous_ar<={arid,araddr,arlen,arsize,arburst,arlock,arcache,arprot};
            if(arvalid&&arready)begin
                if(arid!=0||arsize!=3||arburst!=1||araddr[2:0]!=0||arlen>15||({1'b0,araddr[11:0]}+(arlen+1)*8)>4096)$fatal(1,"Invalid nodule read burst");
                read_active<=1;read_address<=araddr>>3;remaining<=arlen+1;latency<=cycle%5+1;bursts=bursts+1;
            end
            if(read_active&&!rvalid)begin
                if(latency!=0)latency<=latency-1;
                else begin
                    if(!((read_address>=a_base&&read_address<a_base+a_words)||(read_address>=b_base&&read_address<b_base+b_words)))$fatal(1,"Read outside current source tensors: %0d",read_address);
                    if(!tb_nodule_network.written[read_address])$fatal(1,"Read before RTL producer: %0d",read_address);
                    rdata<=tb_nodule_network.memory[read_address];rvalid<=1;rid<=0;rresp<=0;rlast<=remaining==1;
                end
            end
            if(rvalid&&rready)begin
                rvalid<=0;
                if(rlast)read_active<=0;else begin remaining<=remaining-1;read_address<=read_address+1;latency<=cycle%3;end
            end
            if(bvalid&&bready)acks=acks+1;
            if(commit)begin
                wi=address>>3;
                if(!checking||address[2:0]!=0||wi<dst_base||wi>=dst_base+dst_words||(!TENSOR&&(wi-dst_base)%blocks!=block)
                    ||frame_commits>=frame_pixels||strb!==255||data!==tb_nodule_network.gold[wi]
                    ||tb_nodule_network.written[wi])$fatal(1,"Nodule DDR output mismatch tensor=%0d address=%h data=%h gold=%h",TENSOR,address,data,tb_nodule_network.gold[wi]);
                tb_nodule_network.memory[wi]=data;tb_nodule_network.written[wi]=1;frame_commits=frame_commits+1;
            end
        end
    end
    task put(input reg[9:0]address,input reg[31:0]word);
        begin host.write_reg(address,word,15,host.writes%3,0);end
    endtask
    task finish_command;
        reg[31:0]status;
        begin
            while(frame_commits!=frame_pixels)@(negedge clk);
            host.read_reg(8,0,status);
            if(!status[0]||status[1])$fatal(1,"DONE before final nodule B response");
            hold_final=0;
            host.read_reg(8,0,status);while(!status[1]||status[0])host.read_reg(8,0,status);
            if(status[2]||status[8]||read_active||rvalid||bvalid||mem.pending)$fatal(1,"Nodule command failed or premature completion");
            checking=0;
        end
    endtask
endmodule
'''


TOP='''`timescale 1ns/1ps
module tb_nodule_network;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    reg[63:0]memory[0:99999],gold[0:99999];reg written[0:99999],expected_written[0:99999];
    reg[31:0]header[0:4],commands[0:999],bias[0:999],mult[0:999],shift[0:999];
    reg[511:0]weights[0:9999];
    integer f,t,p,k,i,fh,cstart;reg[31:0]status,value;
    nodule_sim_engine #(.TENSOR(0)) conv(clk,rst_n);
    nodule_sim_engine #(.TENSOR(1)) tensor(clk,rst_n);
    initial begin
        fh=$fopen("vectors/header.mem","r");if(!fh)$fatal(1,"Missing nodule vectors");$fclose(fh);
        $readmemh("vectors/header.mem",header,0,4);
        if(header[0]!=15||header[1]>100000||header[2]>10000||header[3]>1000)$fatal(1,"Bad nodule fixture header");
        $readmemh("vectors/commands.mem",commands,0,header[0]*28-1);
        $readmemh("vectors/initial.mem",memory,0,header[1]-1);
        $readmemh("vectors/gold.mem",gold,0,header[1]-1);
        $readmemh("vectors/initial_written.mem",written,0,header[1]-1);
        $readmemh("vectors/expected_written.mem",expected_written,0,header[1]-1);
        $readmemh("vectors/weights.mem",weights,0,header[2]-1);
        $readmemh("vectors/bias.mem",bias,0,header[3]-1);
        $readmemh("vectors/mult.mem",mult,0,header[3]-1);
        $readmemh("vectors/shift.mem",shift,0,header[3]-1);
        repeat(4)@(negedge clk);rst_n=1;
        conv.host.read_reg(0,0,value);if(value!==32'h4c554e47)$fatal(1,"Wrong conv ID");
        tensor.host.read_reg(0,0,value);if(value!==32'h54454e53)$fatal(1,"Wrong tensor ID");
        for(f=0;f<header[0];f=f+1)begin
            t=f*28;
            if(commands[t]==0)begin
                conv.a_base=commands[t+1];conv.a_words=commands[t+25];conv.b_words=0;
                conv.dst_base=commands[t+2];conv.dst_words=commands[t+24];
                conv.blocks=(commands[t+7]+7)/8;conv.block=commands[t+8];
                conv.frame_pixels=commands[t+5]*commands[t+6];conv.frame_commits=0;conv.checking=1;conv.hold_final=1;conv.mode=f%3;
                conv.put(16,commands[t+1]*8);conv.put(20,commands[t+2]*8);conv.put(24,commands[t+5]|(commands[t+6]<<16));
                conv.put(28,commands[t+4]|((commands[t+9]==3)<<8)|(commands[t+10]<<9)|(conv.blocks<<16)|(commands[t+8]<<24));
                conv.put(32,commands[t+27]|(commands[t+11]<<8)|(commands[t+12]<<16)|(commands[t+13]<<24));
                for(k=0;k<8;k=k+1)begin
                    conv.put(10'h140+4*k,bias[commands[t+16]+k]);conv.put(10'h160+4*k,mult[commands[t+16]+k]);conv.put(10'h180+4*k,shift[commands[t+16]+k]);
                end
                for(p=0;p<commands[t+15];p=p+1)begin
                    for(k=0;k<16;k=k+1)conv.put(10'h100+4*k,weights[commands[t+14]+p][32*k+:32]);
                    conv.put(4,4|((p==0)<<3)|((p==commands[t+15]-1)<<4));
                end
                conv.put(4,1);conv.finish_command();conv.host.read_reg(48,0,value);
            end else begin
                tensor.a_base=commands[t+1];tensor.a_words=commands[t+25];tensor.b_base=commands[t+3];tensor.b_words=commands[t+26];
                tensor.dst_base=commands[t+2];tensor.dst_words=commands[t+24];
                tensor.frame_pixels=commands[t+24];tensor.frame_commits=0;tensor.checking=1;tensor.hold_final=1;tensor.mode=f%3;
                tensor.put(16,commands[t+1]*8);tensor.put(20,commands[t+3]*8);tensor.put(24,commands[t+2]*8);
                tensor.put(28,commands[t+5]|(commands[t+6]<<16));
                tensor.put(32,commands[t+8]|(commands[t+4]<<8)|(commands[t+7]<<16));
                tensor.put(36,commands[t+17]|(commands[t+18]<<8)|(commands[t+19]<<16));
                tensor.put(40,commands[t+20]);tensor.put(44,commands[t+21]);tensor.put(48,commands[t+22]);tensor.put(52,commands[t+23]);
                tensor.put(4,1);tensor.finish_command();tensor.host.read_reg(64,0,value);
            end
            if(value==0)$fatal(1,"Missing nodule cycle counter");
            $display("NODULE_COMMAND_PASS command=%0d kind=%0d cycles=%0d",f,commands[t],value);
        end
        for(i=0;i<header[1];i=i+1)begin
            if(written[i]!==expected_written[i])$fatal(1,"Nodule missing/extra write %0d",i);
            if(memory[i]!==gold[i])$fatal(1,"Nodule final memory/canary mismatch %0d",i);
        end
        $display("Nodule continuous network RTL: PASS (15 commands, 11 nodes, seven convolutions, shared DDR producer-consumer, five signed head channels, AXI stalls and final-B checks)");
        $finish;
    end
    initial begin #500000000;$fatal(1,"Nodule RTL simulation timeout");end
endmodule
'''


def prepare(out):
    out=Path(out).resolve();folder,manifest,plan,params=load_package(out/'nodule_board')
    sim=out/'rtl_sim'
    if sim.exists():raise FileExistsError('Keep previous simulation; use a new output/simulation folder')
    sim.mkdir();vectors=sim/'vectors';vectors.mkdir()
    with np.load(out/'simulation_trace.npz',allow_pickle=False) as z:trace={n:z[n] for n in z.files}
    addresses={};offset=512;gold=np.full(100000,0xdeadbeefcafef00d,np.uint64)
    initial=gold.copy();written=np.zeros(100000,int);expected=written.copy()
    for name in ['input']+[n['name'] for n in plan['nodes']]:
        zero=0 if name=='input' else next(n['zero'] for n in plan['nodes'] if n['name']==name)
        words=feature_words(trace[name],zero);addresses[name]=(offset,len(words))
        gold[offset:offset+len(words)]=words;expected[offset:offset+len(words)]=1
        if name=='input':initial[offset:offset+len(words)]=words;written[offset:offset+len(words)]=1
        offset+=len(words)+16
    commands=[];weights=[];bias=[];mult=[];shift=[]
    for node in plan['nodes']:
        a=node['sources'][0];ca,h,w=trace[a].shape
        dst,dwords=addresses[node['name']];src,awords=addresses[a]
        if node['kind']=='conv':
            row=node['layer'];arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
            cout=node['shape'][0]
            for block in range((cout+7)//8):
                p=prepare_block(arrays,block,row);wb=len(weights);pb=len(bias)
                weights.extend(p['packets']);bias.extend(p['bias']);mult.extend(p['mult']);shift.extend(p['shift'])
                cmd=[0,src,dst,0,ca,h,w,cout,block,row['kernel'],row['padding'],row['input_zero'],row['output_zero'],int(row['relu']),
                    wb,len(p['packets']),pb,0,0,0,1,0,1,0,dwords,awords,0,p['keep']]
                commands.extend(cmd)
        else:
            q=node['quant'];b=node['sources'][1] if node['kind']=='fused_concat' else None
            baddr,bwords=addresses[b] if b else (0,0);cb=trace[b].shape[0] if b else 0
            commands.extend([1,src,dst,baddr,ca,h,w,cb,3 if b else 0,0,0,0,0,0,0,0,0,
                q['za'],q.get('zb',0),q['zo'],q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0),dwords,awords,bwords,0])
    mem(vectors/'header.mem',[len(commands)//28,offset,len(weights),len(bias),1],32)
    for name,a,bits in [('commands',commands,32),('initial',initial[:offset],64),('gold',gold[:offset],64),
        ('initial_written',written[:offset],1),('expected_written',expected[:offset],1),('weights',weights,512),
        ('bias',bias,32),('mult',mult,32),('shift',shift,32)]:mem(vectors/(name+'.mem'),a,bits)
    (sim/'nodule_sim_engine.sv').write_text(wrapper(),encoding='utf-8')
    (sim/'tb_nodule_network.sv').write_text(TOP,encoding='utf-8')
    (sim/'run.tcl').write_text('run all\nexit\n',encoding='ascii')
    sources=list((folder/'rtl').glob('*.v'))+[ROOT/'fpga/tb_lung_axil_control.sv',ROOT/'fpga/tb_lung_burst_write_memory.sv',
        sim/'nodule_sim_engine.sv',sim/'tb_nodule_network.sv']
    # Compile only actual dependency modules; unrelated standalone tb modules are not tops.
    return sim,sources


def verify(sim,sources,log_name='rtl_run.log',elapsed=None):
    sim=Path(sim)
    log=(sim/log_name).read_text(encoding='utf-8')
    if 'Nodule continuous network RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):
        raise RuntimeError('RTL mismatch: '+str(sim/log_name))
    rows=[dict(command=int(a),kind=int(b),cycles=int(c)) for a,b,c in re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)]
    if len(rows)!=15 or [r['command'] for r in rows]!=list(range(15)):raise ValueError('Incomplete nodule RTL coverage')
    proof=dict(status='actual_xsim_full_nodule_graph_pass',board_tested=False,
        commands=rows,elapsed_sec=elapsed,log_name=log_name,log_sha256=sha(sim/log_name),
        source_sha256={str(p):sha(p) for p in sources},vectors_sha256={p.name:sha(p) for p in (sim/'vectors').iterdir()},
        scope='One real training patch, all 11 nodes, shared DDR output->input, actual AXI-Lite drivers, not board FPS or clinical validation')
    path=sim/'verification.json'
    if path.exists():raise FileExistsError('Preserve existing proof')
    path.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('Full shared-memory nodule RTL PASS',flush=True)


def simulate(out,bin):
    sim,sources=prepare(out);bin=Path(bin);tick=time.perf_counter()
    calls=[([str(bin/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(bin/'xelab.bat'),'tb_nodule_network','-s','nodule_snapshot','--debug','typical'],'elaborate'),
        ([str(bin/'xsim.bat'),'nodule_snapshot','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]
    for cmd,name in calls:
        print('Nodule actual XSim:',name,flush=True)
        with (sim/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(cmd,cwd=sim,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('XSim failed: '+str(sim/(name+'.log')))
    verify(sim,sources,elapsed=time.perf_counter()-tick)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();simulate(a.out,a.vivado_bin)
