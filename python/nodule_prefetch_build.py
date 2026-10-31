"""Isolated exact-integer window FIFO experiment and actual XSim replay."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq'))
from nodule_plan import sha


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected frozen source: ' + before[:70])
    return text.replace(before, after)


def window_source(text):
    text = replace(text, '    reg loading_weights;', '''    reg loading_weights;
    wire command;
    // Bounded lookahead decouples cache/window reads from pixel-tail drains.
    // All metadata travels with its word; output coordinates advance only
    // when the resident-weight core actually consumes the corresponding word.
    // Distributed RAM has an asynchronous read; no change to MAC arithmetic.
    (* ram_style="distributed" *) reg [93:0] lookahead[0:63];
    reg [5:0] read_pointer,write_pointer;
    reg [6:0] queued;
    wire core_ready;
    wire core_valid=rst_n&&busy&&(queued!=0);
    wire [93:0] front=lookahead[read_pointer];
    wire [63:0] core_data=front[63:0];
    wire [7:0] core_keep=front[71:64];
    wire [8:0] core_x=front[80:72],core_y=front[89:81];
    wire core_first=front[90],core_last=front[91];
    wire core_final=front[92],core_error=front[93];
    // Full may insert one recovery cycle; deliberately no long combinational
    // ready path from the core through the cache hit/miss mux.
    assign win_ready=rst_n&&busy&&(queued<64);
    wire push=win_valid&&win_ready;
    wire pop=core_valid&&core_ready;
    always @(posedge clk)begin
        if(push)lookahead[write_pointer]<={win_error,win_frame_last,win_last,win_first,win_y,win_x,win_keep,win_data};
        if(!rst_n)begin queued<=0;read_pointer<=0;write_pointer<=0;end
        else if(command)begin queued<=0;read_pointer<=0;write_pointer<=0;end
        else begin
            if(push)write_pointer<=write_pointer+1'b1;
            if(pop)read_pointer<=read_pointer+1'b1;
            case({push,pop})
                2'b10:queued<=queued+1'b1;
                2'b01:queued<=queued-1'b1;
                default:queued<=queued;
            endcase
        end
    end''')
    text = replace(text, 'wire command=c_valid&&c_ready;', 'assign command=c_valid&&c_ready;')
    text = replace(text, 'assign c_ready=rst_n&&!busy&&!w_valid&&weights_loaded&&window_c_ready;',
                   'assign c_ready=rst_n&&!busy&&(queued==0)&&!w_valid&&weights_loaded&&window_c_ready;')
    text = replace(text, '.weights_loaded(weights_loaded),.s_valid(win_valid),.s_ready(win_ready),\n        .s_first(win_first),.s_last(win_last),.s_activations(win_data),.s_input_keep(win_keep),',
                   '.weights_loaded(weights_loaded),.s_valid(core_valid),.s_ready(core_ready),\n        .s_first(core_first),.s_last(core_last),.s_activations(core_data),.s_input_keep(core_keep),')
    text = replace(text, '''            if(win_valid&&win_ready)begin
                if(win_first)begin pixel_x<=win_x;pixel_y<=win_y;pixel_error<=win_error;pixel_final<=0;end
                else pixel_error<=pixel_error||win_error;
                if(win_last)pixel_final<=win_frame_last;
                error<=error||win_error;
            end''', '''            if(pop)begin
                if(core_first)begin pixel_x<=core_x;pixel_y<=core_y;pixel_error<=core_error;pixel_final<=0;end
                else pixel_error<=pixel_error||core_error;
                if(core_last)pixel_final<=core_final;
                error<=error||core_error;
            end''')
    return text


def checked(root):
    root=Path(root).resolve()
    prep=json.loads((root/'preparation.json').read_text())
    for group in ['protected_sha256','generated_sha256']:
        for p,h in prep[group].items():
            if sha(p)!=h:raise ValueError('Changed frozen prefetch input: '+p)
    return root,prep


def prepare(root):
    root=Path(root).resolve()
    if root.parent!=ROOT/'data' or root.exists():raise ValueError('Use fresh workspace/data child')
    old=ROOT/'data/nodule_direct_lut_20261023_v1'
    proof=json.loads((old/'rtl_sim/verification.json').read_text())
    if proof['status']!='actual_xsim_nodule_448_exact_lut_pass':raise ValueError('Need LUT baseline')
    protected={**proof['source_sha256'],str(old/'rtl_sim/verification.json'):sha(old/'rtl_sim/verification.json')}
    # Include every frozen reference vector used by both replay sizes.
    for replay in [ROOT/'data/nodule_board_20261023_v1/rtl_sim',old/'rtl_sim']:
        p=json.loads((replay/'verification.json').read_text())
        for n,h in p.get('vectors_sha256',{}).items():
            if sha(replay/'vectors'/n)!=h:raise ValueError('Changed original patch gold')
        for f in (replay/'vectors').iterdir():protected[str(f)]=sha(f)
    for n,h in protected.items():
        if sha(n)!=h:raise ValueError('Changed original source')
    root.mkdir();shutil.copytree(old/'rtl',root/'rtl');shutil.copytree(old/'rtl',root/'baseline_rtl')
    f=root/'rtl/lung_conv8_window.v';f.write_text(window_source(f.read_text()))
    for name in ['ps_settings.tcl','build_flow.tcl']:shutil.copy2(ROOT/'data/lung_lut_20261017'/name,root/name)
    shutil.copytree(ROOT/'data/lung_lut_20261017/benches',root/'benches')
    shutil.copytree(ROOT/'data/lung_lut_20261017/vectors',root/'vectors')
    protected[str(Path(__file__).resolve())]=sha(__file__)
    (root/'preparation.json').write_text(json.dumps(dict(status='prepared_window_fifo64_exact_math',
        functional_changes=['lung_conv8_window.v'],model_changed=False,arithmetic_changed=False,
        protected_sha256=protected,generated_sha256={str(p):sha(p) for p in root.rglob('*') if p.is_file()}),indent=2))
    print('Prepared isolated FIFO64 window prefetch RTL',flush=True)


def simulate(root,size,binary):
    root,prep=checked(root);sim=root/('sim_'+str(size));sim.mkdir()
    old=ROOT/('data/nodule_board_20261023_v1/rtl_sim' if size==128 else 'data/nodule_direct_lut_20261023_v1/rtl_sim')
    shutil.copytree(old/'vectors',sim/'vectors')
    for name in ['nodule_sim_engine.sv','tb_nodule_network.sv','run.tcl']:shutil.copy2(old/name,sim/name)
    # Observe FIFO safety independently from the DUT arithmetic/output oracle.
    f=sim/'tb_nodule_network.sv';text=f.read_text();marker='    initial begin\n'
    text=replace(text,marker,'''    integer queue_full_cycles=0;
    always @(posedge clk)if(rst_n)begin
        if(conv.con.dut.engine.conv.conv.queued>64)$fatal(1,"Prefetch FIFO overflow");
        if(conv.con.dut.engine.conv.conv.queued==64)queue_full_cycles=queue_full_cycles+1;
        if(conv.con.dut.engine.conv.conv.done && conv.con.dut.engine.conv.conv.queued!=0)
            $fatal(1,"Prefetch data left at completion");
    end
    initial begin
''')
    # marker occurs twice because timeout initial is single-line, replacement remains unique.
    text=text.replace('        $finish;', '        $display("FIFO_FULL_CYCLES %0d",queue_full_cycles);\n        $finish;')
    f.write_text(text)
    for n in ['tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']:shutil.copy2(ROOT/'fpga'/n,sim/n)
    sources=list((root/'rtl').glob('*.v'))+list(sim.glob('*.sv'));tick=time.perf_counter();binary=Path(binary)
    for cmd,name in [([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(binary/'xelab.bat'),'tb_nodule_network','-s','prefetch','--debug','typical','--mt','2'],'elaborate'),
        ([str(binary/'xsim.bat'),'prefetch','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        print('Prefetch actual XSim',size,name,flush=True)
        with (sim/(name+'.log')).open('x') as stream:r=subprocess.run(cmd,cwd=sim,stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Prefetch simulation failed: '+str(sim/(name+'.log')))
    log=(sim/'rtl_run.log').read_text()
    if 'RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Prefetch full graph not exact')
    rows=[dict(command=int(a),kind=int(b),cycles=int(c)) for a,b,c in re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)]
    if len(rows)!=15 or [r['command'] for r in rows]!=list(range(15)):raise ValueError('Incomplete graph')
    baseline=json.loads((old/'verification.json').read_text())['commands']
    before=[int(x['cycles']) if isinstance(x,dict) else int(x[2]) for x in baseline]
    gains=[dict(command=r['command'],kind=r['kind'],before=before[i],after=r['cycles'],ratio=before[i]/r['cycles']) for i,r in enumerate(rows)]
    result=dict(status='actual_xsim_prefetch_exact_'+str(size),board_tested=False,arithmetic_changed=False,
        commands=rows,gains=gains,total_cycle_ratio=sum(before)/sum(r['cycles'] for r in rows),
        fifo_full_cycles=int(re.search(r'FIFO_FULL_CYCLES (\d+)',log)[1]),elapsed_sec=time.perf_counter()-tick,
        source_sha256={str(p):sha(p) for p in sources},vectors_sha256={p.name:sha(p) for p in (sim/'vectors').iterdir()},
        log_sha256=sha(sim/'rtl_run.log'),scope='Actual all-node graph, DDR canaries, AXI stalls, final B; not board latency')
    (sim/'verification.json').write_text(json.dumps(result,indent=2));checked(root)
    print(json.dumps(dict(size=size,total_cycle_ratio=result['total_cycle_ratio'],gains=gains),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--simulate',type=int,choices=[128,448]);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.simulate:simulate(a.root,a.simulate,a.vivado_bin)
    else:prepare(a.root)
