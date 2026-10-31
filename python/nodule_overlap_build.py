"""Isolated two-credit pixel/result pipeline, unchanged integer arithmetic."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import sha
from nodule_prefetch_build import replace


def mac_source(t):
    t=replace(t,'// A group is not overlapped with the next group until its LAST output drains.',
        '// Two reserved result slots overlap next-pixel MAC with prior quantization/output.\n// Slot ownership lasts from the first input through the final output handshake.')
    t=replace(t,'assign s_ready = rst_n && !flushing && !emitting;',
        'reg [1:0] inflight;\n    assign s_ready = rst_n && !flushing && (input_active || inflight<2);')
    t=replace(t,'    reg signed [31:0] held_acc[0:7];\n    reg held_error[0:7];', '''    reg signed [31:0] result_acc[0:15];
    reg result_error[0:15];
    reg [1:0] result_ready;
    reg write_slot, send_slot, r1_slot, r2_slot, r3_slot, r4_slot;
    reg [255:0] result_biases[0:1],result_multipliers[0:1];
    reg [47:0] result_shifts[0:1];
    reg [7:0] result_keep[0:1],result_zero[0:1];
    reg result_relu[0:1],result_group_error[0:1];
    reg [63:0] result_partial_errors[0:1];''')
    t=replace(t,'wire signed [31:0] bias = biases[32*o+:32];',
        'wire signed [31:0] bias = result_biases[r3_slot][32*o+:32];')
    t=replace(t,'totals[o]<=0;held_acc[o]<=0;held_error[o]<=0;',
        'totals[o]<=0;result_acc[o]<=0;result_acc[8+o]<=0;result_error[o]<=0;result_error[8+o]<=0;')
    t=replace(t,'''                    held_acc[o]<=final_sum[31:0];
                    held_error[o]<=group_error || (|partial_errors[8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});''',
        '''                    result_acc[{r3_slot,3'b000}+o]<=final_sum[31:0];
                    result_error[{r3_slot,3'b000}+o]<=result_group_error[r3_slot] ||
                        (|result_partial_errors[r3_slot][8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});''')
    t=replace(t,'    reg [5:0] dispatch_shift;',
        '    reg [5:0] dispatch_shift;\n    reg [7:0] dispatch_zero;reg dispatch_relu;')
    t=replace(t,'.s_shift(dispatch_shift),.s_zero_point(output_zero),.s_relu(relu),',
        '.s_shift(dispatch_shift),.s_zero_point(dispatch_zero),.s_relu(dispatch_relu),')
    t=replace(t,'            dispatch_acc<=0;dispatch_multiplier<=0;dispatch_shift<=0;', '''            dispatch_acc<=0;dispatch_multiplier<=0;dispatch_shift<=0;dispatch_zero<=0;dispatch_relu<=0;
            inflight<=0;result_ready<=0;write_slot<=0;send_slot<=0;
            r1_slot<=0;r2_slot<=0;r3_slot<=0;r4_slot<=0;
            for(f=0;f<2;f=f+1)begin
                result_biases[f]<=0;result_multipliers[f]<=0;result_shifts[f]<=0;
                result_keep[f]<=0;result_zero[f]<=0;result_relu[f]<=0;
                result_group_error[f]<=0;result_partial_errors[f]<=0;
            end''')
    t=replace(t,'            if(r4_valid)begin emitting<=1;remaining<=output_keep;end', '''            // Reduction captures the old accumulator BEFORE the next group's
            // first product can update it. Parameters/errors have slot ownership.
            if(reduce_start)begin
                flushing<=0;
                r1_slot<=write_slot;write_slot<=!write_slot;
                result_biases[write_slot]<=biases;result_multipliers[write_slot]<=multipliers;
                result_shifts[write_slot]<=shifts;result_keep[write_slot]<=output_keep;
                result_zero[write_slot]<=output_zero;result_relu[write_slot]<=relu;
                result_group_error[write_slot]<=group_error;
                result_partial_errors[write_slot]<=partial_errors;
            end
            r2_slot<=r1_slot;r3_slot<=r2_slot;r4_slot<=r3_slot;
            if(r4_valid)result_ready[r4_slot]<=1;
            if(!emitting && result_ready[send_slot])begin
                emitting<=1;remaining<=result_keep[send_slot];
            end
            case({accepted&&!input_active,popped&&m_last})
                2'b10:inflight<=inflight+1'b1;
                2'b01:inflight<=inflight-1'b1;
                default:inflight<=inflight;
            endcase''')
    t=replace(t,'''                    dispatch_acc<=held_acc[selected];dispatch_multiplier<=multipliers[32*selected+:32];
                    dispatch_shift<=shifts[6*selected+:6];dispatch_lane<=selected;
                    dispatch_last<=after_selected==0;dispatch_error<=held_error[selected];''', '''                    dispatch_acc<=result_acc[{send_slot,selected}];
                    dispatch_multiplier<=result_multipliers[send_slot][32*selected+:32];
                    dispatch_shift<=result_shifts[send_slot][6*selected+:6];dispatch_lane<=selected;
                    dispatch_zero<=result_zero[send_slot];dispatch_relu<=result_relu[send_slot];
                    dispatch_last<=after_selected==0;dispatch_error<=result_error[{send_slot,selected}];''')
    t=replace(t,'            if(selected_fire)remaining<=after_selected;', '''            if(selected_fire)begin
                remaining<=after_selected;
                if(after_selected==0)begin
                    emitting<=0;result_ready[send_slot]<=0;send_slot<=!send_slot;
                end
            end''')
    t=replace(t,'                if(m_last)begin flushing<=0;emitting<=0;end','                // Output completion returns a credit; it never resets a newer input group.')
    return t


def reuse_source(t):
    t=replace(t,'reg loaded,loading,active,flushing,load_error,group_error;', '''reg loaded,loading,load_error,group_error;
    reg [2:0] pending_groups;
    reg [3:0] group_errors;
    reg [1:0] error_write,error_read;''')
    t=replace(t,'assign w_ready=rst_n && !active && !q_valid;',
        'assign w_ready=rst_n && (pending_groups==0) && !q_valid;')
    t=replace(t,' && !flushing && slot_ready;', ' && slot_ready;')
    t=replace(t,'assign m_error=tile_error || load_error || group_error;',
        'assign m_error=tile_error || load_error || group_errors[error_read];')
    t=replace(t,'loaded<=0;loading<=0;active<=0;flushing<=0;load_error<=0;group_error<=0;',
        'loaded<=0;loading<=0;load_error<=0;group_error<=0;\n            pending_groups<=0;group_errors<=0;error_write<=0;error_read<=0;')
    t=replace(t,'                active<=1;q_first<=beat_index==0;q_last<=s_last;',
        '                q_first<=beat_index==0;q_last<=s_last;')
    t=replace(t,'''                group_error<=group_error || (s_first!=(beat_index==0)) ||
                    (s_last!=(beat_index==weight_count-1'b1)) || beat_index>=weight_count;''', '''                group_error<=(beat_index==0?1'b0:group_error) || (s_first!=(beat_index==0)) ||
                    (s_last!=(beat_index==weight_count-1'b1)) || beat_index>=weight_count;''')
    t=replace(t,'                if(s_last)flushing<=1;', '''                if(s_last)begin
                    beat_index<=0;
                    group_errors[error_write]<=(beat_index==0?1'b0:group_error) ||
                        (s_first!=(beat_index==0)) || (s_last!=(beat_index==weight_count-1'b1)) || beat_index>=weight_count;
                    error_write<=error_write+1'b1;
                end''')
    t=replace(t,'''            if(m_valid && m_ready && m_last)begin
                active<=0;flushing<=0;beat_index<=0;group_error<=0;q_valid<=0;
            end''', '''            if(m_valid && m_ready && m_last)error_read<=error_read+1'b1;
            case({accepted&&(beat_index==0),m_valid&&m_ready&&m_last})
                2'b10:pending_groups<=pending_groups+1'b1;
                2'b01:pending_groups<=pending_groups-1'b1;
                default:pending_groups<=pending_groups;
            endcase''')
    return t


def window_source(t):
    t=replace(t,'    wire pop=core_valid&&core_ready;', '''    wire pop=core_valid&&core_ready;
    // Completed input groups own their coordinates until final output acceptance.
    reg [19:0] metadata[0:3];
    reg [1:0] meta_write,meta_read;
    reg [2:0] meta_count;
    wire [19:0] meta_front=metadata[meta_read];
    wire meta_push=pop&&core_last;
    wire meta_pop=m_valid&&m_ready&&m_last;
    always @(posedge clk)begin
        if(meta_push)metadata[meta_write]<={core_first?core_error:(pixel_error||core_error),core_final,
            core_first?core_y:pixel_y,core_first?core_x:pixel_x};
        if(!rst_n)begin meta_write<=0;meta_read<=0;meta_count<=0;end
        else if(command)begin meta_write<=0;meta_read<=0;meta_count<=0;end
        else begin
            if(meta_push)meta_write<=meta_write+1'b1;
            if(meta_pop)meta_read<=meta_read+1'b1;
            case({meta_push,meta_pop})
                2'b10:meta_count<=meta_count+1'b1;
                2'b01:meta_count<=meta_count-1'b1;
                default:meta_count<=meta_count;
            endcase
        end
    end''')
    t=replace(t,'wire core_valid=rst_n&&busy&&(queued!=0);',
        'wire core_valid;')
    t=replace(t,'    wire meta_pop=m_valid&&m_ready&&m_last;',
        '    wire meta_pop=m_valid&&m_ready&&m_last;\n    assign core_valid=rst_n&&busy&&(queued!=0)&&(!core_first||(meta_count<4));')
    t=replace(t,'assign m_x=pixel_x;assign m_y=pixel_y;',
        'assign m_x=meta_front[8:0];assign m_y=meta_front[17:9];')
    t=replace(t,'assign m_frame_last=pixel_final&&m_last;',
        'assign m_frame_last=meta_front[18]&&m_last;')
    t=replace(t,'assign m_error=tile_error||pixel_error;',
        'assign m_error=tile_error||meta_front[19];')
    return t


def prepare(root):
    root=Path(root).resolve()
    if root.parent!=ROOT/'data' or root.exists():raise ValueError('Use a fresh data child')
    old=ROOT/'data/nodule_prefetch_20261024_v2'
    manifest=json.loads((old/'nodule_prefetch/prefetch_manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if sha(old/'nodule_prefetch'/n)!=h:raise ValueError('Changed frozen parent '+n)
    root.mkdir();shutil.copytree(old/'rtl',root/'rtl')
    for n,fn in [('lung_mac_tile8.v',mac_source),('lung_mac_tile8_reuse.v',reuse_source),('lung_conv8_window.v',window_source)]:
        f=root/'rtl'/n;f.write_text(fn(f.read_text()))
    for n in ['ps_settings.tcl','build_flow.tcl']:shutil.copy2(old/n,root/n)
    shutil.copytree(old/'benches',root/'benches');shutil.copytree(old/'vectors',root/'vectors')
    protected={str(old/'nodule_prefetch/prefetch_manifest.json'):sha(old/'nodule_prefetch/prefetch_manifest.json')}
    for size in [128,448]:
        sim=old/('sim_'+str(size));proof=json.loads((sim/'verification.json').read_text())
        for n,h in proof['vectors_sha256'].items():
            if sha(sim/'vectors'/n)!=h:raise ValueError('Changed graph gold')
            protected[str(sim/'vectors'/n)]=h
        protected[str(sim/'verification.json')]=sha(sim/'verification.json')
    prep=dict(status='prepared_two_credit_pixel_overlap',parent=str(old),model_changed=False,
        arithmetic_changed=False,multiplier_lanes=64,result_slots=2,
        protected_sha256=protected,generated_sha256={str(p):sha(p) for p in root.rglob('*') if p.is_file()},
        generator_sha256=sha(__file__))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2));print(prep['status'],flush=True)


def checked(root):
    root=Path(root).resolve();p=json.loads((root/'preparation.json').read_text())
    for g in ['protected_sha256','generated_sha256']:
        for f,h in p[g].items():
            if sha(f)!=h:raise ValueError('Changed overlap input '+f)
    if sha(__file__)!=p['generator_sha256']:raise ValueError('Changed generator; use a fresh experiment')
    return root,p


def simulate(root,size,binary):
    root,prep=checked(root);old=Path(prep['parent'])/('sim_'+str(size));out=root/('sim_'+str(size));out.mkdir()
    shutil.copytree(old/'vectors',out/'vectors')
    for n in ['nodule_sim_engine.sv','tb_nodule_network.sv','run.tcl']:shutil.copy2(old/n,out/n)
    for n in ['tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']:shutil.copy2(ROOT/'fpga'/n,out/n)
    f=out/'tb_nodule_network.sv';t=f.read_text();marker='    integer queue_full_cycles=0;'
    t=replace(t,marker,'''    integer overlap_cycles=0;
    `define P conv.con.dut.engine.conv.conv
    always @(posedge clk)if(rst_n)begin
        if(`P.core.tile.inflight>2 || `P.meta_count>4 || `P.core.pending_groups>3)
            $fatal(1,"Pixel ownership overflow");
        if(`P.m_valid && `P.meta_count==0)$fatal(1,"Missing output coordinates");
        if(`P.core.tile.accepted && `P.core.tile.inflight!=0 && !`P.core.tile.input_active)
            overlap_cycles=overlap_cycles+1;
        if(`P.done && (`P.meta_count!=0 || `P.core.pending_groups!=0 || `P.core.tile.inflight!=0))
            $fatal(1,"Pixel ownership leaked at command completion");
    end
    integer queue_full_cycles=0;''')
    t=t.replace('        $finish;','        $display("PIXEL_OVERLAP_CYCLES %0d",overlap_cycles);\n        $finish;');f.write_text(t)
    sources=list((root/'rtl').glob('*.v'))+list(out.glob('*.sv'));binary=Path(binary);tick=time.perf_counter()
    for c,n in [([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(binary/'xelab.bat'),'tb_nodule_network','-s','overlap','--debug','typical','--mt','2'],'elaborate'),
        ([str(binary/'xsim.bat'),'overlap','-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        print('Actual pixel-overlap XSim',size,n,flush=True)
        with (out/(n+'.log')).open('x') as s:r=subprocess.run(c,cwd=out,stdout=s,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Tool failed: '+str(out/(n+'.log')))
    log=(out/'rtl_run.log').read_text()
    if 'RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):raise ValueError('Full graph/protocol not exact: '+str(out/'rtl_run.log'))
    rows=[dict(command=int(a),kind=int(b),cycles=int(c)) for a,b,c in re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)',log)]
    if len(rows)!=15 or [r['command'] for r in rows]!=list(range(15)):raise ValueError('Missing commands')
    baseline=json.loads((old/'verification.json').read_text())['commands']
    gains=[dict(command=r['command'],kind=r['kind'],before=baseline[i]['cycles'],after=r['cycles'],ratio=baseline[i]['cycles']/r['cycles']) for i,r in enumerate(rows)]
    result=dict(status='actual_xsim_pixel_overlap_exact_'+str(size),board_tested=False,arithmetic_changed=False,
        commands=rows,gains=gains,total_cycle_ratio=sum(x['before'] for x in gains)/sum(x['after'] for x in gains),
        overlap_events=int(re.search(r'PIXEL_OVERLAP_CYCLES (\d+)',log)[1]),elapsed_sec=time.perf_counter()-tick,
        source_sha256={str(p):sha(p) for p in sources},vectors_sha256={p.name:sha(p) for p in (out/'vectors').iterdir()},log_sha256=sha(out/'rtl_run.log'))
    (out/'verification.json').write_text(json.dumps(result,indent=2));checked(root);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--simulate',type=int,choices=[128,448]);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.simulate:simulate(a.root,a.simulate,a.vivado_bin)
    else:prepare(a.root)
