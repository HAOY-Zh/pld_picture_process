"""Generate isolated, provenance-bound tensor burst derivatives; frozen arithmetic unchanged."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from lung_word_prepare import replace_once
from lung_rtl_vectors import hex_bank
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
BIN=Path('E:/vivado2/Vivado/2022.2/bin')


def transactions(op,h,w,ca,cb,abase,bbase,dst):
    """Independent old logical request order + two single-line read caches."""
    oh,ow=(h//2,w//2) if op==0 else ((h*2,w*2) if op in (1,3) else (h,w))
    blocks=(ca+cb+7)//8 if op>=2 else (ca+7)//8
    sizes=[h*w*((ca+7)//8),oh*ow*((cb+7)//8)]
    bases=[abase,bbase];tags=[None,None];last=None;logical=bursts=beats=0
    def read(branch,y,x,c):
        nonlocal last,logical,bursts,beats
        width=ow if branch else w;channels=cb if branch else ca
        address=bases[branch]+(y*width+x)*((channels+7)//8)+c//8
        if op>=2 and last==address: return
        last=address;logical+=1
        tag=address//16
        if tags[branch]!=tag:
            tags[branch]=tag;bursts+=1
            beats+=min(tag*16+16,bases[branch]+sizes[branch])-max(tag*16,bases[branch])
    for y in range(oh):
        for x in range(ow):
            for block in range(blocks):
                if op==0:
                    for sy,sx in [(2*y,2*x),(2*y,2*x+1),(2*y+1,2*x),(2*y+1,2*x+1)]: read(0,sy,sx,block*8)
                elif op==1: read(0,y//2,x//2,block*8)
                elif ca%8==0:
                    branch=int(block*8>=ca);c=block*8-ca if branch else block*8
                    read(branch,y//2 if op==3 and not branch else y,x//2 if op==3 and not branch else x,c)
                else:
                    for lane in range(8):
                        c=block*8+lane
                        if c>=ca+cb: continue
                        branch=int(c>=ca);c=c-ca if branch else c
                        read(branch,y//2 if op==3 and not branch else y,x//2 if op==3 and not branch else x,c)
    writes=0;remaining=oh*ow*blocks
    while remaining:
        n=min(16,512-dst%512,remaining);dst+=n;remaining-=n;writes+=1
    return logical,bursts,beats,writes


def engine_source(text):
    text=replace_once(text,'module lung_tensor_word_engine(','module lung_tensor_burst_engine(')
    text=replace_once(text,'output wire rd_valid,input wire rd_ready,output wire[31:0]rd_addr,',
        'output wire rd_valid,input wire rd_ready,output wire[31:0]rd_addr,output wire[7:0]rd_beats,')
    text=replace_once(text,'input wire r_valid,output wire r_ready,input wire[63:0]r_data,input wire r_error,',
        'input wire r_valid,output wire r_ready,input wire[63:0]r_data,input wire r_error,input wire r_last,')
    text=replace_once(text,'input wire b_valid,output wire b_ready,input wire b_error',
        'output wire[7:0]wr_beats,output wire stream_valid,input wire stream_ready,output wire stream_last,\n    input wire b_valid,output wire b_ready,input wire b_error')
    text=replace_once(text,'V_PREP=19,V_SEND=20,V_WAIT=21;',
        'V_PREP=19,V_SEND=20,V_WAIT=21,LINE_REQ=22,LINE_WAIT=23,STREAM=24,WAIT_B=25,LINE_CLIP=26,LINE_LENGTH=27,LINE_SETUP=28;')
    insert='''
    // Tiny separate A/B line caches: 2 x 16 x 64 bits, invalidated per command.
    reg[63:0]line_a[0:15],line_b[0:15],output_bank[0:15];
    reg valid_a,valid_b;reg[27:0]tag_a,tag_b;
    reg[31:0]line_base,burst_base;reg[4:0]line_length,line_index,buffer_count,buffer_limit,stream_index;
    reg line_error;reg[63:0]wanted_word,fetched_word;
    wire line_hit=branch_b?(valid_b&&tag_b==address[31:4]):(valid_a&&tag_a==address[31:4]);
    wire[32:0]aligned_lo={1'b0,address[31:4],4'd0};
    reg[32:0]branch_end,line_end,clipped_lo,clipped_hi,clipped_length;
    wire expected_line_last=line_index==line_length-1'b1;
    wire fill_error=line_error||r_error||(r_last!=expected_line_last);
    wire[9:0]page_left=10'd512-{1'b0,output_address[8:0]};
    wire[4:0]first_limit=page_left<16?page_left[4:0]:5'd16;
    wire last_output=block_index==oblocks-1'b1&&pixel_index==opixels-1'b1;
    assign rd_beats={3'd0,line_length};assign wr_beats={3'd0,buffer_count};
    assign stream_valid=rst_n&&state==STREAM;assign stream_last=stream_index==buffer_count-1'b1;
'''
    text=replace_once(text,'wire[32:0]a_end=',insert+'    wire[32:0]a_end=')
    text=replace_once(text,'wire[63:0]next_max=slot==0?r_data:maximum(pool_max,r_data);',
        'wire[63:0]next_max=slot==0?fetched_word:maximum(pool_max,fetched_word);')
    text=replace_once(text,'assign rd_valid=rst_n&&state==READ_REQ;assign rd_addr=address;',
        'assign rd_valid=rst_n&&state==LINE_REQ;assign rd_addr=line_base;')
    text=replace_once(text,'assign r_ready=rst_n&&state==READ_WAIT;','assign r_ready=rst_n&&state==LINE_WAIT;')
    text=replace_once(text,'assign wr_valid=rst_n&&state==WRITE_REQ;assign wr_addr=output_address;assign wr_data=output_word;',
        'assign wr_valid=rst_n&&state==WRITE_WAIT;assign wr_addr=burst_base;assign wr_data=output_bank[stream_index[3:0]];')
    text=replace_once(text,'assign b_ready=rst_n&&state==WRITE_WAIT;','assign b_ready=rst_n&&state==WAIT_B;')
    text=replace_once(text,'state<=IDLE;op<=0;slot<=0;',
        'valid_a<=0;valid_b<=0;tag_a<=0;tag_b<=0;line_base<=0;burst_base<=0;line_length<=0;line_index<=0;\n            branch_end<=0;line_end<=0;clipped_lo<=0;clipped_hi<=0;clipped_length<=0;\n            buffer_count<=0;buffer_limit<=0;stream_index<=0;line_error<=0;wanted_word<=0;fetched_word<=0;\n            state<=IDLE;op<=0;slot<=0;')
    text=replace_once(text,'written_words<=0;read_words<=0;cycles<=0;cache_valid<=0;',
        'written_words<=0;read_words<=0;cycles<=0;cache_valid<=0;valid_a<=0;valid_b<=0;buffer_count<=0;')
    start=text.index('            READ_REQ:if(rd_ready)state<=READ_WAIT;')
    end=text.index('            RQ_PREP:begin',start)
    text=text[:start]+'''            READ_REQ:begin
                if(line_hit)begin fetched_word<=branch_b?line_b[address[3:0]]:line_a[address[3:0]];state<=READ_WAIT;end
                else begin
                    branch_end<={1'b0,source_base}+(branch_b?bwords:awords);line_end<=aligned_lo+16;state<=LINE_CLIP;
                end
            end
            LINE_CLIP:begin
                clipped_lo<=aligned_lo<{1'b0,source_base}?{1'b0,source_base}:aligned_lo;
                clipped_hi<=line_end>branch_end?branch_end:line_end;state<=LINE_LENGTH;
            end
            LINE_LENGTH:begin clipped_length<=clipped_hi-clipped_lo;state<=LINE_SETUP;end
            LINE_SETUP:begin
                if(clipped_lo>address||clipped_hi<=address||clipped_length==0||clipped_length>16)begin error<=1;state<=FINISH;end
                else begin
                    line_base<=clipped_lo[31:0];line_length<=clipped_length[4:0];line_index<=0;line_error<=0;wanted_word<=0;
                    if(branch_b)valid_b<=0;else valid_a<=0;
                    state<=LINE_REQ;
                end
            end
            LINE_REQ:if(rd_ready)state<=LINE_WAIT;
            LINE_WAIT:if(r_valid)begin
                if(branch_b)line_b[(line_base[3:0]+line_index[3:0])&4'hf]<=r_data;
                else line_a[(line_base[3:0]+line_index[3:0])&4'hf]<=r_data;
                if(line_base+line_index==address)wanted_word<=r_data;
                line_error<=fill_error;
                if(r_last||expected_line_last)begin
                    if(fill_error)begin error<=1;state<=FINISH;end
                    else begin
                        if(branch_b)begin valid_b<=1;tag_b<=address[31:4];end else begin valid_a<=1;tag_a<=address[31:4];end
                        fetched_word<=line_base+line_index==address?r_data:wanted_word;state<=READ_WAIT;
                    end
                end else line_index<=line_index+1'b1;
            end
            READ_WAIT:begin
                // ABI read_words counts logical delivered words, NOT prefetched AXI beats.
                read_words<=read_words+1'b1;cache_valid<=1;cache_address<=address;cache_data<=fetched_word;
                if(vector_concat)begin vector_input<=fetched_word;state<=V_PREP;end
                else if(concatenate)begin input_code<=fetched_word[8*source_lane+:8];state<=RQ_PREP;end
                else if(op==0)begin
                    pool_max<=next_max;
                    if(slot==3)begin output_word<=masked(next_max,ca,block_index,zo);state<=WRITE_REQ;end
                    else begin slot<=slot+1'b1;state<=MAP_PIXEL;end
                end else begin output_word<=masked(fetched_word,ca,block_index,zo);state<=WRITE_REQ;end
            end
'''+text[end:]
    start=text.index('            WRITE_REQ:if(wr_ready)state<=WRITE_WAIT;')
    end=text.index('            ADVANCE:begin',start)
    text=text[:start]+'''            WRITE_REQ:begin
                output_bank[buffer_count[3:0]]<=output_word;
                if(buffer_count==0)begin burst_base<=output_address;buffer_limit<=first_limit;end
                buffer_count<=buffer_count+1'b1;
                if((buffer_count==0&&first_limit==1)||(buffer_count!=0&&buffer_count+1'b1==buffer_limit)||last_output)state<=WRITE_WAIT;
                else state<=ADVANCE;
            end
            WRITE_WAIT:if(wr_ready)begin stream_index<=0;state<=STREAM;end
            STREAM:if(stream_ready)begin
                if(stream_last)state<=WAIT_B;else stream_index<=stream_index+1'b1;
            end
            WAIT_B:if(b_valid)begin
                if(b_error)begin error<=1;buffer_count<=0;state<=FINISH;end
                else begin written_words<=written_words+buffer_count;buffer_count<=0;state<=ADVANCE;end
            end
'''+text[end:]
    return '// Generated from frozen word engine: unchanged Q31/RNE, mapping, masks, geometry checks.\n'+text


def prepare(root):
    root=Path(root).resolve()
    if WORKSPACE/'data' not in root.parents: raise ValueError('Use new workspace/data experiment')
    root=fresh_output(root).resolve();old=WORKSPACE/'data/lung_word_20261015';cache=WORKSPACE/'data/lung_cache_fast_20261016'
    previous=json.loads((cache/'verification.json').read_text(encoding='utf-8'))
    check_hashes(previous['source_sha256']);check_hashes(previous['evidence_sha256'])
    oldprep=json.loads((old/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(oldprep['source_sha256']);check_hashes({str(old/n):h for n,h in oldprep['generated_sha256'].items()})
    rtl=root/'rtl';rtl.mkdir()
    (rtl/'lung_tensor_burst_engine.v').write_text(engine_source((WORKSPACE/'fpga/lung_tensor_word_engine.v').read_text(encoding='utf-8')),encoding='utf-8')
    top=(old/'rtl/word/lung_tensor_axil.v').read_text(encoding='utf-8')
    top=replace_once(top,'lung_tensor_word_engine engine','lung_tensor_burst_engine engine')
    top=replace_once(top,'wire wr_valid,wr_ready,b_valid,b_ready,b_error;',
        'wire wr_valid,wr_ready,b_valid,b_ready,b_error,r_last,stream_valid,stream_ready,stream_last;wire[7:0]rd_beats,wr_beats;')
    top=replace_once(top,'.b_valid(b_valid),.b_ready(b_ready),.b_error(b_error));',
        '.b_valid(b_valid),.b_ready(b_ready),.b_error(b_error),.rd_beats(rd_beats),.r_last(r_last),\n        .wr_beats(wr_beats),.stream_valid(stream_valid),.stream_ready(stream_ready),.stream_last(stream_last));')
    top=replace_once(top,'.b_addr(rd_addr),.b_beats(8\'d1)', '.b_addr(rd_addr),.b_beats(rd_beats)')
    top=replace_once(top,'.q_error(r_error),.q_last()', '.q_error(r_error),.q_last(r_last)')
    top=replace_once(top,'lung_axi_write64 writer','lung_axi_burst_write64 writer')
    top=replace_once(top,'.b_addr(wr_addr),.b_data(wr_data),.b_strb(8\'hff)',
        '.b_addr(wr_addr),.b_beats(wr_beats),.s_valid(stream_valid),.s_ready(stream_ready),.s_data(wr_data),.s_last(stream_last)')
    (rtl/'lung_tensor_axil.v').write_text(top,encoding='utf-8')
    vectors=root/'vectors';vectors.mkdir()
    for p in (old/'vectors').iterdir(): shutil.copy2(p,vectors/p.name)
    values=[int(s,16) for s in (vectors/'commands.mem').read_text().split()];stats=[]
    shifts=[0,1,509,511]
    for f in range(35):
        row=values[f*18:f*18+18]
        stats.append(transactions(*row[:5],4096+shifts[f%4],1048576+shifts[(f+1)%4],2097152+shifts[(f+2)%4]))
    if sum(s[0] for s in stats)!=382152: raise ValueError('Independent logical traffic differs from frozen original RTL')
    for i,name in enumerate(['logical_reads','read_bursts','physical_reads','write_bursts']): hex_bank(vectors/(name+'.mem'),[s[i] for s in stats],32)
    bench=(old/'tb_lung_tensor_word_axil.sv').read_text(encoding='utf-8')
    bench=replace_once(bench,'module tb_lung_tensor_word_axil;','module tb_lung_tensor_burst_axil;')
    bench=replace_once(bench,'lung_test_write_memory mem(.*);','lung_test_burst_memory mem(.*);')
    bench=bench.replace('4096','sim_a_base').replace('1048576','sim_b_base').replace('2097152','sim_dst_base')
    bench=replace_once(bench,'reg[31:0]status,value;', '''reg[31:0]status,value;
    integer sim_a_base=4096,sim_b_base=1048576,sim_dst_base=2097152,prior_read_beats;
    reg[31:0]logical_reads[0:34],expected_bursts[0:34],physical_reads[0:34],write_bursts[0:34];
    function integer frame_shift(input integer i);
        case(i%4)0:frame_shift=0;1:frame_shift=1;2:frame_shift=509;3:frame_shift=511;endcase
    endfunction''')
    bench=replace_once(bench,'checking=1;\n        for(f=0;', '''$readmemh("vectors/logical_reads.mem",logical_reads,0,34);
        $readmemh("vectors/read_bursts.mem",expected_bursts,0,34);
        $readmemh("vectors/physical_reads.mem",physical_reads,0,34);
        $readmemh("vectors/write_bursts.mem",write_bursts,0,34);
        checking=1;
        for(f=0;''')
    bench=replace_once(bench,'t=f*18;a_offset=',
        'sim_a_base=4096+frame_shift(f);sim_b_base=1048576+frame_shift(f+1);sim_dst_base=2097152+frame_shift(f+2);\n            t=f*18;a_offset=')
    bench=replace_once(bench,'arlen!=0||arlock', 'arlen>15||({1\'b0,araddr[11:0]}+(arlen+1)*8)>4096||arlock')
    bench=replace_once(bench,'remaining<=read_fault==3?2:1;',
        'remaining<=read_fault==3?arlen+2:(read_fault==4?1:arlen+1);')
    bench=replace_once(bench,'else begin remaining<=remaining-1;latency<=5;end',
        'else begin remaining<=remaining-1;if(!(read_mode==3&&remaining==2))read_address<=read_address+1;latency<=cycle%3;end')
    bench=replace_once(bench,'prior_ar=bursts;prior_acks=acks;\n            for(a=',
        'prior_ar=bursts;prior_acks=acks;prior_read_beats=read_beats;\n            for(a=')
    bench=replace_once(bench,'acks-prior_acks!=frame_pixels','acks-prior_acks!=write_bursts[f]')
    bench=replace_once(bench,'if(value!==bursts-prior_ar)$fatal(1,"Tensor read counter mismatch");',
        'if(value!==logical_reads[f]||bursts-prior_ar!=expected_bursts[f]||read_beats-prior_read_beats!=physical_reads[f])$fatal(1,"Independent tensor logical/physical read traffic mismatch frame=%0d logical=%0d/%0d bursts=%0d/%0d beats=%0d/%0d",f,value,logical_reads[f],bursts-prior_ar,expected_bursts[f],read_beats-prior_read_beats,physical_reads[f]);')
    bench=replace_once(bench,'if(writes!=header[3]||acks!=writes)', 'if(writes!=header[3])')
    bench=bench.replace('WORD_CYCLES','BURST_CYCLES')
    # Both pool and vector paths now also test early LAST; length>1 is needed.
    bench=bench.replace('p<=3;p=p+1','p<=4;p=p+1')
    (root/'tb_lung_tensor_burst_axil.sv').write_text(bench,encoding='utf-8')
    (root/'run.tcl').write_text('run all\nexit\n',encoding='utf-8')
    builder=(cache/'build_cache.tcl').read_text(encoding='utf-8')
    builder=replace_once(builder,'lung_tensor_word_engine.v lung_tensor_control.v','lung_tensor_burst_engine.v lung_axi_burst_write64.v lung_tensor_control.v')
    builder=replace_once(builder,'"'+(old/'rtl/word/lung_tensor_axil.v').as_posix()+'"','"'+(rtl/'lung_tensor_axil.v').as_posix()+'"')
    anchor='[file join $task_root fpga $f]'
    choice='[expr {$f eq "lung_tensor_burst_engine.v" ? "'+(rtl/'lung_tensor_burst_engine.v').as_posix()+'" : [file join $task_root fpga $f]}]'
    (root/'build_burst.tcl').write_text(replace_once(builder,anchor,choice),encoding='utf-8')
    synth=(WORKSPACE/'fpga/synth_lung_tensor.tcl').read_text(encoding='utf-8')
    synth=replace_once(synth,'set task_root [file normalize [file join [file dirname [info script]] ..]]','set task_root {'+WORKSPACE.as_posix()+'}')
    synth=replace_once(synth,'lung_tensor_engine.v','lung_tensor_burst_engine.v lung_axi_burst_write64.v')
    select='[expr {$f eq "lung_tensor_axil.v" ? "'+(rtl/'lung_tensor_axil.v').as_posix()+'" : ($f eq "lung_tensor_burst_engine.v" ? "'+(rtl/'lung_tensor_burst_engine.v').as_posix()+'" : [file join $task_root fpga $f])}]'
    (root/'synth_burst.tcl').write_text(replace_once(synth,anchor,select),encoding='utf-8')
    shutil.copy2(cache/'ps_settings.tcl',root/'ps_settings.tcl')
    sources=[Path(__file__).resolve()]+[WORKSPACE/'fpga'/n for n in ['lung_tensor_word_engine.v','lung_axi_burst_write64.v',
        'tb_lung_burst_write_memory.sv','lung_axi_read64.v','lung_requant.v','lung_tensor_control.v','tb_lung_axil_control.sv']]
    write_json(root/'preparation.json',dict(status='prepared_tensor_burst_experiment',cases=oldprep['cases'],
        independent_traffic=stats,previous_cache_proof_sha256=file_sha256(cache/'verification.json'),
        source_sha256={str(p):file_sha256(p) for p in sources},
        generated_sha256={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()},
        old_word_vectors_sha256={p.name:file_sha256(p) for p in (old/'vectors').iterdir()},
        warning='Generated derivatives only; no RTL/route/board proof until actual tools pass. Logical READS ABI preserved; physical reads separately checked.'))
    print('Prepared burst experiment:',root,flush=True)


def simulate(root):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    folder=root/'sim_burst'
    if folder.exists(): raise FileExistsError('Preserve previous attempt')
    folder.mkdir();shutil.copytree(root/'vectors',folder/'vectors');shutil.copy2(root/'run.tcl',folder/'run.tcl')
    files=[WORKSPACE/'fpga'/n for n in ['lung_requant.v','lung_axi_read64.v','lung_axi_burst_write64.v','lung_tensor_control.v',
        'tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']]
    files += [root/'rtl/lung_tensor_burst_engine.v',root/'rtl/lung_tensor_axil.v',root/'tb_lung_tensor_burst_axil.sv']
    commands=[[str(BIN/'xvlog.bat'),'--sv',*[str(p) for p in files]],
        [str(BIN/'xelab.bat'),'tb_lung_tensor_burst_axil','-s','burst_snapshot','--debug','typical'],
        [str(BIN/'xsim.bat'),'burst_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    start=time.perf_counter()
    for i,command in enumerate(commands):
        print('burst: '+command[0],flush=True)
        with (folder/('tool_'+str(i)+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('RTL tool failed: '+str(i))
    text=(folder/'rtl_run.log').read_text(encoding='utf-8')
    if 'Lung tensor AXI-Lite RTL: PASS (35 frames, 379310 words,' not in text or '26 error cases;' not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',text,re.I):
        raise ValueError('Incomplete/failing tensor burst RTL')
    rows=[dict(zip(['frame','op','ca','cycles','words'],map(int,m))) for m in re.findall(
        r'BURST_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)',text)]
    if len(rows)!=35 or [r['frame'] for r in rows]!=list(range(35)): raise ValueError('Missing cycle records')
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_tensor_burst',results=rows,
        frames=35,output_words=379310,error_cases=26,independent_traffic=prep['independent_traffic'],
        wall_seconds=time.perf_counter()-start,log_sha256=file_sha256(folder/'rtl_run.log'),
        source_sha256=prep['source_sha256'],scope='Actual XSim; not real-board latency'))
    print('Burst tensor RTL PASS: 35 frozen frames, 379310 words, 26 fault cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_tensor_burst_20261016')
    parser.add_argument('--simulate',action='store_true')
    args=parser.parse_args();simulate(args.root) if args.simulate else prepare(args.root)
