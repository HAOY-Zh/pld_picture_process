"""Add simultaneous cache hit response/next-window-request to pixel overlap."""
import argparse
import json
from pathlib import Path
from nodule_overlap_build import ROOT,prepare,simulate,replace
from nodule_plan import sha


def cache_source(t):
    t=replace(t,'assign s_ready=rst_n&&configured&&state==IDLE&&!cfg_valid;', '''// A consumed hit/held response frees the synchronous read port immediately.
    // Miss/refill remains single outstanding; no speculative DDR transaction.
    assign s_ready=rst_n&&configured&&!cfg_valid&&
        (state==IDLE||((direct_hit||state==RETURN)&&m_ready));''')
    t=replace(t,'''                end else if(requested)begin
                    request<=s_addr;set_index<=s_addr[AB-1:4];tag<=s_addr[31:AB];
                    if(s_addr<base||{1'b0,s_addr}>=frame_end)begin held_data<=0;held_error<=1;state<=RETURN;end
                    else state<=CHECK;
                end''','''                end''')
    t=replace(t,'''            default:state<=IDLE;
            endcase''','''            default:state<=IDLE;
            endcase
            // Last assignment wins only on an actual next-request handshake.
            // At this edge the old response has already been accepted.
            if(requested)begin
                request<=s_addr;set_index<=s_addr[AB-1:4];tag<=s_addr[31:AB];
                if(s_addr<base||{1'b0,s_addr}>=frame_end)begin held_data<=0;held_error<=1;state<=RETURN;end
                else state<=CHECK;
            end''')
    return t


def feature_source(t):
    t=replace(t,'// padding is checked before issuing, and at most one request is outstanding.',
        '// padding is checked before issuing. A consumed response may issue the next\n// request at the same edge; at most one unconsumed response is outstanding.')
    t=replace(t,'    assign req_valid=rst_n&&state==ISSUE&&!outside;\n    assign req_addr=address[31:0];', '''    // The lookahead address uses exactly the same stepping/order as advance_window.
    wire next_channel=cb!=blocks-1'b1;
    wire next_column=!next_channel&&kernel3&&kx!=2;
    wire next_row=!next_channel&&!next_column&&kernel3&&ky!=2;
    wire next_pixel=!next_channel&&!next_column&&!next_row;
    wire final_word=pixel_last&&frame_last;
    wire [3:0] next_cb=next_channel?cb+1'b1:4'd0;
    wire [1:0] next_kx=next_channel?kx:next_column?kx+1'b1:2'd0;
    wire [1:0] next_ky=(next_channel||next_column)?ky:next_row?ky+1'b1:2'd0;
    wire [8:0] next_x=next_pixel?(x!=out_width-1'b1 ? x+1'b1 : 9'd0):x;
    wire [8:0] next_y=next_pixel&&x==out_width-1'b1 ? y+1'b1 : y;
    wire [33:0] next_address=(next_channel||next_column)?address+1'b1:
        next_row?kernel_row_address+row_stride:
        x!=out_width-1'b1 ? pixel_address+blocks : next_row_address;
    wire signed [10:0] next_sy=$signed({2'b0,next_y})+$signed({9'd0,next_ky})-$signed({10'd0,padding});
    wire signed [10:0] next_sx=$signed({2'b0,next_x})+$signed({9'd0,next_kx})-$signed({10'd0,padding});
    wire next_outside=next_sy<0||next_sx<0||next_sy>=$signed({1'b0,height})||next_sx>=$signed({1'b0,width});
    wire consume_window=(state==OUTPUT||(state==RESPONSE&&rsp_valid))&&m_ready;
    wire chain_request=consume_window&&!final_word&&!next_outside;
    assign req_valid=rst_n&&((state==ISSUE&&!outside)||chain_request);
    assign req_addr=state==ISSUE?address[31:0]:next_address[31:0];''')
    t=replace(t,'            default:state<=IDLE;\n            endcase', '''            default:state<=IDLE;
            endcase
            // Retain the existing holding-register path whenever downstream is
            // stalled. No geometry can advance before that word is accepted.
            if(consume_window&&!final_word)begin
                if(next_outside)begin held_data<={8{input_zero}};held_error<=0;state<=OUTPUT;end
                else if(chain_request&&req_ready)state<=RESPONSE;
            end''')
    return t


def make(root):
    prepare(root);root=Path(root).resolve()
    for n,fn in [('lung_feature_cache.v',cache_source),('lung_feature_window.v',feature_source)]:
        p=root/'rtl'/n;p.write_text(fn(p.read_text()))
    p=root/'preparation.json';d=json.loads(p.read_text())
    d['status']='prepared_pixel_overlap_and_hit_feed';d['protected_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    d['generated_sha256']={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='preparation.json'}
    p.write_text(json.dumps(d,indent=2));print(d['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--simulate',type=int,choices=[128,448]);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.simulate:simulate(a.root,a.simulate,a.vivado_bin)
    else:make(a.root)
