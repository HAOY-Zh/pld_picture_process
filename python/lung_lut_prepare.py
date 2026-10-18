"""Exact concat LUT experiment, isolated from the board-accepted flow package."""
import argparse
import json
from pathlib import Path
import shutil

import lung_flow_prepare as flow
from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_word_prepare import replace_once
from verify_lung_axil import check_hashes

ROOT = flow.WORKSPACE / 'data/lung_lut_20261017'


def tensor_source(text):
    text = replace_once(text, 'LINE_SETUP=28;', 'LINE_SETUP=28,TABLE_PREP=29;')
    text = replace_once(text, '    reg signed[8:0]vector_centered[0:7];',
        '    reg table_loading;reg[8:0]table_index;')
    start = text.index('    wire[7:0]v_ready,v_valid,v_error;')
    end = text.index('    assign c_ready=', start)
    text = text[:start]+'''    // Each lane has an independent read port. The original scalar requantizer
    // builds BOTH complete 256-code tables for every aligned concat command.
    // No host ABI, floating approximation, or cached quantization parameters.
    wire[63:0]lookup_codes;
    genvar n;
    generate for(n=0;n<8;n=n+1)begin: lookup_lanes
        (* ram_style="distributed" *) reg[7:0]codes[0:511];
        always @(posedge clk)
            if(rst_n&&table_loading&&state==RQ_WAIT&&rq_valid&&!rq_error)
                codes[table_index]<=rq_code;
        assign lookup_codes[8*n+:8]=codes[{branch_b,vector_input[8*n+:8]}];
    end endgenerate
'''+text[end:]
    text = replace_once(text, '            for(i=0;i<8;i=i+1)vector_centered[i]<=0;',
        '            table_loading<=0;table_index<=0;')
    text = replace_once(text, '                x<=0;y<=0;pixel_index<=0;block_index<=0;state<=SHAPE;',
        '                table_loading<=0;table_index<=0;\n                x<=0;y<=0;pixel_index<=0;block_index<=0;state<=SHAPE;')
    text = replace_once(text, '                else state<=WORD_START;\n            end',
        '                else if(vector_concat)begin table_loading<=1;table_index<=0;state<=TABLE_PREP;end\n                else state<=WORD_START;\n            end')
    text = replace_once(text, '            RQ_PREP:begin', '''            TABLE_PREP:begin
                centered_reg<=$signed({1'b0,table_index[7:0]})-$signed({1'b0,(table_index[8]?zb:za)});
                multiplier_reg<=table_index[8]?mb:ma;shift_reg<=table_index[8]?sb:sa;state<=RQ_SEND;
            end
            RQ_PREP:begin''')
    text = replace_once(text, '                else begin\n                    output_word[8*lane+:8]<=rq_code;', '''                else if(table_loading)begin
                    if(table_index==511)begin table_loading<=0;state<=WORD_START;end
                    else begin table_index<=table_index+1'b1;state<=TABLE_PREP;end
                end else begin
                    output_word[8*lane+:8]<=rq_code;''')
    start = text.index('            V_PREP:begin')
    end = text.index('            WRITE_REQ:begin', start)
    text = text[:start]+'''            V_PREP:begin
                output_word<=masked(lookup_codes,total[6:0],block_index,zo);state<=WRITE_REQ;
            end
'''+text[end:]
    return text


def prepare(root):
    root = Path(root).resolve()
    if root.parent != flow.WORKSPACE/'data' or root.exists():
        raise ValueError('Use a fresh direct workspace/data child')
    old = flow.checked(flow.ROOT)
    proof = json.loads((flow.ROOT/'lung_flow_result_threeway_verification.json').read_text())
    if proof['status'] != 'verified_returned_threeway_actual_fp32_and_int8_heads' or not proof['int8_head_bit_exact']:
        raise ValueError('Need independently accepted flow board baseline')
    accepted = json.loads((flow.ROOT/'verification.json').read_text())
    check_hashes(accepted['packaged_source_sha256'])
    check_hashes(accepted['evidence_sha256'])
    root.mkdir()
    for name in ['rtl','baseline_rtl','benches','vectors']:
        shutil.copytree(flow.ROOT/('rtl' if name=='baseline_rtl' else name), root/name)
    (root/'rtl/lung_tensor_burst_engine.v').write_text(tensor_source(
        (root/'baseline_rtl/lung_tensor_burst_engine.v').read_text()), encoding='utf-8')
    for name in ['run.tcl','ps_settings.tcl','build_flow.tcl','tb_lung_tensor_burst_axil.sv']:
        shutil.copy2(flow.ROOT/name,root/name)
    # The flow harness verifies the same frozen outputs, errors and physical traffic.
    sources = [Path(__file__).resolve(),flow.WORKSPACE/'python/lung_lut_deploy.py',
               flow.WORKSPACE/'fpga/tb_lung_concat_lut.sv']
    write_json(root/'preparation.json',dict(status='prepared_exact_concat_lut',model_changed=False,
        arithmetic_changed=False,source_sha256={str(f):file_sha256(f) for f in sources},
        frozen_vectors_sha256=old['frozen_vectors_sha256'],
        protected_sources_sha256=old['protected_sources_sha256'],
        baseline_package_sha256={str(f):file_sha256(f) for f in (flow.ROOT/'lung_flow').iterdir() if f.is_file()},
        accepted_flow_proof_sha256=file_sha256(flow.ROOT/'lung_flow_result_threeway_verification.json'),
        independent_traffic={v:old['independent_traffic']['flow'] for v in ['baseline','flow']},
        generated_sha256={str(f.relative_to(root)):file_sha256(f) for f in root.rglob('*') if f.is_file()}))
    print('Prepared exact concat LUT:',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--simulate',choices=['baseline','flow'])
    a=p.parse_args()
    flow.simulate(a.root,a.simulate) if a.simulate else prepare(a.root)
