"""Stage a SOURCE-ONLY Vivado project. Never starts Vivado or external tools."""
import argparse
import hashlib
import json
from pathlib import Path

from lung_tensor_burst_prepare import engine_source
from lung_word_prepare import replace_once

WORKSPACE=Path(__file__).resolve().parents[1]
RTL=['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v','lung_feature_window.v',
     'lung_conv8_window.v','lung_feature_cache.v','lung_axi_read64.v','lung_conv8_cached_axi.v',
     'lung_axi_write64.v','lung_output_writer.v','lung_conv8_memory_axi.v','lung_axil_control.v',
     'lung_conv8_axil.v','lung_tensor_burst_engine.v','lung_axi_burst_write64.v','lung_tensor_control.v','lung_tensor_axil.v']


def burst_top(top):
    top=replace_once(top,'lung_tensor_engine engine','lung_tensor_burst_engine engine')
    top=replace_once(top,'wire wr_valid,wr_ready,b_valid,b_ready,b_error;',
        'wire wr_valid,wr_ready,b_valid,b_ready,b_error,r_last,stream_valid,stream_ready,stream_last;wire[7:0]rd_beats,wr_beats;')
    top=replace_once(top,'.b_valid(b_valid),.b_ready(b_ready),.b_error(b_error));',
        '.b_valid(b_valid),.b_ready(b_ready),.b_error(b_error),.rd_beats(rd_beats),.r_last(r_last),\n        .wr_beats(wr_beats),.stream_valid(stream_valid),.stream_ready(stream_ready),.stream_last(stream_last));')
    top=replace_once(top,".b_addr(rd_addr),.b_beats(8'd1)",'.b_addr(rd_addr),.b_beats(rd_beats)')
    top=replace_once(top,'.q_error(r_error),.q_last()','.q_error(r_error),.q_last(r_last)')
    top=replace_once(top,'lung_axi_write64 writer','lung_axi_burst_write64 writer')
    return replace_once(top,".b_addr(wr_addr),.b_data(wr_data),.b_strb(8'hff)",
        '.b_addr(wr_addr),.b_beats(wr_beats),.s_valid(stream_valid),.s_ready(stream_ready),.s_data(wr_data),.s_last(stream_last)')


def project_script(template):
    # Remove, rather than merely disable, the whole automatic hardware-run block.
    anchor='if {$argc == 2 && [lindex $argv 1] eq "build"} {'
    if template.count(anchor)!=1: raise ValueError('Builder run gate changed')
    text=template.split(anchor)[0]
    text=text[text.index('if {$argc < 1'):]
    start=text.index('set task_root ')
    text='# SOURCE ONLY: creates/validates BD and output products; NEVER launches hardware runs.\n'+\
        'if {$argc != 1} { error "Usage: create_project.tcl PREPARED_DIR (no build argument)" }\n'+text[start:]
    text=replace_once(text,'[file normalize [file join [file dirname [info script]] ..]]',
        '[file normalize [file join [file dirname [info script]] .. ..]]')
    text=replace_once(text,'lung_tensor_engine.v lung_tensor_control.v','lung_tensor_burst_engine.v lung_axi_burst_write64.v lung_tensor_control.v')
    text=replace_once(text,'[file join $task_root fpga $f]','[file join $task_out rtl $f]')
    text=replace_once(text,'validate_bd_design','source [file join $task_out validate_bd.tcl]')
    text=replace_once(text,'[file join $task_root fpga lung_axil_config_timing.xdc]',
        '[file join $task_out constraints lung_axil_config_timing.xdc]')
    text+='\nputs "SOURCE-ONLY project ready. No synthesis, implementation or bitstream was run. Project left open."\n'
    if any(word in text for word in ['launch_runs','wait_on_run','synth_design','opt_design','place_design','route_design','write_bitstream']):
        raise ValueError('Source-only builder contains a hardware execution command')
    return text


def prepare(root,ps_settings):
    root=Path(root).resolve();ps_settings=Path(ps_settings).resolve()
    if root.parent!=WORKSPACE/'data': raise ValueError('Use a new direct child of workspace/data')
    if root.exists(): raise FileExistsError('Do not overwrite any existing project or evidence')
    def read(name): return (WORKSPACE/'fpga'/name).read_text(encoding='utf-8')
    content={n:read(n) for n in RTL if n not in ['lung_feature_window.v','lung_feature_cache.v','lung_tensor_burst_engine.v']}
    content['lung_feature_window.v']=replace_once(read('lung_feature_window_step.v'),'module lung_feature_window_step','module lung_feature_window')
    content['lung_feature_cache.v']=replace_once(read('lung_feature_cache_fast.v'),'module lung_feature_cache_fast','module lung_feature_cache')
    content['lung_tensor_burst_engine.v']=engine_source(read('lung_tensor_word_engine.v'))
    content['lung_tensor_axil.v']=burst_top(content['lung_tensor_axil.v'])
    builder=project_script(read('build_lung_network_board.tcl'))
    settings=ps_settings.read_bytes()
    root.mkdir();(root/'rtl').mkdir();(root/'constraints').mkdir()
    for n,text in content.items(): (root/'rtl'/n).write_text(text,encoding='utf-8')
    (root/'create_project.tcl').write_text(builder,encoding='utf-8')
    (root/'validate_bd.tcl').write_text(read('validate_lung_window_step.tcl'),encoding='utf-8')
    (root/'constraints/lung_axil_config_timing.xdc').write_text(read('lung_axil_config_timing.xdc'),encoding='utf-8')
    (root/'ps_settings.tcl').write_bytes(settings)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (root/'source_manifest.json').write_text(json.dumps(dict(status='source_only_not_executed_in_vivado',
        files_sha256=hashes,synthesis_run=False,implementation_run=False,bitstream_generated=False,
        warning='Static staging only. BD/elaboration/timing/physical acceptance must be run locally.'),indent=2)+'\n',encoding='utf-8')
    print('Source-only files ready:',root)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_window_step_source_20261016')
    parser.add_argument('--ps-settings',default='data/lung_tensor_burst_20261016_v3/ps_settings.tcl')
    a=parser.parse_args();prepare(a.root,a.ps_settings)
