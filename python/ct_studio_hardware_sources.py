"""Maintainer: export exact deployed RTL and portable rebuild entrypoints."""
import argparse
from pathlib import Path
import re
import shutil

from ct_studio_runtime import ROOT, read, sha, write


def export(output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve existing hardware source release')
    seg=output/'segmentation'; rtl=seg/'rtl'; rtl.mkdir(parents=True)
    original=ROOT/'data/lung_cache_return_20261016_v2'
    proof=read(original/'verification.json')
    sources={}
    for filename,digest in proof['packaged_source_sha256'].items():
        path=Path(filename)
        if sha(path)!=digest: raise ValueError('Actual imported RTL changed')
        dest=rtl/path.name
        if dest.exists() and sha(dest)!=digest: raise ValueError('Conflicting flat RTL names')
        if not dest.exists(): shutil.copy2(path,dest)
        sources[path.relative_to(ROOT).as_posix()]=digest
    for name in ('validate_lung_window_step.tcl','lung_axil_config_timing.xdc'):
        shutil.copy2(ROOT/'fpga'/name,seg/name)
    shutil.copy2(original/'ps_settings.tcl',seg/'ps_settings.tcl')
    text=(original/'build_return.tcl').read_text('utf-8')
    text=text.replace('set task_root {D:/pld_picture_process}',
        'set snapshot [file normalize [file dirname [info script]]]\nset task_root [file normalize [file join $snapshot ../../..]]')
    text=text.replace('source [file join $task_out ps_settings.tcl]',
        'source [file join $snapshot ps_settings.tcl]')
    text=text.replace('if {![file exists [file join $task_out ps_settings.tcl]] || [file exists [file join $task_out project]]}',
        'if {[file exists [file join $task_out project]]}')
    lines=text.splitlines()
    lines=[('foreach f $rtl_names { add_files -norecurse [file join $snapshot rtl $f] }'
        if line.startswith('foreach f $rtl_names { add_files') else line) for line in lines]
    text='\n'.join(lines)+'\n'
    text=text.replace('[file join $task_root fpga validate_lung_window_step.tcl]',
        '[file join $snapshot validate_lung_window_step.tcl]')
    text=text.replace('[file join $task_root fpga lung_axil_config_timing.xdc]',
        '[file join $snapshot lung_axil_config_timing.xdc]')
    if 'D:/' in text: raise ValueError('Unresolved machine path in portable rebuild')
    (seg/'build.tcl').write_text(text,encoding='utf-8')
    match=re.search(r'set rtl_names \{([^}]+)\}',text)
    if not match or any(not (rtl/n).is_file() for n in match[1].split()):
        raise ValueError('Missing segmentation build dependency')
    basic=output/'basic_shared'; (basic/'rtl').mkdir(parents=True)
    original_basic=ROOT/'data/ct_basic_shared_20261017_v2'
    text=(ROOT/'fpga/build_ct_basic_board.tcl').read_text('utf-8')
    text=text.replace('set root [file normalize [file join [file dirname [info script]] ..]]',
        'set snapshot [file normalize [file dirname [info script]]]\nset root [file normalize [file join $snapshot ../../..]]')
    text=text.replace('source [file join $out ps_settings.tcl]', 'source [file join $snapshot ps_settings.tcl]')
    text=text.replace('[file join $root fpga $f]', '[file join $snapshot rtl $f]')
    names=re.search(r'foreach f \{([^}]+)\}',text)[1].split()
    for name in names: shutil.copy2(ROOT/'fpga'/name,basic/'rtl'/name)
    shutil.copy2(original_basic/'ps_settings.tcl',basic/'ps_settings.tcl')
    (basic/'build.tcl').write_text(text,encoding='utf-8')
    # Textual engineering reports only: no image-derived vector arrays.
    for dest,src in ((seg,original),(basic,original_basic)):
        (dest/'evidence').mkdir()
        for name in ('timing_routed.rpt','utilization_routed.rpt','drc.rpt'):
            text=(src/name).read_text('utf-8',errors='replace')
            text=text.replace(str(ROOT),'<WORKSPACE>').replace(ROOT.as_posix(),'<WORKSPACE>')
            (dest/'evidence'/(name+'.txt')).write_text(text,encoding='utf-8')
    for name, src in [('shared_synthetic_simulation.txt',ROOT/'data/ct_basic_shared_sim_20261017/rtl_test.log'),
                      ('native_shape_simulation.txt',ROOT/'data/ct_basic_native_sim_20261017/rtl_test.log')]:
        text=src.read_text('utf-8',errors='replace').replace(str(ROOT),'<WORKSPACE>').replace(ROOT.as_posix(),'<WORKSPACE>')
        (basic/'evidence'/name).write_text(text,encoding='utf-8')
    write(output/'source_manifest.json',dict(schema='ct_studio_hardware_source_release_v1',
        exact_segmentation_imported_sources=sources,
        segmentation_routed_design=proof['routed_design'], segmentation_utilization=proof['utilization'],
        original_bit_rebuilt_this_delivery=False, portable_build_executed_this_delivery=False,
        note='Exact imported RTL; portable Tcl changes paths only. Runtime uses original bit bytes. Rebuilding creates a new unaccepted bit identity.',
        files_sha256={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print('Exact hardware sources exported:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
