"""Run the PREVIOUS word engine against the SAME burst-capable memory BFM."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from lung_tensor_burst_prepare import WORKSPACE,BIN
from lung_rtl_vectors import hex_bank
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes


def run(root):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(prep['source_sha256']);check_hashes({str(root/n):h for n,h in prep['generated_sha256'].items()})
    folder=root/'sim_baseline'
    if folder.exists(): raise FileExistsError('Preserve prior baseline attempt')
    folder.mkdir();shutil.copytree(root/'vectors',folder/'vectors');shutil.copy2(root/'run.tcl',folder/'run.tcl')
    traffic=prep['independent_traffic']
    for name in ['read_bursts','physical_reads']: hex_bank(folder/'vectors'/(name+'.mem'),[t[0] for t in traffic],32)
    hex_bank(folder/'vectors/write_bursts.mem',[c['words'] for c in prep['cases']],32)
    text=(root/'tb_lung_tensor_burst_axil.sv').read_text(encoding='utf-8').replace('module tb_lung_tensor_burst_axil;','module tb_lung_tensor_burst_baseline;')
    # A single-beat read cannot have early LAST; retain the original three
    # RRESP/RID/late-LAST faults, all other normal/fault checks unchanged.
    text=text.replace('p<=4;p=p+1','p<=3;p=p+1')
    bench=folder/'tb_baseline.sv';bench.write_text(text,encoding='utf-8')
    files=[WORKSPACE/'fpga'/n for n in ['lung_requant.v','lung_axi_read64.v','lung_axi_write64.v','lung_tensor_word_engine.v',
        'lung_tensor_control.v','tb_lung_axil_control.sv','tb_lung_burst_write_memory.sv']]
    files += [WORKSPACE/'data/lung_word_20261015/rtl/word/lung_tensor_axil.v',bench]
    commands=[[str(BIN/'xvlog.bat'),'--sv',*[str(p) for p in files]],
        [str(BIN/'xelab.bat'),'tb_lung_tensor_burst_baseline','-s','baseline_snapshot','--debug','typical'],
        [str(BIN/'xsim.bat'),'baseline_snapshot','-tclbatch','run.tcl','-log','rtl_run.log']]
    for i,command in enumerate(commands):
        with (folder/('tool_'+str(i)+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('Matched baseline tool failed')
    log=(folder/'rtl_run.log').read_text(encoding='utf-8')
    rows=[dict(zip(['frame','op','ca','cycles','words'],map(int,m))) for m in re.findall(
        r'BURST_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)',log)]
    if len(rows)!=35 or '24 error cases;' not in log or 'Lung tensor AXI-Lite RTL: PASS (35 frames, 379310 words,' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',log,re.I):
        raise ValueError('Incomplete/failing matched baseline')
    write_json(folder/'verification.json',dict(status='passed_actual_xsim_word_same_burst_bfm',results=rows,
        same_burst_bfm=True,frames=35,output_words=379310,error_cases=24,
        log_sha256=file_sha256(folder/'rtl_run.log'),source_sha256={str(p):file_sha256(p) for p in files+[Path(__file__).resolve()]}))
    print('Previous word engine with same burst-capable BFM: actual RTL PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_tensor_burst_20261016_v3')
    run(parser.parse_args().root)
