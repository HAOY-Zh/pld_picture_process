"""Run the targeted overlapping-group error/reload isolation RTL oracle."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from nodule_overlap_build import ROOT,checked
from nodule_plan import sha


def run(root,binary):
    root,_=checked(root);out=root/'ownership';out.mkdir();binary=Path(binary)
    top='tb_nodule_overlap_ownership';shutil.copy2(ROOT/'fpga'/(top+'.sv'),out/(top+'.sv'))
    (out/'run.tcl').write_text('run all\nexit\n')
    sources=[root/'rtl'/n for n in ['lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v']]+[out/(top+'.sv')]
    for c,n in [([str(binary/'xvlog.bat'),'--sv',*map(str,sources)],'compile'),
        ([str(binary/'xelab.bat'),top,'-s',top,'--debug','typical','--mt','2'],'elaborate'),
        ([str(binary/'xsim.bat'),top,'-tclbatch','run.tcl','-log','rtl_run.log'],'simulate')]:
        with (out/(n+'.log')).open('x') as s:r=subprocess.run(c,cwd=out,stdout=s,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Tool failed: '+str(out/(n+'.log')))
    log=(out/'rtl_run.log').read_text()
    if 'ownership RTL: PASS' not in log or re.search(r'Fatal:|ERROR:',log):raise ValueError('Ownership failed: '+str(out/'rtl_run.log'))
    d=dict(status='actual_xsim_overlap_ownership_pass',source_sha256={str(p):sha(p) for p in sources},log_sha256=sha(out/'rtl_run.log'))
    (out/'verification.json').write_text(json.dumps(d,indent=2));checked(root);print(d['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args();run(a.root,a.vivado_bin)
