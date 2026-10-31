"""Continue the frozen expansion after already-running training masks finish."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from lung_candidate_expand import verify
from lung_candidate_data import write_json


def run(out):
    out=out.resolve();plan=verify(out);logs=out/'logs';logs.mkdir(exist_ok=True)
    state=out/'pipeline_state.json'
    while not all((out/'cases'/uid/'mask.json').exists() for uid in plan['added_training_48']):
        completed=sum((out/'cases'/uid/'mask.json').exists() for uid in plan['added_training_48'])
        write_json(state,dict(status='WAITING_FOR_RUNNING_MASK_PHASE',completed=completed,total=48))
        time.sleep(5)
    # The mask command finishes its final identity check and releases CUDA first.
    time.sleep(5)
    stages=[('features_training',['features','--cohort','training']),('training',['train']),
        ('masks_validation',['masks','--cohort','validation']),('features_validation',['features','--cohort','validation']),
        ('evaluation',['evaluate']),('audit',['audit'])]
    for name,args in stages:
        write_json(state,dict(status='RUNNING',stage=name))
        command=[sys.executable,str(Path(__file__).with_name('lung_candidate_expand.py')),*args,'--outdir',str(out)]
        with (logs/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
            write_json(state,dict(status='RUNNING',stage=name,pid=process.pid))
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        if code:
            write_json(state,dict(status='FAILED',stage=name,returncode=code))
            raise RuntimeError(f'{name} failed with code {code}; preserve completed caches')
    write_json(state,dict(status='COMPLETE',development_scans=90,heldout_scans_run=0))
    print('ALL DEVELOPMENT STAGES COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir)
