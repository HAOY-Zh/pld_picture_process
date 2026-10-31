"""First CNN development run; stop after internal calibration, not final test."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[1]
out=root/'data/lung_nodule_cnn_20261019_v1'
logs=out/'logs';logs.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
for stage in ('prepare','train','infer','calibrate'):
    command=[sys.executable,str(root/'python/lung_nodule_cnn_experiment.py'),stage,'--outdir',str(out)]
    with (logs/(stage+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        state=dict(status='RUNNING',stage=stage,pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat())
        (out/'pipeline_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
    if code:
        (out/'pipeline_state.json').write_text(json.dumps(dict(status='FAILED',stage=stage,exit_code=code)),encoding='utf-8')
        sys.exit(code)
(out/'pipeline_state.json').write_text(json.dumps(dict(status='COMPLETE_INTERNAL_CALIBRATION',review_12_run=False,
    heldout_scans_run=0)),encoding='utf-8')
print('FIRST CNN DEVELOPMENT RUN COMPLETE; review12+6 and heldout45 not run',flush=True)
