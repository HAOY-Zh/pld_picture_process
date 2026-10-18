"""Continue predeclared candidates through frozen, complete numerical confirmation.

No training, calibration, threshold search, original-model replacement or board
programming occurs here. Previously observed cohorts are development confirmation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from lung_prune_refine import PLANS, WORKSPACE
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256


def run(parent, outdir, timeout_minutes=100):
    parent = Path(parent).resolve()
    output = fresh_output(outdir).resolve()
    start = time.monotonic()
    protocol = dict(
        status='predeclared_sequential_candidate_confirmation',
        parent=str(parent), parent_protocol_sha256=file_sha256(parent/'protocol.json'),
        candidates=list(PLANS),
        decision='First predeclared candidate with all 27 exact-INT8 final Dice drops <=0.01; additional failure metrics remain reportable',
        skip='Skip full confirmation if a candidate fails the six-development-case Dice or structural-work screens',
        test_previously_observed=True, replaces_model=False, board_tested=False,
        source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)})
    write_json(output/'protocol.json', protocol)
    results = []

    def wait_for(path):
        while not path.exists():
            if time.monotonic()-start > timeout_minutes*60:
                raise TimeoutError('Parent candidate did not complete within bounded wait: '+str(path))
            time.sleep(10)

    def command(script, *args):
        argv=[sys.executable, '-u', '-X', 'utf8', str(WORKSPACE/'python'/script), *map(str,args)]
        print('RUN', subprocess.list2cmdline(argv), flush=True)
        subprocess.run(argv, cwd=WORKSPACE, check=True)

    chosen=None
    for candidate in PLANS:
        wait_for(parent/candidate/'result.json')
        development=json.loads((parent/candidate/'result.json').read_text(encoding='utf-8'))
        row=dict(candidate=candidate, development=development)
        results.append(row)
        if chosen is not None:
            row['status']='development_complete_full_confirmation_not_needed'
        elif not (development['passes_mean_and_work_screen'] and development['passes_each_case_dice_screen']):
            row['status']='development_dice_or_work_screen_failed'
        else:
            snapshot=output/(candidate+'_snapshot')
            full=output/(candidate+'_fulltest')
            command('lung_prune_refine_snapshot.py', '--parent',parent,'--candidate',candidate,'--outdir',snapshot)
            command('lung_prune_review.py','--root',snapshot,'--baseline-review',WORKSPACE/'data/lung_prune_moderate_20261017/review.json')
            command('lung_prune_freeze.py','--root',snapshot)
            command('lung_prune_refine_fulltest.py','--recovery',snapshot,'--candidate',candidate,'--outdir',full)
            command('lung_prune_fulltest_audit.py','--root',full)
            command('lung_prune_refine_fulltest_report.py','--root',full)
            report=json.loads((full/'evaluation.json').read_text(encoding='utf-8'))
            deltas=[dict(case=r['case'],dice_delta=r['scores']['pruned_fixed_final']['metrics']['dice']-
                         r['scores']['original_fixed_final']['metrics']['dice']) for r in report['cases']]
            passed=len(deltas)==27 and min(r['dice_delta'] for r in deltas)>=-.01-1e-12
            row.update(status='full_confirmation_complete',snapshot=str(snapshot),fulltest=str(full),
                       every_case_dice_passed=passed,per_case=deltas,
                       cohorts={k:v['pairs']['fixed_final'] for k,v in report['cohorts'].items()},
                       audit_sha256=file_sha256(full/'returned_mask_audit.json'))
            print('CONFIRMED',candidate,'all_case_Dice_gate',passed,'worst',min(r['dice_delta'] for r in deltas),flush=True)
            if passed:
                chosen=candidate
                review=json.loads((snapshot/'review.json').read_text(encoding='utf-8'))
                args=['--experiment',snapshot,'--outdir',output/(candidate+'_board')]
                if review['selected_for_board_experiment'] is None:
                    args.extend(['--exploratory-candidate',candidate])
                command('lung_prune_refine_package.py',*args)
                row['board_package']=str(output/(candidate+'_board'))
                row['additional_development_review_passed']=review['reviews'][0]['passes_conservative_development_review']
        write_json(output/'progress.json',dict(results=results,first_all_case_dice_pass=chosen))
    wait_for(parent/'verification.json')
    write_json(output/'verification.json',dict(
        status='complete_predeclared_pruning_confirmation', results=results,
        first_all_case_dice_pass=chosen, parent_verification_sha256=file_sha256(parent/'verification.json'),
        protocol_sha256=file_sha256(output/'protocol.json'),
        no_test_training_or_calibration=True,test_previously_observed=True,
        original_web_model_unchanged=True,board_tested=False,actual_speed_pending=True,
        wall_seconds=time.monotonic()-start))
    print('CONFIRMATION COMPLETE first all-case Dice pass:',chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',default='data/lung_prune_refine_20261017')
    p.add_argument('--outdir',default='data/lung_prune_refine_confirm_20261017')
    a=p.parse_args();run(a.parent,a.outdir)
