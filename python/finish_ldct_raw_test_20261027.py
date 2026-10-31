"""Reject a slower transport candidate and normally restore its accepted owner."""
import json
import shlex
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from deploy_ldct_raw_candidate_20261027 import connect, run, NEW, OLD, UNIT, OLD_UNIT, PACKAGE, OUT


def main():
    result = json.loads((OUT / 'physical_pilot/result.json').read_text())
    if result['status'] != 'PHYSICAL_PAIRED_OUTPUTS_EXACT' or result['latency_reduction_percent'] >= 0:
        raise ValueError('This recovery is only for the measured slower candidate')
    sys.path.insert(0, str(ROOT / 'data/CT_Studio_ExactSpeed_20261027_04/python'))
    from start_ct_studio_demo import read_web_info, configuration, web_info, connect_web
    web = read_web_info('http://127.0.0.1:8099/')
    if web and (web[2]['phase'] in ('loading', 'connecting', 'processing', 'computing', 'reconstructing')
                or web[2].get('ldct', {}).get('running')
                or (web[2].get('roi', {}).get('job') or {}).get('running')
                or (web[2].get('nodule', {}).get('job') or {}).get('status') in ('RUNNING', 'PENDING')):
        raise RuntimeError('Preserve current web task')
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token = (PACKAGE / 'platform_token.txt').read_text().strip()
    health = json.load(op.open(urllib.request.Request(
        'http://192.168.2.99:8766/v1/shared/platform',
        headers={'Authorization': 'Bearer ' + token}), timeout=10))
    expected = json.loads((OUT / 'raw_candidate.json').read_text())['service_bundle_sha256']
    if health.get('service_bundle_sha256') != expected or not health.get('ready') or health.get('busy') or health.get('error'):
        raise RuntimeError('Candidate must be matched, healthy and idle; no forced reset')
    c = connect()
    try:
        run(c, 'systemctl stop ' + UNIT, sudo=True)
        stopped = run(c, 'systemctl show ' + UNIT + ' -p MainPID -p SubState')
        if 'MainPID=0' not in stopped or 'SubState=dead' not in stopped:
            raise RuntimeError('Candidate did not normally exit')
        run(c, 'systemctl start ' + OLD_UNIT, sudo=True)
        print('RESTORING_ACCEPTED_OWNER_ALL_ORIGINAL_SELFTESTS', flush=True)
        ready = False
        for _ in range(12):
            script = ('import pathlib,json;p=pathlib.Path(' + repr(OLD) + ")/'owner_state.json';"
                      "s=json.loads(p.read_text());print(json.dumps({k:s.get(k)for k in ['phase','error','elapsed_seconds','pid']}))")
            value = json.loads(run(c, '/usr/local/share/pynq-venv/bin/python3 -c ' + shlex.quote(script), sudo=True))
            print(json.dumps(value), flush=True)
            if value.get('error'):
                raise RuntimeError('Owner reported fault; leave state intact')
            if value.get('phase') == 'ready':
                ready = True
                break
            time.sleep(18)
        if not ready:
            raise TimeoutError('Preserve owner while selftests continue')
        config = ROOT / 'data/CT_Studio_ExactSpeed_20261027_04/data/ct_studio_integration_20261025/host_config.json'
        client, manifest = configuration(config)
        client.health()
        old_health = client.call('/v1/shared/platform', timeout=10)
        if not old_health.get('ready') or old_health.get('busy') or old_health.get('error'):
            raise RuntimeError('Accepted owner not idle/healthy')
        current = web_info('http://127.0.0.1:8099/', manifest)
        if current:
            connect_web('http://127.0.0.1:8099/', current, manifest)
            current = web_info('http://127.0.0.1:8099/', manifest)[2]
        record = dict(status='CANDIDATE_REJECTED_NO_SPEEDUP_ACCEPTED_OWNER_RESTORED',
                      baseline_mean_ms=result['baseline_mean_ms'], candidate_mean_ms=result['candidate_mean_ms'],
                      latency_reduction_percent=result['latency_reduction_percent'],
                      compared_outputs=60, unique_slices=10, all_output_sha256_exact=True,
                      full_case_not_run_reason='No speed improvement in three alternating pilot repetitions',
                      service_bundle_sha256=manifest['service_bundle_sha256'],
                      board_ready=True, board_idle=True, web_connected=current.get('board_connected') if current else None,
                      boot_default=run(c, 'systemctl show ' + OLD_UNIT + ' -p UnitFileState --value').strip(),
                      candidate_boot=run(c, 'systemctl show ' + UNIT + ' -p UnitFileState --value').strip())
        (OUT / 'physical_acceptance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        c.close()


if __name__ == '__main__':
    main()
