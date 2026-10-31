"""Compare actual new PL heads against every saved old head, without UI reruns."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
sys.path.insert(0, str(APP/'python'))
from start_ct_studio_demo import configuration
from shared_segmentation_contract import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    state = json.loads(args.state.read_text(encoding='utf8'))
    folder = ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'/state['case_id']
    meta = json.loads((folder/(state['volume_job_id']+'_volume.json')).read_text(encoding='utf8'))
    client, _ = configuration(APP/'data/ct_studio_integration_20261025/host_config.json')
    client.health()
    arrays = {}
    rows = []
    started = time.perf_counter()
    for z, rid in sorted(meta['slice_receipts'].items(), key=lambda x: int(x[0])):
        with np.load(folder/rid/'actual_head.npz', allow_pickle=False) as a:
            image = a['input_codes']
            expected = a['head_codes']
        head, receipt = client.infer(image)
        np.testing.assert_array_equal(head, expected)
        arrays['z'+z] = head
        row = dict(slice_index=int(z), old_request_id=rid, new_receipt=receipt,
            old_head_sha256=digest(expected.tobytes()), different_head_codes=int(np.count_nonzero(head != expected)))
        rows.append(row)
        (args.out/'progress.json').write_text(json.dumps(dict(status='RUNNING', completed=len(rows), total=len(meta['slice_receipts']),
            all_completed_exact=True, elapsed_s=time.perf_counter()-started), indent=2))
        if len(rows) % 25 == 0 or len(rows) == 1:
            print('NEW PL EXACT', len(rows), '/', len(meta['slice_receipts']),
                'network_ms', round(receipt['stats']['host_total_ms'], 3), flush=True)
    np.savez_compressed(args.out/'actual_new_heads.npz', **arrays)
    times = [r['new_receipt']['stats']['host_total_ms'] for r in rows]
    steady = [r for r in rows if not r['new_receipt']['stats']['overlay_switch_ms']]
    old_times = []
    for r in steady:
        old = json.loads((folder/r['old_request_id']/'receipt.json').read_text(encoding='utf8'))
        if not old['stats']['overlay_switch_ms']:
            old_times.append(old['stats']['host_total_ms'])
    result = dict(status='PASS_ALL_SAVED_SLICES_EXACT', case_id=state['case_id'], slices=len(rows),
        input_model_unchanged=True, different_head_codes=0, reference_expert_present=state['expert_present'],
        new_network_mean_ms=float(np.mean(times)),
        new_steady_network_mean_ms=float(np.mean([r['new_receipt']['stats']['host_total_ms'] for r in steady])),
        new_steady_pl_active_mean_ms=float(np.mean([r['new_receipt']['stats']['pl_command_cycles']/100000 for r in steady])),
        old_saved_steady_network_mean_ms=float(np.mean(old_times)) if old_times else None,
        scope='Actual new PL on every saved original input versus old saved PL head; not a new GT Dice test; timings from different runs.',
        arrays_sha256=digest((args.out/'actual_new_heads.npz').read_bytes()), rows=rows)
    (args.out/'result.json').write_text(json.dumps(result, indent=2))
    (args.out/'progress.json').write_text(json.dumps(dict(status=result['status'], completed=len(rows), total=len(rows)), indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
