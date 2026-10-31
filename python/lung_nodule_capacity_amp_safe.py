"""Resume both capacity arms from their last common epoch with bounded AMP scale.

The aborted partial epoch is discarded for BOTH arms. Preserve the original v1
source, first five complete epochs and RNG states; do not change data or budget.
"""
import argparse
import copy
from pathlib import Path

from lung_candidate_data import sha_file
from lung_nodule_cnn_balanced import read, save
import lung_nodule_cnn_experiment as first
import lung_nodule_capacity as capacity

OUT = capacity.ROOT/'data/lung_nodule_capacity_20261020_amp_safe_v2'
SOURCE = capacity.OUT
SCALE = 32.


def bounded_scaler_type(torch):
    parent = torch.amp.GradScaler

    class BoundedScaler(parent):
        def __init__(self, *args, **kwargs):
            kwargs['init_scale'] = SCALE
            kwargs['growth_interval'] = 2**31-1
            super().__init__(*args, **kwargs)

        def load_state_dict(self, state):
            state = dict(state, scale=SCALE, growth_interval=2**31-1, _growth_tracker=0)
            super().load_state_dict(state)

    return BoundedScaler


def freeze(out):
    if out.exists():
        raise FileExistsError('Resume existing v2 instead')
    old = first.verify(SOURCE)
    import torch
    state = torch.load(SOURCE/'paired_resume.pt',map_location='cpu',weights_only=False)
    if state['plan_sha256'] != sha_file(SOURCE/'plan.json') or len(state['history']) != 5:
        raise ValueError('Expected five jointly completed epochs')
    if any(any(h['skipped_updates'].values()) for h in state['history']):
        raise ValueError('Cannot reuse unmatched updates')
    plan = copy.deepcopy(old)
    sources = [Path(__file__).resolve(),capacity.ROOT/'python/tests/test_lung_nodule_capacity_amp_safe.py',
        SOURCE/'plan.json',SOURCE/'plan.sha256',SOURCE/'paired_resume.pt',
        SOURCE/'initial_function_check.json',SOURCE/'training_progress.json',SOURCE/'pipeline_state.json']
    for p in sources:
        plan['protected_sha256'][str(p)] = sha_file(p)
    plan['version'] = 'paired_capacity_v2_common_epoch_resume_bounded_amp'
    plan['training'].update(initial_grad_scale=SCALE,gradient_scale_growth_interval=2**31-1,
        scale_policy='first five shared epochs retain original 1024 scale; remaining25 both use fixed32; zero skipped updates required')
    plan['resume'] = dict(source=str(SOURCE),source_resume_sha256=sha_file(SOURCE/'paired_resume.pt'),
        reused_complete_epochs=5,discarded_partial_epoch=6,
        reason='Small arm overflow would skip an update; neither arm of that partial epoch is reused',
        changes='Only shared numerical AMP scaling after common epoch5; sampling, optimizer, schedule and total30 epochs retained')
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in sources:
        if p.suffix=='.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    state['plan_sha256'] = sha_file(out/'plan.json')
    torch.save(state,out/'paired_resume.pt')
    save(out/'resume_provenance.json',plan['resume'])
    print('RESUME CAPACITY: preserve5 common epochs; remaining25 AMP scale32, no growth',flush=True)


def run(out):
    if not out.exists():
        freeze(out)
    original = capacity.setup_torch

    def setup(seed):
        torch = original(seed)
        if not getattr(torch.amp.GradScaler,'capacity_bounded',False):
            cls = bounded_scaler_type(torch)
            cls.capacity_bounded = True
            torch.amp.GradScaler = cls
        return torch

    capacity.setup_torch = setup
    capacity.run(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    run(parser.parse_args().outdir.resolve())
