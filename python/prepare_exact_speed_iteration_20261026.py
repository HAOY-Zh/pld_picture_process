"""Prepare a separate PC-only candidate; existing board and app stay untouched."""
import ast
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/CT_Studio_DualExact_20261026_02'
DST=ROOT/'data/CT_Studio_ExactSpeed_20261026_03'


def once(s,a,b):
    if s.count(a)!=1:raise ValueError('Unexpected source anchor: '+a[:80])
    return s.replace(a,b)


def main():
    if DST.exists():raise FileExistsError('Preserve existing candidate')
    shutil.copytree(SRC,DST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'python/nodule_exact_host.py',DST/'python/nodule_exact_host.py')
    # Wire the already-accepted binary segmentation client into the real entry.
    p=DST/'python/ct_workspace_web.py';s=p.read_text(encoding='utf8')
    s=once(s,'from ct_shared_seg_client import SharedSegStudioClient',
        'from ct_dual_io_client import DualSegStudioClient as SharedSegStudioClient')
    s=once(s,"            value['basic'] = copy.deepcopy(self.basic_result)",
        "            value['host_speed_profile']='exact_host_r1'\n            value['basic'] = copy.deepcopy(self.basic_result)")
    p.write_text(s,encoding='utf8')
    p=DST/'python/start_ct_studio_demo.py';s=p.read_text(encoding='utf8')
    s=once(s,'    opener,headers,state=info\n    if (',
        "    opener,headers,state=info\n    if state.get('host_speed_profile')!='exact_host_r1':\n        raise ValueError('Port belongs to another host version; preserve it and select a new port')\n    if (")
    p.write_text(s,encoding='utf8')
    # Same frozen nodule envelope and sampling, with bounded case-local reuse.
    p=DST/'python/nodule_detector.py';s=p.read_text(encoding='utf8')
    s=once(s,'from nodule_canvas_contract import quantize_canvas, digest, shapes',
        'from nodule_canvas_contract import digest, shapes\nfrom nodule_exact_host import quantize_exact as quantize_canvas, search_mask_exact as search_mask, ExactSamplerCache')
    s=once(s,'from nodule_vendor.lung_nodule_cnn import config_from_dict, search_mask, tiles, decode, consolidate',
        'from nodule_vendor.lung_nodule_cnn import config_from_dict, tiles, decode, consolidate')
    s=once(s,'        def prepare(index):','        sampler_cache=ExactSamplerCache(hu,geometry[\'spacing_zyx_mm\'])\n        def prepare(index):')
    s=once(s,"sampler = SharedPlaneSampler(hu, geometry['spacing_zyx_mm'], group, self.config)",
        'sampler = sampler_cache.sample(group,self.config)')
    s=once(s,"[Path(__file__),Path(__file__).with_name('nodule_frozen_scoring.py')]",
        "[Path(__file__),Path(__file__).with_name('nodule_frozen_scoring.py'),Path(__file__).with_name('nodule_exact_host.py')]")
    p.write_text(s,encoding='utf8')
    # Exactly one writer in flight; all futures drain before registration,
    # cancellation or errors. Board calls remain strictly serial.
    p=DST/'python/ldct_workspace.py';s=p.read_text(encoding='utf8')
    s=once(s,'import base64\n','import base64\nfrom concurrent.futures import ThreadPoolExecutor\n')
    lo=s.index('            for done,z in enumerate(indices,1):',s.index('    def _enhance('))
    save=s.index('                if output.dtype',lo)
    end=s.index('            with self.lock:\n                self._register_enhanced_source',save)
    infer=s[lo:save];store=s[save:end]
    parallel=('            def save_slice(done,z,original,output,elapsed,receipt):\n'+store+
        "            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='ldct-save') as writer:\n"
        '                pending=None\n'+''.join('    '+line if line.strip() else line for line in infer.splitlines(keepends=True))+
        '                    if pending is not None:pending.result()\n'
        '                    pending=writer.submit(save_slice,done,z,original,output.copy(),elapsed,receipt)\n'
        '                if pending is not None:pending.result()\n')
    s=s[:lo]+parallel+s[end:];p.write_text(s,encoding='utf8')
    for p in (DST/'python').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    (DST/'exact_speed_candidate.json').write_text(json.dumps(dict(status='PREPARED_NOT_BOARD_VALIDATED',
        source=str(SRC),github_baseline='58ec1d49cfdcceb6e70ddcde41af2a7c7ad763bd',
        board_files_unchanged=True,models_weights_and_thresholds_unchanged=True,
        cleanup_unchanged=True,production_not_restarted=True),indent=2),encoding='utf8')
    print('PC_CANDIDATE_PREPARED',DST)


if __name__=='__main__':main()
