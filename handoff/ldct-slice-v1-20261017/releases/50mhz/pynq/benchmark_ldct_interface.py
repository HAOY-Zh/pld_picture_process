"""Owned-resource acceptance/benchmark harness, separate from the borrowed API.

This CLI plays the role of the service: serial lock, Overlay/CMA/clock ownership.
Only the harness changes clocks or recovers hardware. LDCTProcessor never does.
Expected arrays are generated independently on PC and hash-bound to each bundle.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import sleep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ldct_processor import LDCTProcessor
from ct_denoise_tile_pl_test import check_bundle, sha, reset_dma


def serializable(result):
    return {k:v for k,v in result.items() if k != 'hu'}


def owned_cleanup(processor, dma, gpio, tx, rx):
    """Caller-side recovery only; do not add this policy to LDCTProcessor."""
    try:
        if processor is not None:
            processor.assert_idle()
    except Exception:
        try:
            reset_dma(dma)
            if not (dma.mmio.read(4)&1 and dma.mmio.read(0x34)&1):
                raise RuntimeError('DMA channels not halted after owner reset')
            gpio.write(0,0x80000000); sleep(.001); gpio.write(0,0)
        except Exception as error:
            print('OWNER RECOVERY FAILED; retaining CMA in this live process. External recovery required: '+str(error),flush=True)
            while True:
                sleep(30)
    if rx is not None:
        rx.freebuffer()
    tx.freebuffer()


def main(a):
    from pynq import Overlay, allocate, Clocks
    import pynq
    if a.phase_repeats < 1 or a.warmup < 0:
        raise ValueError('Invalid repetition count')
    bundles = {}
    for item in a.bundle:
        label, path = item.split('=',1)
        if label in bundles:
            raise ValueError('Duplicate label')
        bundle = Path(path).resolve()
        manifest = check_bundle(bundle)
        if manifest.get('interface_version') != 'ldct_slice_v1':
            raise ValueError('Interface v1 release required')
        bundles[label] = (bundle, manifest)
    if len({v[1]['model']['weights_sha256'] for v in bundles.values()}) != 1:
        raise ValueError('Frequency comparison requires identical integer weights')
    out = Path(a.outdir).resolve()
    if out.exists():
        raise ValueError('Choose a new output directory')
    out.mkdir(parents=True)
    lock = open('/tmp/ldct_slice_validation.lock','w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    labels = list(bundles)
    sequence = labels + list(reversed(labels))  # ABBA; each label tested twice
    rows, checks, phases = [], [], []
    accepted = set()
    for phase, label in enumerate(sequence):
        bundle, manifest = bundles[label]
        phase_dir = out/f'phase_{phase:02d}_{label}'
        phase_dir.mkdir()
        print('LOAD',phase,label,'clock',manifest['clock_mhz'],flush=True)
        # Hardware owner acts only after preceding processor was asserted idle
        # and its buffers released. No LDCTProcessor code sets any clocks.
        overlay = Overlay(str(bundle/'overlay/ct_denoise_tile.bit'))
        Clocks.fclk0_mhz = float(manifest['clock_mhz'])
        actual = float(Clocks.fclk0_mhz)
        if abs(actual-float(manifest['clock_mhz'])) > .5:
            raise RuntimeError('Owner clock setup failed')
        if abs(float(Clocks.cpu_mhz)-650.) > .5:
            raise RuntimeError('Keep ARM at 650 MHz for this comparison')
        dma, gpio = overlay.axi_dma_0, overlay.axi_gpio_0
        tx = allocate(shape=(4096,),dtype=np.uint32)
        rx = None
        processor = None
        try:
            rx = allocate(shape=(4096,),dtype=np.uint32)
            gpio.write(0,0x80000000); sleep(.001); gpio.write(0,0)
            processor = LDCTProcessor(dma=dma,gpio=gpio,tx_buffer=tx,rx_buffer=rx,
                model_dir=bundle/'model',read_clock_mhz=lambda:float(Clocks.fclk0_mhz),
                expected_clock_mhz=manifest['clock_mhz'],timeout_s=a.timeout)
            processor.assert_idle()
            fixtures = json.loads((bundle/'fixtures/cases.json').read_text())
            retained = None
            if label not in accepted:
                for case in fixtures:
                    image = np.load(bundle/case['input'],allow_pickle=False)
                    expected = np.load(bundle/case['expected'],allow_pickle=False)
                    result = processor.process_slice(image,case['method'])
                    np.testing.assert_array_equal(result['hu'],expected)
                    assert result['hu'].dtype == np.int16 and result['hu'].shape == image.shape
                    assert not np.shares_memory(result['hu'],rx)
                    if retained is not None:
                        np.testing.assert_array_equal(retained[0],retained[1])
                    retained = result['hu'], result['hu'].copy()
                    processor.assert_idle()
                    target = phase_dir/(case['name']+'_actual.npy')
                    np.save(target,result['hu'],allow_pickle=False)
                    checks.append(dict(label=label,phase=phase,case=case['name'],
                        output=str(target.relative_to(out)),output_sha256=sha(target),
                        expected_sha256=sha(bundle/case['expected']),max_error_hu=0,
                        **serializable(result),state_after=processor.get_state()))
                    print('ACCEPT',label,case['name'],'tiles',result['tiles'],'exact',flush=True)
                accepted.add(label)
            for case in [c for c in fixtures if c['name'] in ('C067','C077')]:
                image = np.load(bundle/case['input'],allow_pickle=False)
                expected = np.load(bundle/case['expected'],allow_pickle=False)
                for repeat in range(a.warmup+a.phase_repeats):
                    result = processor.process_slice(image,'cnn')
                    np.testing.assert_array_equal(result['hu'],expected)
                    processor.assert_idle()
                    row = dict(label=label,phase=phase,patient=case['name'],repeat=repeat,
                        warmup=repeat<a.warmup,bit_exact=True,**serializable(result))
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                target = phase_dir/(case['name']+'_timed_actual.npy')
                np.save(target,result['hu'],allow_pickle=False)
            phases.append(dict(phase=phase,label=label,release_id=manifest['release_id'],
                bit_sha256=sha(bundle/'overlay/ct_denoise_tile.bit'),hwh_sha256=sha(bundle/'overlay/ct_denoise_tile.hwh'),
                model_id=manifest['model_id'],weights_sha256=manifest['model']['weights_sha256'],
                expected_pl_mhz=manifest['clock_mhz'],actual_pl_mhz=float(Clocks.fclk0_mhz),
                cpu_mhz=float(Clocks.cpu_mhz),state=processor.get_state(),
                fixture_file_sha256=sha(bundle/'fixtures/cases.json')))
        finally:
            owned_cleanup(processor,dma,gpio,tx,rx)
    summaries = []
    for label in labels:
        for patient in ['C067','C077']:
            selected = [r for r in rows if r['label']==label and r['patient']==patient and not r['warmup']]
            values = [r['total_ms'] for r in selected]
            summaries.append(dict(label=label,patient=patient,count=len(values),mean_ms=float(np.mean(values)),
                median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                actual_pl_mhz=selected[0]['actual_pl_mhz_before']))
    result = dict(status='PASS',actual_FPGA_execution=True,interface_version='ldct_slice_v1',
        environment=dict(pynq=pynq.__version__,python=platform.python_version(),numpy=np.__version__,cpu_mhz=float(Clocks.cpu_mhz)),
        sequence=sequence,warmup_each_phase=a.warmup,measured_each_phase=a.phase_repeats,
        per_patient_measured_per_version=a.phase_repeats*2,phases=phases,checks=checks,rows=rows,summary=summaries,
        timing_scope='pack+cache+DMA+PL+poll+stitch; excludes load/clock setup/files/reference validation/display',
        reference='Expected NPY arrays generated by independent PC NumPy integer reference, hash-bound to release',
        supported_methods_tested=list(next(iter(bundles.values()))[1]['methods']))
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'timings.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='PASS',summary=summaries)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',action='append',required=True,help='label=/absolute/path; repeat for each verified frequency release')
    p.add_argument('--outdir',required=True)
    p.add_argument('--phase-repeats',type=int,default=10)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--timeout',type=float,default=10.)
    main(p.parse_args())
