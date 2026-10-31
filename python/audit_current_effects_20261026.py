"""Offline audit of the saved current case; never changes the web or hardware."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT/'data/CT_Studio_DualExact_20261026_02'
OLD = ROOT/'data/CT_Studio_PC_Paired_20261025_01'
OUT = ROOT/'data/current_effect_audit_20261026'
PACKAGE = APP/'data/ct_studio_integration_20261025'
sys.path[:0] = [str(APP/'python'), str(PACKAGE/'ct_studio')]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    import numpy as np
    import torch
    from lung_live_data import read_ct, input_codes, native_masks
    from lung_island_cleanup import clean_prediction, PRIMARY
    from ldct_contract import quantize_slice
    from shared_cnn.lung_cpu import NativeUNet
    from shared_cnn.lung_nodule_shared import load_model
    from ctdenoise.cnn_fixed import load_quantized, infer_fixed
    from nodule_frozen_scoring import load_scoring, score_candidates
    from nodule_workspace import validate_packet

    torch.set_num_threads(2)
    started = time.perf_counter()
    state = json.loads((OUT/'state.json').read_text(encoding='utf8'))
    results = OLD/'data/ct_workspace_demo/results'
    case = results/state['case_id']
    original_case = results/state['series_id']
    info = json.loads((original_case/'input.json').read_text(encoding='utf8'))
    original, geom, details = read_ct(info['source'], confirm_hu=True,
        series_uid=info.get('series_uid', ''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied') is True)
    assert details['hu_sha256'] == info['hu_sha256']
    volume = json.loads((case/(state['volume_job_id']+'_volume.json')).read_text(encoding='utf8'))
    assert geom == volume['input_geometry']
    ldct_rows = sorted(state['ldct']['results'], key=lambda row: row['index'])
    assert [row['index'] for row in ldct_rows] == list(range(len(original)))
    enhanced = np.empty_like(original)
    hashes = []
    for row in ldct_rows:
        z = row['index']
        with np.load(original_case/'ldct'/(row['result_id']+'.npz'), allow_pickle=False) as saved:
            plane = saved['hu']
        assert sha(plane.tobytes()) == row['output_sha256']
        assert sha(quantize_slice(original[z]).tobytes()) == row['input_sha256']
        enhanced[z] = plane
        hashes.append(dict(index=z, input_exact=True, output_hash_exact=True))
    assert sha(enhanced.tobytes()) == state['analysis_source']['hu_sha256']
    write('ldct_all_saved_source_hashes.json', dict(slices=len(hashes), rows=hashes,
        source_hash=sha(enhanced.tobytes()), geometry_exact=True))
    print('LDCT_ALL_351_INPUT_OUTPUT_HASHES_AND_GEOMETRY_PASS', flush=True)

    arrays, spec = load_quantized(PACKAGE/'ct_studio/ldct/model')
    reference_rows = []
    selected = sorted(set([0, len(original)-1, *range(35, len(original), 35)]))
    for z in selected:
        actual, saturation = infer_fixed(quantize_slice(original[z]), arrays, spec)
        np.testing.assert_array_equal(actual, enhanced[z])
        reference_rows.append(dict(index=z, exact=True, different_pixels=0, saturation=saturation))
        print('LDCT_FIXED_REFERENCE_EXACT', z, flush=True)
    write('ldct_extended_reference.json', dict(status='PASS', slices=reference_rows,
        scope='Fresh integer software-reference check of selected saved actual FPGA planes; other planes hash-checked.'))

    # Verify a previously built local integer oracle against its build metadata
    # and against the current release sources before loading the DLL.
    native = ROOT/'data/ct_studio_delivery_20261018/assets_build/native'
    build = json.loads((native/'lung_cpu_native_build.json').read_text(encoding='utf8'))
    library = native/'lung_cpu_native.dll'
    assert sha(library.read_bytes()) == build['binary_sha256']
    for name, expected in build['sources_sha256'].items():
        assert sha((PACKAGE/'ct_studio/shared_cnn'/name).read_bytes()) == expected
    dll_directory = os.add_dll_directory(str(Path(build['command'][0]).parent))
    model, parameters = load_model(PACKAGE/'ct_studio/shared_cnn', 'unet_original')
    original_raw = np.zeros(original.shape, bool)
    original_body = np.zeros(original.shape, bool)
    reference_body = np.zeros(original.shape, bool)
    with np.load(case/(state['volume_job_id']+'_masks.npz'), allow_pickle=False) as saved:
        saved_masks = {name: saved[name] for name in saved.files}
    rows = []
    with NativeUNet(library, model, parameters, threads=2) as net:
        for z in range(len(original)):
            rid = volume['slice_receipts'][str(z)]
            with np.load(case/rid/'actual_head.npz', allow_pickle=False) as saved:
                codes = input_codes(enhanced[z], model)
                np.testing.assert_array_equal(codes, saved['input_codes'])
                head = net.forward(codes)
                np.testing.assert_array_equal(head, saved['head_codes'])
                raw, reference_body[z], _ = native_masks(head, enhanced[z], geom, model)
                np.testing.assert_array_equal(raw, saved_masks['raw'][z])
                np.testing.assert_array_equal(reference_body[z], saved_masks['body'][z])
            original_codes = input_codes(original[z], model)
            original_head = net.forward(original_codes)
            original_raw[z], original_body[z], _ = native_masks(original_head, original[z], geom, model)
            rows.append(dict(index=z, input_codes_exact=True, full_head_exact=True,
                raw_mask_exact=True, body_mask_exact=True,
                original_vs_enhanced_input_code_differences=int(np.count_nonzero(original_codes != codes)),
                original_vs_enhanced_body_mask_differences=int(np.count_nonzero(original_body[z] != reference_body[z]))))
            if z % 25 == 0 or z == len(original)-1:
                print('CURRENT_CASE_FULL_INTEGER_GRAPH_AND_MASK_EXACT', z+1, '/', len(original), flush=True)
                write('followup_progress.json', dict(stage='segmentation', completed=z+1,
                    total=len(original), elapsed_s=time.perf_counter()-started))
    enhanced_cleaned, _ = clean_prediction(reference_body, geom['spacing_zyx_mm'], **PRIMARY)
    np.testing.assert_array_equal(enhanced_cleaned, saved_masks['cleaned'])
    original_cleaned, original_cleanup = clean_prediction(original_body, geom['spacing_zyx_mm'], **PRIMARY)
    ml_per_voxel = float(np.prod(geom['spacing_zyx_mm'])/1000)
    original_ml = float(original_cleaned.sum()*ml_per_voxel)
    enhanced_ml = float(enhanced_cleaned.sum()*ml_per_voxel)
    assert enhanced_ml == volume['volume_ml']
    write('segmentation_all351_offline_reference.json', dict(status='PASS_FULL_CASE_INTEGER_REFERENCE_AND_POSTPROCESS',
        slices=len(rows), rows=rows, enhanced_volume_ml=enhanced_ml,
        original_cpu_reference_volume_ml=original_ml, saved_head_code_differences=0,
        saved_native_mask_differences=0, saved_3d_cleaned_mask_differences=0,
        original_vs_enhanced_cleaned_different_voxels=int(np.count_nonzero(original_cleaned != enhanced_cleaned)),
        scope='Current actual saved FPGA heads against unchanged native integer CPU oracle on identical enhanced inputs. Original-input CPU reference is a source comparison, not GT accuracy.'))
    np.savez_compressed(OUT/'original_input_cpu_masks.npz', raw=original_raw, body=original_body, cleaned=original_cleaned)

    # Recompute the final frozen score from the retained raw candidates, without
    # running the CNN, communicating with the board, or choosing a new threshold.
    packet_path = case/'nodule'/state['nodule']['job']['job_id']/'prediction.json'
    packet = json.loads(packet_path.read_text(encoding='utf8'))
    context = state['nodule']['context']
    validate_packet(packet, context, allow_enhanced=True)
    folder = packet_path.parent
    with gzip.open(folder/'all_ranked_candidates.json.gz', 'rt', encoding='utf8') as f:
        ranked = json.load(f)
    candidates = [dict(c, score=c['raw_score']) for c in ranked]
    for c in candidates:
        c.pop('raw_score', None)
        c.pop('score_kind', None)
    scoring, ranker = load_scoring(PACKAGE)
    scoring_out = OUT/'recomputed_frozen_scoring'
    scoring_out.mkdir(exist_ok=False)
    selected, scoring_record = score_candidates(enhanced, geom['spacing_zyx_mm'], candidates,
        scoring, ranker, scoring_out)
    assert selected == packet['candidates']
    with gzip.open(scoring_out/'all_ranked_candidates.json.gz', 'rt', encoding='utf8') as f:
        assert ranked == json.load(f)
    write('nodule_scores_and_geometry.json', dict(status='PASS_ALL_RETAINED_SCORES_AND_SELECTED_GEOMETRY',
        full_ranked_candidates=len(ranked), selected_candidates=len(selected),
        context_geometry_valid=True, retained_scores_exact=True, selected_candidates_exact=True,
        score_record=scoring_record, scope='Recompute unchanged scores from saved CNN candidates; no new CNN forward or GT evaluation.'))

    # A fixed-window static comparison makes source changes visible without
    # touching the production page's selection or display settings.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    slice_ids = [int(np.argmax(original_body.sum(axis=(1,2)))), len(original)//2, min(len(original)-1, 250)]
    fig, axes = plt.subplots(len(slice_ids), 4, figsize=(14, 10), facecolor='#10202c')
    for i,z in enumerate(slice_ids):
        for j,(image,mask,title) in enumerate([
            (original[z],None,'Original HU'), (enhanced[z],None,'Enhanced HU'),
            (original[z],original_cleaned[z],'Original: same INT8 U-Net'),
            (enhanced[z],enhanced_cleaned[z],'Enhanced: saved FPGA verified')]):
            ax=axes[i,j];ax.imshow(image,cmap='gray',vmin=-1350,vmax=150)
            if mask is not None and mask.any():ax.contour(mask,levels=[.5],colors=['#23e1af'],linewidths=.65)
            ax.set_title(title+' / z='+str(z),fontsize=9,color='white');ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'same_case_same_window_sources.png',dpi=140,facecolor=fig.get_facecolor());plt.close(fig)
    write('followup_complete.json', dict(status='OFFLINE_AUDIT_PASS_WITH_KNOWN_PIPELINE_DIFFERENCES',
        elapsed_s=time.perf_counter()-started, case_id=state['case_id'],
        numpy_version=np.__version__, torch_version=torch.__version__,
        web_session_unchanged=True, board_not_accessed=True,
        segmentation_all351_integer_and_postprocess_exact=True,
        nodule_all_retained_scores_and_selected_geometry_exact=True,
        fresh_ldct_reference_slices=len(reference_rows),
        original_cpu_reference_volume_ml=original_ml,enhanced_volume_ml=enhanced_ml))
    print('OFFLINE_AUDIT_COMPLETE', round(time.perf_counter()-started,2), flush=True)
    dll_directory.close()


if __name__ == '__main__':
    main()
