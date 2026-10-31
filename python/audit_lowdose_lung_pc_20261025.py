"""Local-only review of the user-supplied CT with the existing frozen U-Net.

No network requests, board access, training, threshold selection or GT metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'data/CT_Studio_PC_Paired_20261025_01'
sys.path[:0] = [str(APP / 'python'), str(APP / 'pynq')]
import numpy as np
from lung_live_data import read_ct, input_codes, native_masks
from lung_unet import make_model, resize_stack, torch_module
from lung_body_gate import gate_slice
from lung_island_cleanup import clean_prediction, PRIMARY


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def run(source, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint = ROOT / 'data/lung_training_20261011/early_stop_run/best.pt'
    checkpoint_sha = sha(checkpoint)
    write(output / 'progress.json', dict(status='READING_CT', backend='PC_ONLY', no_board_access=True))
    # The reader checks common patient/study/series and physical grid before
    # accepting the known per-slice anonymized FrameOfReferenceUID variation.
    hu, geom, details = read_ct(source, allow_inconsistent_frame_uids=True)
    write(output / 'input_audit.json', details)
    torch = torch_module()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = make_model(cp['base_channels'], cp.get('context_slices', 1)).eval().to(device)
    net.load_state_dict(cp['model'])
    assert cp.get('image_size', 256) == 256 and cp.get('threshold', .5) == .5
    tick = time.perf_counter()
    images = ((resize_stack(np.clip(hu, -1000, 400), 256, 1) + 1000) / 1400).astype(np.float16).astype(np.float32)
    preprocessing_ms = (time.perf_counter() - tick) * 1000
    raw = np.zeros(hu.shape, bool)
    tick = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(hu), 8):
            stop = min(begin + 8, len(hu))
            probability = torch.sigmoid(net(torch.from_numpy(images[begin:stop, None]).to(device)))
            probability = torch.nn.functional.interpolate(probability, size=hu.shape[1:], mode='bilinear', align_corners=False)
            raw[begin:stop] = (probability[:, 0] >= .5).cpu().numpy()
            if stop % 64 == 0 or stop == len(hu):
                write(output / 'progress.json', dict(status='PC_FP32_INFERENCE', completed=stop, total=len(hu), device=device))
                print('PC FP32', stop, '/', len(hu), flush=True)
    inference_ms = (time.perf_counter() - tick) * 1000
    del images, net
    if device == 'cuda':
        torch.cuda.empty_cache()
    tick = time.perf_counter()
    body = np.empty_like(raw)
    for z in range(len(hu)):
        body[z], _ = gate_slice(hu[z], raw[z], geom['spacing_zyx_mm'][1:])
    body_ms = (time.perf_counter() - tick) * 1000
    write(output / 'progress.json', dict(status='PC_3D_CLEANUP', completed=len(hu), total=len(hu)))
    tick = time.perf_counter()
    cleaned, cleanup = clean_prediction(body, geom['spacing_zyx_mm'], **PRIMARY)
    cleanup_ms = (time.perf_counter() - tick) * 1000
    # Packed masks preserve every voxel without duplicating the full CT.
    np.savez_compressed(output / 'masks_packed.npz', shape=np.array(hu.shape),
                        raw=np.packbits(raw.ravel(), bitorder='little'),
                        body=np.packbits(body.ravel(), bitorder='little'),
                        cleaned=np.packbits(cleaned.ravel(), bitorder='little'))
    write(output / 'cleanup.json', cleanup)
    selected = [z for z in [30, 65, 80, 110, 150, 180, 220, 260, 300, 325] if z < len(hu)]
    # Offline exact-integer reference for representative slices. This checks
    # quantization differences, not physical PL correctness on this new case.
    from lung_cpu import NativeUNet
    compiler_dir = Path('E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin')
    dll_dir = os.add_dll_directory(str(compiler_dir)) if os.name == 'nt' else None
    build = ROOT / 'data/lung_nodule_shared_20261024_v1/oracle/build'
    proof = json.loads((build / 'lung_cpu_native_build.json').read_text(encoding='utf-8'))
    library = build / 'lung_cpu_native.dll'
    assert sha(library) == proof['binary_sha256']
    for name, expected in proof['sources_sha256'].items():
        assert sha(build / name) == expected
    model_folder = ROOT / 'data/lung_nodule_shared_20261024_v1/lung_nodule_shared/models/unet_original'
    model = json.loads((model_folder / 'model.json').read_text(encoding='utf-8'))
    with np.load(model_folder / 'parameters.npz', allow_pickle=False) as p:
        params = {n: p[n] for n in p.files}
    write(output / 'progress.json', dict(status='PC_INT8_SPOT_CHECK', slices=selected))
    integer_masks = {}; differences = []; head_bank = {}
    with NativeUNet(library, model, params, 2) as integer:
        for z in selected:
            codes = input_codes(hu[z], model)
            head = integer.forward(codes).copy()
            iraw, ibody, _ = native_masks(head, hu[z], geom, model)
            integer_masks[z] = ibody
            differences.append(dict(index=z, fp32_body_pixels=int(body[z].sum()), int8_body_pixels=int(ibody.sum()),
                                    differing_body_pixels=int((body[z] != ibody).sum()),
                                    note='FP32 vs offline INT8 predictions; no expert reference and no PL run'))
            head_bank['slice_' + str(z)] = head
            print('PC INT8 comparison', z, 'different native pixels', differences[-1]['differing_body_pixels'], flush=True)
    if dll_dir is not None:
        dll_dir.close()
    np.savez_compressed(output / 'int8_sample_heads.npz', **head_bank)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def draw(ax, image, mask, title, aspect=1):
        ax.imshow(image, cmap='gray', vmin=-1350, vmax=150, aspect=aspect)
        if mask is not None:
            color = np.zeros((*mask.shape, 4)); color[mask] = [1., .4, 0., .35]
            ax.imshow(color, aspect=aspect)
            if mask.any():
                ax.contour(mask, levels=[.5], colors=['#ffae32'], linewidths=.55)
        ax.set_title(title); ax.axis('off')

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, len(selected) * 3.6), squeeze=False)
    for row, z in enumerate(selected):
        draw(axes[row, 0], hu[z], None, f'Z {z} | Original low-dose CT')
        draw(axes[row, 1], hu[z], cleaned[z], 'PC FP32 + body / 3D cleanup')
        draw(axes[row, 2], hu[z], integer_masks[z], 'PC INT8 + body constraint')
    fig.suptitle('Unchanged lung U-Net | PC-only test | no expert ground truth', fontsize=14)
    fig.tight_layout(); fig.savefig(output / 'slice_comparison.png', dpi=105); plt.close(fig)
    area = cleaned.sum((1, 2))
    zz = int(np.argmax(area)); yy = hu.shape[1] // 2; xx = hu.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, axis, index, name in zip(axes, [0, 1, 2], [zz, yy, xx], ['Axial', 'Coronal', 'Sagittal']):
        plane = np.take(hu, index, axis); mask = np.take(cleaned, index, axis)
        sp = [geom['spacing_zyx_mm'][i] for i in range(3) if i != axis]
        draw(ax, plane, mask, name + ' | PC FP32 + cleanup', aspect=sp[0] / sp[1])
    fig.tight_layout(); fig.savefig(output / 'mpr.png', dpi=120); plt.close(fig)
    np.savetxt(output / 'area_by_slice.csv', np.column_stack((np.arange(len(hu)), raw.sum((1, 2)), body.sum((1, 2)), area)),
               fmt='%d', delimiter=',', header='index_zero_based,raw_pixels,body_pixels,cleaned_pixels', comments='')
    assert sha(checkpoint) == checkpoint_sha
    result = dict(status='COMPLETE_PC_ONLY', backend='PC_CUDA_FP32' if device == 'cuda' else 'PC_CPU_FP32',
                  gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None, shape=list(hu.shape), geometry=geom,
                  expert_reference_present=False, expert_dice=None, checkpoint_sha256=checkpoint_sha,
                  checkpoint_epoch=cp['epoch'], model_parameters_changed=False, threshold=.5,
                  full_volume_fp32_slices=len(hu), offline_integer_spot_checks=differences,
                  predicted_volume_ml=float(cleaned.sum() * np.prod(geom['spacing_zyx_mm']) / 1000),
                  body_removed_voxels=int((raw & ~body).sum()), cleanup_removed_voxels=int((body & ~cleaned).sum()),
                  timing_ms=dict(preprocessing=preprocessing_ms, pc_forward_and_native_resize=inference_ms,
                                 body_constraint=body_ms, cleanup=cleanup_ms, whole_audit=(time.perf_counter() - started) * 1000),
                  timing_scope='One PC review run; includes loading and artifact generation in whole_audit. Not a controlled benchmark or FPGA time.',
                  no_board_access=True, no_expert_annotations_used=True,
                  source_hu_sha256=details['hu_sha256'])
    write(output / 'result.json', result)
    write(output / 'progress.json', dict(status='COMPLETE_PC_ONLY', completed=len(hu), total=len(hu)))
    print('COMPLETE:', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.source, a.output)
