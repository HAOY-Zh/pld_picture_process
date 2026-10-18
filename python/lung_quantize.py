"""Training-only static PTQ and development evaluation of the retained 2D U-Net.

Real PyTorch CPU quantized convolutions, NOT RTL/FPGA or a bit-exact RTL oracle.
Convolution weights are signed INT8; activations are unsigned UINT8. Sigmoid and
native-resolution probability interpolation remain floating point outside core.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from lung_segmentation import fresh_output, write_json
from lung_unet import (_case_prediction, aggregate_slice_quality, context_batch,
                       file_sha256, load_cached_source, make_model, mapped_npy,
                       read_manifest, save_learned_preview, select_device,
                       slice_quality_metrics, torch_module)
from lungseg.metrics import overlap_metrics


BLOCKS = ('enc1', 'enc2', 'enc3', 'neck', 'dec3', 'dec2', 'dec1')


def make_quantizable(source):
    """Copy parameters and preserve skip/resize semantics; never mutate source."""
    torch = torch_module()
    if source.context_slices != 1 or source.training:
        raise ValueError('Static PTQ requires the retained eval-mode 2D model')

    class QuantizableUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_slices = 1
            for name in BLOCKS:
                setattr(self, name, copy.deepcopy(getattr(source, name)))
            self.pool = copy.deepcopy(source.pool)
            self.head = copy.deepcopy(source.head)
            self.input_quant = torch.ao.quantization.QuantStub()
            self.output_dequant = torch.ao.quantization.DeQuantStub()
            self.cats = torch.nn.ModuleList([torch.ao.nn.quantized.FloatFunctional() for _ in range(3)])

        def forward(self, image):
            image = self.input_quant(image)
            e1 = self.enc1(image)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            x = self.neck(self.pool(e3))
            for number, (skip, decoder) in enumerate(((e3, self.dec3), (e2, self.dec2), (e1, self.dec1))):
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='nearest')
                x = decoder(self.cats[number].cat((x, skip), dim=1))
            return self.output_dequant(self.head(x))

    return QuantizableUNet().eval()


def fold_batchnorm(model):
    torch = torch_module()
    result = copy.deepcopy(model).eval()
    for name in BLOCKS:
        block = getattr(result, name)
        for conv_index, bn_index in ((0, 1), (3, 4)):
            block[conv_index] = torch.nn.utils.fuse_conv_bn_eval(block[conv_index], block[bn_index])
            block[bn_index] = torch.nn.Identity()
    if any(isinstance(m, torch.nn.BatchNorm2d) for m in result.modules()):
        raise ValueError('Unexpected residual BatchNorm in folded core')
    return result


def prepare_static(folded, engine='x86'):
    torch = torch_module()
    if engine not in torch.backends.quantized.supported_engines or engine == 'none':
        raise ValueError(f'Quantized CPU engine is unavailable: {engine}')
    result = copy.deepcopy(folded).eval()
    for name in BLOCKS:
        torch.ao.quantization.fuse_modules(getattr(result, name), [['0', '2'], ['3', '5']], inplace=True)
    result.qconfig = torch.ao.quantization.get_default_qconfig(engine)
    # MaxPool and nearest resize preserve the quantized input grid; no FP fallback.
    result.pool.qconfig = None
    return torch.ao.quantization.prepare(result, inplace=False)


def calibrate_training(prepared, original, root, records, device, batch_size=8, image_size=None):
    """Observe ALL training images; never open labels, validation, or test inputs."""
    torch = torch_module()
    if (not records or any(r['split'] != 'train' or not r['case'].startswith('LCTSC-Train-') for r in records)
            or len({r['patient_key'] for r in records}) != len(records)
            or batch_size < 1):
        raise ValueError('Calibration accepts unique training patients only and positive batch size')
    prepared.to(device).eval()
    original.to(device).eval()
    cases, largest_error, slices = [], 0., 0
    with torch.inference_mode():
        for record in records:
            path = Path(root) / record['cache_folder'] / 'images.npy'
            before = file_sha256(path)
            print(f'Calibrating TRAIN {record["case"]}: all {record["slices"]} slices', flush=True)
            with mapped_npy(path) as images:
                size = image_size if image_size is not None else (images.shape[1] if images.ndim == 3 else 0)
                if size < 32 or size % 8 or images.shape != (record['slices'], size, size):
                    raise ValueError('Calibration cache shape/slice count differs')
                for begin in range(0, len(images), batch_size):
                    centers = np.arange(begin, min(begin + batch_size, len(images)))
                    array = context_batch(images, centers, 1)
                    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                        raise ValueError('Calibration inputs must be finite normalized HU in [0,1]')
                    image = torch.from_numpy(array).to(device)
                    expected, folded = original(image), prepared(image)
                    torch.testing.assert_close(folded, expected, rtol=2e-4, atol=2e-4)
                    largest_error = max(largest_error, float((folded - expected).abs().max()))
                    slices += len(centers)
            if file_sha256(path) != before:
                raise ValueError('Training image cache changed during calibration')
            cases.append({'case': record['case'], 'split': 'train', 'slices': record['slices'],
                          'patient_key': record['patient_key'], 'images_sha256': before})
    return {'cases': cases, 'patients': len(cases), 'slices': slices,
            'source': 'All normalized training CT slices, including empty-lung layers; NO labels read',
            'validation_or_test_used': False, 'fold_max_abs_logit_error': largest_error,
            'fold_tolerance': {'rtol': 2e-4, 'atol': 2e-4}}


def convert_static(prepared, engine='x86'):
    torch = torch_module()
    torch.backends.quantized.engine = engine
    cpu_model = copy.deepcopy(prepared).cpu().eval()
    # prepare() adds observer factories whose closures refer to their original
    # modules. deepcopy retains those closure references: after GPU calibration
    # a new weight observer could be created on the old GPU despite cpu_model.cpu().
    # Rebind factories to CPU without replacing calibrated activation observers.
    cpu_config = torch.ao.quantization.get_default_qconfig(engine)
    for module in cpu_model.modules():
        if getattr(module, 'qconfig', None) is not None:
            module.qconfig = cpu_config
    result = torch.ao.quantization.convert(cpu_model, inplace=True)
    convolutions = [m for m in result.modules() if isinstance(m, torch.ao.nn.quantized.Conv2d)]
    if len(convolutions) != 15 or any(isinstance(m, torch.nn.Conv2d) for m in result.modules()):
        raise ValueError('Expected all 15 convolutions to become real CPU quantized convolutions')
    return result


def export_core(model, size, output):
    """Actual weights/scales plus a conservative proposed INT32 MAC bound.

    Bias FP32 is exported unchanged; rounding bias into INT32 is a proposed RTL
    convention, NOT a claim that all PyTorch engines use that convention.
    """
    torch = torch_module()
    layers, arrays, handles, boundaries = [], {}, [], {}

    def boundary(name):
        def hook(module, inputs, result):
            if not result.is_quantized:
                raise ValueError(f'Expected quantized boundary: {name}')
            boundaries[name] = {'shape': list(result.shape), 'dtype': str(result.dtype),
                                'scale': result.q_scale(), 'zero_point': result.q_zero_point()}
        return hook

    def convolution(name):
        def hook(module, inputs, result):
            image, weight = inputs[0], module.weight()
            if not image.is_quantized or not result.is_quantized or weight.dtype != torch.qint8:
                raise ValueError('Quantized convolution fell back to float')
            integer = weight.int_repr().numpy().copy()
            if weight.qscheme() not in (torch.per_channel_affine, torch.per_channel_symmetric):
                raise ValueError('Expected per-output-channel weight quantization')
            scales = weight.q_per_channel_scales().numpy().copy()
            zeros = weight.q_per_channel_zero_points().numpy().copy()
            if weight.q_per_channel_axis() != 0:
                raise ValueError('Expected output-channel weight axis 0')
            bias = module.bias().detach().numpy().copy()
            bias_scale = image.q_scale() * scales
            proposed_bias = np.rint(bias / bias_scale)
            if not np.isfinite(proposed_bias).all() or np.max(np.abs(proposed_bias)) >= 2**31:
                raise ValueError('Proposed INT32 bias overflows')
            centered_w = integer.astype(np.int64) - zeros[:, None, None, None]
            max_input = max(image.q_zero_point(), 255 - image.q_zero_point())
            bounds = np.abs(centered_w).sum(axis=(1, 2, 3)) * max_input + np.abs(proposed_bias.astype(np.int64))
            if np.max(bounds) >= 2**31:
                raise ValueError('Conservative centered INT32 accumulator bound fails')
            prefix = name.replace('.', '_')
            arrays[prefix + '_weight_int8'] = integer
            arrays[prefix + '_weight_scale'] = scales
            arrays[prefix + '_weight_zero_point'] = zeros
            arrays[prefix + '_bias_fp32'] = bias
            arrays[prefix + '_proposed_bias_int32'] = proposed_bias.astype(np.int32)
            layers.append({'name': name, 'weight_shape': list(integer.shape),
                'weight_dtype': 'int8', 'weight_scale_axis': 0, 'weight_bytes': integer.nbytes,
                'bias_fp32_bytes': bias.nbytes, 'input_shape': list(image.shape),
                'output_shape': list(result.shape), 'input_dtype': str(image.dtype),
                'output_dtype': str(result.dtype), 'input_scale': image.q_scale(),
                'input_zero_point': image.q_zero_point(), 'output_scale': result.q_scale(),
                'output_zero_point': result.q_zero_point(),
                'output_bytes_per_slice': result[0].numel(),
                'MACs_per_slice': int(result[0].numel() * integer.shape[1] * integer.shape[2] * integer.shape[3]),
                'stride': list(module.stride), 'padding': list(module.padding),
                'fused_relu': isinstance(module, torch.ao.nn.intrinsic.quantized.ConvReLU2d),
                'proposed_int32_accumulator_max_abs_bound': int(bounds.max())})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.Conv2d):
            handles.append(module.register_forward_hook(convolution(name)))
        if name == 'input_quant' or name.startswith('cats.') and name.count('.') == 1:
            # QFunctional.cat does not invoke Module.__call__; capture cats separately below.
            if name == 'input_quant':
                handles.append(module.register_forward_hook(boundary(name)))
    try:
        with torch.inference_mode():
            logits = model(torch.zeros((1, 1, size, size)))
        if logits.is_quantized or len(layers) != 15:
            raise ValueError('Unexpected quantized core/export output')
    finally:
        for handle in handles:
            handle.remove()
    for number, module in enumerate(model.cats):
        boundaries[f'cats.{number}'] = {'scale': float(module.scale), 'zero_point': int(module.zero_point),
                                        'operation': 'Quantized concat with rescaling of differing skip/input grids'}
    np.savez(output / 'int8_core_parameters.npz', **arrays)
    weights = sum(r['weight_bytes'] for r in layers)
    biases = sum(r['bias_fp32_bytes'] for r in layers)
    skips = [r for r in layers if r['name'] in ('enc1.3', 'enc2.3', 'enc3.3')]
    report = {'layers': layers, 'boundaries': boundaries, 'int8_weight_bytes': weights,
              'bias_fp32_bytes': biases, 'weight_and_bias_bytes_excluding_scales': weights + biases,
              'same_folded_core_fp32_weight_and_bias_bytes': weights * 4 + biases,
              'conv_MACs_per_slice': sum(r['MACs_per_slice'] for r in layers),
              'naive_full_skip_buffer_bytes': sum(r['output_bytes_per_slice'] for r in skips),
              'skip_buffers': [{'name': r['name'], 'bytes': r['output_bytes_per_slice']} for r in skips],
              'integer_convolution_count': len(layers), 'FP32_convolution_count': 0,
              'precision': 'Signed INT8 per-channel weights, UINT8 affine activation tensors; '
                           'x86 default histogram observers use reduce_range for calibration',
              'MAC_equation': 'sum((qx-zx)*(qw-zw)) + proposed_round_even(bias/(sx*sw))',
              'important': 'A nonzero activation zero point requires offset correction; centered UINT8 '
                           'may require 9 bits. Do not reinterpret uint8 as signed int8.',
              'outside_core': ['FP32 HU normalization/input resize', 'FP32 dequantized logits and sigmoid',
                               'FP32 bilinear probability resize to native grid', 'threshold >=0.5', 'display'],
              'hardware_status': 'No RTL, HLS, bitstream or PL inference. No bit-exact RTL equivalence; '
                                 'bias/requantization rounding and fixed-point multipliers still need specification. '
                                 'Buffer bytes are naive full skips, not a minimum or a BRAM/DSP/LUT estimate.'}
    write_json(output / 'core_manifest.json', report)
    return report


def save_quantized(model, path, checkpoint, protocol):
    torch = torch_module()
    torch.save({'schema': 1, 'quantization': 'static_ptq_x86_w8_uint8',
                'state_dict': model.state_dict(), 'engine': 'x86',
                'base_channels': checkpoint['base_channels'], 'context_slices': 1,
                'image_size': checkpoint['image_size'], 'threshold': .5,
                'source_checkpoint_sha256': protocol['checkpoint_sha256'],
                'dataset_fingerprint': checkpoint['dataset_fingerprint']}, path)


def restore_quantized(path):
    torch = torch_module()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['quantization'] != 'static_ptq_x86_w8_uint8' or saved['context_slices'] != 1:
        raise ValueError('Unsupported quantized checkpoint')
    source = make_model(saved['base_channels'], 1).eval()
    prepared = prepare_static(fold_batchnorm(make_quantizable(source)), saved['engine'])
    # The placeholder observers are calibrated only to construct the module
    # skeleton. All weights and calibrated qparameters are replaced by the saved
    # state; this is NOT a new calibration of the saved model.
    with torch.inference_mode():
        prepared(torch.zeros(1, 1, 32, 32))
    result = convert_static(prepared, saved['engine'])
    result.load_state_dict(saved['state_dict'], strict=True)
    return result, saved


def quality_summary(rows, key):
    return {'mean_case_dice': float(np.mean([r[key]['dice'] for r in rows])),
            'mean_case_iou': float(np.mean([r[key]['iou'] for r in rows])),
            'slice_quality': aggregate_slice_quality([r[key + '_slice_quality'] for r in rows])}


def validate_pair(original, quantized, root, records, device, args, output):
    torch = torch_module()
    if not records or any(r['split'] != 'validation' for r in records):
        raise ValueError('PTQ precision evaluation accepts validation cases only')
    rows = []
    for record in records:
        print(f'Comparing VALIDATION {record["case"]}', flush=True)
        fp32, fp32_ms = _case_prediction(original, root, record, device, args.batch_size, .5)
        int8, int8_ms = _case_prediction(quantized, root, record, torch.device('cpu'), args.batch_size, .5)
        with mapped_npy(root / record['cache_folder'] / 'reference_native.npy') as array:
            reference = array != 0
        original_scores, quantized_scores = overlap_metrics(fp32, reference), overlap_metrics(int8, reference)
        case_dir = output / record['case']
        case_dir.mkdir()
        np.save(case_dir / 'float_prediction_native.npy', fp32.astype(np.uint8)*255)
        np.save(case_dir / 'int8_prediction_native.npy', int8.astype(np.uint8)*255)
        fp32_slices = [overlap_metrics(fp32[i], reference[i]) for i in range(len(reference))]
        int8_slices = [overlap_metrics(int8[i], reference[i]) for i in range(len(reference))]
        nonempty = [i for i in range(len(reference)) if reference[i].any()]
        indices = {int(np.argmax(reference.sum(axis=(1, 2)))),
                   min(nonempty, key=lambda i: fp32_slices[i]['dice']),
                   min(nonempty, key=lambda i: int8_slices[i]['dice']),
                   int(np.argmax((int8 != fp32).sum(axis=(1, 2))))}
        empty_fp = [i for i in range(len(reference)) if not reference[i].any() and int8[i].any()]
        if empty_fp:
            indices.add(max(empty_fp, key=lambda i: int(int8[i].sum())))
        hu = load_cached_source(record)
        for index in sorted(indices):
            save_learned_preview(hu[index], reference[index], int8[index], fp32[index],
                case_dir / f'quantization_slice_{index:03d}.png', record['case'], index,
                baseline_title='Original FP32 (yellow), CPU/GPU',
                learned_title='PTQ INT8 core (green), CPU; FP32 postprocess')
        row = {'case': record['case'], 'split': 'validation', 'slices': len(reference),
               'fp32': original_scores, 'int8': quantized_scores,
               'dice_delta': quantized_scores['dice'] - original_scores['dice'],
               'different_pixels': int((fp32 != int8).sum()), 'total_pixels': int(int8.size),
               'fp32_slice_quality': slice_quality_metrics(fp32, reference),
               'int8_slice_quality': slice_quality_metrics(int8, reference),
               'fp32_per_slice': fp32_slices, 'int8_per_slice': int8_slices,
               'preview_slices': sorted(indices), 'fp32_cached_input_to_native_mask_ms': fp32_ms,
               'int8_cached_input_to_native_mask_ms': int8_ms,
               'timing_warning': 'FP32 on '+str(device)+' and INT8 CPU; not a fair speedup or FPGA FPS claim'}
        rows.append(row)
        write_json(case_dir / 'metrics.json', row)
        write_json(output / 'evaluation.json', {'status': 'evaluating_validation', 'cases': rows})
        print(f'  FP32 Dice {original_scores["dice"]:.6f}; INT8 {quantized_scores["dice"]:.6f}; '
              f'delta {row["dice_delta"]:+.6f}', flush=True)
        del fp32, int8, reference, hu
    return rows


def run_quantization(args):
    torch = torch_module()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and CPU threads must be positive')
    root, manifest = read_manifest(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint['dataset_fingerprint'] != manifest['fingerprint'] or
            checkpoint['image_size'] != manifest['image_size'] or checkpoint.get('context_slices', 1) != 1 or
            checkpoint.get('threshold', .5) != .5 or manifest['hu_clip'] != [-1000, 400] or
            manifest['input_normalization'] != '(clip(HU,-1000,400)+1000)/1400'):
        raise ValueError('Require retained 2D weights, matching development dataset and original preprocessing')
    train = [r for r in manifest['cases'] if r['split'] == 'train']
    validation = [r for r in manifest['cases'] if r['split'] == 'validation']
    if {r['patient_key'] for r in train} & {r['patient_key'] for r in validation}:
        raise ValueError('Calibration and validation patients overlap')
    torch.set_num_threads(args.threads)
    old_engine = torch.backends.quantized.engine
    old_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    device = select_device(args.device)
    source = make_model(checkpoint['base_channels'], 1).eval()
    source.load_state_dict(checkpoint['model'])
    folded = fold_batchnorm(make_quantizable(source))
    prepared = prepare_static(folded)
    output = fresh_output(args.outdir)
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': file_sha256(args.checkpoint),
        'checkpoint_epoch': checkpoint['epoch'], 'dataset': str(Path(args.dataset).resolve()),
        'dataset_sha256': file_sha256(args.dataset), 'dataset_fingerprint': manifest['fingerprint'],
        'image_size': checkpoint['image_size'], 'threshold': .5, 'base_channels': checkpoint['base_channels'],
        'calibration_cases': [r['case'] for r in train], 'validation_cases': [r['case'] for r in validation],
        'test_used': False, 'engine': 'x86', 'torch_version': str(torch.__version__),
        'qconfig': str(prepared.qconfig), 'calibration_device': str(device), 'integer_inference_device': 'cpu',
        'policy': 'Post-training static quantization, no optimizer/no retraining. All training images only. '
                  'No validation/test observer updates. No layer-selection search based on scores.',
        'engineering_warning_gates': {'max_mean_dice_drop': .005, 'max_small_slice_dice_drop': .03,
                                     'warning': 'Project screening thresholds, NOT clinical acceptance criteria'},
        'outside_int8_core': ['HU input preprocessing', 'dequantized logits sigmoid',
                              'native-grid bilinear probability resize', 'threshold and visualization'],
        'code_sha256': {name: file_sha256(Path(__file__).resolve().parent / name)
                        for name in ('lung_quantize.py', 'lung_unet.py', 'lungseg/metrics.py')}}
    write_json(output / 'protocol.json', protocol)
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        calibration = calibrate_training(prepared, source, root, train, device, args.batch_size,
                                         checkpoint['image_size'])
        write_json(output / 'calibration.json', calibration)
        quantized = convert_static(prepared)
        export = export_core(quantized, checkpoint['image_size'], output)
        save_quantized(quantized, output / 'int8_model.pt', checkpoint, protocol)
        restored, _ = restore_quantized(output / 'int8_model.pt')
        with mapped_npy(root / train[0]['cache_folder'] / 'images.npy') as images, torch.inference_mode():
            sample = torch.from_numpy(context_batch(images, [0], 1))
            torch.testing.assert_close(restored(sample), quantized(sample), rtol=0, atol=0)
        del restored, prepared, folded
        rows = validate_pair(source, quantized, root, validation, device, args, output)
        original_summary, int8_summary = quality_summary(rows, 'fp32'), quality_summary(rows, 'int8')
        dice_drop = original_summary['mean_case_dice'] - int8_summary['mean_case_dice']
        original_small = original_summary['slice_quality']['groups']['small']['mean_slice_dice']
        int8_small = int8_summary['slice_quality']['groups']['small']['mean_slice_dice']
        small_drop = original_small - int8_small if original_small is not None and int8_small is not None else None
        unchanged = file_sha256(args.checkpoint) == protocol['checkpoint_sha256']
        if not unchanged or file_sha256(args.dataset) != protocol['dataset_sha256']:
            raise ValueError('Source checkpoint/development manifest changed')
        report = {'status': 'complete_validation_ptq', 'scope': 'Development quantization verification; '
                  'NOT independent test, RTL bit-exact proof or FPGA inference',
                  'checkpoint_sha256': protocol['checkpoint_sha256'], 'source_checkpoint_unchanged': unchanged,
                  'cases': rows, 'validation_patients': len(rows), 'validation_slices': sum(r['slices'] for r in rows),
                  'fp32': original_summary, 'int8': int8_summary,
                  'mean_case_dice_drop': dice_drop, 'small_slice_dice_drop': small_drop,
                  'passes_predeclared_dice_warning_gates': dice_drop <= .005 and
                    (small_drop is None or small_drop <= .03),
                  'calibration': calibration, 'core_summary': {k: v for k, v in export.items() if k != 'layers'}}
        write_json(output / 'evaluation.json', report)
        with (output / 'case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case', 'slices', 'fp32_dice', 'int8_dice', 'dice_delta', 'different_pixels'])
            for row in rows:
                writer.writerow([row['case'], row['slices'], row['fp32']['dice'], row['int8']['dice'],
                                 row['dice_delta'], row['different_pixels']])
        print(f'Completed static PTQ: {output / "evaluation.json"}\n'
              f'FP32 Dice {original_summary["mean_case_dice"]:.6f}; '
              f'INT8 Dice {int8_summary["mean_case_dice"]:.6f}; drop {dice_drop:+.6f}', flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'aborted', 'error': str(error)})
        raise
    finally:
        torch.backends.quantized.engine = old_engine
        torch.backends.cudnn.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='FP32 folding check and calibration device; actual INT8 inference always CPU')
    return parser


if __name__ == '__main__':
    run_quantization(build_parser().parse_args())
