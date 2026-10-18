"""Offline expert/current 2D U-Net 3D comparison with native MPR and Slicer export.

Visualization only: no training, label editing, hole filling, surface smoothing,
component removal or mesh decimation. Original CT geometry is retained.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pydicom

from lung_segmentation import fresh_output, write_json
from lung_unet import (_case_prediction, file_sha256, load_cached_source,
    make_model, mapped_npy, read_manifest, select_device, torch_module)
from lungseg.dicom_io import orientation_vectors
from lungseg.metrics import overlap_metrics


def geometry(origin, orientation, spacing):
    origin, spacing = np.asarray(origin, float), np.asarray(spacing, float)
    if origin.shape != (3,) or spacing.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Need finite origin and positive native ZYX spacing')
    col, row, normal = orientation_vectors(orientation)
    # The viewer's native planes are called axial/coronal/sagittal only here.
    if min(abs(col[0]), abs(row[1]), abs(normal[2])) < .999:
        raise ValueError('This viewer requires axis-aligned axial CT; oblique data need explicit MPR resampling')
    basis = np.column_stack((normal * spacing[0], row * spacing[1], col * spacing[2]))
    return dict(origin_lps_mm=origin.tolist(), basis_lps_zyx=basis.tolist(),
                spacing_zyx_mm=spacing.tolist(), coordinates='DICOM LPS, millimetres')


def to_world(vertices_zyx, geom):
    return np.asarray(vertices_zyx) @ np.asarray(geom['basis_lps_zyx']).T + geom['origin_lps_mm']


def mesh(mask, geom):
    from skimage.measure import marching_cubes
    array = np.asarray(mask)
    if array.ndim != 3 or not np.isin(array, [0, 1, 255]).all():
        raise ValueError('Need a native binary 3-D mask')
    if not array.any():
        return dict(voxels=np.empty((0, 3), np.float32), world=np.empty((0, 3), np.float32),
                    faces=np.empty((0, 3), np.uint32))
    # Padding closes masks touching the acquisition boundary without editing GT.
    vertices, faces, _, _ = marching_cubes(np.pad(array != 0, 1).astype(np.uint8),
        level=.5, step_size=1, allow_degenerate=False)
    vertices = vertices - 1
    return dict(voxels=vertices.astype(np.float32), world=to_world(vertices, geom).astype(np.float32),
                faces=faces.astype(np.uint32))


def packed(array, binary=False):
    a = np.asarray(array)
    if binary:
        if not np.isin(a, [0, 1, 255]).all():
            raise ValueError('Binary packing requires a mask')
        raw = np.packbits((a != 0).ravel(), bitorder='little').tobytes()
        dtype = 'bits'
    else:
        target = np.dtype(a.dtype).newbyteorder('<')
        if target not in [np.dtype('<f4'), np.dtype('<i2'), np.dtype('<u4')]:
            raise ValueError('Unsupported viewer array type')
        raw = np.ascontiguousarray(a, dtype=target).tobytes()
        dtype = {np.dtype('<f4'): 'float32', np.dtype('<i2'): 'int16', np.dtype('<u4'): 'uint32'}[target]
    return dict(dtype=dtype, shape=list(a.shape), elements=int(a.size),
        gzip_base64=base64.b64encode(gzip.compress(raw, compresslevel=3, mtime=0)).decode('ascii'))


def hu_storage(hu):
    a = np.asarray(hu)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Need finite ZHW HU data')
    # This is lossless storage, NOT a conversion to display grayscale.
    if a.min() >= -32768 and a.max() <= 32767 and np.array_equal(a, np.rint(a)):
        return a.astype(np.int16)
    return a.astype(np.float32)


def nifti_export(array_zyx, geom, path):
    import nibabel as nib
    lps = np.eye(4)
    # NIfTI array axes are XYZ, while our arrays are native ZYX.
    lps[:3, :3] = np.asarray(geom['basis_lps_zyx'])[:, [2, 1, 0]]
    lps[:3, 3] = geom['origin_lps_mm']
    affine = np.diag([-1., -1., 1., 1.]) @ lps
    image = nib.Nifti1Image(np.asarray(array_zyx).transpose(2, 1, 0), affine)
    image.header.set_xyzt_units('mm')
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path)


def slice_metrics(prediction, expert, axis, index):
    scores = overlap_metrics(np.take(prediction, index, axis), np.take(expert, index, axis))
    return dict(index=index, reference_pixels=int(np.count_nonzero(np.take(expert, index, axis))),
                prediction_pixels=int(np.count_nonzero(np.take(prediction, index, axis))), **scores)


def payload(hu, expert, prediction, geom, meshes, metadata):
    if hu.shape != expert.shape or hu.shape != prediction.shape:
        raise ValueError('Native CT/reference/prediction grids must match')
    return dict(schema=1, shape=list(hu.shape), geometry=geom, metadata=metadata,
        hu=packed(hu_storage(hu)), expert=packed(expert, True), prediction=packed(prediction, True),
        meshes={name: {key: packed(value) for key, value in item.items()} for name, item in meshes.items()})


def html_export(data, path):
    from plotly.offline import get_plotlyjs
    template = Path(__file__).with_name('lungseg') / 'volume_viewer.html'
    text = template.read_text(encoding='utf-8')
    # Prevent a dataset title/string from escaping its JSON script element.
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    bundle = get_plotlyjs().replace('</script', '<\\/script')
    path.write_text(text.replace('__PAYLOAD__', encoded).replace('__PLOTLY__', bundle), encoding='utf-8')


def html_payload(text):
    marker = '<script id="dataset" type="application/json">'
    if text.count(marker) != 1:
        raise ValueError('Expected one embedded viewer dataset')
    raw, end, _ = text.split(marker, 1)[1].partition('</script>')
    if not end:
        raise ValueError('Unclosed embedded viewer dataset')
    data = json.loads(raw)
    if data.get('schema') != 1 or not all(k in data for k in ['shape', 'hu', 'expert', 'prediction', 'meshes', 'metadata']):
        raise ValueError('Not an exported native volume viewer')
    return raw, data


def refresh_html(path):
    """Replace only presentation; reuse exact embedded arrays/geometry/metrics."""
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() != '.html':
        raise ValueError('Expected an existing .html viewer')
    raw, data = html_payload(path.read_text(encoding='utf-8'))
    before = file_sha256(path)
    temporary = path.with_name(path.name + '.layout.tmp')
    if temporary.exists():
        raise ValueError('Layout temporary output already exists; inspect it first')
    html_export(data, temporary)
    updated, _ = html_payload(temporary.read_text(encoding='utf-8'))
    if updated != raw or file_sha256(path) != before:
        raise ValueError('Viewer dataset or source changed; original HTML was not replaced')
    after = file_sha256(temporary)
    temporary.replace(path)
    report = dict(status='presentation_only_updated', layout='four_panel_responsive',
        html_sha256_before=before, html_sha256_after=after,
        embedded_dataset_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        embedded_dataset_byte_identical=True, inference_rerun=False,
        template_sha256=file_sha256(Path(__file__).with_name('lungseg') / 'volume_viewer.html'),
        note='Original report.json is historical build provenance; this record identifies the updated HTML.')
    write_json(path.with_suffix('.layout.json'), report)
    print(f'Updated layout only: {path}; embedded dataset unchanged', flush=True)
    return report


def unpack_viewer(item):
    """Exact native transport decoder for integrity checks and optional exports."""
    raw = gzip.decompress(base64.b64decode(item['gzip_base64'], validate=True))
    if item['dtype'] == 'bits':
        expected = (item['elements']+7)//8
        if len(raw) != expected:
            raise ValueError('Binary transport length mismatch')
        a = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')[:item['elements']]
    else:
        dtype = {'int16':'<i2','float32':'<f4','uint32':'<u4'}[item['dtype']]
        a = np.frombuffer(raw,dtype)
    if len(a)!=item['elements'] or int(np.prod(item['shape']))!=item['elements']:
        raise ValueError('Transport shape mismatch')
    return a.reshape(item['shape'])


def cleanup_export(source, screen, outdir):
    """Add an OPTIONAL cleaned layer in a new HTML, never replace original data."""
    source, screen = Path(source).resolve(strict=True), Path(screen).resolve(strict=True)
    tracked={str(p):file_sha256(p) for p in [source,screen,Path(__file__).resolve(),
        Path(__file__).with_name('lungseg').joinpath('volume_viewer.html').resolve()]}
    _,data=html_payload(source.read_text(encoding='utf-8'))
    report=json.loads(screen.read_text(encoding='utf-8'))
    if (report.get('status')!='complete_development_screen' or
        not report.get('primary_accepted_for_optional_development_view') or
        not report.get('protected_inputs_unchanged')):
        raise ValueError('Completed accepted development screen required')
    if data['metadata']['checkpoint_sha256']!=report['protocol']['model_sha256']:
        raise ValueError('Viewer/model mismatch')
    rows=[r for r in report['cases'] if r['case']==data['metadata']['case'] and r['split']=='validation']
    if len(rows)!=1:
        raise ValueError('Matching development validation case required')
    row=rows[0]
    artifact=(screen.parent/row['case']/'predictions_native.npz').resolve(strict=True)
    if artifact.parent.parent!=screen.parent or file_sha256(artifact)!=row['prediction_sha256']:
        raise ValueError('Prediction artifact path/hash mismatch')
    tracked[str(artifact)]=row['prediction_sha256']
    expert=unpack_viewer(data['expert'])!=0
    with np.load(artifact) as archive:
        original=archive['original'];cleaned=archive['primary']
    if not np.array_equal(original,unpack_viewer(data['prediction'])!=0):
        raise ValueError('Screen original prediction differs from current viewer')
    if (original.shape!=cleaned.shape or list(original.shape)!=data['shape'] or
        not np.isin(cleaned,[0,1]).all() or (cleaned&~original).any() or
        not np.allclose(row['spacing_zyx_mm'],data['geometry']['spacing_zyx_mm'],atol=1e-6,rtol=0)):
        raise ValueError('Cleaned grid/spacing/subset mismatch')
    for name,mask in [('original',original),('primary',cleaned)]:
        actual=overlap_metrics(mask,expert)
        for key in ['dice','true_positive','false_positive','false_negative']:
            if actual[key]!=row['scores'][name][key]:
                raise ValueError('Screen score differs from embedded reference')
    # Critical assertion: no true lung was removed in this development example.
    if (expert&original&~cleaned).any():
        raise ValueError('Accepted cleanup unexpectedly removed expert-positive voxels')
    output=fresh_output(outdir)
    print('Exporting optional cleanup mesh and new offline viewer...',flush=True)
    cleaned_mesh=mesh(cleaned,data['geometry'])
    data['cleaned']=packed(cleaned,True)
    data['meshes']['cleaned']={k:packed(v) for k,v in cleaned_mesh.items()}
    volume_ml=int(cleaned.sum())*float(np.prod(data['geometry']['spacing_zyx_mm']))/1000
    data['metadata']['cleanup']=dict(metrics=overlap_metrics(cleaned,expert),prediction_volume_ml=volume_ml,
        volume_relative_difference=(volume_ml-data['metadata']['expert_volume_ml'])/data['metadata']['expert_volume_ml'],
        parameters=report['protocol']['primary'],cpu_analysis_ms=row['cpu_analysis_ms'],
        removed_components=row['scores']['primary']['removed_components'],
        removed_voxels=int((original&~cleaned).sum()),screen_sha256=tracked[str(screen)],
        warning='Optional conservative 3D postprocess, development validation only, not new neural model or PL.')
    html_export(data,output/'viewer.html')
    nifti_export(cleaned.astype(np.uint8),data['geometry'],output/'cleaned_mask.nii.gz')
    # Real full-resolution surfaces; no smoothing/decimation for a prettier image.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    meshes={name:{key:unpack_viewer(value) for key,value in item.items()}
            for name,item in data['meshes'].items() if name in ['expert','prediction']}
    meshes['cleaned']=cleaned_mesh
    limits=np.concatenate([m['world'] for m in meshes.values() if len(m['world'])])
    low,high=limits.min(0),limits.max(0)
    fig=plt.figure(figsize=(16,7))
    for j,(name,title,color) in enumerate([('expert','Expert','cyan'),('prediction','Original','#f6a33b'),('cleaned','Conservative cleanup','#6fd78d')],1):
        ax=fig.add_subplot(1,3,j,projection='3d');item=meshes[name]
        if len(item['faces']):
            ax.add_collection3d(Poly3DCollection(item['world'][item['faces']],facecolor=color,edgecolor='none',alpha=.8))
        ax.set_xlim(low[0]-5,high[0]+5);ax.set_ylim(low[1]-5,high[1]+5);ax.set_zlim(low[2]-5,high[2]+5)
        ax.set_box_aspect(high-low);ax.view_init(elev=15,azim=-75);ax.set_title(title)
        ax.set_xlabel('LPS X / mm');ax.set_ylabel('LPS Y / mm');ax.set_zlabel('LPS Z / mm')
    fig.suptitle(row['case']+' | full native surfaces, same axes | education only')
    fig.tight_layout();fig.savefig(output/'preview_3d.png',dpi=120);plt.close(fig)
    for path,expected in tracked.items():
        if file_sha256(path)!=expected:
            raise ValueError('Protected viewer/screen input changed: '+path)
    write_json(output/'export_report.json',dict(status='complete_optional_cleanup_viewer',input_sha256=tracked,
        original_viewer_unchanged=True,original_embedded_prediction_unchanged=True,default_variant='original',
        original_metrics=row['scores']['original'],cleanup=data['metadata']['cleanup'],
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in output.iterdir()}))
    print('Completed optional cleanup viewer: '+str(output/'viewer.html'),flush=True)


def preview(hu, expert, prediction, meshes, geom, index, path, case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 9))
    for j, name in enumerate(['expert', 'prediction'], 1):
        ax = fig.add_subplot(2, 2, j, projection='3d')
        item = meshes[name]
        faces = item['faces']
        if len(faces):
            triangles = item['world'][faces]
            collection = Poly3DCollection(triangles, facecolor='cyan' if name == 'expert' else '#f6a33b',
                                         edgecolor='none', alpha=.75)
            ax.add_collection3d(collection)
        limits = np.concatenate([m['world'] for m in meshes.values() if len(m['world'])])
        low, high = limits.min(0), limits.max(0)
        for method, k in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            method(low[k] - 5, high[k] + 5)
        ax.set_box_aspect(high - low)
        ax.view_init(elev=15, azim=-75)
        ax.set_xlabel('LPS X (mm)'); ax.set_ylabel('LPS Y (mm)'); ax.set_zlabel('LPS Z (mm)')
        ax.set_title('Expert native mask' if name == 'expert' else 'Retained 2D U-Net, computer inference')
        z = to_world([[index, 0, 0]], geom)[0, 2]
        corners = [[low[0], low[1], z], [high[0], low[1], z], [high[0], high[1], z], [low[0], high[1], z]]
        ax.add_collection3d(Poly3DCollection([corners], facecolor='magenta', edgecolor='magenta', alpha=.16))
    for j, title in [(3, 'CT + native expert/model contours'), (4, 'Red extra / blue missed / gray correct')]:
        ax = fig.add_subplot(2, 2, j)
        if j == 3:
            ax.imshow(hu[index], cmap='gray', vmin=-1000, vmax=400)
            for mask, color in [(expert[index], 'cyan'), (prediction[index], '#f6a33b')]:
                if mask.any(): ax.contour(mask, levels=[.5], colors=[color], linewidths=.8)
        else:
            rgb = np.zeros((*expert.shape[1:], 3))
            rgb[expert[index] & prediction[index]] = .6
            rgb[expert[index] & ~prediction[index]] = [0, .4, 1]
            rgb[~expert[index] & prediction[index]] = [1, 0, 0]
            ax.imshow(rgb)
        ax.axis('off'); ax.set_title(f'{title} | axial slice {index}')
    fig.suptitle(f'{case} | UNSMOOTHED label surfaces | education only\n'
                 'Open HTML for interactive full surfaces, MPR and clipping')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def build(args):
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Positive batch and thread counts required')
    root, manifest = read_manifest(args.dataset)
    records = [r for r in manifest['cases'] if r['case'] == args.case]
    if len(records) != 1:
        raise ValueError('Pick one existing development case; no automatic Test/external access')
    record = records[0]
    torch = torch_module()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint.get('context_slices', 1) != 1 or checkpoint.get('kind') == 'soft_level_v1' or
            checkpoint['dataset_fingerprint'] != manifest['fingerprint'] or
            checkpoint['image_size'] != manifest['image_size'] or checkpoint['hu_clip'] != [-1000, 400] or
            checkpoint['threshold'] != .5):
        raise ValueError('Retained original full-field 2D model and unchanged data required')
    files = [Path(args.dataset), Path(args.checkpoint), Path(__file__),
             Path(__file__).with_name('lungseg') / 'volume_viewer.html',
             root / record['cache_folder'] / 'images.npy', root / record['cache_folder'] / 'reference_native.npy']
    files += [Path(p) for p in record['source_ct_files_sorted']]
    tracked = {str(p.resolve()): file_sha256(p) for p in files}
    output = fresh_output(args.outdir)
    report = dict(status='running', case=record['case'], input_sha256=tracked,
                  original_model_retained=True, warning='Computer visualization, not PL inference or clinical use')
    write_json(output / 'report.json', report)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_num_threads(args.threads)
        device = select_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        print('Reading unchanged CT and native expert mask...', flush=True)
        hu = load_cached_source(record)
        with mapped_npy(root / record['cache_folder'] / 'reference_native.npy') as reference:
            expert = reference != 0
        header = pydicom.dcmread(record['source_ct_files_sorted'][0], stop_before_pixels=True)
        geom = geometry(header.ImagePositionPatient, header.ImageOrientationPatient, record['spacing_zyx_mm'])
        if args.slice < 0 or args.slice >= len(hu):
            raise ValueError('Initial slice outside the case')
        print('Running retained original model (no training)...', flush=True)
        model = make_model(checkpoint['base_channels'], 1).to(device)
        model.load_state_dict(checkpoint['model']); model.eval()
        prediction, inference_ms = _case_prediction(model, root, record, device, args.batch_size)
        del model
        scores = overlap_metrics(prediction, expert)
        factor = float(np.prod(geom['spacing_zyx_mm']) / 1000.)
        expert_ml = int(expert.sum()) * factor
        prediction_ml = int(prediction.sum()) * factor
        print('Reconstructing FULL native surfaces, without smoothing/fill/removal...', flush=True)
        start = perf_counter()
        meshes = {name: mesh(mask, geom) for name, mask in [('expert', expert), ('prediction', prediction)]}
        reconstruction_ms = (perf_counter() - start) * 1000
        metadata = dict(case=record['case'], split=record['split'], initial_slice=args.slice,
            model='Retained Small 2D U-Net, original epoch ' + str(checkpoint['epoch']),
            checkpoint_sha256=file_sha256(args.checkpoint), metrics=scores,
            expert_volume_ml=expert_ml, prediction_volume_ml=prediction_ml,
            volume_relative_difference=(prediction_ml - expert_ml) / expert_ml if expert_ml else None,
            computer_inference_ms=inference_ms, native_reconstruction_ms=reconstruction_ms,
            inference_device=str(device), full_resolution_mesh=True, smoothing=False,
            hole_filling=False, component_removal=False, mesh_decimation=False,
            disclaimer='公开数据；仅用于工程教学与算法演示，不作为临床诊断依据。电脑推理与展示，不是板端性能。')
        print('Exporting offline HTML and native NIfTI files...', flush=True)
        html_export(payload(hu, expert, prediction, geom, meshes, metadata), output / 'viewer.html')
        nifti_export(hu_storage(hu), geom, output / 'ct_hu.nii.gz')
        nifti_export(expert.astype(np.uint8), geom, output / 'expert_mask.nii.gz')
        nifti_export(prediction.astype(np.uint8), geom, output / 'prediction_mask.nii.gz')
        np.savez_compressed(output / 'surfaces_native.npz',
            expert_vertices_lps=meshes['expert']['world'], expert_faces=meshes['expert']['faces'],
            prediction_vertices_lps=meshes['prediction']['world'], prediction_faces=meshes['prediction']['faces'])
        preview(hu, expert, prediction, meshes, geom, args.slice, output / 'preview.png', args.case)
        metadata['selected_axial_slice'] = slice_metrics(prediction, expert, 0, args.slice)
        for path, expected in tracked.items():
            if file_sha256(path) != expected:
                raise ValueError(f'Protected original input changed: {path}')
        report.update(status='complete_native_3d_comparison', metadata=metadata,
            protected_inputs_unchanged=True, mesh_counts={name: dict(vertices=len(item['world']), faces=len(item['faces']))
                for name, item in meshes.items()},
            outputs={p.name: dict(bytes=p.stat().st_size, sha256=file_sha256(p)) for p in output.iterdir() if p.name != 'report.json'})
        write_json(output / 'report.json', report)
        print(f'Completed: {output / "viewer.html"} | whole-volume Dice={scores["dice"]:.6f}', flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete', error=str(error)); write_json(output / 'report.json', report); raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--checkpoint', default='data/lung_training_20261011/early_stop_run/best.pt')
    p.add_argument('--case', default='LCTSC-Train-S3-008')
    p.add_argument('--slice', type=int, default=72)
    p.add_argument('--outdir')
    p.add_argument('--refresh-html', help='Update existing HTML presentation without inference or data changes')
    p.add_argument('--cleanup-screen',help='Accepted island-cleanup evaluation.json for an optional comparison export')
    p.add_argument('--source-viewer',help='Existing original viewer.html to preserve in optional comparison')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    return p


if __name__ == '__main__':
    cli = parser()
    args = cli.parse_args()
    if args.cleanup_screen:
        if not args.source_viewer or not args.outdir or args.refresh_html:
            cli.error('--cleanup-screen needs --source-viewer and --outdir, not --refresh-html')
        cleanup_export(args.source_viewer,args.cleanup_screen,args.outdir)
    elif args.refresh_html:
        if args.outdir:
            cli.error('--refresh-html and --outdir cannot be combined')
        refresh_html(args.refresh_html)
    else:
        if not args.outdir:
            cli.error('--outdir is required when exporting a new case')
        build(args)
