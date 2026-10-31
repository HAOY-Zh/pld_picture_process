"""Lung-guided five-plane YOLOv5n experiment, isolated from production models.

Uses pinned official YOLOv5 v7.0 model/CIoU loss. Sphere-derived 2D training
boxes are approximate labels, not expert contours. No sealed-test entrypoint.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'data/lung_nodule_yolo_20261020_v2'
REPO = ROOT/'data/third_party/yolov5_v7'
COMMIT = '915bbf294bb74c859f0b41f1c23bc395014ea679'
MANIFEST = ROOT/'data/luna16_expansion_20261020_v1/manifest.json'
LUNG = ROOT/'data/lung_training_20261011/early_stop_run/best.pt'
WEIGHTS = REPO/'yolov5n.pt'
CONTEXT = (-2.5, -1.25, 0., 1.25, 2.5)
SIZE = 384


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    def convert(item):
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,np.ndarray):return item.tolist()
        raise TypeError('Unsupported JSON value: '+str(type(item)))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,default=convert)+'\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def cohort(manifest, role):
    if role not in ('train', 'calibration'):
        raise ValueError('YOLO development accepts train/calibration only; sealed test refused')
    records = [r for r in manifest['cases'] if r['role'] == role]
    uids = [r['series_uid'] for r in records]
    if len(uids) != len(set(uids)):
        raise ValueError('Duplicate scan UID')
    forbidden = {r['series_uid'] for r in manifest['cases'] if r['role'] != role}
    if set(uids) & forbidden:
        raise ValueError('Overlapping cohort roles')
    return records


def upstream():
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    os.environ['YOLOv5_VERBOSE'] = 'false'
    if not REPO.is_dir():
        raise FileNotFoundError('Clone official YOLOv5 v7.0 into '+str(REPO))
    commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError('Unexpected YOLOv5 source revision')
    if subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'], text=True).strip():
        raise ValueError('Official YOLO source modified')
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(2)
    return torch


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def square_roi(body, spacing, size=SIZE, margin=5.):
    """Only predicted lungs and geometry determine the ROI; no annotations."""
    spacing = np.asarray(spacing, float)
    zany = np.flatnonzero(body.any(axis=(1,2)))
    if not len(zany):
        lo = np.array([0.,0.])
        hi = (np.array(body.shape[1:])-1)*spacing[1:]
        zlim = [0., (len(body)-1)*spacing[0]]
    else:
        ys = np.flatnonzero(body.any(axis=(0,2)))
        xs = np.flatnonzero(body.any(axis=(0,1)))
        lo = np.array([ys[0],xs[0]])*spacing[1:]-margin
        hi = np.array([ys[-1],xs[-1]])*spacing[1:]+margin
        zlim = [max(0.,zany[0]*spacing[0]-margin),
                min((len(body)-1)*spacing[0],zany[-1]*spacing[0]+margin)]
    side = float(max(256., max(hi-lo)))
    first = (lo+hi)/2-side/2
    return dict(y0_mm=float(first[0]), x0_mm=float(first[1]), side_mm=side,
                pixel_mm=side/size, size=size, z_limits_mm=zlim,
                empty_predicted_lung=not bool(len(zany)))


def z_grid(record, roi, step=1.25):
    z = np.arange(int(np.ceil((record['shape_zyx'][0]-1)*record['spacing_zyx_mm'][0]/step))+1)*step
    return z[(z >= roi['z_limits_mm'][0]-1e-7) & (z <= roi['z_limits_mm'][1]+1e-7)]


def plane_boxes(nodules, zm, roi, spacing):
    """Approximate sphere cross sections, normalized xywh. Not voxel masks."""
    result = []
    for n in nodules:
        center = np.asarray(n['center_index_zyx'])*spacing
        radius = max(0., float(n['diameter_mm'])/2)
        dz = float(zm-center[0])
        if radius <= 0 or abs(dz) >= radius:
            continue
        d = max(2*math.sqrt(max(0., radius*radius-dz*dz)), 2*roi['pixel_mm'])
        x = (center[2]-roi['x0_mm'])/roi['side_mm']
        y = (center[1]-roi['y0_mm'])/roi['side_mm']
        if not (0 <= x < 1 and 0 <= y < 1):
            continue
        w = d/roi['side_mm']
        # Clip the box to the input image while retaining its center consistently.
        x0,y0,x1,y1 = max(0.,x-w/2),max(0.,y-w/2),min(1.,x+w/2),min(1.,y+w/2)
        result.append([0.,(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0])
    return np.asarray(result, np.float32).reshape(-1,5)


def context_input(planes, zm, spacing_z, context=CONTEXT):
    output = []
    for offset in context:
        z = float(np.clip((zm+offset)/spacing_z, 0, len(planes)-1))
        lo, hi = int(math.floor(z)), int(math.ceil(z))
        a = planes[lo].astype(np.float32)
        if hi != lo:
            a = a*(1-(z-lo))+planes[hi]*(z-lo)
        output.append(a)
    return np.asarray(output,dtype=np.float32)/np.float32(255.)


def orient(image, labels, ignored, rotation, flip_x, flip_z):
    image = np.rot90(image, rotation, axes=(-2,-1))
    outputs = []
    for value in (labels, ignored):
        value = value.copy()
        for _ in range(rotation):
            x = value[:,1].copy()
            value[:,1] = value[:,2]
            value[:,2] = 1-x
            value[:,[3,4]] = value[:,[4,3]]
        if flip_x:
            value[:,1] = 1-value[:,1]
        outputs.append(value)
    if flip_x:
        image = image[:,:,::-1]
    if flip_z:
        image = image[::-1]
    return np.ascontiguousarray(image), outputs[0], outputs[1]


def build_model(plan, pretrained=False):
    torch = upstream()
    from models.yolo import DetectionModel
    import yaml
    cfg = yaml.safe_load((REPO/'models/yolov5n.yaml').read_text())
    cfg['anchors'] = plan['anchors']
    net = DetectionModel(cfg, ch=5, nc=1)
    transferred = 0
    if pretrained:
        # Trusted official Ultralytics release, hash recorded in the frozen plan.
        checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
        state = checkpoint['model'].float().state_dict()
        target = net.state_dict()
        for name in list(state):
            if name == 'model.0.conv.weight':
                state[name] = state[name].sum(dim=1, keepdim=True).repeat(1,5,1,1)/5
            if name not in target or state[name].shape != target[name].shape or 'anchors' in name:
                state.pop(name)
        net.load_state_dict(state, strict=False)
        transferred = len(state)
    hyp = yaml.safe_load((REPO/'data/hyps/hyp.scratch-low.yaml').read_text())
    hyp['obj'] *= (plan['image_size']/640)**2
    hyp['cls'] *= 1/80
    net.hyp = hyp
    net.names = ['nodule_candidate']
    return net, transferred


def ignored_objectness(base):
    """Retain upstream CIoU/anchor matching; mask uncertain findings in obj loss."""
    import torch
    from torch import nn
    base.reduction = 'none'

    class MaskedObjectness(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.ignored = None

        def forward(self, prediction, target):
            terms = self.base(prediction, target)
            valid = torch.ones_like(target, dtype=torch.bool)
            if self.ignored is not None:
                ny, nx = target.shape[-2:]
                # All anchors are ignored in the uncertain projected rectangle.
                for box in self.ignored:
                    b, _, x, y, w, h = box.tolist()
                    x0 = max(0, int(math.floor((x-w/2)*nx))-1)
                    y0 = max(0, int(math.floor((y-h/2)*ny))-1)
                    x1 = min(nx, int(math.ceil((x+w/2)*nx))+1)
                    y1 = min(ny, int(math.ceil((y+h/2)*ny))+1)
                    valid[int(b),:,y0:y1,x0:x1] = False
            valid |= target > 0  # Reference positives take precedence.
            return (terms*valid).sum()/valid.sum().clamp(min=1)
    return MaskedObjectness()


def freeze(out):
    from luna16_multibundle import read_manifest
    from lung_candidate_rules_run import EXCLUDED
    m = read_manifest(MANIFEST)
    training, calibration = cohort(m,'train'), cohort(m,'calibration')
    if len(training) != 247 or len(calibration) != 47:
        raise ValueError('Unexpected common comparison cohort')
    upstream()
    if not WEIGHTS.is_file():
        raise FileNotFoundError('Download official v7.0 yolov5n.pt first')
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve frozen YOLO experiment')
    source_paths = [Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_yolo.py',
                    MANIFEST, LUNG, WEIGHTS, EXCLUDED]
    source_paths += [ROOT/'python'/n for n in ('luna16_multibundle.py','lung_candidate_run.py',
        'lung_unet.py','lung_body_gate.py','lung_nodule_cnn.py','lung_candidate_froc.py')]
    source_paths += [REPO/n for n in ('models/yolo.py','models/common.py','utils/loss.py','models/yolov5n.yaml')]
    plan = dict(version='lung_guided_yolov5n_5plane_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seed=2026102010,upstream_commit=COMMIT,upstream_url='https://github.com/ultralytics/yolov5',
        image_size=SIZE,context_mm=CONTEXT,z_step_mm=1.25,roi_margin_mm=5.,
        train_uids=[r['series_uid'] for r in training],calibration_uids=[r['series_uid'] for r in calibration],
        training_references=sum(len(r['nodules']) for r in training),
        anchors=[[4,4,6,6,9,9],[12,12,16,16,22,22],[30,30,42,42,60,60]],
        epochs=30,batch_size=8,learning_rate=.0003,final_learning_rate=.00001,
        optimizer='AdamW',weight_decay=.0005,positive_repeats=4,random_planes_per_scan=32,
        preprocessing='predicted-lung square ROI; clipped HU[-1000,400] uint8 compact native-Z planes; physical linear Z sampling',
        labeling='approximate sphere cross-section boxes from centers/diameters; excluded findings masked in objectness loss',
        augmentation='XY right-angle rotation/reflection, Z reversal; no RGB HSV or mosaic',
        initialization='official COCO v7.0 YOLOv5n; first convolution channel sum/5; new one-class head and small-object anchors',
        checkpoint='fixed epoch30 EMA, no validation-based epoch selection',
        evaluation='all calibration47 entire CT; 2D NMS + physical 3D consolidation; same distance-based FROC; no Top-K',
        candidate_floor=.001,nms_iou=.3,merge_mm=2.5,
        limitations='adapted YOLOv5n, not full reproduction of published medical pipeline; different initialization/sampler/budget from 13k model',
        fpga_validated=False,sealed_test_scans_accessed=0,default_model_changed=False,
        protected_sha256={str(p):sha(p) for p in source_paths})
    out.mkdir(parents=True,exist_ok=True)
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n',encoding='utf-8')
    save(out/'environment.json',dict(python=sys.version,torch=__import__('torch').__version__,
         packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()))
    (out/'sources').mkdir(exist_ok=True)
    for p in source_paths:
        if p.suffix == '.py' and REPO not in p.parents:
            shutil.copy2(p,out/'sources'/p.name)
    print('YOLO FROZEN train247 refs341 / calibration47 / sealed56 untouched',flush=True)
    return plan,m


def verify(out):
    from luna16_multibundle import read_manifest
    plan = read(out/'plan.json')
    if sha(out/'plan.json') != (out/'plan.sha256').read_text().strip():
        raise ValueError('YOLO plan changed')
    for name,digest in plan['protected_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Protected YOLO input changed: '+name)
    m = read_manifest(MANIFEST)
    for role,key in [('train','train_uids'),('calibration','calibration_uids')]:
        if {r['series_uid'] for r in cohort(m,role)} != set(plan[key]):
            raise ValueError('YOLO cohort changed')
    return plan,m


def predicted_body(record,hu,geometry,lung_factory):
    uid = record['series_uid']
    p = ROOT/'data/lung_nodule_expanded_20261020_v1/patches'/(uid+'.mask.npz')
    meta = p.with_name(uid+'.json')
    if p.is_file() and meta.is_file():
        info = read(meta)
        if info['raw_sha256'] != record['raw_sha256'] or sha(p) != info['mask_packed_sha256']:
            raise ValueError('Expanded lung mask identity mismatch')
        with np.load(p,allow_pickle=False) as a:
            body = np.unpackbits(a['body'],count=int(np.prod(a['shape_zyx']))).reshape(a['shape_zyx']).astype(bool)
        if hashlib.sha256(body.tobytes()).hexdigest() != info['lung_mask_sha256']:
            raise ValueError('Lung pixel identity mismatch')
        return body,dict(source=str(p),sha256=sha(p),reused=True)
    from lung_candidate_run import predict_lungs
    raw,body,ms = predict_lungs(hu,geometry,lung_factory(),'cuda',16)
    del raw
    return body,dict(source='frozen original U-Net',model_sha256=sha(LUNG),milliseconds=ms,reused=False)


def prepare(out,role='train',limit=None):
    plan,m = verify(out)
    torch = upstream()
    from luna16_multibundle import load_development_volume
    from lung_candidate_run import excluded_findings
    from lung_candidate_rules_run import EXCLUDED
    from lung_unet import make_model
    import cv2
    lung = []
    def lung_factory():
        if not lung:
            cp = torch.load(LUNG,map_location='cpu',weights_only=True)
            net = make_model(cp['base_channels'],cp.get('context_slices',1)).cuda().eval()
            net.load_state_dict(cp['model']);lung.append(net)
        return lung[0]
    records = cohort(m,role)
    if limit is not None:
        records = records[:limit]
    excluded = excluded_findings(EXCLUDED)
    folder = out/('cache_'+role)
    folder.mkdir(exist_ok=True)
    rows = []
    for i,r in enumerate(records,1):
        uid = r['series_uid'];p = folder/(uid+'.npz');meta_path = folder/(uid+'.json')
        if meta_path.exists():
            meta = read(meta_path)
            if meta['plan_sha256'] != sha(out/'plan.json') or sha(p) != meta['cache_sha256']:
                raise ValueError('Changed prepared YOLO planes')
        else:
            if shutil.disk_usage(out).free < 3*1024**3:
                raise OSError('Less than 3 GiB free; stop before generating more cache')
            tick = perf_counter()
            hu,g = load_development_volume(m,uid,role)
            body,source = predicted_body(r,hu,g,lung_factory)
            roi = square_roi(body,g['spacing_zyx_mm'],plan['image_size'],plan['roi_margin_mm'])
            yy,xx = np.meshgrid((roi['y0_mm']+(np.arange(SIZE)+.5)*roi['pixel_mm'])/g['spacing_zyx_mm'][1],
                (roi['x0_mm']+(np.arange(SIZE)+.5)*roi['pixel_mm'])/g['spacing_zyx_mm'][2],indexing='ij')
            planes = np.empty((len(hu),SIZE,SIZE),np.uint8)
            for z in range(len(hu)):
                a = cv2.remap(hu[z].astype(np.float32),xx.astype(np.float32),yy.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1000)
                planes[z] = np.rint((a.clip(-1000,400)+1000)*(255/1400)).astype(np.uint8)
            with p.with_suffix('.tmp.npz').open('wb') as f:
                np.savez_compressed(f,planes=planes,body=np.packbits(body),shape_zyx=np.array(body.shape))
            os.replace(p.with_suffix('.tmp.npz'),p)
            ignored = []
            basis = np.asarray(g['basis_lps_zyx']);origin = np.asarray(g['origin_lps_mm'])
            for n in excluded.get(uid,[]):
                ignored.append(dict(n,center_index_zyx=np.linalg.solve(basis,np.asarray(n['center_lps_xyz_mm'])-origin).tolist(),
                    diameter_mm=n['diameter_mm'] if n['diameter_mm']>0 else 10.))
            grid = z_grid(r,roi,plan['z_step_mm'])
            positive = [float(z) for z in grid if len(plane_boxes(r['nodules'],z,roi,np.array(r['spacing_zyx_mm'])))]
            centers_in_roi = sum(roi['x0_mm'] <= n['center_index_zyx'][2]*r['spacing_zyx_mm'][2] < roi['x0_mm']+roi['side_mm']
                and roi['y0_mm'] <= n['center_index_zyx'][1]*r['spacing_zyx_mm'][1] < roi['y0_mm']+roi['side_mm']
                and roi['z_limits_mm'][0] <= n['center_index_zyx'][0]*r['spacing_zyx_mm'][0] <= roi['z_limits_mm'][1]
                for n in r['nodules'])
            meta = dict(series_uid=uid,role=role,raw_sha256=r['raw_sha256'],plan_sha256=sha(out/'plan.json'),
                cache_sha256=sha(p),cache_bytes=p.stat().st_size,roi=roi,geometry=g,shape_zyx=r['shape_zyx'],
                predicted_mask=source,positive_planes=positive,excluded=ignored,seconds=perf_counter()-tick,
                references=len(r['nodules']),reference_centers_in_roi=centers_in_roi)
            save(meta_path,meta)
            del planes,body,hu
            gc.collect()
        rows.append(dict(series_uid=uid,cache_bytes=meta['cache_bytes'],seconds=meta['seconds'],positive_planes=len(meta['positive_planes'])))
        save(out/('preparation_'+role+'.json'),dict(status='RUNNING',completed=i,expected=len(records),cases=rows,
            cache_bytes=sum(x['cache_bytes'] for x in rows),seconds=sum(x['seconds'] for x in rows),sealed_test_scans_accessed=0))
        print('YOLO PREP',role,i,'/',len(records),'sec',round(meta['seconds'],1),'MiB',round(meta['cache_bytes']/2**20,1),flush=True)
    save(out/('preparation_'+role+'.json'),dict(status='COMPLETE' if limit is None else 'LIMITED_PREFLIGHT',
        completed=len(rows),expected=len(cohort(m,role)),cases=rows,cache_bytes=sum(x['cache_bytes'] for x in rows),
        seconds=sum(x['seconds'] for x in rows),sealed_test_scans_accessed=0))


def batch_data(planes,meta,record,zs,augment):
    import torch
    images,labels,ignored = [],[],[]
    for b,zm in enumerate(zs):
        image = context_input(planes,zm,record['spacing_zyx_mm'][0])
        a = plane_boxes(record['nodules'],zm,meta['roi'],np.array(record['spacing_zyx_mm']))
        e = plane_boxes(meta['excluded'],zm,meta['roi'],np.array(record['spacing_zyx_mm']))
        if augment:
            image,a,e = orient(image,a,e,random.randrange(4),random.random()<.5,random.random()<.5)
        images.append(image)
        for collection,value in ((labels,a),(ignored,e)):
            if len(value):
                collection.append(np.column_stack((np.full(len(value),b),value)))
    return (torch.from_numpy(np.stack(images)),
            torch.from_numpy(np.concatenate(labels).astype(np.float32) if labels else np.zeros((0,6),np.float32)),
            torch.from_numpy(np.concatenate(ignored).astype(np.float32) if ignored else np.zeros((0,6),np.float32)))


def requests(record,meta,plan,epoch):
    digest = hashlib.sha256(f"{plan['seed']}:{epoch}:{record['series_uid']}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16],16))
    values = list(meta['positive_planes'])*plan['positive_repeats']
    grid = z_grid(record,meta['roi'],plan['z_step_mm'])
    if len(grid):
        values += rng.choice(grid,plan['random_planes_per_scan'],replace=len(grid)<plan['random_planes_per_scan']).tolist()
    else:
        values += [0.]*plan['random_planes_per_scan']
    rng.shuffle(values)
    return values


def make_loss(net):
    from utils.loss import ComputeLoss
    criterion = ComputeLoss(net)
    criterion.BCEobj = ignored_objectness(criterion.BCEobj)
    return criterion


def atomic_torch(path,value):
    import torch
    temp = path.with_suffix('.tmp.pt')
    torch.save(value,temp)
    os.replace(temp,path)


def smoke(out):
    plan,m = verify(out)
    torch = upstream();seed_all(plan['seed'])
    record = cohort(m,'train')[0]
    uid = record['series_uid'];meta = read(out/'cache_train'/(uid+'.json'))
    with np.load(out/'cache_train'/(uid+'.npz'),allow_pickle=False) as a:
        planes = a['planes']
    zs = meta['positive_planes'][:2] or [float(z_grid(record,meta['roi'])[0])]*2
    if len(zs)==1: zs *= 2
    x,y,ignored = batch_data(planes,meta,record,zs,False)
    net,transferred = build_model(plan,pretrained=True)
    net = net.cuda().train(); criterion = make_loss(net)
    criterion.BCEobj.ignored = ignored
    optimizer = torch.optim.AdamW(net.parameters(),lr=plan['learning_rate'])
    tick = perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):
        total,parts = criterion(net(x.cuda()),y.cuda())
    total.backward()
    finite = bool(torch.isfinite(total)) and all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
    if not finite:
        raise FloatingPointError('YOLO real-input forward/backward failed')
    optimizer.step();torch.cuda.synchronize()
    net.eval()
    with torch.inference_mode():
        predictions = net(x.cuda())[0]
    if predictions.shape[0]!=len(x) or predictions.shape[-1]!=6 or not torch.isfinite(predictions).all():
        raise ValueError('YOLO decoder output invalid')
    save(out/'smoke.json',dict(status='PASS',training_uid=uid,input_shape=list(x.shape),
        positive_labels=len(y),prediction_shape=list(predictions.shape),loss=float(total.detach()),
        components=parts.tolist(),parameters=sum(p.numel() for p in net.parameters()),transferred_tensors=transferred,
        seconds=perf_counter()-tick,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        updates_discarded=True,sealed_test_scans_accessed=0))
    print('YOLO SMOKE PASS',flush=True)


def train(out):
    plan,m = verify(out)
    if (out/'model_freeze.json').exists():
        if sha(out/'model.pt') != read(out/'model_freeze.json')['model_sha256']:
            raise ValueError('Completed YOLO model changed')
        return
    if read(out/'preparation_train.json')['status'] != 'COMPLETE':
        raise ValueError('Prepare all 247 training scans before fit')
    torch = upstream();seed_all(plan['seed'])
    from utils.torch_utils import ModelEMA
    net,_ = build_model(plan,pretrained=True)
    net.cuda().train();ema = ModelEMA(net)
    optimizer = torch.optim.AdamW(net.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    scaler = torch.amp.GradScaler('cuda',init_scale=32.,growth_interval=2**31-1)
    records = cohort(m,'train')
    metas = {r['series_uid']:read(out/'cache_train'/(r['series_uid']+'.json')) for r in records}
    for r in records:
        uid=r['series_uid']
        if sha(out/'cache_train'/(uid+'.npz')) != metas[uid]['cache_sha256']:
            raise ValueError('Training image cache changed')
    expected_per_epoch = sum(math.ceil(len(requests(r,metas[r['series_uid']],plan,1))/plan['batch_size']) for r in records)
    done,updates,history = 0,0,[]
    if (out/'resume.pt').exists():
        cp = torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        if cp['plan_sha256'] != sha(out/'plan.json'):
            raise ValueError('Wrong YOLO resume identity')
        net.load_state_dict(cp['state']);ema.ema.load_state_dict(cp['ema'])
        ema.updates=cp['updates'];optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        done,updates,history=cp['epoch'],cp['updates'],cp['history']
        random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng'])
        torch.cuda.set_rng_state_all(cp['cuda_rng'])
    criterion = make_loss(net)
    for epoch in range(done+1,plan['epochs']+1):
        tick = perf_counter();order=list(records)
        random.Random(plan['seed']+epoch).shuffle(order)
        loss_sum=np.zeros(3);batches=0;data_seconds=0.
        net.train()
        for r in order:
            uid=r['series_uid'];start=perf_counter()
            with np.load(out/'cache_train'/(uid+'.npz'),allow_pickle=False) as a:
                planes=a['planes']
            data_seconds += perf_counter()-start
            meta=metas[uid];zs=requests(r,meta,plan,epoch)
            for begin in range(0,len(zs),plan['batch_size']):
                x,y,ignored=batch_data(planes,meta,r,zs[begin:begin+plan['batch_size']],True)
                fraction=updates/max(1,expected_per_epoch*plan['epochs']-1)
                lr=plan['final_learning_rate']+.5*(plan['learning_rate']-plan['final_learning_rate'])*(1+math.cos(math.pi*fraction))
                lr*=min(1.,(updates+1)/expected_per_epoch)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                criterion.BCEobj.ignored=ignored
                with torch.autocast('cuda',dtype=torch.float16):
                    value,parts=criterion(net(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite YOLO loss')
                scaler.scale(value).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite YOLO gradients')
                scaler.step(optimizer);scaler.update();ema.update(net)
                updates+=1;batches+=1;loss_sum+=parts.detach().cpu().numpy()
                if batches % 100 == 0:
                    save(out/'training_progress.json',dict(status='RUNNING',epoch=epoch,epochs=plan['epochs'],
                        batches=batches,expected_batches=expected_per_epoch,updates=updates,history=history,
                        epoch_elapsed_seconds=perf_counter()-tick,sealed_test_scans_accessed=0))
                    print('YOLO TRAIN',epoch,'/',plan['epochs'],'batch',batches,'/',expected_per_epoch,flush=True)
            del planes
        torch.cuda.synchronize()
        if batches!=expected_per_epoch:raise ValueError('Unexpected number of training updates')
        seconds=perf_counter()-tick
        history.append(dict(epoch=epoch,batches=batches,losses=(loss_sum/batches).tolist(),seconds=seconds,
            cache_read_seconds=data_seconds,updates=updates))
        atomic_torch(out/'resume.pt',dict(plan_sha256=sha(out/'plan.json'),epoch=epoch,updates=updates,
            state=net.state_dict(),ema=ema.ema.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()))
        save(out/'training_progress.json',dict(status='RUNNING',epoch=epoch,epochs=plan['epochs'],updates=updates,
            history=history,training_eta_seconds=(plan['epochs']-epoch)*sum(h['seconds'] for h in history)/len(history),
            sealed_test_scans_accessed=0))
        print('YOLO EPOCH',epoch,'/',plan['epochs'],'seconds',round(seconds,1),'losses',history[-1]['losses'],flush=True)
    atomic_torch(out/'model.pt',dict(state=ema.ema.state_dict(),plan_sha256=sha(out/'plan.json'),epoch=plan['epochs']))
    save(out/'model_freeze.json',dict(status='COMPLETE',model_sha256=sha(out/'model.pt'),plan_sha256=sha(out/'plan.json'),
        parameters=sum(p.numel() for p in net.parameters()),epochs=history,optimizer_updates=updates,
        expected_updates=expected_per_epoch*plan['epochs'],checkpoint='last epoch EMA',fpga_validated=False,
        sealed_test_scans_accessed=0))
    progress=read(out/'training_progress.json');progress['status']='COMPLETE';save(out/'training_progress.json',progress)
    print('YOLO TRAIN COMPLETE',flush=True)


def detections_to_points(prediction,zm,roi,geometry,floor=.001,iou=.3):
    """Per-plane one-class NMS without arbitrary candidate-count truncation."""
    import torch
    from torchvision.ops import nms
    prediction=prediction.detach().float()
    score=prediction[:,4]*prediction[:,5]
    prediction=prediction[score>=floor];score=score[score>=floor]
    if not len(score):return [],[],[]
    xy=prediction[:,:2];wh=prediction[:,2:4]
    boxes=torch.cat((xy-wh/2,xy+wh/2),1)
    keep=nms(boxes,score,iou)
    xy=xy[keep].cpu().numpy();wh=wh[keep].cpu().numpy();scores=score[keep].cpu().numpy()
    spacing=np.array(geometry['spacing_zyx_mm'])
    points=np.column_stack((np.full(len(xy),zm/spacing[0]),
        (roi['y0_mm']+xy[:,1]*roi['pixel_mm'])/spacing[1],
        (roi['x0_mm']+xy[:,0]*roi['pixel_mm'])/spacing[2]))
    diameters=np.sqrt(wh[:,0]*wh[:,1])*roi['pixel_mm']
    return points,scores,diameters


def evaluate(out):
    plan,m=verify(out)
    frozen=read(out/'model_freeze.json')
    if sha(out/'model.pt')!=frozen['model_sha256']:raise ValueError('YOLO model identity mismatch')
    torch=upstream()
    from lung_nodule_cnn import DetectorConfig, consolidate, search_mask
    from lung_candidate_froc import evaluate as froc
    from lung_candidate_run import excluded_findings
    from lung_candidate_rules_run import EXCLUDED
    net,_=build_model(plan);net.cuda().eval()
    net.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['state'])
    records=cohort(m,'calibration');exclusions=excluded_findings(EXCLUDED)
    folder=out/'calibration';folder.mkdir(exist_ok=True)
    rows=[];predictions={};timings=[]
    for i,r in enumerate(records,1):
        uid=r['series_uid'];result_path=folder/(uid+'.json')
        if result_path.exists():
            result=read(result_path)
            if result['model_sha256']!=frozen['model_sha256'] or result['raw_sha256']!=r['raw_sha256']:
                raise ValueError('Wrong cached YOLO evaluation')
        else:
            tick=perf_counter()
            meta=read(out/'cache_calibration'/(uid+'.json'))
            with np.load(out/'cache_calibration'/(uid+'.npz'),allow_pickle=False) as a:
                planes=a['planes'];body=np.unpackbits(a['body'],count=int(np.prod(a['shape_zyx']))).reshape(a['shape_zyx']).astype(bool)
            input_seconds=perf_counter()-tick;g=meta['geometry']
            search=search_mask(body,np.array(g['spacing_zyx_mm']),DetectorConfig())
            grid=z_grid(r,meta['roi'],plan['z_step_mm'])
            points=[];scores=[];diameters=[]
            torch.cuda.synchronize();start=perf_counter()
            with torch.inference_mode():
                for begin in range(0,len(grid),plan['batch_size']):
                    zs=grid[begin:begin+plan['batch_size']]
                    x=np.stack([context_input(planes,z,g['spacing_zyx_mm'][0]) for z in zs])
                    output=net(torch.from_numpy(x).cuda())[0]
                    for pred,z in zip(output,zs):
                        p,s,d=detections_to_points(pred,z,meta['roi'],g,plan['candidate_floor'],plan['nms_iou'])
                        if len(s):points.extend(p);scores.extend(s);diameters.extend(d)
            torch.cuda.synchronize();network_seconds=perf_counter()-start
            points=np.asarray(points,float).reshape(-1,3);scores=np.asarray(scores);diameters=np.asarray(diameters)
            indices=np.rint(points).astype(int)
            keep=((indices>=0)&(indices<np.array(body.shape))).all(1)
            in_lung=np.zeros(len(points),bool)
            in_lung[keep]=search[tuple(indices[keep].T)]
            start=perf_counter()
            candidates=consolidate(points[in_lung],scores[in_lung],diameters[in_lung],g,DetectorConfig(merge_mm=plan['merge_mm']))
            result=dict(series_uid=uid,model_sha256=frozen['model_sha256'],raw_sha256=r['raw_sha256'],
                candidates=candidates,processed_center_planes=len(grid),input_seconds=input_seconds,
                sample_transfer_cnn_decode_nms_seconds=network_seconds,merge_seconds=perf_counter()-start,
                source='five-plane 2D YOLO detections merged in 3D; not contour segmentation',
                fpga_validated=False,candidate_limit=None)
            save(result_path,result)
            del planes,body,search
        candidates=result['candidates']
        rows.append(dict(uid=uid,world=np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3),
            record=r,excluded=exclusions.get(uid,[])))
        predictions[uid]=np.array([c['score'] for c in candidates])
        timings.append(result['sample_transfer_cnn_decode_nms_seconds'])
        save(out/'evaluation_progress.json',dict(status='RUNNING',completed=i,expected=len(records),sealed_test_scans_accessed=0))
        print('YOLO CALIBRATION',i,'/',len(records),'candidates',len(candidates),flush=True)
    report=froc(rows,predictions)
    save(out/'evaluation.json',dict(status='COMPLETE',model_sha256=frozen['model_sha256'],froc=report,
        mean_sampling_transfer_cnn_decode_nms_seconds=float(np.mean(timings)),calibration_is_independent_test=False,
        timing_limit='RTX4060 Laptop GPU + PC; cached input; excludes original ZIP input, U-Net, web; not FPGA',
        fpga_validated=False,sealed_test_scans_accessed=0,default_model_changed=False))
    save(out/'evaluation_progress.json',dict(status='COMPLETE',completed=len(records),expected=len(records),sealed_test_scans_accessed=0))
    print('YOLO CALIBRATION COMPLETE',json.dumps(report['operating_points']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','prepare','smoke','train','evaluate','run'])
    p.add_argument('--outdir',type=Path,default=OUT)
    p.add_argument('--role',choices=['train','calibration'],default='train')
    p.add_argument('--limit',type=int)
    args=p.parse_args();out=args.outdir.resolve()
    if args.command=='freeze':freeze(out)
    elif args.command=='prepare':prepare(out,args.role,args.limit)
    elif args.command=='smoke':smoke(out)
    elif args.command=='train':train(out)
    elif args.command=='evaluate':evaluate(out)
    else:
        try:
            if not (out/'plan.json').exists():freeze(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='prepare_train',sealed_test_scans_accessed=0))
            prepare(out,'train')
            if not (out/'smoke.json').exists():smoke(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='train',sealed_test_scans_accessed=0))
            train(out)
            save(out/'pipeline.json',dict(status='RUNNING',stage='prepare_calibration',sealed_test_scans_accessed=0))
            prepare(out,'calibration')
            save(out/'pipeline.json',dict(status='RUNNING',stage='evaluate',sealed_test_scans_accessed=0))
            evaluate(out)
            save(out/'pipeline.json',dict(status='COMPLETE',stage='complete',sealed_test_scans_accessed=0))
        except Exception as error:
            save(out/'pipeline.json',dict(status='FAILED',error=repr(error),sealed_test_scans_accessed=0))
            raise


if __name__=='__main__':
    main()
