"""Export the frozen 445-scan FP32 detector for the existing flow-v1 PL bit.

Calibration reads TRAIN patches only. No test labels/images, no retraining.
Export is experimental until full detection quality and physical-board checks.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import SCHEMA,TOPOLOGY,multiplier,validate,quantize_hu,lower_stem,dequantize_head,sha
from nodule_integer_reference import forward
from lung_nodule_cnn import model,config_from_dict

MODEL=ROOT/'data/lung_nodule_five_bundle_20261021_v1/model.pt'
EXPECTED_SHA='c1fefe55a74d1cdc81ede2f2e249190aef7d0b71a8a81378c950427ddcd8dc06'
FLOW=ROOT/'data/lung_flow_20261017'
LAYERS={'stem0':'stem.0.0','stem1':'stem.1.0','enc1':'enc1.1.0','enc2':'enc2.1.0',
        'dec1':'dec1.0','dec0':'dec0.0','head':'head'}


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def training_patches(count):
    catalog=ROOT/'data/lung_nodule_five_bundle_20261021_v1/training_catalogue.json'
    data=json.loads(catalog.read_text(encoding='utf-8'))
    plan=json.loads((catalog.parent/'plan.json').read_text(encoding='utf-8'))
    training=set(plan['training_scans'])
    items=data['items']
    if any(x['series_uid'] not in training for x in items): raise ValueError('Non-training item in calibration source')
    # Half lesion-near patches and half background, spread deterministically over CTs.
    groups=[[x for x in items if bool(x['label']['positive'])==positive] for positive in [True,False]]
    chosen=[]
    for group in groups:
        for i in np.linspace(0,len(group)-1,count//2,dtype=int): chosen.append(group[i])
    patches=[];receipts=[]
    for item in chosen:
        p=Path(item['image_path']);a=np.load(p,mmap_mode='r',allow_pickle=False)
        patch=np.array(a[item['patch_index']],copy=True)
        if patch.shape!=(5,128,128) or patch.dtype!=np.int16: raise ValueError('Unexpected training patch')
        patches.append(patch)
        receipts.append(dict(series_uid=item['series_uid'],patch_index=item['patch_index'],path=str(p),
            patch_sha256=__import__('hashlib').sha256(patch.tobytes()).hexdigest(),positive=bool(item['label']['positive'])))
    return np.stack(patches),receipts,sha(catalog)


def prepare(out,count):
    if count<16 or count%2: raise ValueError('Even calibration count >=16')
    out=Path(out).resolve()
    if out.exists() or out.parent!=ROOT/'data': raise ValueError('Use a fresh workspace/data directory')
    if sha(MODEL)!=EXPECTED_SHA: raise ValueError('Wrong nodule checkpoint')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    cp=torch.load(MODEL,map_location='cpu',weights_only=True);cfg=config_from_dict(cp['config'])
    net=model(cfg).to(device).float().eval();net.load_state_dict(cp['state'])
    out.mkdir();pack=out/'nodule_board';pack.mkdir();tick=time.perf_counter()
    patches,receipts,catalog_sha=training_patches(count)
    maxima={name:0. for name in LAYERS if name!='head'};head_max=np.zeros(5)
    handles=[]
    for name,path in LAYERS.items():
        # Observe after ReLU rather than the signed conv boundary for hidden layers.
        module=net.get_submodule(path if name=='head' else path.rsplit('.',1)[0])
        def observe(module,args,result,name=name):
            nonlocal head_max
            if name=='head': head_max=np.maximum(head_max,result.detach().abs().amax((0,2,3)).cpu().numpy())
            else: maxima[name]=max(maxima[name],float(result.detach().amax()))
        handles.append(module.register_forward_hook(observe))
    predictions=[]
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():
        for start in range(0,len(patches),16):
            a=torch.as_tensor(patches[start:start+16].astype(np.float32),device=device).clamp(-1000,400).add(1000).div(1400)
            predictions.append(net(a).cpu().numpy())
    for h in handles:h.remove()
    fp=np.concatenate(predictions)
    scales={n:max(v/255,1e-7) for n,v in maxima.items()}
    # Five signed grids: center logit does not consume the regression precision.
    head_scales=np.maximum(head_max/127,1e-7)
    plan=dict(schema=SCHEMA,input_shape=[45,64,64],input_zero=0,input_scale=1/255.,
        original_input_shape=[5,128,128],head_scales=head_scales.tolist(),nodes=[],commands=0,
        checkpoint_sha256=EXPECTED_SHA,detector_config=cp['config'],
        head_fields=['center_logit','dx','dy','dz','log_diameter_over_6mm'],
        deployment_status='experimental_int8_not_detection_quality_accepted')
    available={'input':([45,64,64],0,1/255.)};params={}
    for name,kind,sources,cout,kernel in TOPOLOGY:
        ishape,iz,inscale=available[sources[0]];cin,h,w=ishape
        if kind=='conv':
            conv=net.get_submodule(LAYERS[name]);weights=conv.weight.detach().cpu().numpy()
            if name=='stem0':weights=weights.reshape(8,45,1,1)
            ws=np.maximum(np.abs(weights).max((1,2,3))/127,1e-12)
            qw=np.rint(weights/ws[:,None,None,None]).clip(-127,127).astype(np.int8)
            bs=np.rint(conv.bias.detach().cpu().numpy()/(inscale*ws))
            if not np.isfinite(bs).all() or np.max(np.abs(bs))>=2**31:raise OverflowError('Bias exceeds INT32')
            oscale=head_scales if name=='head' else scales[name];oz=128 if name=='head' else 0
            ratios=inscale*ws/oscale;ms=[multiplier(r) for r in ratios]
            params.update({name+'_weight':qw,name+'_bias':bs.astype(np.int32),
                name+'_multiplier':np.array([r[0] for r in ms],np.int32),name+'_shift':np.array([r[1] for r in ms],np.int32)})
            shape=[cout,h,w];row=dict(name=name,prefix=name,kernel=kernel,padding=kernel//2,
                input_shape=ishape,input_zero=iz,output_shape=shape,output_zero=oz,relu=name!='head',
                input_scale=float(inscale),output_scale=np.asarray(oscale).tolist())
            node=dict(name=name,kind=kind,sources=sources,shape=shape,zero=oz,layer=row)
            plan['commands']+=(cout+7)//8
        elif kind=='pool':
            oz,oscale=iz,inscale;shape=[cout,h//2,w//2]
            node=dict(name=name,kind=kind,sources=sources,shape=shape,zero=oz,quant=dict(za=iz,zo=oz))
            plan['commands']+=1
        else:
            bshape,bz,bs=available[sources[1]];oz=0;oscale=max(inscale,bs)
            ma,sa=multiplier(inscale/oscale);mb,sb=multiplier(bs/oscale);shape=[cout,h*2,w*2]
            node=dict(name=name,kind=kind,sources=sources,shape=shape,zero=oz,
                quant=dict(za=iz,zb=bz,zo=oz,ma=ma,sa=sa,mb=mb,sb=sb))
            plan['commands']+=1
        plan['nodes'].append(node);available[name]=(shape,oz,oscale)
    validate(plan,params);save(pack/'nodule_model.json',plan);np.savez(pack/'nodule_parameters.npz',**params)
    # Independent integer references for four real TRAIN patches, retained FP32 maps.
    refs={};errors=[]
    for i in [0,1,len(patches)//2,len(patches)//2+1]:
        codes,trace,acc=forward(plan,params,lower_stem(quantize_hu(patches[i])),device,capture=True)
        pred=dequantize_head(codes,plan)
        for name,v in trace.items(): refs[f'case{len(errors)}_{name}']=v
        refs[f'case{len(errors)}_hu']=patches[i];refs[f'case{len(errors)}_fp32']=fp[i]
        errors.append(dict(training_patch=i,mae_by_channel=np.abs(pred-fp[i]).mean((1,2)).tolist(),
            max_abs_by_channel=np.abs(pred-fp[i]).max((1,2)).tolist()))
        if i==0:np.savez(out/'simulation_trace.npz',**trace,**{n+'_acc':v for n,v in acc.items()})
        print('Integer reference patch',len(errors),'PASS',flush=True)
    np.savez(pack/'nodule_reference.npz',**refs)
    save(out/'calibration.json',dict(train_only=True,test_used=False,patches=len(patches),
        training_scans=len(set(r['series_uid'] for r in receipts)),catalog_sha256=catalog_sha,
        samples=receipts,hidden_scales=scales,head_scales=head_scales.tolist(),patch_diagnostics=errors,
        warning='Patch diagnostics only; not a sensitivity/FP-per-CT acceptance or speed measurement'))
    shutil.copy2(ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json',pack/'nodule_ranker.json')
    # The existing flow bit already implements every lowered graph operation.
    for name in ['lung_network.bit','lung_network.hwh']:
        shutil.copy2(FLOW/'lung_flow'/name,pack/name)
    names=['nodule_plan.py','nodule_pl.py','nodule_cpu.py','nodule_benchmark.py','nodule_detect.py',
        'lung_axil.py','lung_axil_smoke.py','lung_continuous.py','lung_fast.py','lung_fast_io.cpp',
        'lung_fast_io.h','lung_model_layers_test.py','lung_tensor.py','lung_cpu_native.cpp','lung_cpu_kernel.h']
    for name in names:shutil.copy2(ROOT/'pynq'/name,pack/name)
    for name in ['lung_nodule_cnn.py','lung_nodule_sampling_cache.py','lung_nodule_continuity.py','lung_nodule_continuity_boundary.py']:
        shutil.copy2(ROOT/'python'/name,pack/name)
    shutil.copytree(FLOW/'rtl',pack/'rtl')
    manifest=dict(schema='nodule_board_package_v1',model_sha256=EXPECTED_SHA,board_tested=False,
        bit_reused=True,clock_mhz=100.,calibration_sha256=sha(out/'calibration.json'),
        commands_per_patch=plan['commands'],patches_for_board_check=4,
        note='Experimental. Verify exact head AND timing on PYNQ-Z2. Clinical detection quality after INT8 is pending.',
        files_sha256={p.relative_to(pack).as_posix():sha(p) for p in pack.rglob('*') if p.is_file()})
    save(pack/'nodule_manifest.json',manifest)
    save(out/'preparation.json',dict(status='exported_experimental_int8',elapsed_sec=time.perf_counter()-tick,
        source_sha256={str(ROOT/'pynq'/n):sha(ROOT/'pynq'/n) for n in names},
        model_sha256=sha(MODEL),package_manifest_sha256=sha(pack/'nodule_manifest.json')))
    print('Prepared:',pack,'commands:',plan['commands'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--calibration-patches',type=int,default=128);a=p.parse_args();prepare(a.out,a.calibration_patches)
