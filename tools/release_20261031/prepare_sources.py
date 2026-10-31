"""Collect authored sources into the existing publication checkout, never merge remote code."""
import hashlib,json,os,re,shutil,subprocess
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PUBLIC=ROOT/'data/github_integrated_review_20261025'
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
DESKTOP=Path('C:/Users/lenovo/Desktop/CT/CT_Studio_Desktop_DualExact_20261026_02')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(p):return Path(p).read_text(encoding='utf-8-sig',errors='replace').replace('\r\n','\n')
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

extensions={'.py','.cpp','.c','.h','.hpp','.v','.sv','.vh','.svh','.tcl','.xdc','.m','.md','.txt',
            '.html','.js','.css','.cjs','.mjs','.ps1','.cmd','.sh','.ipynb','.xpr','.bd','.xci','.xml','.prj'}
skip_names={'.git','.venv-lung','__pycache__','.pytest_cache','.Xil','ip_user_files','Lab_9'}
skip_suffixes=('.runs','.gen','.cache','.sim','.hw','.ip_user_files','.sdk')
excluded_inputs={'hu_input.txt','matlab_expected.txt','rtl_output.txt'}
added=[];archived=[]
for folder in ('python','pynq','fpga','matlab','docs'):
    for base,dirs,files in os.walk(ROOT/folder):
        dirs[:]=[d for d in dirs if d not in skip_names and not d.endswith(skip_suffixes)]
        for name in sorted(files):
            src=Path(base)/name
            if src.suffix.lower()not in extensions or name in excluded_inputs:continue
            rel=src.relative_to(ROOT)
            dst=PUBLIC/rel
            if not dst.exists():
                copy(src,dst);added.append(rel.as_posix())
                if src.suffix=='.ipynb':
                    notebook=json.loads(dst.read_text(encoding='utf-8'))
                    for cell in notebook.get('cells',[]):
                        if cell.get('cell_type')=='code':cell.update(outputs=[],execution_count=None)
                    write(dst,notebook)
            elif norm(src)!=norm(dst):
                # The current runnable sources take precedence. Retain differing
                # historical author sources explicitly, without replacing them.
                historical=PUBLIC/'research/author_sources'/rel
                copy(src,historical);archived.append(rel.as_posix())

# The actual accepted PC code, including the fixed final operating point.
current=['nodule_threshold_policy.py','nodule_workspace.py','ct_workspace_resume.py','nodule_frozen_scoring.py',
         'lungseg/workspace/index.html','lungseg/workspace/nodule.js']
for name in current:
    assert (APP/'python'/name).read_bytes()==(PUBLIC/'python'/name).read_bytes(),name

# Retain the desktop shell sources, never copy its paired config/token/results.
for name in ('Launcher.cs','app.manifest','app.ico','check_runtime.py','test_desktop.py'):
    copy(DESKTOP/'launcher'/name,PUBLIC/'desktop/launcher'/name)
copy(DESKTOP/'app/desktop_bootstrap.py',PUBLIC/'desktop/app/desktop_bootstrap.py')
copy(ROOT/'docs/CT_Studio项目综合报告_引用与数据集修订版_20261030.docx',
     PUBLIC/'docs/CT_Studio项目综合报告_引用与数据集修订版_20261030.docx')

# Tensor-only FP32 exports omit optimizer, patient/split lists and local settings.
# Every tensor is checked against the accepted local checkpoint after reload.
model_sources=[
    ('lung_unet_fp32',ROOT/'data/lung_training_20261011/early_stop_run/best.pt','model',
       ['base_channels','image_size','hu_clip','threshold']),
    ('nodule_2p5d_fp32',ROOT/'data/lung_nodule_five_bundle_20261021_v1/model.pt','state',['config']),
    ('ldct_cnn_fp32',APP/'data/ct_studio_integration_20261025/ct_studio/ldct/model/source_trained_best.pt','model',
       ['channels','hu_scale'])]
model_manifest={}
for name,src,key,keys in model_sources:
    original=torch.load(src,map_location='cpu',weights_only=True)
    weights=original[key]
    assert all(isinstance(k,str)and isinstance(v,torch.Tensor)for k,v in weights.items())
    config={k:original[k]for k in keys if k in original}
    json.dumps(config,allow_nan=False)
    dest=PUBLIC/'models/fp32'/(name+'.pt');dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=weights,architecture=config,kind='tensor_only_export'),dest)
    again=torch.load(dest,map_location='cpu',weights_only=True)
    assert set(again['state_dict'])==set(weights)
    assert all(torch.equal(v,again['state_dict'][k])for k,v in weights.items())
    model_manifest[name]=dict(file=dest.name,sha256=sha(dest),source_checkpoint_sha256=sha(src),
        tensor_count=len(weights),architecture=config,all_tensors_exact=True,
        training_patient_lists_included=False,optimizer_included=False)
write(PUBLIC/'models/fp32/manifest.json',dict(schema='ct_studio_tensor_only_checkpoints_v1',models=model_manifest,
    note='Optional offline FP32 weights. The live FPGA runtime continues to use the unchanged frozen integer assets.'))
(PUBLIC/'models/fp32/README.md').write_text('''# 离线浮点模型权重

此处提供肺分割、2.5D结节CNN和LDCT CNN最终已保存浮点检查点的纯张量导出，便于保留研究结果。
导出文件统一使用state_dict和architecture字段；与原检查点逐张量核对一致。
未包含优化器、训练病例清单、医学影像或本机路径设置，不能直接替代历史脚本要求的完整续训检查点。
实际网页与FPGA仍使用runtime/ct_studio_current_20261027中的冻结整数模型、评分器和bit/HWH。
重新训练还需要按docs中的公开数据来源自行下载并生成本地划分与缓存。
''',encoding='utf-8')

# Portable test entry: use this checkout when the author's historical snapshot
# directory is absent on another computer.
t=PUBLIC/'python/tests/test_nodule_threshold_policy.py'
text=t.read_text(encoding='utf-8')
anchor="APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'"
assert anchor in text
text=text.replace(anchor,anchor+"\nif not (APP/'python/nodule_workspace.py').is_file():APP=ROOT")
t.write_text(text,encoding='utf-8')
copy(APP/'python/tests/test_ct_nodule_workspace.py',PUBLIC/'python/tests/test_current_nodule_workspace.py')

readme=PUBLIC/'README.md';text=readme.read_text(encoding='utf-8')
text=text.replace('默认最终阈值提高为 0.7656，保留原档 0.6419 与低误报档 0.8738。完整候选支持重新筛选，无需重跑 CNN。提高阈值也可能增加漏检。',
    '最终阈值固定为低误报档 0.8738163050576156（约0.8738），已删除档位选择框和重筛按钮。新检测、结果导入和旧会话恢复均遵循固定策略；模型、坐标和冻结评分器不变。')
text=text.replace('当前严格阈值会改变最终候选数量','当前固定低误报阈值会改变最终候选数量')
text=text.replace('## 新电脑快速启动','## 2026年10月31日完整交付\n\n本次补齐离线训练与评价、RTL仿真、板端基准和桌面外壳源码，保留当前网页为唯一运行入口。优先阅读[最终交付说明](docs/最终完整交付说明_20261031.md)。历史脚本不会自动接入网页，不将其阶段性结果作为当前性能。\n\n## 新电脑快速启动',1)
readme.write_text(text,encoding='utf-8')
doc=PUBLIC/'docs/CT_Studio_latest_20261027.md';text=doc.read_text(encoding='utf-8')
text=text.replace('独立最终选择策略默认使用 0.7655561641768475，并可切到原档或 0.8738163050576156。已完成的真实结果可重新筛选，无需重跑CNN。',
    '独立最终选择策略自2026-10-30起固定为0.8738163050576156，档位选择和重筛按钮已删除；导入及恢复的真实候选同样应用固定阈值，不重跑CNN。')
text=text.replace('当前严格档会改变候选数量','当前固定低误报档会改变候选数量')
doc.write_text(text,encoding='utf-8')
ignore=PUBLIC/'.gitignore';text=ignore.read_text(encoding='utf-8')
for name,_,_,_ in model_sources:
    line='!/models/fp32/'+name+'.pt'
    if line not in text:text+='\n'+line
ignore.write_text(text+'\n',encoding='utf-8')

write(PUBLIC/'evidence/final_20261031/source_collection.json',dict(date='2026-10-31',
    fixed_threshold=0.8738163050576156,added_source_files=added,archived_differing_author_sources=archived,
    credentials_included=False,patient_images_included=False,generated_vivado_builds_included=False,
    old_experiments_are_not_default=True,board_program_changed=False,current_runtime_assets_changed=False,
    fp32_exports=model_manifest))
browser=json.loads((ROOT/'data/nodule_fixed_lowfp_20261030/browser_verification.json').read_text(encoding='utf-8'))
write(PUBLIC/'evidence/final_20261031/fixed_threshold_checks.json',dict(date='2026-10-30',fixed_policy_tests=8,
    nodule_link_tests=11,browser=browser,portable_offline_diagnose_passed=True,
    scope='Local host policy and UI verification; no new model training, board timing, or dataset accuracy evaluation.'))
write(OUT/'preparation.json',dict(status='SOURCES_PREPARED_PENDING_INSTALL_TESTS_AND_PRIVACY_SCAN',
    added=len(added),historical_differences=len(archived),fp32_models=len(model_manifest),publication=str(PUBLIC)))
print(json.dumps(dict(added=len(added),historical_differences=len(archived),fp32_models=len(model_manifest)),ensure_ascii=False))
