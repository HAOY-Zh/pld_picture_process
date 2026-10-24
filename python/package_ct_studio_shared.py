"""Fresh bundle: add reviewed shared Verilog runtime, preserve legacy assets/ABI."""
import argparse,ast,json,shutil,zipfile
from pathlib import Path
from shared_patch_contract import INTERFACE,INPUT_SHAPE,OUTPUT_SHAPE,digest,validate_assets,BATCH_INTERFACE,MAX_BATCH

ROOT=Path(__file__).resolve().parents[1]


def adapt_imports(source,local_names):
    lines=source.splitlines(keepends=True)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.ImportFrom) and not node.level and node.module in local_names:
            old='from '+node.module+' import';new='from .'+node.module+' import'
            line=lines[node.lineno-1]
            if old not in line:raise ValueError('Unsupported local import')
            lines[node.lineno-1]=line.replace(old,new,1)
        elif isinstance(node,ast.Import) and any(a.name in local_names for a in node.names):
            raise ValueError('Handle explicit package import before deployment: '+ast.unparse(node))
    result=''.join(lines);ast.parse(result)
    return result


def package(base,source,out):
    base,source,out=[Path(p).resolve() for p in (base,source,out)]
    if out.exists():raise FileExistsError(out)
    board=base/'ct_studio';old=json.loads((board/'platform_manifest.json').read_text())
    for name,h in old['files_sha256'].items():
        p=(board/name).resolve()
        if board not in p.parents or digest(p.read_bytes())!=h:raise ValueError('Changed legacy package '+name)
    frozen=json.loads((source/'shared_manifest.json').read_text())
    for name,h in frozen['files_sha256'].items():
        p=(source/name).resolve()
        if source not in p.parents or digest(p.read_bytes())!=h:raise ValueError('Changed reviewed shared source '+name)
    if frozen['clock_mhz']!=100:raise ValueError('Wrong frozen shared clock')
    shutil.copytree(board,out/'ct_studio',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=out/'ct_studio'
    shutil.copy2(board/'platform_manifest.json',target/'legacy_platform_manifest.json')
    shared=target/'shared_cnn';shared.mkdir()
    adaptations={};local_names={p.stem for p in source.glob('*.py')}
    for name in sorted(set(frozen['files_sha256'])|{'shared_manifest.json'}):
        p=source/name;q=shared/name;q.parent.mkdir(parents=True,exist_ok=True)
        raw=p.read_bytes()
        if p.suffix=='.py' and p.parent==source:
            result=adapt_imports(raw.decode('utf8'),local_names).encode('utf8')
            adaptations[name]=dict(source_sha256=digest(raw),adapted_sha256=digest(result),
                change='only package-relative imports' if result!=raw else 'unchanged')
        else:result=raw
        q.write_bytes(result)
    (shared/'__init__.py').write_text('"""Isolated reviewed shared-engine drivers; model/RTL bytes frozen."""\n')
    (shared/'import_adaptation.json').write_text(json.dumps(adaptations,indent=2)+'\n')
    for name in ('ct_studio_shared_service.py',):shutil.copy2(ROOT/'pynq'/name,target/name)
    for name in ('shared_patch_contract.py',):shutil.copy2(ROOT/'python'/name,target/name)
    owner=(ROOT/'pynq/ct_studio_roi_owner.py').read_text()
    owner=owner.replace('from ct_studio_roi_service import start','from ct_studio_shared_service import start')
    (target/'ct_studio_shared_owner.py').write_text(owner,encoding='utf8')
    model=json.loads((shared/'models/nodule/model.json').read_text())
    row=dict(interface=INTERFACE,clock_mhz=100,model_kind='nodule',input_shape=list(INPUT_SHAPE),
        output_shape=list(OUTPUT_SHAPE),commands_per_patch=15,
        bit='shared_cnn/lung_network.bit',hwh='shared_cnn/lung_network.hwh',
        model='shared_cnn/models/nodule/model.json',parameters='shared_cnn/models/nodule/parameters.npz',
        source_manifest='shared_cnn/shared_manifest.json',checkpoint_sha256=model['checkpoint_sha256'],
        detection_quality_accepted=False,whole_case_service_available=False)
    for key in ('bit','hwh','model','parameters','source_manifest'):row[key+'_sha256']=digest((target/row[key]).read_bytes())
    batch=dict(interface=BATCH_INTERFACE,max_patches=MAX_BATCH)
    manifest=dict(old,shared_nodule=row,shared_batch=batch,legacy_contract='legacy_platform_manifest.json',
        legacy_contract_sha256=digest((target/'legacy_platform_manifest.json').read_bytes()))
    manifest['files_sha256']={p.relative_to(target).as_posix():digest(p.read_bytes()) for p in sorted(target.rglob('*'))
        if p.is_file() and p.name!='platform_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    # Every legacy byte is pinned separately; new drivers cannot shadow old ones.
    for name,h in old['files_sha256'].items():
        if digest((target/name).read_bytes())!=h:raise ValueError('Preserved legacy artifact changed '+name)
    write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
    write(target/'platform_manifest.json',manifest);pin=digest((target/'platform_manifest.json').read_bytes())
    descriptor=dict(service_bundle_sha256=pin,legacy_contract_sha256=manifest['legacy_contract_sha256'],shared_nodule=row,shared_batch=batch)
    write(out/'shared_host_platform.json',descriptor)
    # Old web sessions and clients remain bound to the unchanged legacy contract.
    for name in ('host_platform.json','host_config.json','model.json','platform_token.txt'):
        shutil.copy2(base/name,out/name)
    validate_assets(target,row)
    archive=out/'ct_studio_shared_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(target).as_posix())
    evidence=dict(status='PACKAGED_BOARD_ACCEPTANCE_PENDING',service_bundle_sha256=pin,
        legacy_contract_sha256=manifest['legacy_contract_sha256'],shared_nodule=row,
        source_shared_manifest_sha256=digest((source/'shared_manifest.json').read_bytes()),
        legacy_files_preserved=len(old['files_sha256']),legacy_model_and_bits_unchanged=True,
        rtl_modified=False,bit_rebuilt=False,reviewed_bit_reused=True,import_adaptations=adaptations,
        archive_sha256=digest(archive.read_bytes()),physical_board_verified=False)
    write(out/'shared_integration.json',evidence)
    return dict(package=str(out),service_bundle_sha256=pin,archive_sha256=evidence['archive_sha256'],
        legacy_contract_sha256=manifest['legacy_contract_sha256'],legacy_files_preserved=len(old['files_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'data/ct_studio_roi_statistics_20261024_01')
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.base,a.shared,a.outdir),indent=2))
