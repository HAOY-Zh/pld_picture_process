"""Export/install a credential-free integration runtime without modifying a live board."""
import argparse,copy,hashlib,json,secrets,shutil,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'runtime/ct_studio_current_20261027'
CREDENTIALS=('platform_token.txt','segmentation/live_token.txt')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def bind_manifests(board,add_credentials=False):
    """Rebind the acyclic nested file manifests, leaving numerical assets byte-exact."""
    replacements={}
    for _ in range(30):
        changed=False
        for path in sorted(board.rglob('*.json'),key=lambda p:len(p.parts),reverse=True):
            value=read(path)
            def update(node):
                if isinstance(node,dict):
                    result={k:update(v)for k,v in node.items()if k!='files_sha256'}
                    if 'files_sha256'in node:
                        hashes={n:sha(path.parent/n)for n in node['files_sha256']if (path.parent/n).is_file()}
                        if path.name=='manifest.json' and path.parent.name=='ldct':
                            for p in sorted((path.parent/'fixtures').glob('*')):
                                if p.is_file():hashes[p.relative_to(path.parent).as_posix()]=sha(p)
                        if add_credentials and path.parent==board and path.name in ('platform_manifest.json','legacy_platform_manifest.json'):
                            for n in CREDENTIALS:hashes[n]=sha(board/n)
                        result['files_sha256']=hashes
                    return result
                if isinstance(node,list):return [update(x)for x in node]
                if isinstance(node,str):
                    seen=set()
                    while node in replacements and node not in seen:seen.add(node);node=replacements[node]
                return node
            updated=update(value)
            if updated!=value:
                before=sha(path);write(path,updated);replacements[before]=sha(path);changed=True
        if not changed:break
    else:raise ValueError('Manifest dependencies did not converge')
    return read(board/'platform_manifest.json')

def validate(board):
    sys.path[:0]=[str(board),str(board/'ldct/pynq'),str(board/'ldct/python')]
    import ct_studio_shared_service
    return ct_studio_shared_service.validate_package(board)

def build(source,out):
    source,out=source.resolve(),out.resolve()
    if out.exists():raise FileExistsError('Use a new release directory')
    old=read(source/'platform_manifest.json');out.mkdir(parents=True)
    staging=out/'staging';board=staging/'ct_studio';board.mkdir(parents=True)
    keep={'.py','.cpp','.h','.v','.sv','.vh','.tcl','.xdc','.xml','.ps1','.hex','.json','.npz','.bit','.hwh'}
    for name,expected in old['files_sha256'].items():
        src=source/name
        if sha(src)!=expected:raise ValueError('Changed accepted source: '+name)
        if (name in CREDENTIALS or name.startswith(('ldct/fixtures/','ldct/verification/','ldct/samples/'))
                or (src.suffix not in keep and 'LICENSE'not in src.name.upper())):continue
        dst=board/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copy2(source/'platform_manifest.json',board/'platform_manifest.json')
    # Replace every historical LDCT image fixture with deterministic analytic inputs.
    from ctdenoise.cnn_fixed import load_quantized,infer_fixed
    weights,spec=load_quantized(board/'ldct/model');fixtures=board/'ldct/fixtures';fixtures.mkdir()
    cases=[]
    for name,shape,method in [('synthetic_67x69',(67,69),'cnn'),('synthetic_row',(1,69),'cnn'),
            ('synthetic_endpoints',(1,7),'cnn'),('synthetic_bypass',(17,19),'bypass')]:
        y,x=np.indices(shape);image=((x*47+y*71)%2400-1200).astype(np.int16)
        if name=='synthetic_endpoints':image=np.array([[-32768,-1024,-1,0,1,1024,32767]],np.int16)
        expected=image.copy()if method=='bypass'else infer_fixed(image,weights,spec)[0]
        np.save(fixtures/(name+'_input.npy'),image,allow_pickle=False)
        np.save(fixtures/(name+'_expected.npy'),expected,allow_pickle=False)
        cases.append(dict(name=name,method=method,input='fixtures/'+name+'_input.npy',expected='fixtures/'+name+'_expected.npy',shape=list(shape)))
    write(fixtures/'cases.json',cases)
    # Each delivery fixture has explicit provenance; patient images never enter this archive.
    write(fixtures/'synthetic_provenance.json',dict(kind='deterministic_analytic_integer_patterns',patient_data=False,
        random_source=False,reference='frozen integer CNN reference',physical_board_tested=False))
    for name in ('nodule_scoring.json','nodule_ranker.json'):
        shutil.copy2(source.parent/name,staging/name)
    manifest=bind_manifests(board)
    protected=['segmentation/model.json','segmentation/lung_network.bit','ldct/model/weights_integer.npz',
        'ldct/overlay/ct_denoise_tile.bit','shared_cnn/models/nodule/model.json',
        'shared_cnn/models/nodule/parameters.npz','shared_cnn/lung_network.bit']
    for name in protected:
        if (board/name).read_bytes()!=(source/name).read_bytes():raise ValueError('Numerical asset changed: '+name)
    validate(board)
    archive=out/'ct_studio_integration_runtime.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(staging).as_posix())
    files={p.relative_to(staging).as_posix():sha(p)for p in sorted(staging.rglob('*'))if p.is_file()}
    write(out/'asset_manifest.json',dict(schema='ct_studio_integration_release_v1',archive=archive.name,archive_sha256=sha(archive),
        files_sha256=files,credentials_included=False,patient_data_included=False,
        synthetic_fixtures_only=True,new_fixture_board_tested=False,numerical_assets_unchanged=True,
        source_service_bundle_sha256=sha(source/'platform_manifest.json'),
        note='Same accepted numerical assets; sanitized self-tests and generated credentials require recipient startup acceptance.'))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),patient_data=False,credentials=False),indent=2))

def install(release,out,url):
    from ct_studio_runtime import validate_board_url
    url=validate_board_url(url);out=out.resolve();release=release.resolve();meta=read(release/'asset_manifest.json')
    if out.exists():raise FileExistsError('Preserve existing deployments; choose a new output directory')
    archive=release/meta['archive']
    if sha(archive)!=meta['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
    with zipfile.ZipFile(archive)as z:
        names=z.namelist()
        if len(names)!=len(set(names))or set(names)!=set(meta['files_sha256']):raise ValueError('Unexpected archive members')
        for name in names:
            path=PurePosixPath(name)
            if path.is_absolute()or '..'in path.parts or '\\'in name:raise ValueError('Unsafe archive path')
        out.mkdir(parents=True);z.extractall(out)
    for name,expected in meta['files_sha256'].items():
        if sha(out/name)!=expected:raise ValueError('Extracted asset SHA256 mismatch: '+name)
    board=out/'ct_studio';token=secrets.token_hex(32)+'\n'
    for name in CREDENTIALS:(board/name).write_text(token,encoding='utf8')
    (out/'platform_token.txt').write_text(token,encoding='utf8')
    manifest=bind_manifests(board,add_credentials=True);validate(board)
    legacy=read(board/'legacy_platform_manifest.json');legacy_sha=sha(board/'legacy_platform_manifest.json')
    current_fields={k:manifest[k]for k in ('shared_nodule','shared_batch','shared_segmentation',
        'canvas_interface','compact_interface','dual_io_interface','ldct_lossless_interface')if k in manifest}
    write(out/'host_platform.json',dict(legacy,platform_manifest_sha256=legacy_sha,
        service_bundle_sha256=sha(board/'platform_manifest.json'),**current_fields))
    write(out/'shared_host_platform.json',dict(service_bundle_sha256=sha(board/'platform_manifest.json'),
        legacy_contract_sha256=legacy_sha,**current_fields))
    shutil.copy2(board/'segmentation/model.json',out/'model.json')
    def relative(path):
        try:return path.relative_to(ROOT).as_posix()
        except ValueError:return path.as_posix()
    write(out/'host_config.json',dict(board_url=url,token_file='platform_token.txt',model='model.json',
        platform_manifest='host_platform.json',studio_service=True,ldct_release=relative(board/'ldct'),nodule_package=relative(out)))
    write(out/'installation.json',dict(status='FILES_AND_CONTRACTS_VERIFIED_NO_BOARD_ACCESSED',
        service_bundle_sha256=sha(board/'platform_manifest.json'),legacy_contract_sha256=legacy_sha,
        new_fixture_board_tested=False,credentials_generated_locally=True))
    print('INSTALLED:',out,'\nConfiguration:',out/'host_config.json','\nBoard package:',board,'\nCredentials generated locally; no board accessed.')

def verify(out):
    out=out.resolve();manifest=validate(out/'ct_studio');host=read(out/'host_platform.json');shared=read(out/'shared_host_platform.json')
    assert host['platform_manifest_sha256']==manifest['legacy_contract_sha256']
    assert shared['service_bundle_sha256']==sha(out/'ct_studio/platform_manifest.json')
    assert shared['shared_nodule']==manifest['shared_nodule']
    assert (out/'platform_token.txt').read_bytes()==(out/'ct_studio/platform_token.txt').read_bytes()
    print('VERIFY PASS: all files and nested contracts; no hardware accessed.')

if __name__=='__main__':
    p=argparse.ArgumentParser();commands=p.add_subparsers(dest='command',required=True)
    b=commands.add_parser('build');b.add_argument('--source',type=Path,required=True);b.add_argument('--out',type=Path,default=PUBLIC)
    i=commands.add_parser('install');i.add_argument('--release',type=Path,default=PUBLIC);i.add_argument('--output',type=Path,default=ROOT/'data/ct_studio_current_20261027');i.add_argument('--board-url',default='http://192.168.2.99:8766')
    v=commands.add_parser('verify');v.add_argument('--output',type=Path,default=ROOT/'data/ct_studio_current_20261027')
    a=p.parse_args()
    if a.command=='build':build(a.source,a.out)
    elif a.command=='install':install(a.release,a.output,a.board_url)
    else:verify(a.output)
