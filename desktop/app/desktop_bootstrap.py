"""Portable PC launcher. Uses the existing paired board, never installs or resets it."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
CONFIG=ROOT/'data/ct_studio_integration_20261025/host_config.json'
TOKEN=''

class LaunchError(Exception):
    def __init__(self,code,message): self.code=code; super().__init__(message)

def safe(message):
    value=str(message)
    return value.replace(TOKEN,'[凭据已隐藏]') if TOKEN else value

def emit(event,message='',**extra):
    print(json.dumps(dict(event=event,message=safe(message),**extra),ensure_ascii=False),flush=True)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))

def validate_local():
    global TOKEN
    config=read(CONFIG)
    TOKEN=(CONFIG.parent/config['token_file']).read_text(encoding='utf-8').strip()
    if len(TOKEN)<32: raise LaunchError('FILES','配套凭据文件不完整，请重新解压完整上位机包。')
    board=CONFIG.parent/'ct_studio'
    installation=read(CONFIG.parent/'installation.json')
    if sha(board/'platform_manifest.json')!=installation['service_bundle_sha256']:
        raise LaunchError('FILES','板端配套资产身份不一致，请使用同一份完整部署包。')
    for name in ['platform_manifest.json','legacy_platform_manifest.json','ldct/manifest.json']:
        manifest=board/name
        for rel,expected in read(manifest)['files_sha256'].items():
            target=(manifest.parent/rel).resolve()
            if board not in target.parents or sha(target)!=expected:
                raise LaunchError('FILES','运行资产缺失或被修改，请重新解压完整上位机包。')
    if TOKEN!=(board/'platform_token.txt').read_text(encoding='utf-8').strip():
        raise LaunchError('FILES','电脑与运行资产的凭据不配套，请恢复原配套文件。')
    # Verify current accepted program files too. Relocated configuration is handled separately.
    source=read(ROOT/'source_provenance.json')
    for name,expected in source['validated_export_files_sha256'].items():
        if name in source['relocated_config_files']:continue
        if sha(ROOT/name)!=expected:
            raise LaunchError('FILES','程序文件缺失或被修改，请重新解压完整上位机包。')
    if (ROOT/config['ldct_release']).resolve()!=board/'ldct' or (ROOT/config['nodule_package']).resolve()!=CONFIG.parent:
        raise LaunchError('FILES','运行资产路径不完整，请保留整个解压目录。')
    compat=ROOT.parent/'runtime/compat_site'
    descriptor=compat/'compat_manifest.json'
    if sha(descriptor)!=source['runtime_compat_manifest_sha256']:
        raise LaunchError('RUNTIME','数值兼容环境不完整，请重新解压配套包。')
    for rel,expected in read(descriptor)['files_sha256'].items():
        target=(compat/rel).resolve()
        if compat not in target.parents or sha(target)!=expected:
            raise LaunchError('RUNTIME','数值兼容文件缺失或改变。')
    import numpy as np
    if np.__version__!='1.26.4' or compat not in Path(np.__file__).resolve().parents:
        raise LaunchError('RUNTIME','结节采样需要已验收的 NumPy 1.26.4 兼容环境。')
    from start_ct_studio_demo import configuration
    from ct_dual_io_client import CachedCanvasClient as SharedPatchClient
    from ldct_contract import validate_release
    from nodule_detector import NoduleCaseProvider
    client,manifest=configuration(CONFIG)
    if validate_release(ROOT/config['ldct_release'])[1]!=manifest['ldct']:
        raise LaunchError('IDENTITY','电脑 LDCT 资产与板端契约不匹配。')
    if manifest.get('dual_io_interface')!='ct_dual_exact_io_v3':
        raise LaunchError('IDENTITY','缺少已验收的 DualExact V3 配套身份。')
    provider=NoduleCaseProvider(CONFIG.parent,client)
    shared=SharedPatchClient(client.url,client.token,read(CONFIG.parent/'shared_host_platform.json'))
    return client,shared,manifest

def classify_exception(exc):
    if isinstance(exc,LaunchError): return exc
    if isinstance(exc,urllib.error.HTTPError):
        if exc.code in (401,403):
            return LaunchError('AUTH','已连接到板子，但配套凭据被拒绝。请使用与当前板端服务配对的部署包，无需重装或重启板子。')
        return LaunchError('BOARD','板端服务返回异常，请查看服务状态后点击“重新连接”。')
    if isinstance(exc,(ConnectionError,TimeoutError,urllib.error.URLError,OSError)):
        return LaunchError('NETWORK','没有连上板子。请检查板子电源、网线和电脑网段；板子地址为 192.168.2.99，电脑应能访问它的 8766 端口。')
    if isinstance(exc,(ImportError,ModuleNotFoundError)):
        return LaunchError('RUNTIME','内置运行环境不完整。请重新解压整个软件包，不要只复制 exe 文件。')
    return LaunchError('START','启动未完成：'+safe(exc))

def ready_once(client,shared):
    legacy=client.call('/v1/platform',timeout=4)
    identity=shared.call('/v1/shared/platform',timeout=4)
    if (legacy.get('platform_manifest_sha256')!=client.manifest['platform_manifest_sha256'] or
        identity.get('service_bundle_sha256')!=shared.bundle['service_bundle_sha256'] or
        identity.get('shared_nodule')!=shared.bundle['shared_nodule'] or
        identity.get('shared_batch')!=shared.bundle['shared_batch']):
        raise LaunchError('IDENTITY','已连接的 8766 服务与这份上位机不配套。请保留板端程序，核对配套包和服务身份。')
    if shared.bundle.get('dual_io_interface') and (
            identity.get('dual_io_interface')!=shared.bundle['dual_io_interface'] or
            identity.get('shared_segmentation')!=shared.bundle['shared_segmentation'] or
            identity.get('dual_io_startup')!='all-nodes-and-native-layout-exact'):
        raise LaunchError('IDENTITY','板端尚未通过新版分割与结节接口自检。')
    if legacy.get('error') or identity.get('error'):
        raise LaunchError('BOARD_FAULT','板端报告运行故障。请检查板端状态；上位机不会自动重载硬件或重复提交任务。')
    if legacy.get('busy') or identity.get('busy'):
        return False,'busy'
    if not legacy.get('ready') or not identity.get('ready'):
        return False,'starting'
    client.health();shared.health()
    return True,'ready'

def wait_ready(client,shared,seconds,existing=None,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+seconds
    while True:
        try:
            ok,state=ready_once(client,shared)
            if ok:return
            if state=='busy':
                if existing is not None:return
                emit('progress','板子正在处理已有任务，等待空闲后自动连接…',step=2)
            else:emit('progress','板端正在完成启动自检，等待就绪…',step=2)
        except urllib.error.HTTPError as exc: raise classify_exception(exc)
        except LaunchError:raise
        except ValueError as exc:raise LaunchError('IDENTITY','板端身份或启动自检与配套版本不一致。请核对服务后重试。') from exc
        except (urllib.error.URLError,ConnectionError,TimeoutError,OSError):
            emit('progress','正在等待板子连接，请确认电源和网线已接好…',step=2)
        if clock()>=deadline:
            raise LaunchError('NETWORK','等待板子就绪超时。请检查电源、网线及电脑网段；如有任务正在运行，请等待完成后点击“重新连接”。')
        sleep(min(2,max(0,deadline-clock())))

def choose_port(manifest):
    from start_ct_studio_demo import web_info
    settings=ROOT/'data/desktop_launcher/last_port.json'
    ports=[]
    try:ports.append(int(read(settings)['port']))
    except (OSError,ValueError,KeyError):pass
    ports+=list(range(8099,8110))
    for port in dict.fromkeys(ports):
        if not 8099<=port<=8109:continue
        try:
            info=web_info('http://127.0.0.1:'+str(port)+'/',manifest)
            if info is not None:
                state=info[2]
                release_id=read(CONFIG)['desktop_release_id']
                # Preserve other source/desktop copies even when their hardware
                # identities match. Reuse only our own accepted local backend.
                if state.get('desktop_release_id')!=release_id:continue
                if not settings.is_file():continue
                marker=read(settings)
                if (marker.get('port')!=port or marker.get('release_id')!=release_id or
                        marker.get('launch_root')!=str(ROOT.resolve())):continue
            return port,info
        except (ValueError,RuntimeError,urllib.error.URLError,TimeoutError,OSError):continue
    raise LaunchError('PORT','本机网页端口均被占用，请关闭其它上位机窗口后重试。已有进程已保留。')

def start(seconds):
    emit('progress','正在检查内置环境与配套运行资产…',step=1)
    client,shared,manifest=validate_local()
    import torch
    from plotly.offline import get_plotlyjs
    torch.set_num_threads(2)
    get_plotlyjs()
    port,existing=choose_port(manifest)
    emit('progress','正在连接板子 192.168.2.99…',step=2)
    wait_ready(client,shared,seconds,existing)
    emit('progress','板子已配对，正在启动处理界面…',step=3)
    import start_ct_studio_demo as launcher
    from types import SimpleNamespace
    # Existing launcher retains sessions and never installs/restarts the board owner.
    with contextlib.redirect_stdout(sys.stderr):
        code=launcher.launch(SimpleNamespace(config=CONFIG,port=port,check_only=False,
            no_browser=True,wait_seconds=0))
    if code:raise LaunchError('START','网页尚未连接成功，请稍后点击“重新连接”。原任务和板端程序均保留。')
    folder=ROOT/'data/desktop_launcher';folder.mkdir(parents=True,exist_ok=True)
    (folder/'last_port.json').write_text(json.dumps(dict(port=port,release_id=read(CONFIG)['desktop_release_id'],launch_root=str(ROOT.resolve()))),encoding='utf-8')
    emit('ready','已就绪，可以导入 CT 开始处理。',url='http://127.0.0.1:'+str(port)+'/',step=3)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--wait',type=int,default=300)
    args=parser.parse_args()
    try:
        if args.diagnose:
            emit('progress','正在检查内置环境与运行资产…',step=1)
            validate_local()
            import torch,scipy,pydicom,nibabel,skimage,matplotlib,plotly
            torch.set_num_threads(2)
            if torch.add(torch.tensor([2]),3).item()!=5:raise ValueError('Torch runtime failed')
            emit('diagnosed','离线环境与配套资产检查通过；未连接板子、未启动网页。')
        else:start(max(0,min(300,args.wait)))
        return 0
    except Exception as exc:
        failure=classify_exception(exc)
        emit('error',str(failure),code=failure.code)
        return 1

if __name__=='__main__':raise SystemExit(main())
