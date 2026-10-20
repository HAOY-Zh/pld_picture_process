"""Competition PC launcher: validate the board, reuse sessions, never reset PL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib import error, request
import uuid
import webbrowser

from ct_studio_client import StudioClient
from lung_live_web import MODEL_SHA, NoRedirect

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=('loading','connecting','processing','computing','reconstructing')


def configuration(path):
    path=Path(path).resolve()
    if not path.is_file():
        raise ValueError('缺少本机运行配置：'+str(path)+'；先按集成运行说明恢复并生成运行包。')
    config=json.loads(path.read_text(encoding='utf-8'))
    model=path.parent/config['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError('分割模型契约不匹配，请保留并检查原运行包。')
    manifest=json.loads((path.parent/config['platform_manifest']).read_text(encoding='utf-8'))
    if (manifest.get('basic_transport')!='ct_basic_zlib_cache_v1'
            or manifest.get('ldct',{}).get('clock_mhz')!=100.):
        raise ValueError('竞赛入口要求已验收的共享基础版和100 MHz LDCT配置。')
    token=(path.parent/config['token_file']).read_text(encoding='utf-8').strip()
    return StudioClient(config['board_url'],token,manifest),manifest


def read_web_info(url):
    """Return None only when no listener exists; never close another server."""
    opener=request.build_opener(request.ProxyHandler({}),NoRedirect())
    try:
        with opener.open(url,timeout=3) as response:html=response.read(1024*1024).decode('utf-8')
    except error.URLError as exc:
        if isinstance(exc.reason,ConnectionRefusedError):return None
        if getattr(exc.reason,'winerror',None)==10061:return None
        raise RuntimeError('网页端口无响应；未停止或重启已有进程。') from exc
    match=re.search(r'<meta name="ui-token" content="([^"]+)"',html)
    if not match:raise ValueError('该端口已有其他程序，请使用另一端口；不会关闭已有程序。')
    headers={'X-UI-Token':match[1]}
    with opener.open(request.Request(url.rstrip('/')+'/api/state',headers=headers),timeout=5) as response:
        state=json.load(response)
    return opener,headers,state


def web_info(url,manifest):
    info=read_web_info(url)
    if info is None:return None
    opener,headers,state=info
    if (state.get('desktop_only') is not False or not state.get('fast_platform_active')
            or state.get('ldct',{}).get('identity')!=manifest['ldct']
            or not state['ldct'].get('fpga_configured')):
        raise ValueError('此端口不是匹配的实板集成网页；请保留旧会话，另选端口。')
    return opener,headers,state


def connect_web(url,info,manifest):
    opener,headers,state=info
    # Refreshing/opening the launcher must not disrupt any existing operation.
    if state.get('board_connected') or state['phase'] in ACTIVE or state['ldct'].get('running'):
        return
    req=request.Request(url+'/api/connect',data=b'{}',
        headers=dict(headers,**{'Content-Type':'application/json'}))
    with opener.open(req,timeout=10) as response:json.load(response)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        current=web_info(url,manifest)[2]
        if current.get('board_connected'):return
        if current['phase']=='error':raise RuntimeError(current.get('error','网页连接失败'))
        time.sleep(.2)
    raise TimeoutError('网页连接检查尚未完成；服务保留，请查看网页。')


def spawn_web(config,port,desktop=False):
    folder=ROOT/'data/ct_studio_demo_launcher'/uuid.uuid4().hex
    folder.mkdir(parents=True)
    task_env=os.environ.copy()
    temporary=folder/'tmp';temporary.mkdir()
    task_env.update(TEMP=str(temporary),TMP=str(temporary))
    flags=0
    if os.name=='nt':flags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
    with (folder/'web.stdout.log').open('xb') as out,(folder/'web.stderr.log').open('xb') as err:
        mode=['--desktop'] if desktop else ['--config',str(config)]
        process=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/'python/ct_workspace_web.py'),
            *mode,'--port',str(port),
            '--output',str(ROOT/('data/ct_workspace_pc/results' if desktop else 'data/ct_workspace_demo/results'))],
            cwd=ROOT,env=task_env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
    (folder/'web.pid').write_text(str(process.pid),encoding='ascii')
    return process,folder


def wait_board(client, seconds):
    """Poll readiness only; never retry an FPGA command or a reported fault."""
    deadline = time.monotonic() + seconds
    announced = False
    while True:
        try:
            return client.health()
        except error.HTTPError:
            raise
        except (error.URLError, TimeoutError, ConnectionError, OSError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if not announced:
                print('开发板正在启动或尚未连通，自动等待开机服务就绪（最多%d秒）…' % seconds, flush=True)
                announced = True
            time.sleep(min(3, remaining))


def launch_desktop(args):
    from roi_workspace import reference_identity
    identity=reference_identity()
    url=f'http://127.0.0.1:{args.port}/'
    def inspect():
        info=read_web_info(url)
        if info is not None:
            state=info[2]
            if state.get('desktop_only') is not True or state.get('roi',{}).get('identity')!=identity:
                raise ValueError('该端口已有其他版本或实板网页，已保留原会话；请选择另一端口。')
        return info
    existing=inspect()
    if args.check_only:
        print('CHECK PASS：电脑环境及ROI参考通过；未访问开发板、未启动网页。',flush=True)
        return 0
    if existing is None:
        process,logs=spawn_web(None,args.port,desktop=True)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('电脑网页未启动，查看：'+str(logs/'web.stderr.log'))
            existing=inspect()
            if existing is not None:break
            time.sleep(.4)
        if existing is None:raise TimeoutError('电脑网页启动尚未完成，日志：'+str(logs))
    else:
        print('复用现有电脑网页，保留当前病例、原图/增强图和结果。',flush=True)
    print('CT Studio 电脑端：'+url+'；ROI/基础参考/LDCT电脑参考可用，FPGA分割待接板。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    return 0


def launch(args):
    config=Path(args.config).resolve();url=f'http://127.0.0.1:{args.port}/'
    client,manifest=configuration(config)
    existing=web_info(url,manifest)
    print('正在检查 PYNQ 连接、运行包和真实启动自检…',flush=True)
    try:
        wait_board(client,0 if args.check_only else getattr(args,'wait_seconds',0))
    except Exception as exc:
        # An accepted owner doing an existing job is not a startup failure.
        # Only read its identity; never submit another operation or reload it.
        try:
            detail=client.call('/v1/platform',timeout=3)
            busy=(detail.get('schema')=='ct_platform_identity_v1'
                  and detail.get('platform_manifest_sha256')==manifest.get('platform_manifest_sha256')
                  and detail.get('ldct')==manifest['ldct']
                  and detail.get('basic')==manifest.get('basic')
                  and detail.get('integration_board_acceptance')=='startup-selftests-passed'
                  and detail.get('ready') is True and detail.get('busy') is True
                  and detail.get('error') is None)
        except Exception:busy=False
        if busy and existing is not None and not args.check_only:
            print('程序已经运行，开发板正在处理已有任务。已保留任务，请在原网页查看进度：'+url,flush=True)
            if not args.no_browser:webbrowser.open(url)
            return 0
        print('开发板尚未通过就绪检查：'+str(exc),flush=True)
        print('先给开发板上电、连接网线，等待开机服务完成自检；排查方法见运行说明。'
              '若有正在计算的任务，请等待；不会自动复位、重载或改用电脑计算。',flush=True)
        if existing is not None and not args.check_only and not args.no_browser:
            webbrowser.open(url)
            print('已打开原网页供查看，原任务和结果保留。',flush=True)
        return 1
    print('开发板已就绪：共享基础 + 100 MHz LDCT + 原分割模型。',flush=True)
    if args.check_only:
        print('CHECK PASS：仅检查，未启动网页、未提交计算。',flush=True)
        return 0
    if existing is None:
        process,logs=spawn_web(config,args.port)
        print('正在启动独立网页服务；日志：'+str(logs),flush=True)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('网页未启动，查看日志：'+str(logs/'web.stderr.log'))
            existing=web_info(url,manifest)
            if existing is not None:break
            time.sleep(.4)
        if existing is None:
            raise TimeoutError('网页启动检查超时；进程保留，查看日志：'+str(logs))
    else:
        print('复用现有网页会话，保留当前病例和全部结果。',flush=True)
    connect_web(url,existing,manifest)
    print('CT Studio 展示入口：'+url,flush=True)
    if not args.no_browser:webbrowser.open(url)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'data/ct_studio_ldct_fast_03/host_config.json')
    parser.add_argument('--desktop',action='store_true',help='电脑参考模式，不连接开发板；默认端口8096')
    parser.add_argument('--port',type=int,help='实板默认8095，电脑模式默认8096')
    parser.add_argument('--check-only',action='store_true',help='只读检查，不启动或打开网页')
    parser.add_argument('--no-browser',action='store_true',help='检查或启动服务，但不打开浏览器')
    parser.add_argument('--wait-seconds',type=int,default=300,help='等待板端开机就绪的最长秒数，默认300')
    args=parser.parse_args()
    if args.port is None:args.port=8096 if args.desktop else 8095
    if not 1<=args.port<=65535:parser.error('端口须为1至65535')
    if not 0<=args.wait_seconds<=900:parser.error('等待时间须为0至900秒')
    try:return launch_desktop(args) if args.desktop else launch(args)
    except Exception as exc:
        print('启动未完成：'+str(exc),flush=True)
        print('不会停止原进程或覆盖模型/结果。请查看 docs/CT_Studio_LDCT集成运行.md。',flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
