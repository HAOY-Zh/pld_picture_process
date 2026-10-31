"""CT workbench with explicit PC references and authenticated real basic PL.

The optional platform service owns ALL overlays and switches only when idle.
No neural CPU fallback, modified HU/model or benchmark-as-live metrics.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from urllib import parse

import numpy as np

from ct_basic_processing import OPERATIONS, morphology, parameters, process
from ct_platform_client import PlatformClient
from ct_nodule_inference import NoduleService, array_sha
from lung_live_web import BoardClient, Controller, MODEL_SHA, handler_for, write_new

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent/'lungseg'/'workspace'
ACTIVE = ('loading', 'connecting', 'computing', 'reconstructing', 'processing', 'detecting')
MORPHOLOGY = ('erosion', 'dilation', 'opening', 'closing')
FAST_CONFIG = ROOT/'data/ct_platform_fast_20261017_v2/host_config.json'


def pack_image(image):
    return dict(shape=list(image.shape), dtype='uint8',
        pixels=base64.b64encode(np.ascontiguousarray(image).tobytes()).decode('ascii'))


def benchmarks():
    """Only expose independently accepted historical numbers; no CT paths/secrets."""
    rows = []
    for title, folder, stem, experimental in [
        ('原模型 · flow', 'lung_flow_20261017', 'lung_flow_result', False),
        ('原模型 · 精确查表', 'lung_lut_20261017', 'lung_lut_result', False),
        ('剪枝恢复版 · flow', 'lung_prune_board_20261017', 'lung_pruned_result', True)]:
        path = ROOT/'data'/folder/(stem+'_verification.json')
        if not path.exists(): continue
        proof = json.loads(path.read_text(encoding='utf-8'))
        if (proof.get('status') != 'verified_returned_prepared_board_actual_head_and_profile'
                or proof.get('actual_head_codes_bit_exact') is not True): continue
        if any(hashlib.sha256((path.parent/n).read_bytes()).hexdigest() != h
               for n, h in proof['returned_files_sha256'].items()):
            continue
        rows.append(dict(title=title, cpu_ms=proof['cpu_ms'], pl_ms=proof['pl_ms'],
            speedup=proof['cpu_ms']/proof['pl_ms'], bit_sha256=proof['bit_sha256'],
            experimental=experimental, scope='同一验证切片重复20次；非本次网页测速'))
    return dict(rows=rows, note='历史同板 ARM INT8 / FPGA INT8；不代表当前服务版本、整例或网页端到端性能。')


class WorkspaceController(Controller):
    def __init__(self, client, model, output, nodule_model=None, nodule_service=None):
        super().__init__(client, model, output)
        self.basic_result = self.basic_image = None
        self.threshold_result = self.threshold_image = None
        self.history = []
        self.basic_condition = threading.Condition(self.lock)
        self.basic_request_id = None
        self.nodule_service = nodule_service or NoduleService(nodule_model)
        self.nodule_result = None
        self.nodule_generation = 0
        self.nodule_stage = None

    def _nodule_ready(self):
        if self.hu is None: return False, '先加载CT并完成肺分割'
        if len(self.hu) < 2 or self.details.get('partial_input'):
            return False, '请加载完整CT序列；单层或截取片段暂不执行结节检测'
        if self.known is None or not self.known.all(): return False, '先完成当前完整序列的肺分割'
        if set(self.receipts) != set(range(len(self.hu))): return False, '肺分割逐层来源记录不完整'
        return True, None

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            value['basic'] = copy.deepcopy(self.basic_result)
            value['threshold_input'] = copy.deepcopy(self.threshold_result)
            value['workspace_history'] = copy.deepcopy(self.history[-30:])
            value['basic_backend'] = 'FPGA_BASIC' if isinstance(self.client, PlatformClient) else 'PC_REFERENCE'
            value['basic_pl_configured'] = isinstance(self.client, PlatformClient)
            value['basic_pl_operations'] = list(self.client.manifest['basic']) if isinstance(self.client, PlatformClient) else []
            value['desktop_only'] = self.client is None
            value['fast_platform_available'] = self.client is not None and FAST_CONFIG.is_file()
            value['fast_platform_active'] = (isinstance(self.client, PlatformClient) and
                self.client.manifest.get('basic_transport') == 'ct_basic_zlib_cache_v1')
            ready, reason = self._nodule_ready()
            availability = self.nodule_service.availability()
            value['nodule'] = {**availability, 'ready':ready and availability['available'],
                'reason':reason or availability['reason'], 'stage':self.nodule_stage,
                'result':copy.deepcopy({k:v for k,v in self.nodule_result.items() if k != 'candidates'})
                    if self.nodule_result is not None else None}
            value['warning'] = '肺分割为实际FPGA结果；基础参考及肺结节检测为明确标注的电脑计算，二者耗时分别记录。'
            return value

    def submit(self, action, values):
        submitted_at = time.perf_counter()
        with self.lock:
            if self.status['phase'] in ACTIVE:
                raise BlockingIOError('任务正在运行，请等待完成；分割可在层间停止')
            if self.client is None and action in ('connect', 'connect_fast', 'run'):
                raise ValueError('当前为电脑浏览模式；真实FPGA分割需使用 --config 指定板端配置后重新启动')
            if action == 'nodule':
                if values.get('case_id') != self.status['case_id'] or self.hu is None:
                    raise ValueError('当前病例已变化，请重新读取状态')
                if values.get('backend') != 'PC_CNN_FP32' or set(values)-{'case_id','backend'}:
                    raise ValueError('请明确选择电脑FP32结节检测；模型和阈值不可由请求覆盖')
                ready, reason = self._nodule_ready()
                available = self.nodule_service.availability()
                if not ready or not available['available']: raise ValueError(reason or available['reason'])
                if self.nodule_result is not None: return self.snapshot()
                generation = self.nodule_generation; case_id = self.status['case_id']
                source = dict(mask_stage='body', segmentation_backend='FPGA_INT8',
                    segmentation_model_contract_sha256=MODEL_SHA, slice_receipts=copy.deepcopy(self.receipts),
                    source_slice_indices=list(self.details['source_slice_indices']), geometry=copy.deepcopy(self.geom))
                self.cancel.clear(); self.nodule_stage = '准备检测'
                self._update(phase='detecting', error=None, completed=0, total=1)
                self.nodule_thread = threading.Thread(target=self._nodule_work,
                    args=(self.hu,self.body,self.geom,self.case_folder,case_id,generation,source),daemon=True)
                self.nodule_thread.start()
                return self.snapshot()
            if action == 'connect_fast':
                self._update(phase='connecting', error=None, completed=0, total=0)
                threading.Thread(target=self._connect_fast, daemon=True).start()
                return self.snapshot()
            if action != 'basic':
                state = super().submit(action, values)
                if action in ('load','run'):
                    self.nodule_generation += 1; self.nodule_result = None; self.nodule_stage = None
                    return self.snapshot()
                return state
            if self.hu is None or values.get('case_id') != self.status['case_id']:
                raise ValueError('请先加载 CT；旧病例请求已失效')
            index = values.get('index')
            if type(index) is not int or not 0 <= index < len(self.hu):
                raise ValueError('切片索引越界')
            params = parameters(values)
            request_id = values.get('ui_request_id', uuid.uuid4().hex)
            if (not isinstance(request_id, str) or len(request_id) != 32 or
                    any(c not in '0123456789abcdef' for c in request_id)):
                raise ValueError('无效界面请求标识')
            mask_input = None
            if params['operation'] in MORPHOLOGY:
                source = self.threshold_result
                if (source is None or source['case_id'] != self.status['case_id'] or
                        source['index'] != index or values.get('mask_result_id') != source['result_id']):
                    raise ValueError('请先对当前切片执行 HU 范围阈值，再选择该二值结果进行形态学处理')
                # Freeze an explicitly selected result, not the current HU controls.
                params = dict(source['parameters'], operation=params['operation'])
                mask_input = (copy.deepcopy(source), self.threshold_image.copy())
            backend = values.get('backend', 'PC_REFERENCE')
            if backend not in ('PC_REFERENCE', 'FPGA_BASIC'): raise ValueError('无效执行位置')
            if backend == 'FPGA_BASIC':
                if not isinstance(self.client, PlatformClient):
                    raise ValueError('请使用统一PL服务配置启动网页；不会自动改用电脑计算')
                if not self.status['board_connected']: raise ValueError('请先连接统一FPGA服务')
                if params['operation'] not in self.client.manifest['basic']:
                    raise ValueError('该方法尚无现有PL模块，请明确选择电脑参考预览')
                if self.hu[index].shape != (512,512):
                    raise ValueError('现有基础PL要求原生512×512切片，不自动缩放')
            hu = self.hu[index].copy(); case_id = self.status['case_id']
            source_index = self.details['source_slice_indices'][index]
            folder = self.case_folder
            self.basic_request_id = request_id
            self._update(phase='processing', error=None, completed=0, total=1)
            threading.Thread(target=self._basic_work,
                args=(hu, index, source_index, case_id, folder, params, backend,
                      mask_input, request_id, submitted_at), daemon=True).start()
            return self.snapshot()

    def _connect_fast(self):
        """Explicit host connection to a pinned release. Does not load FPGA."""
        try:
            config = json.loads(FAST_CONFIG.read_text(encoding='utf-8'))
            folder = FAST_CONFIG.parent
            model_path = folder/config['model']
            token = (folder/config['token_file']).read_text(encoding='utf-8').strip()
            if (hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA or
                    config['board_url'] != self.client.url or token != self.client.token):
                raise ValueError('加速服务必须保持原分割模型、板端地址及凭据')
            manifest = json.loads((folder/config['platform_manifest']).read_text(encoding='utf-8'))
            if manifest.get('basic_transport') != 'ct_basic_zlib_cache_v1':
                raise ValueError('不是预期的基础加速包')
            client = PlatformClient(config['board_url'], token, manifest)
            identity = client.health()  # Only a read; old service stays untouched on mismatch.
            for key in ('input_quantization','output_scale','output_zero'):
                if identity[key] != self.model[key]: raise ValueError('分割量化契约不同')
            with self.lock:
                self.client = client
                self._update(board_connected=True, phase='idle', error=None)
        except Exception as exc:
            self._update(phase='error', error='加速服务尚未就绪：'+str(exc), board_connected=False)

    def _work(self, action, values):
        previous = self.status.get('case_id')
        super()._work(action, values)
        with self.lock:
            if action == 'load' and self.status.get('case_id') != previous:
                self.basic_result = self.basic_image = None
                self.threshold_result = self.threshold_image = None
                self.history = []
                self.nodule_result = None; self.nodule_stage = None

    def _nodule_work(self, hu, mask, geometry, folder, case_id, generation, source):
        started = time.perf_counter(); result_id = uuid.uuid4().hex
        def progress(stage, completed, total):
            with self.lock:
                if case_id != self.status['case_id'] or generation != self.nodule_generation:
                    raise InterruptedError('当前分割结果已变化')
                self.nodule_stage = stage
                self._update(completed=completed,total=total)
        try:
            before = (array_sha(hu),array_sha(mask))
            prediction = self.nodule_service.predict(hu,mask,geometry,progress,self.cancel)
            if self.cancel.is_set(): raise InterruptedError('结节检测已停止')
            if before != (array_sha(hu),array_sha(mask)):
                raise ValueError('结节检测改变了原CT或肺掩膜，结果未发布')
            if (prediction['input_hu_sha256'],prediction['lung_mask_sha256']) != before:
                raise ValueError('检测结果输入身份不一致')
            prediction.update(case_id=case_id,result_id=result_id,segmentation_generation=generation,source=source)
            prediction['host_result_ready_ms'] = (time.perf_counter()-started)*1000
            write_new(folder/(result_id+'_nodule.json'), prediction)
            with self.lock:
                if case_id != self.status['case_id'] or generation != self.nodule_generation:
                    raise InterruptedError('旧病例结果不再发布')
                self.nodule_result = prediction; self.nodule_stage = '完成'
                self._update(phase='done',error=None)
        except InterruptedError:
            with self.lock:
                if generation == self.nodule_generation and case_id == self.status['case_id']:
                    self.nodule_stage = '已停止'; self._update(phase='cancelled',error=None)
        except Exception as exc:
            with self.lock:
                if generation == self.nodule_generation and case_id == self.status['case_id']:
                    self.nodule_stage = '检测失败'
                    self._update(phase='error',error=type(exc).__name__+': '+str(exc))

    def nodule_payload(self, case_id, result_id):
        with self.lock:
            if (case_id != self.status['case_id'] or self.nodule_result is None or
                    result_id != self.nodule_result['result_id']):
                raise ValueError('结节结果已失效，请对当前分割结果重新检测')
            return copy.deepcopy(self.nodule_result)

    def _basic_work(self, hu, index, source_index, case_id, folder, params, backend,
                    mask_input, request_id, submitted_at):
        try:
            start = time.perf_counter()
            if mask_input is None:
                reference = process(hu, params)
            else:
                source, mask = mask_input
                # The existing board wire protocol accepts HU. Reproduce only
                # this frozen threshold on PS, and verify it is the selected mask.
                reproduced = process(hu, dict(params, operation='threshold'))
                if not np.array_equal(mask, reproduced):
                    raise ValueError('已保存二值输入与原 HU/阈值来源不一致，停止处理')
                binary = mask != 0
                op = params['operation']
                if op == 'erosion': binary = morphology(binary)
                elif op == 'dilation': binary = morphology(binary, True)
                elif op == 'opening': binary = morphology(morphology(binary), True)
                else: binary = morphology(morphology(binary, True))
                reference = binary.astype(np.uint8)*255
            reference_ms = (time.perf_counter()-start)*1000
            receipt = None; image = reference; elapsed = reference_ms
            if backend == 'FPGA_BASIC':
                image, receipt = self.client.basic(hu, params)
                if not np.array_equal(image, reference):
                    # Keep actual data on a mismatch. Never replace it with the reference.
                    failure_id = uuid.uuid4().hex
                    with (folder/(failure_id+'_basic_mismatch.npz')).open('xb') as stream:
                        np.savez_compressed(stream, actual=image, reference=reference)
                    write_new(folder/(failure_id+'_basic_receipt.json'), receipt)
                    raise ValueError('实际PL输出与独立软件参考不同，已保存真实输出，未使用参考替代')
                elapsed = receipt['stats']['dma_ms']
            result_id = uuid.uuid4().hex
            metadata = dict(schema='ct_workspace_basic_v1', case_id=case_id,
                result_id=result_id, index=index, source_slice_index=source_index,
                operation=params['operation'], label=OPERATIONS[params['operation']],
                parameters=params, backend=backend, processing_ms=elapsed,
                ui_request_id=request_id,
                input_mask=(dict(result_id=source['result_id'], output_sha256=source['output_sha256'],
                    parameters=source['parameters'], backend=source['backend'], index=source['index'],
                    note='显式选定已保存阈值结果；兼容板端协议由PS复现相同掩膜，形态学在所选后端执行')
                    if mask_input is not None else None),
                reference_ms=reference_ms, reference_bit_exact=True if receipt else None,
                hardware_receipt=receipt,
                shape=list(image.shape), output_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                input_hu_sha256=hashlib.sha256(np.ascontiguousarray(hu).tobytes()).hexdigest(),
                note=('实际DMA/PL输出；与参考逐像素一致；切换/往返另计' if receipt else '电脑参考预览，非PL实测')+
                    '；不修改原HU或分割输入')
            with (folder/(result_id+'_basic.npz')).open('xb') as stream:
                np.savez_compressed(stream, image=image)
            metadata['host_result_ready_ms'] = (time.perf_counter()-submitted_at)*1000
            metadata['timing_scope'] = dict(
                processing_ms='PL/DMA（含驱动轮询），或电脑参考运算；不含bit切换与通信',
                host_result_ready_ms='电脑收到任务至图像落盘；含调度、参考校验、板端往返；不含最终JSON回执写入及浏览器取图显示')
            write_new(folder/(result_id+'_basic.json'), metadata)
            with self.lock:
                self.basic_image = image; self.basic_result = metadata
                if params['operation'] == 'threshold':
                    self.threshold_image = image.copy(); self.threshold_result = copy.deepcopy(metadata)
                self.history.append(metadata)
                self._update(phase='done', completed=1, error=None)
                self.basic_condition.notify_all()
        except Exception as exc:
            self._update(phase='error', error=type(exc).__name__+': '+str(exc),
                         **({'board_connected':False} if backend == 'FPGA_BASIC' else {}))
            with self.basic_condition: self.basic_condition.notify_all()

    def wait_basic(self, case_id, request_id, timeout=15.):
        """Wait for an already submitted job. Never submit/retry hardware work."""
        deadline = time.monotonic()+timeout
        with self.basic_condition:
            while True:
                if case_id != self.status['case_id'] or request_id != self.basic_request_id:
                    raise ValueError('等待的基础任务已过期')
                if (self.basic_result is not None and self.basic_result.get('ui_request_id') == request_id
                        or self.status['phase'] == 'error'):
                    return self.snapshot()
                remaining = deadline-time.monotonic()
                if remaining <= 0: return self.snapshot()
                self.basic_condition.wait(remaining)

    def basic_payload(self, case_id, result_id):
        with self.lock:
            if (case_id == self.status['case_id'] and self.threshold_result is not None and
                    result_id == self.threshold_result['result_id']):
                return dict(**self.threshold_result, **{'image': pack_image(self.threshold_image)})
            if (case_id != self.status['case_id'] or self.basic_result is None or
                    self.basic_result['case_id'] != case_id or
                    result_id != self.basic_result['result_id']):
                raise ValueError('结果已过期，请重新计算当前病例')
            return dict(**self.basic_result, **{'image': pack_image(self.basic_image)})


def workspace_handler(controller, token, plotly):
    old_html = (Path(__file__).parent/'lungseg/live_viewer.html').read_text(encoding='utf-8')
    # Reuse the same session/API without changing the preserved standalone page.
    embed = '''<style>body{height:100vh}header>h1,header>.row:nth-of-type(1),
header>.row:nth-of-type(2),header>.row:nth-of-type(3){display:none}
header{padding:5px 10px}footer{display:none}main{padding:6px}
#receipt{display:none}.row{margin:3px 0}#message{font-size:12px}
</style>'''
    old_html = old_html.replace('</head>', embed+'</head>').replace('__UI_TOKEN__', token)
    old_html = old_html.replace("['loading','connecting','computing','reconstructing']",
                                "['loading','connecting','computing','reconstructing','processing','detecting']")
    old_html = old_html.replace("async function poll(){try{apply(await api('/api/state'));}",
        "async function poll(){try{if(window.ctWorkspaceVisible!==false)apply(await api('/api/state'));}")
    bridge = '''<script>window.ctWorkspaceView={snapshot:()=>state,currentIndices:()=>indices.slice(),ready:()=>sceneReady};
let workspacePendingSlice=null;
function workspaceApplySlice(){const m=workspacePendingSlice;
if(!m||m.case_id!==state?.case_id||!state?.shape||m.index<0||m.index>=state.shape[0])return;
workspacePendingSlice=null;indices[0]=m.index;$('s0').value=$('n0').value=m.index;
fetchPlane(0);planes.forEach((_,a)=>draw(a));updatePlane3D();}
const workspaceOriginalApply=apply;apply=s=>{workspaceOriginalApply(s);workspaceApplySlice();};
window.addEventListener('message', e=>{
if(e.origin!==location.origin||e.source!==parent)return;
const m=e.data;if(m?.type==='ct-workspace-visible'){window.ctWorkspaceVisible=m.visible!==false;
if(!window.ctWorkspaceVisible)return;
api('/api/state').then(s=>apply(s)).catch(e=>message(e.message,true));requestAnimationFrame(()=>{
planes.forEach((_,a)=>draw(a));if(sceneReady)Plotly.Plots.resize('scene');});}
if(m?.type==='ct-workspace-slice'&&typeof m.case_id==='string'&&Number.isInteger(m.index)){
workspacePendingSlice=m;workspaceApplySlice();}
});</script>'''
    old_html = old_html.replace('</body>', bridge+'</body>')
    base = handler_for(controller, token, old_html.encode('utf-8'), plotly)
    html = (ASSETS/'index.html').read_text(encoding='utf-8').replace('__UI_TOKEN__', token).encode('utf-8')
    assets = {'/workspace.css': ('style.css', 'text/css; charset=utf-8'),
              '/workspace.js': ('app.js', 'text/javascript; charset=utf-8')}

    class Handler(base):
        def do_GET(self):
            parts = parse.urlsplit(self.path)
            if parts.path in ('/', '/segmentation', '/api/benchmarks', '/api/basic/image',
                              '/api/export','/api/nodule/result') or parts.path in assets:
                if not self.allowed(parts.path.startswith('/api/')): return
                try:
                    if parts.path == '/': return self.send(200, html, 'text/html; charset=utf-8')
                    if parts.path == '/segmentation': return self.send(200, old_html.encode('utf-8'), 'text/html; charset=utf-8')
                    if parts.path in assets:
                        name, mime = assets[parts.path]
                        return self.send(200, (ASSETS/name).read_bytes(), mime)
                    if parts.path == '/api/benchmarks': return self.send(200, benchmarks())
                    if parts.path == '/api/export':
                        snapshot = controller.snapshot()
                        return self.send(200, dict(schema='ct_workspace_session_v1', state=snapshot,
                            note='当前会话状态及计算来源；基础PL/DMA与后台耗时分别记录；浏览器导出另附点击至显示计时；分割只显示实际回执'))
                    q = parse.parse_qs(parts.query)
                    if parts.path == '/api/nodule/result':
                        return self.send(200, controller.nodule_payload(q['case'][0],q['result'][0]))
                    return self.send(200, controller.basic_payload(q['case'][0], q['result'][0]))
                except (ValueError, KeyError) as exc:
                    return self.send(400, dict(error=str(exc)))
                except OSError:
                    return self.send(503, dict(error='历史证据文件不可用，未显示未核验的指标'))
            return super().do_GET()

        def do_POST(self):
            if self.path not in ('/api/basic/run','/api/connect-fast','/api/nodule/run','/api/nodule/cancel'): return super().do_POST()
            if not self.allowed(True): return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if (self.headers.get('Transfer-Encoding') or not 0 < length <= 16384 or
                        self.headers.get('Content-Type') != 'application/json'):
                    raise ValueError('需要小型JSON请求')
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict): raise ValueError('需要JSON对象')
                if self.path == '/api/nodule/cancel':
                    current = controller.snapshot()
                    if values.get('case_id') != current['case_id']:
                        raise ValueError('当前病例已变化')
                    if current['phase'] != 'detecting': raise ValueError('当前没有结节检测任务')
                    controller.cancel.set()
                    return self.send(200,dict(message='结节检测将在图块批次边界停止'))
                if self.path == '/api/nodule/run':
                    state = controller.submit('nodule',values)
                    return self.send(202 if state['phase']=='detecting' else 200,state)
                if self.path == '/api/connect-fast':
                    return self.send(202, controller.submit('connect_fast', values))
                if values.get('wait_for_result') is True and not values.get('ui_request_id'):
                    raise ValueError('等待完成需要显式界面请求标识')
                state = controller.submit('basic', values)
                if values.get('wait_for_result') is True:
                    state = controller.wait_basic(values['case_id'], values['ui_request_id'])
                return self.send(202 if state['phase'] == 'processing' else 200, state)
            except BlockingIOError as exc: return self.send(409, dict(error=str(exc)))
            except (ValueError, KeyError) as exc: return self.send(400, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--config', type=Path, help='真实板端服务配置；配置错误时直接报错，不回退')
    mode.add_argument('--desktop', action='store_true', help='电脑浏览与基础参考，不加载模型或连接FPGA')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--nodule-model',type=Path,help='可选：已核验445例结节model.pt的位置；仍检查固定SHA256')
    parser.add_argument('--output', type=Path, default=ROOT/'data/ct_workspace_20261017/results')
    args = parser.parse_args()
    default_config = ROOT/'data/ct_platform_20261017_v2/host_config.json'
    config_path = args.config or (default_config if not args.desktop and default_config.is_file() else None)
    if config_path is None:
        client, model = None, {}
    else:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        model_path = config_path.parent/config['model']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError('保留版分割模型契约不匹配，未启动服务')
        token = (config_path.parent/config['token_file']).read_text(encoding='utf-8').strip()
        if config.get('platform_manifest'):
            manifest = json.loads((config_path.parent/config['platform_manifest']).read_text(encoding='utf-8'))
            client = PlatformClient(config['board_url'], token, manifest)
        else:
            client = BoardClient(config['board_url'], token)
        model = json.loads(model_path.read_text(encoding='utf-8'))
        from lung_unet import torch_module
        torch_module().set_num_threads(2)
    controller = WorkspaceController(client, model, args.output,nodule_model=args.nodule_model)
    from plotly.offline import get_plotlyjs
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
        workspace_handler(controller, secrets.token_hex(32), get_plotlyjs().encode('utf-8')))
    print(f'CT工作台 http://127.0.0.1:{args.port}/', flush=True)
    if client is None:
        print('电脑浏览模式：可读取CT、浏览切片与运行基础参考；不连接FPGA，不执行神经分割。', flush=True)
    else:
        print('基础处理支持真实PL与独立电脑参考；统一板端服务仅在空闲时切换bit，不替换模型。', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
