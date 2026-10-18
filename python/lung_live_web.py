"""Loopback-only interactive CT viewer/proxy; every forward is requested from Z2.

The PC only loads HU, quantizes, restores native masks, cleans and renders.
Optional expert masks are used only for display/evaluation, never as predictions.
"""
import argparse
import base64
import copy
import hashlib
import hmac
import ipaddress
import json
from pathlib import Path
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request, parse

import numpy as np

from lung_live_data import read_ct, input_codes, native_masks, parse_slices
from lung_live_reference import read_reference
from lung_island_cleanup import clean_prediction, PRIMARY
from lung_volume_viewer import mesh
from lungseg.metrics import overlap_metrics

BIT_SHA = '567ddc0e47e3556e21b248a2d0bab85f26381c8056295f88ddd14335e695c89b'
MODEL_SHA = '70ad14405228ac08421c5d9a732a2825f46c1e364e4cf927488a90cdd0650dbd'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)


def reconstruction_plan(known, details, geom):
    """One contiguous, genuinely computed slab; never bridge unknown planes."""
    indices = np.flatnonzero(known)
    if len(indices) < 2 or not np.all(np.diff(indices) == 1): return None
    lo, hi = int(indices[0]), int(indices[-1]) + 1
    source_ids = details['source_slice_indices'][lo:hi]
    if len(source_ids) != hi-lo or not np.all(np.diff(source_ids) == 1): return None
    partial = bool(details.get('partial_input') or not np.all(known))
    local_geom = copy.deepcopy(geom)
    local_geom['origin_lps_mm'] = (np.asarray(geom['origin_lps_mm']) +
        np.asarray(geom['basis_lps_zyx'])[:, 0] * lo).tolist()
    return dict(scope='partial' if partial else 'input_volume', local_start=lo, local_stop=hi,
        source_first=source_ids[0], source_last=source_ids[-1], slice_count=hi-lo,
        geometry=local_geom, clipped_to_input=partial,
        boundary_note='局部三维；上下截断/封口由输入范围产生，不是肺的真实解剖边界。' if partial else
            '全部已提供切片的重建；是否覆盖完整肺脏须由操作者确认。')


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Board redirect refused; credentials are not forwarded')


class BoardClient:
    def __init__(self, url, token):
        parts = parse.urlsplit(url)
        address = ipaddress.ip_address(parts.hostname or '')
        if (parts.scheme != 'http' or not (address.is_private or address.is_loopback) or
                parts.path not in ('', '/') or parts.query or parts.fragment or parts.username or parts.password):
            raise ValueError('Use an explicit private-LAN board IP and HTTP port, no path/credentials')
        self.url, self.token = url.rstrip('/'), token
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def call(self, path, data=None, headers=None, timeout=60):
        req = request.Request(self.url + path, data=data, headers={
            'Authorization': 'Bearer ' + self.token, **(headers or {})})
        with self.opener.open(req, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024: raise ValueError('Oversized board reply')
        return json.loads(raw)

    def health(self):
        value = self.call('/v1/health', timeout=5)
        if (value.get('schema') != 'lung_live_identity_v1' or value.get('backend') != 'FPGA_INT8' or
                value.get('bit_sha256') != BIT_SHA or value.get('model_sha256') != MODEL_SHA or
                len(value.get('startup_nodes_checked', [])) != 21 or value.get('startup_command_count') != 51):
            raise ValueError('Board identity/self-test differs from preserved FPGA version')
        if not value.get('ready') or value.get('busy'):
            raise RuntimeError('Board is faulted or busy; inspect its notebook, do not reload automatically')
        return value

    def infer(self, codes):
        if codes.dtype != np.uint8 or codes.shape != (1, 256, 256): raise ValueError('Invalid input codes')
        raw = codes.tobytes(); rid = uuid.uuid4().hex; start = time.perf_counter()
        value = self.call('/v1/infer', raw, {'Content-Type': 'application/octet-stream',
            'X-Request-ID': rid, 'X-Input-SHA256': sha(raw)})
        if (value.get('schema') != 'lung_live_result_v1' or value.get('request_id') != rid or
                value.get('input_sha256') != sha(raw) or value.get('backend') != 'FPGA_INT8' or
                value.get('bit_sha256') != BIT_SHA or value.get('model_sha256') != MODEL_SHA or
                value.get('shape') != [1, 256, 256]):
            raise ValueError('Board result identity/request binding mismatch')
        head = base64.b64decode(value['head_base64'], validate=True)
        if len(head) != 65536 or sha(head) != value['head_sha256']:
            raise ValueError('Invalid actual FPGA head payload')
        stats = value['stats']
        if stats.get('command_count') != 51 or len(stats.get('commands', [])) != 51:
            raise ValueError('Incomplete FPGA execution receipt')
        before, after = stats['clock_before'], stats['clock_after']
        for clock in (before, after):
            if (not np.isfinite([clock['fclk0_mhz'], clock['cpu_mhz']]).all() or
                    abs(clock['fclk0_mhz'] - 100.) > .001 or clock['cpu_mhz'] <= 0):
                raise ValueError('Wrong FPGA clock evidence')
        if abs(before['cpu_mhz'] - after['cpu_mhz']) > .05: raise ValueError('CPU clock changed')
        value.pop('head_base64')
        value['roundtrip_ms'] = (time.perf_counter() - start) * 1000
        return np.frombuffer(head, np.uint8).reshape(1, 256, 256).copy(), value


class Controller:
    def __init__(self, client, model, output):
        self.client, self.model, self.output = client, model, Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.hu = self.raw = self.body = self.cleaned = self.known = None
        self.geom = self.details = self.case_folder = None
        self.surfaces = {}; self.receipts = {}
        self.expert = self.expert_info = None
        self.scores = {name: {} for name in ('raw', 'body', 'cleaned')}
        self.cancel = threading.Event()
        self.status = dict(phase='empty', revision=0, case_id=None, error=None,
            board_connected=False, completed=0, total=0, latest=None, full_volume_ready=False,
            volume_ready=False, volume_scope=None, reconstruction=None, region_volume_ml=None)

    def snapshot(self):
        with self.lock:
            return dict(self.status, shape=list(self.hu.shape) if self.hu is not None else None,
                geometry=self.geom, source=self.details, known_slices=int(self.known.sum()) if self.known is not None else 0,
                volume_ml=float(self.cleaned.sum() * np.prod(self.geom['spacing_zyx_mm']) / 1000)
                    if self.cleaned is not None else None,
                expert_present=self.expert is not None, expert_source=self.expert_info,
                comparison=self.comparison(),
                warning='工程教学演示，非临床诊断。电脑只做预/后处理和显示，网络前向由 FPGA 执行。')

    def comparison(self):
        """Aggregate cached per-slice counts; unknown slices never count as negatives."""
        if self.expert is None: return None
        result = {}
        for name, rows in self.scores.items():
            if not rows: continue
            tp, fp, fn = [sum(row[key] for row in rows.values()) for key in
                          ('true_positive', 'false_positive', 'false_negative')]
            denom = 2*tp+fp+fn
            result[name] = dict(dice=2*tp/denom if denom else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None, both_empty=denom == 0,
                true_positive=tp, false_positive=fp, false_negative=fn,
                computed_slices=len(rows),
                source_slice_indices=[self.details['source_slice_indices'][z] for z in sorted(rows)],
                scope='input_volume' if len(rows) == len(self.hu) and not self.details.get('partial_input')
                    else 'computed_slices')
        return result

    def _update(self, **fields):
        with self.lock:
            self.status.update(fields)
            self.status['revision'] += 1

    def submit(self, action, values):
        with self.lock:
            if self.status['phase'] in ('loading', 'connecting', 'computing', 'reconstructing'):
                raise BlockingIOError('A task is active; wait or cancel between slices')
            if action == 'connect':
                phase = 'connecting'
            elif action == 'load':
                if not isinstance(values.get('path'), str) or not values['path'].strip():
                    raise ValueError('请输入这台电脑上的 CT 路径')
                phase = 'loading'
            elif action == 'run':
                if self.hu is None or not self.status['board_connected']: raise ValueError('先加载 CT 并连接 FPGA')
                if values.get('mode') not in ('slice', 'volume', 'selection'): raise ValueError('Invalid inference mode')
                if type(values.get('index')) is not int or not 0 <= values['index'] < len(self.hu):
                    raise ValueError('Slice index outside volume')
                values = dict(values)
                values['_indices'] = (parse_slices(values.get('slices', ''), len(self.hu)) if values['mode'] == 'selection'
                    else list(range(len(self.hu))) if values['mode'] == 'volume' else [values['index']])
                phase = 'computing'
            else: raise ValueError('Unknown action')
            self.cancel.clear()
            self._update(phase=phase, error=None, completed=0, total=0)
            threading.Thread(target=self._work, args=(action, values), daemon=True).start()
            return self.snapshot()

    def _work(self, action, values):
        try:
            if action == 'connect':
                identity = self.client.health()
                # The loaded contract is local and hash-bound; don't trust a
                # server-supplied alternative input/output quantization grid.
                for key in ('input_quantization', 'output_scale', 'output_zero'):
                    if identity[key] != self.model[key]: raise ValueError('Board quantization contract differs')
                self._update(board_connected=True, phase='idle')
            elif action == 'load':
                hu, geom, details = read_ct(values['path'].strip().strip('"'),
                    confirm_hu=values.get('confirm_hu') is True, series_uid=values.get('series_uid', ''),
                    slices=values.get('load_slices', ''))
                expert, expert_info = read_reference(values.get('expert_path', '').strip().strip('"'),
                    hu.shape, geom, details, enabled=values.get('load_expert', True) is True)
                details.update(annotations_loaded=expert is not None, expert_reference=expert_info)
                case_id = uuid.uuid4().hex
                folder = self.output / case_id; folder.mkdir()
                write_new(folder / 'input.json', details)
                if expert is not None:
                    with (folder / 'expert_mask.npz').open('xb') as out:
                        np.savez_compressed(out, mask=expert, spacing_zyx_mm=geom['spacing_zyx_mm'],
                            origin_lps_mm=geom['origin_lps_mm'], basis_lps_zyx=geom['basis_lps_zyx'])
                with self.lock:
                    self.hu, self.geom, self.details, self.case_folder = hu, geom, details, folder
                    self.raw = np.zeros(hu.shape, bool); self.body = np.zeros(hu.shape, bool)
                    self.known = np.zeros(len(hu), bool); self.cleaned = None
                    self.surfaces = {}; self.receipts = {}
                    self.expert, self.expert_info = expert, expert_info
                    self.scores = {name: {} for name in ('raw', 'body', 'cleaned')}
                    self._update(phase='idle', case_id=case_id, full_volume_ready=False, latest=None,
                        volume_ready=False, volume_scope=None, reconstruction=None, region_volume_ml=None,
                        elapsed_ms=None, active_slice=None)
            else:
                self._run(values)
        except Exception as exc:
            self._update(phase='error', error=type(exc).__name__ + ': ' + str(exc),
                **({'board_connected': False} if action in ('connect', 'run') else {}))

    def _run(self, values):
        indices = values['_indices']
        with self.lock:
            self.cleaned = None; self.surfaces = {}
            self.scores['cleaned'] = {}
            if values['mode'] == 'volume':
                self.known[:] = False; self.raw[:] = False; self.body[:] = False; self.receipts = {}
            else:
                self.known[indices] = False; self.raw[indices] = False; self.body[indices] = False
                for z in indices: self.receipts.pop(z, None)
            for name in ('raw', 'body'):
                if values['mode'] == 'volume': self.scores[name] = {}
                else:
                    for z in indices: self.scores[name].pop(z, None)
            self._update(total=len(indices), full_volume_ready=False, volume_ready=False,
                         volume_scope=None, reconstruction=None, region_volume_ml=None)
        job_id = uuid.uuid4().hex
        start = time.perf_counter()
        for done, z in enumerate(indices, 1):
            if self.cancel.is_set():
                self._update(phase='cancelled'); return
            self._update(active_slice=int(z))
            codes = input_codes(self.hu[z], self.model)
            head, receipt = self.client.infer(codes)  # No retries or fallback.
            raw, body, plan = native_masks(head, self.hu[z], self.geom, self.model)
            scores = ({name: overlap_metrics(mask, self.expert[z]) for name, mask in
                [('raw', raw), ('body', body)]} if self.expert is not None else {})
            receipt.update(slice_index=int(z), body_plan=plan, case_id=self.status['case_id'],
                           analysis_source=copy.deepcopy(self.details.get('analysis_source',{'source_id':'original'})),
                           job_id=job_id, source_hu_sha256=self.details['hu_sha256'],
                           source_slice_index=self.details['source_slice_indices'][z])
            if scores:
                receipt['expert_comparison'] = dict(reference_sha256=self.expert_info['mask_sha256'],
                    scope='single_slice', source_slice_index=self.details['source_slice_indices'][z], stages=scores)
            folder = self.case_folder / receipt['request_id']; folder.mkdir()
            with (folder / 'actual_head.npz').open('xb') as out:
                np.savez_compressed(out, head_codes=head, input_codes=codes)
            write_new(folder / 'receipt.json', receipt)
            with self.lock:
                self.raw[z], self.body[z], self.known[z] = raw, body, True
                self.receipts[int(z)] = receipt['request_id']
                for name, row in scores.items(): self.scores[name][int(z)] = row
                self._update(completed=done, latest=receipt, elapsed_ms=(time.perf_counter()-start)*1000)
        if self.cancel.is_set():
            self._update(phase='cancelled'); return
        plan = reconstruction_plan(self.known, self.details, self.geom)
        if plan is not None:
            self._update(phase='reconstructing')
            lo, hi = plan['local_start'], plan['local_stop']
            partial = plan['scope'] == 'partial'
            masks = dict(raw=self.raw[lo:hi], body=self.body[lo:hi])
            if partial:
                # Do not apply whole-volume island criteria to truncated anatomy.
                cleaned = None
                cleanup = dict(applied=False, reason='Local slab: keep CT-only body mask; whole-volume island cleanup disabled')
                surfaces = dict(body=mesh(masks['body'], plan['geometry']))
                region_volume = float(masks['body'].sum()*np.prod(self.geom['spacing_zyx_mm'])/1000)
            else:
                cleaned, cleanup = clean_prediction(self.body, self.geom['spacing_zyx_mm'], **PRIMARY)
                masks['cleaned'] = cleaned
                surfaces = {name: mesh(mask, self.geom) for name, mask in [('body', self.body), ('cleaned', cleaned)]}
                region_volume = float(cleaned.sum()*np.prod(self.geom['spacing_zyx_mm'])/1000)
            if self.expert is not None:
                surfaces['expert'] = mesh(self.expert[lo:hi], plan['geometry'])
                if cleaned is not None:
                    with self.lock:
                        self.scores['cleaned'] = {z: overlap_metrics(cleaned[z], self.expert[z]) for z in range(len(cleaned))}
            with (self.case_folder / (job_id + '_masks.npz')).open('xb') as out:
                np.savez_compressed(out, **masks)
            write_new(self.case_folder / (job_id + '_volume.json'), dict(schema='lung_live_volume_v1',
                job_id=job_id, geometry=plan['geometry'], input_geometry=self.geom,
                source=self.details, slice_receipts=self.receipts,
                reconstruction=plan, volume_scope=plan['scope'], region_volume_ml=region_volume,
                cleanup=cleanup, volume_ml=None if partial else region_volume,
                backend='FPGA_INT8', bit_sha256=BIT_SHA, model_sha256=MODEL_SHA,
                total_elapsed_ms=(time.perf_counter()-start)*1000, expert_present=self.expert is not None,
                expert_reference=self.expert_info, expert_comparison=self.comparison(),
                warning='Actual on-demand FPGA heads; PC pre/postprocessing and geometry. Not a clinical result.'))
            with self.lock:
                self.cleaned, self.surfaces = cleaned, surfaces
                self._update(full_volume_ready=not partial, volume_ready=True, volume_scope=plan['scope'],
                             reconstruction=plan, region_volume_ml=region_volume, volume_job_id=job_id)
        if self.expert is not None:
            write_new(self.case_folder / (job_id + '_comparison.json'), dict(
                schema='lung_live_expert_comparison_v1', job_id=job_id, case_id=self.status['case_id'],
                expert_reference=self.expert_info, stages=self.comparison(), slice_receipts=self.receipts,
                note='Only actually computed slices; partial metrics are not whole-lung metrics.'))
        self._update(phase='done', elapsed_ms=(time.perf_counter()-start)*1000)

    def plane(self, case_id, axis, index, stage):
        with self.lock:
            if self.hu is None or case_id != self.status['case_id']: raise ValueError('Stale/no case')
            if axis not in (0, 1, 2) or not 0 <= index < self.hu.shape[axis]: raise ValueError('Invalid plane')
            prediction = {'body': self.body, 'raw': self.raw, 'cleaned': self.cleaned}.get(stage)
            if prediction is None: raise ValueError('Selected mask stage not available yet')
            hu = np.take(self.hu, index, axis)
            mask = np.take(prediction, index, axis)
            known = np.take(np.broadcast_to(self.known[:, None, None], self.hu.shape), index, axis)
            expert = np.take(self.expert, index, axis) if self.expert is not None else None
            scores = overlap_metrics(mask[known], expert[known]) if expert is not None and known.any() else None
            return dict(shape=list(hu.shape), hu=base64.b64encode(hu.astype('<f4').tobytes()).decode('ascii'),
                mask=base64.b64encode(np.packbits(mask.ravel(), bitorder='little').tobytes()).decode('ascii'),
                known=base64.b64encode(np.packbits(known.ravel(), bitorder='little').tobytes()).decode('ascii'),
                expert=base64.b64encode(np.packbits(expert.ravel(), bitorder='little').tobytes()).decode('ascii')
                    if expert is not None else None,
                expert_comparison=scores, computed_pixels=int(known.sum()),
                axis=axis, index=index, spacing=[self.geom['spacing_zyx_mm'][a] for a in range(3) if a != axis],
                source_slice_index=self.details['source_slice_indices'][index] if axis == 0 else None,
                revision=self.status['revision'])

    def surface(self, case_id, stage):
        with self.lock:
            if case_id != self.status['case_id'] or not self.status['volume_ready']:
                raise ValueError('至少两层连续切片实际推理完成后才能重建；未计算层不会当作背景')
            if stage not in self.surfaces: raise ValueError('局部三维仅支持人体约束后；完整输入另支持三维清理后')
            value = self.surfaces[stage]
            return dict(vertices=value['world'].tolist(), faces=value['faces'].tolist(),
                revision=self.status['revision'], geometry=self.status['reconstruction']['geometry'],
                reconstruction=self.status['reconstruction'])


def handler_for(controller, ui_token, html, plotly):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def send(self, status, raw, content_type='application/json; charset=utf-8'):
            if isinstance(raw, dict): raw = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(raw)

        def allowed(self, api=False):
            expected = '127.0.0.1:' + str(self.server.server_port)
            if (self.headers.get('Host') != expected or self.headers.get('Sec-Fetch-Site') == 'cross-site'
                    or self.headers.get('Origin') not in (None, 'http://' + expected)):
                self.send(403, dict(error='Loopback same-origin access only')); return False
            if api and not hmac.compare_digest(self.headers.get('X-UI-Token', ''), ui_token):
                self.send(403, dict(error='Invalid UI token')); return False
            self.connection.settimeout(10); return True

        def do_GET(self):
            parts = parse.urlsplit(self.path)
            if not self.allowed(parts.path.startswith('/api/')): return
            try:
                if parts.path == '/': return self.send(200, html, 'text/html; charset=utf-8')
                if parts.path == '/plotly.js': return self.send(200, plotly, 'text/javascript; charset=utf-8')
                if parts.path == '/api/state': return self.send(200, controller.snapshot())
                q = parse.parse_qs(parts.query)
                one = lambda k: q[k][0]
                if parts.path == '/api/plane':
                    return self.send(200, controller.plane(one('case'), int(one('axis')), int(one('index')), one('stage')))
                if parts.path == '/api/surface':
                    return self.send(200, controller.surface(one('case'), one('stage')))
                self.send(404, dict(error='Not found'))
            except (ValueError, KeyError) as exc: self.send(400, dict(error=str(exc)))

        def do_POST(self):
            if not self.allowed(True): return
            try:
                if (self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type') != 'application/json'
                        or not 0 < int(self.headers.get('Content-Length', '0')) <= 16384):
                    raise ValueError('Need a small JSON request')
                values = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not isinstance(values, dict): raise ValueError('Expected JSON object')
                if self.path == '/api/cancel':
                    controller.cancel.set()
                    return self.send(200, dict(message='当前 FPGA 命令正常完成后，在层间停止；不会强行复位。'))
                if self.path not in ('/api/load', '/api/connect', '/api/run'): raise ValueError('Unknown action')
                self.send(202, controller.submit(self.path.rsplit('/', 1)[1], values))
            except BlockingIOError as exc: self.send(409, dict(error=str(exc)))
            except (ValueError, KeyError) as exc: self.send(400, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    model_path = (args.config.parent / config['model']).resolve()
    if sha(model_path.read_bytes()) != MODEL_SHA: raise ValueError('Changed frozen host model contract')
    model = json.loads(model_path.read_text(encoding='utf-8'))
    client = BoardClient(config['board_url'], (args.config.parent / config['token_file']).read_text(encoding='utf-8').strip())
    controller = Controller(client, model, args.config.parent / 'results')
    from lung_unet import torch_module
    torch_module().set_num_threads(2)  # Quantize/sigmoid/interpolate only, not model inference.
    token = secrets.token_hex(32)
    html = (Path(__file__).parent / 'lungseg' / 'live_viewer.html').read_text(encoding='utf-8').replace('__UI_TOKEN__', token).encode('utf-8')
    from plotly.offline import get_plotlyjs
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(controller, token, html, get_plotlyjs().encode('utf-8')))
    print('打开 http://127.0.0.1:{}/ （保持此终端和板端专用 Jupyter 内核运行）'.format(args.port), flush=True)
    print('网页不会读取旧预测。FPGA 不在线则不能计算；不会自动改用 CPU。', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt:
        # The remote service remains alive and owns any in-progress hardware.
        print('电脑网页已停止；板端继续完成当前请求，不会自动复位或释放 CMA。', flush=True)
    finally: server.server_close()


if __name__ == '__main__': main()
