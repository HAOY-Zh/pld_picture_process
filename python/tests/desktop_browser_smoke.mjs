// Real desktop server + synthetic HU; no fake neural output or board service.
import fs from 'node:fs';
import path from 'node:path';

const [url, input, output, port = '9337'] = process.argv.slice(2);
if (!url || !input || !output) throw Error('Usage: URL SYNTHETIC_NPZ OUTPUT_DIR [CDP_PORT]');
fs.mkdirSync(output, {recursive: true});
const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const tab = tabs.find(t => t.type === 'page');
if (!tab) throw Error('Missing dedicated browser page');
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, {once: true}));
let serial = 0;
const pending = new Map(), errors = [];
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const promise = pending.get(message.id);
    if (!promise) return;
    pending.delete(message.id);
    message.error ? promise.reject(Error(JSON.stringify(message.error))) : promise.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
});
const rpc = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++serial; pending.set(id, {resolve, reject});
  ws.send(JSON.stringify({id, method, params}));
});
async function evaluate(expression) {
  const result = await rpc('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(expression) {
  for (let count = 0; count < 150; count++) {
    if (await evaluate(expression)) return;
    await pause(200);
  }
  throw Error('Timeout: ' + expression);
}
async function screenshot(name) {
  const result = await rpc('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  fs.writeFileSync(path.join(output, name), Buffer.from(result.data, 'base64'));
}
try {
  await rpc('Runtime.enable'); await rpc('Page.enable');
  await rpc('Emulation.setDeviceMetricsOverride', {width: 1500, height: 1050, deviceScaleFactor: 1, mobile: false});
  await rpc('Page.navigate', {url});
  await until('typeof state!=="undefined"&&state?.desktop_only===true');
  if (!await evaluate("$('backend').value==='PC_REFERENCE'&&$('backend').disabled&&$('connect').hidden&&state.latest===null")) {
    throw Error('Desktop mode not clearly selected');
  }
  await screenshot('desktop_empty.png');
  await evaluate(`$('path').value=${JSON.stringify(path.resolve(input))};$('confirm').checked=true;$('loadExpert').checked=false;$('load').click()`);
  await until("state.phase==='idle'&&state.shape&&plane!==null");
  for (const operation of ['window', 'threshold', 'erosion', 'dilation', 'opening', 'closing', 'histogram', 'pseudocolor']) {
    await evaluate(`$('operation').value='${operation}';$('operation').dispatchEvent(new Event('change'));$('apply').click()`);
    await until(`state.phase==='done'&&result?.operation==='${operation}'`);
    if (!await evaluate("result.backend==='PC_REFERENCE'&&state.known_slices===0&&state.latest===null")) {
      throw Error('Computer reference mislabeled as inference');
    }
  }
  await screenshot('desktop_processing.png');
  await evaluate("setIndex(2);$('syncSlice').click()");
  const frame = "$('segmentationFrame').contentWindow";
  await until(`${frame}.ctWorkspaceView?.snapshot()?.case_id===state.case_id`);
  await until(`${frame}.ctWorkspaceView.currentIndices()[0]===2`);
  if (!await evaluate(`${frame}.document.getElementById('slice').disabled&&${frame}.document.getElementById('connect').hidden`)) {
    throw Error('Hardware controls enabled in desktop mode');
  }
  await screenshot('desktop_mpr.png');
  const blocked = await evaluate("(async()=>{try{await api('/api/connect',{});return false}catch(e){return e.message.includes('--config')}})()");
  if (!blocked) throw Error('Desktop accepted a board action');
  await evaluate("showTab('performance')");
  const exported = await evaluate("(async()=>{const r=await api('/api/export');return r.state.desktop_only&&r.state.known_slices===0&&r.state.basic.backend==='PC_REFERENCE'})()");
  if (!exported) throw Error('Wrong exported provenance');
  await screenshot('desktop_performance.png');
  await rpc('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await evaluate("showTab('basic')"); await pause(300);
  if (await evaluate('document.documentElement.scrollWidth>innerWidth+1')) throw Error('Mobile horizontal overflow');
  await screenshot('desktop_mobile.png');
  if (errors.length) throw Error(JSON.stringify(errors));
  const proof = {status: 'passed_real_desktop_server_synthetic_data_no_board', basicOperations: 8,
    mprSliceSynchronization: true, hardwareActionsRejected: true, export: true,
    mobileNoOverflow: true, javascriptErrors: errors, scope: 'Desktop software UI only; no neural or PL performance claim'};
  fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(proof, null, 2));
  console.log(JSON.stringify(proof));
} finally {
  ws.close();
}
