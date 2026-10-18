// Dedicated headless browser and explicit synthetic fixture only.
import fs from 'node:fs';
import path from 'node:path';
const [folder,port='9324']=process.argv.slice(2);
if(!folder)throw Error('Need output folder');
const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const tab=tabs.find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:8091'))||tabs.find(t=>t.type==='page'&&t.url==='about:blank');if(!tab)throw Error('No isolated page');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let counter=0;const pending=new Map(),errors=[];
const basicTimings=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const rpc=(method,params={})=>new Promise((resolve,reject)=>{const id=++counter;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(expression){for(let n=0;n<150;n++){if(await evaluate(expression))return;await pause(200);}throw Error('Timed out: '+expression);}
async function shot(name){const r=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(folder,name),Buffer.from(r.data,'base64'));}
try{
await rpc('Runtime.enable');await rpc('Page.enable');
await rpc('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false});
await rpc('Page.navigate',{url:'http://127.0.0.1:8091/'});
await until('typeof state!=="undefined"&&state!==null');
if(await evaluate("state.known_slices!==0||$('metricPL').textContent!=='—'"))throw Error('Invented initial metrics');
await shot('empty_TEST_ONLY.png');
await evaluate(`$('path').value=${JSON.stringify(path.resolve(folder,'synthetic_ct.npz'))};$('expertPath').value=${JSON.stringify(path.resolve(folder,'synthetic_expert.npz'))};$('confirm').checked=true;$('load').click();`);
await until("state.shape&&state.phase==='idle'&&plane!==null");
if(await evaluate("state.latest!==null||state.known_slices!==0"))throw Error('Loading ran a neural forward');
await evaluate("$('connect').click()");await until("state.board_connected&&state.phase==='idle'");
await evaluate("$('operation').value='opening';$('operation').dispatchEvent(new Event('change'))");
if(!await evaluate("$('apply').disabled&&$('thresholdFields').hidden&&!$('morphologyFields').hidden"))throw Error('Morphology accepted an implicit threshold');
const originalCanvas=await evaluate("$('original').toDataURL()");
await evaluate("$('preset').value='soft';$('preset').dispatchEvent(new Event('change'))");
if(originalCanvas!==await evaluate("$('original').toDataURL()"))throw Error('Output window settings changed original CT');
for(const op of ['window','threshold','erosion','dilation','opening','closing','histogram','pseudocolor']){
 await evaluate("$('backend').value="+JSON.stringify(['histogram','pseudocolor'].includes(op)?'PC_REFERENCE':'FPGA_BASIC')+";$('backend').dispatchEvent(new Event('change'))");
 await evaluate(`$('operation').value='${op}';$('operation').dispatchEvent(new Event('change'));$('apply').click();`);
 await until(`state.phase==='done'&&result?.operation==='${op}'&&browserTimings.has(result.result_id)&&!pendingBasic`);
 if(await evaluate("state.known_slices!==0||result.backend!=="+JSON.stringify(['histogram','pseudocolor'].includes(op)?'PC_REFERENCE':'FPGA_BASIC')))throw Error('Wrong basic execution backend');
 if(op==='window'&&!await evaluate("result.parameters.wl===40&&result.parameters.ww===400&&$('resultTitle').textContent.includes('WL 40')"))throw Error('Window result missing output setting');
 if(['erosion','dilation','opening','closing'].includes(op)){
  await until('maskPlane!==null');
  if(!await evaluate("result.input_mask.result_id===state.threshold_input.result_id&&maskPlane.output_sha256===result.input_mask.output_sha256"))throw Error('Morphology source not bound to saved binary result');
  if(op==='opening')await shot('morphology_TEST_ONLY.png');
 }
 if(!await evaluate("browserTimings.get(result.result_id).click_to_display_ms>=result.host_result_ready_ms"))throw Error('Click timing excludes backend work');
 basicTimings.push(await evaluate("({operation:result.operation,backend:result.backend,click_ms:browserTimings.get(result.result_id).click_to_display_ms,host_ms:result.host_result_ready_ms,transport:result.hardware_receipt?.transport||null})"));
}
await shot('basic_TEST_ONLY.png');
await evaluate("setIndex(2);$('syncSlice').click()");
await until("$('segmentationFrame').contentWindow.ctWorkspaceView?.snapshot()?.case_id===state.case_id");
await until("$('segmentationFrame').contentWindow.ctWorkspaceView.currentIndices()[0]===2");
await evaluate("$('connect').click()");await until("state.board_connected&&state.phase==='idle'");
const frame="$('segmentationFrame').contentWindow";
await until(`${frame}.ctWorkspaceView.snapshot().board_connected`);
await evaluate(`${frame}.document.getElementById('slice').click()`);
await until("state.phase==='done'&&state.known_slices===1");
await until(`!${frame}.document.getElementById('volume').disabled`);
await evaluate(`${frame}.document.getElementById('volume').click()`);
await until("state.full_volume_ready");await until(`${frame}.ctWorkspaceView.ready()`);
await shot('segmentation_TEST_ONLY.png');
await evaluate("showTab('performance')");
await until("$('metricPL').textContent!=='—'");
if(await evaluate("!$('receipt').textContent.includes(state.latest.head_sha256)||document.querySelectorAll('#historyRows tr').length!==8||$('metricBasic').textContent==='—'||!$('basicTiming').textContent.includes('后台至图像落盘')"))throw Error('Missing live evidence/history/timing scopes');
await shot('performance_TEST_ONLY.png');
const exportOK=await evaluate("(async()=>{const e=await api('/api/export');return e.state.basic.backend==='PC_REFERENCE'&&e.state.known_slices===8})()");
if(!exportOK)throw Error('Export missing correct source');
await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
await evaluate("showTab('basic')");await pause(500);
if(await evaluate('document.documentElement.scrollWidth>innerWidth+1'))throw Error('Mobile horizontal overflow');
await shot('mobile_TEST_ONLY.png');
await evaluate("setIndex(3);$('operation').value='closing';$('operation').dispatchEvent(new Event('change'))");
if(!await evaluate("$('apply').disabled&&maskPlane===null"))throw Error('Reused a mask from a different slice');
await evaluate("$('load').click()");await until("state.known_slices===0&&state.basic===null&&result===null");
if(errors.length)throw Error(JSON.stringify(errors));
const proof={status:'passed_unified_pl_protocol_browser_synthetic_inputs_fake_hardware_only',basicPLProtocolOperations:6,explicitPCOperations:2,operations:8,explicitBinaryInput:true,fixedOriginalWindow:true,outputWindowApplied:true,clickToDisplayTiming:true,timingScopesSeparate:true,wrongSliceMaskRejected:true,sharedCase:true,sliceSynchronization:true,segmentationAnd3D:true,sessionExport:true,mobileNoOverflow:true,caseIsolation:true,basicTimings,errors};
fs.writeFileSync(path.join(folder,'browser_TEST_ONLY.json'),JSON.stringify(proof,null,2));console.log(JSON.stringify(proof));
}finally{ws.close();}
