// Isolated CDP test browser only. Fake hardware server is explicitly TEST ONLY.
import fs from 'node:fs';
import path from 'node:path';
const [folder,port='9323']=process.argv.slice(2);
if(!folder)throw Error('Need test output folder');
const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const tab=tabs.find(t=>t.type==='page');if(!tab)throw Error('Missing dedicated test browser');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let counter=0;const pending=new Map(),errors=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const rpc=(method,params={})=>new Promise((resolve,reject)=>{const id=++counter;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function until(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await sleep(200);}throw Error('Timed out: '+expression);}
try{
await rpc('Runtime.enable');await rpc('Page.enable');await rpc('Emulation.setDeviceMetricsOverride',{width:1600,height:1000,deviceScaleFactor:1,mobile:false});
await rpc('Page.navigate',{url:'http://127.0.0.1:8089/'});
await until('typeof state!=="undefined"&&state!==null');
if(await evaluate('state.known_slices!==0||state.latest!==null'))throw Error('Initial UI shows predictions');
const input=path.resolve(folder,'synthetic_ct.npz');
const expert=path.resolve(folder,'synthetic_expert.npz');
await evaluate(`$('path').value=${JSON.stringify(input)};$('expertPath').value=${JSON.stringify(expert)};$('confirm').checked=true;$('load').click();`);
await until('state.shape&&state.phase==="idle"');
await until('planes.every(p=>p!==null)');
if(await evaluate('planes.some(p=>p.knownBits.some(v=>v!==0))'))throw Error('Uncomputed pixels shown as known');
if(await evaluate('!state.expert_present||planes.some(p=>!p.expertBits)||Object.keys(state.comparison).length!==0'))throw Error('Expert preview should be available without inference or scores');
const cyanCount=`(()=>{const d=$('c0').getContext('2d').getImageData(0,0,$('c0').width,$('c0').height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]===35&&d[i+1]===225&&d[i+2]===240)n++;return n;})()`;
if(await evaluate(cyanCount)===0)throw Error('Expert contour is not rendered on the actual canvas');
await evaluate("$('showExpert').click()");
if(await evaluate(cyanCount)!==0)throw Error('Expert overlay toggle did not hide the contour');
await evaluate("$('showExpert').click()");
await evaluate("$('connect').click()");await until('state.board_connected&&state.phase==="idle"');
await evaluate("$('slice').click()");await until('state.phase==="done"&&state.known_slices===1');
if(await evaluate('state.full_volume_ready||sceneReady'))throw Error('One slice falsely rendered as complete 3D');
if(await evaluate("state.comparison.body.computed_slices!==1||state.comparison.body.scope!=='computed_slices'"))throw Error('Single-slice expert score counts uncomputed layers');
await evaluate("$('selection').value='0,2-3,2';$('selected').click()");
await until('state.phase==="done"&&state.completed===3&&state.known_slices===4');
if(await evaluate('state.full_volume_ready'))throw Error('Sparse selection falsely claimed full volume');
await evaluate("$('volume').click()");await until('state.phase==="done"&&state.full_volume_ready');
await until('sceneReady');
await evaluate("$('stage').value='cleaned';$('stage').dispatchEvent(new Event('change'));");
await until("sceneReady&&surfaceKey.includes(':cleaned:')");
if(await evaluate("!$('scene').data.some(t=>t.name==='专家标注（同一重建范围）')||!$('stats').textContent.includes('Dice')"))throw Error('Missing 3D expert overlay or Dice');
await evaluate("$('s0').value=2;$('s0').dispatchEvent(new Event('input'));$('alpha').value=.6;$('alpha').dispatchEvent(new Event('input'));");
await until('planes[0].index===2');
const info=await evaluate(`({phase:state.phase,known:state.known_slices,volume:state.volume_ml,stage:$('stage').value,head:state.latest.head_sha256,panels:[...document.querySelectorAll('.panel')].map(p=>{const r=p.getBoundingClientRect();return {top:r.top,bottom:r.bottom,width:r.width,height:r.height}}),height:innerHeight,scrollHeight:document.documentElement.scrollHeight,meshFaces:$('scene').data[0].i.length})`);
if(info.panels.some(p=>p.bottom>info.height||p.height<150)||info.scrollHeight>info.height+1)throw Error('Four-panel layout does not fit: '+JSON.stringify(info));
if(info.known!==8||info.meshFaces===0)throw Error('Missing actual test results');
const shot=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
fs.writeFileSync(path.join(folder,'live_ui_TEST_STUB.png'),Buffer.from(shot.data,'base64'));
await evaluate("$('loadSlices').value='2-4';$('load').click()");
await until('state.phase==="idle"&&state.shape[0]===3');
if(await evaluate('state.known_slices!==0||!state.source.partial_input||state.source.source_slice_indices[0]!==2'))throw Error('Partial input mapping incorrect');
await evaluate("$('volume').click()");await until('state.phase==="done"&&state.known_slices===3&&sceneReady');
if(await evaluate("state.full_volume_ready||state.volume_ml!==null||!state.volume_ready||state.volume_scope!=='partial'||$('scene').data.length!==5||!$('scopeBadge').textContent.includes('上下截断')||!$('stats').textContent.includes('片段内分割体积')"))throw Error('Local reconstruction scope/cut planes incorrect');
if(await evaluate('state.reconstruction.source_first!==2||state.reconstruction.source_last!==4'))throw Error('Lost original source slice range');
const localShot=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
fs.writeFileSync(path.join(folder,'local_3d_TEST_STUB.png'),Buffer.from(localShot.data,'base64'));
info.partialInputTestPassed=true;
info.local3DTestPassed=true;
info.expertOverlayTestPassed=true;
info.expertComputedScopeTestPassed=true;
// A computed continuous slab inside a larger input uses offset geometry. The
// active plane disappears outside that slab and returns correctly on re-entry.
await evaluate("$('loadSlices').value='';$('load').click()");await until('state.phase==="idle"&&state.shape[0]===8');
await evaluate("$('selection').value='2-4';$('selected').click()");await until('state.phase==="done"&&sceneReady&&state.volume_scope==="partial"');
await evaluate("$('s0').value=0;$('s0').dispatchEvent(new Event('input'))");
await until("$('scene').data[1].x.length===0");
await evaluate("$('s0').value=2;$('s0').dispatchEvent(new Event('input'))");
await until("$('scene').data[1].x.length===4&&$('scene').data[1].i.length===2");
if(errors.length)throw Error('Browser exceptions: '+JSON.stringify(errors));
fs.writeFileSync(path.join(folder,'browser_TEST_ONLY.json'),JSON.stringify({status:'passed_local_browser_with_explicit_stub_NOT_real_fpga',...info,errors},null,2));
console.log(JSON.stringify({status:'PASS — UI/HTTP TEST STUB ONLY, not board integration',...info}));
}finally{ws.close();}
