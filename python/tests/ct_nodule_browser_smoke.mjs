// Isolated headless test profile. All fixture content is synthetic and labelled.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
const [folder,port='9335',webPort='8093']=process.argv.slice(2);
const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const tab=tabs.find(t=>t.type==='page');if(!tab)throw Error('No isolated test page');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let seq=0;const pending=new Map(),errors=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const rpc=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(expression){for(let i=0;i<200;i++){if(await evaluate(expression))return;await pause(200);}throw Error('Timeout: '+expression);}
async function shot(name){const r=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(folder,name),Buffer.from(r.data,'base64'));}
try{
 await rpc('Runtime.enable');await rpc('Page.enable');
 await rpc('Emulation.setDeviceMetricsOverride',{width:1500,height:1100,deviceScaleFactor:1,mobile:false});
 await rpc('Page.navigate',{url:`http://127.0.0.1:${webPort}/`});
 await until('typeof state!=="undefined"&&state!==null');
 if(!await evaluate("$('detectNodules').disabled"))throw Error('Detection enabled before segmentation');
 await evaluate(`$('path').value=${JSON.stringify(path.resolve(folder,'synthetic_ct.npz'))};$('confirm').checked=true;$('loadExpert').checked=false;$('load').click();`);
 await until("state.shape&&state.phase==='idle'");
 await evaluate("showTab('segmentation');$('connect').click()");
 await until("state.board_connected&&state.phase==='idle'");
 const frame="$('segmentationFrame').contentWindow";
 await until(`${frame}.ctWorkspaceView?.snapshot()?.board_connected`);
 await evaluate(`${frame}.document.getElementById('slice').click()`);
 await until("state.phase==='done'&&state.known_slices===1");
 if(!await evaluate("$('detectNodules').disabled"))throw Error('Detection enabled for incomplete mask');
 await evaluate(`${frame}.document.getElementById('volume').click()`);
 await until("state.phase==='done'&&state.known_slices===8&&state.nodule.ready");
 await until("!$('detectNodules').disabled");
 const before=await evaluate("api('/api/test-fixture')");
 await evaluate("$('detectNodules').click()");
 await until("state.phase==='detecting'");
 if(!await evaluate("$('load').disabled&&$('detectNodules').disabled&&!$('cancelNodules').hidden"))throw Error('Active detection controls unsafe');
 await until("state.nodule.result&&noduleResult&&nodulePlane");
 const after=await evaluate("api('/api/test-fixture')");
 if(after.hardware_calls!==before.hardware_calls||after.nodule_calls!==before.nodule_calls+1)throw Error('Detection reran hardware or model');
 if(!await evaluate("!$('noduleResults').hidden&&$('noduleRows').rows.length===1&&noduleIndex===3&&noduleResult.backend==='PC_CNN_FP32'&&noduleResult.lung_inference_rerun===false"))throw Error('Candidate rendering/source mismatch');
 await evaluate("$('noduleRows').querySelector('button').click()");await until('nodulePlane!==null&&noduleIndex===3');
 await shot('nodule_result_TEST_ONLY.png');
 await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await pause(300);
 if(await evaluate('document.documentElement.scrollWidth>innerWidth+1'))throw Error('Mobile overflow');
 await shot('nodule_mobile_TEST_ONLY.png');
 await evaluate("$('load').click()");
 await until("state.known_slices===0&&noduleResult===null&&$('noduleResults').hidden&&$('detectNodules').disabled");
 if(errors.length)throw Error(JSON.stringify(errors));
 const assets=Object.fromEntries(['index.html','app.js','style.css'].map(name=>[name,createHash('sha256').update(fs.readFileSync(path.resolve('python/lungseg/workspace',name))).digest('hex')]));
 const proof={status:'PASS_SYNTHETIC_ONLY',requiresCompleteSegmentation:true,hardwareNotRerun:true,candidateClick:true,staleCaseCleared:true,mobileNoOverflow:true,assets_sha256:assets,errors};
 fs.writeFileSync(path.join(folder,'browser_TEST_ONLY.json'),JSON.stringify(proof,null,2));console.log(JSON.stringify(proof));
}finally{ws.close();}
