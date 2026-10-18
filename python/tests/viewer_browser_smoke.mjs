// Optional real-browser smoke test. Node >=22, an isolated local CDP browser.
// No npm packages. Do not point this at a user's existing browser session.
import fs from 'node:fs';
import path from 'node:path';

const [url, output, port = '9321'] = process.argv.slice(2);
if (!url?.startsWith('file:///') || !output || !/^\d+$/.test(port)) {
  throw Error('Usage: node viewer_browser_smoke.mjs file:///.../viewer.html output-folder [port]');
}
fs.mkdirSync(output, {recursive: true});
const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = tabs.find(p => p.type === 'page');
if (!page) throw Error('No page in the dedicated test browser');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, {once: true}));
let nextID = 0;
const pending = new Map(), contexts = new Map(), errors = [], requests = [];
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const call = pending.get(message.id);
    if (call) {
      pending.delete(message.id);
      message.error ? call.reject(Error(JSON.stringify(message.error))) : call.resolve(message.result);
    }
  } else if (message.method === 'Runtime.executionContextCreated') {
    const context=message.params.context;if(context.auxData?.isDefault)contexts.set(context.auxData.frameId,context.id);
  } else if (message.method === 'Runtime.executionContextDestroyed') {
    for(const [frame,id]of contexts)if(id===message.params.executionContextId)contexts.delete(frame);
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  else if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
});
function rpc(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextID;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression, contextId) {
  const result = await rpc('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true,...(contextId?{contextId}:{})});
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function screenshot(name) {
  const shot = await rpc('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  fs.writeFileSync(path.join(output, name), Buffer.from(shot.data, 'base64'));
}
async function portalSmoke(){
  let initial;
  for(let n=0;n<120;n++){
    await sleep(500);initial=await evaluate('window.portalState?portalState():null');
    if(initial?.status==='error')throw Error('Initial portal load failed');
    if(initial?.status==='ready')break;
    if(n===119)throw Error('Initial portal timeout');
  }
  if(initial.frames!==1||initial.cases.length!==6)throw Error('Expected six cases and one loaded frame');
  const rows=await evaluate('catalog.cases');
  const sequence=[initial.case,...initial.cases.filter(c=>c!==initial.case),initial.case,initial.cases[0]];
  const checks=[];
  for(let index=0;index<sequence.length;index++){
    const name=sequence[index],entry=rows.find(r=>r.case===name);
    if(index)await evaluate(`document.getElementById('caseSelect').value=${JSON.stringify(name)};document.getElementById('caseSelect').dispatchEvent(new Event('change',{bubbles:true}));`);
    let state;
    for(let n=0;n<120;n++){
      await sleep(500);state=await evaluate('portalState()');
      if(state.status==='error')throw Error('Case load failed '+name);
      if(state.status==='ready'&&state.case===name)break;
      if(n===119)throw Error('Case readiness timeout '+name);
    }
    if(state.frames!==1)throw Error('Old case frame not removed');
    const tree=await rpc('Page.getFrameTree');
    const child=tree.frameTree.childFrames?.[0]?.frame.id,context=contexts.get(child);
    if(!context)throw Error('Missing case execution context');
    const data=await evaluate(`({case:D.metadata.case,info:viewerInfo(),originalDice:D.metadata.metrics.dice,
      cleanedDice:D.metadata.cleanup.metrics.dice,wl:document.getElementById('wl').value,
      subtitle:document.getElementById('subtitle').textContent,
      pageHeight:document.documentElement.scrollHeight,height:innerHeight,
      panels:['view3d','slice0','slice1','slice2'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {top:r.top,bottom:r.bottom}})})`,context);
    if(data.case!==name||data.info.variant!=='original'||data.wl!=='-300'||data.info.position[0]!==entry.initial_slice||
       JSON.stringify(data.info.shape)!==JSON.stringify(entry.shape)||data.originalDice!==entry.original_dice||data.cleanedDice!==entry.cleaned_dice)throw Error('Wrong/stale case content '+name);
    if(data.pageHeight>data.height+1||data.panels.some(p=>p.top<0||p.bottom>data.height))throw Error('Embedded four-panel layout scrolls');
    const switched=await evaluate(`(()=>{
      const selector=document.getElementById('variant');selector.value='cleaned';selector.dispatchEvent(new Event('input',{bubbles:true}));
      if(P!==cleanedP||meshData.prediction!==cleanedMesh)throw Error('Cleanup mask/mesh not synchronized');
      if(!document.getElementById('metrics').textContent.includes(D.metadata.cleanup.metrics.dice.toFixed(4)))throw Error('Case metrics stale');
      clearTimeout(plotTimer);return updatePlot().then(()=>({info:viewerInfo(),faces:document.getElementById('plot').data[1].i.length,expected:cleanedMesh.faces.length/3}));
    })()`,context);
    if(switched.faces!==switched.expected||switched.info.variant!=='cleaned')throw Error('Wrong rendered cleanup surface');
    if(entry.body_dice!==undefined){
      const bodyChecks=await evaluate(`(async()=>{
        const result=[];
        for(const name of ['body','body_cleaned']){
          const selector=document.getElementById('variant');selector.value=name;selector.dispatchEvent(new Event('input',{bubbles:true}));
          if(P!==variantMasks[name]||meshData.prediction!==variantMeshes[name])throw Error('Body variant mask/mesh not synchronized');
          if(!document.getElementById('metrics').textContent.includes(D.metadata.variants[name].metrics.dice.toFixed(4)))throw Error('Body variant metrics stale');
          clearTimeout(plotTimer);await updatePlot();
          const faces=document.getElementById('plot').data[1].i.length;
          if(faces!==variantMeshes[name].faces.length/3)throw Error('Body surface not rendered');
          result.push({name,dice:D.metadata.variants[name].metrics.dice,faces});
        }
        if(D.metadata.case==='LCTSC-Train-S3-002'){
          const z=186,start=z*shape[1]*shape[2],end=start+shape[1]*shape[2];
          const count=a=>a.slice(start,end).reduce((sum,v)=>sum+v,0);
          const pixels={expert:count(E),original:count(rawP),old_cleaned:count(cleanedP),body:count(variantMasks.body),combined:count(variantMasks.body_cleaned)};
          if(pixels.expert!==0||pixels.original!==5257||pixels.old_cleaned!==5200||pixels.body!==0||pixels.combined!==0)throw Error('Slice 186 fixture not cleared in 2D stage');
          document.getElementById('number0').value=z;document.getElementById('number0').dispatchEvent(new Event('input',{bubbles:true}));
          document.getElementById('bodyBox').checked=true;document.getElementById('bodyBox').dispatchEvent(new Event('input',{bubbles:true}));
          if(!document.getElementById('readout0').textContent.includes('模型 0 像素'))throw Error('Slice 186 readout wrong');
          clearTimeout(plotTimer);await updatePlot();result.push({slice186:pixels,bodyPlan:D.body_geometry.plans[z]});
        }
        return result;
      })()`,context);
      if(bodyChecks[0].dice!==entry.body_dice||bodyChecks[1].dice!==entry.body_cleaned_dice)throw Error('Catalog body metrics mismatch');
      data.bodyChecks=bodyChecks;
      if(name==='LCTSC-Train-S3-002')await screenshot('body_gate_slice186.png');
    }
    if(index===0)await screenshot('portal_initial_cleaned.png');
    if(index===1)await screenshot('portal_other_case.png');
    // Deliberately change controls, then revisit a case to verify full reset.
    await evaluate(`document.getElementById('wl').value=123;document.getElementById('wl').dispatchEvent(new Event('input',{bubbles:true}));clearTimeout(plotTimer);`,context);
    checks.push({case:name,generation:state.generation,shape:data.info.shape,initialSlice:data.info.position[0],originalDice:data.originalDice,cleanedDice:data.cleanedDice,cleanupFaces:switched.faces,...(data.bodyChecks?{bodyChecks:data.bodyChecks}:{})});
    console.log('Portal case passed',JSON.stringify(checks.at(-1)));
  }
  if(requests.some(s=>/^https?:/.test(s))||errors.length)throw Error('Online request or runtime error during case switching');
  const result={status:'passed_offline_multi_case_smoke',initial,checks,errors,requests};
  fs.writeFileSync(path.join(output,'portal_smoke.json'),JSON.stringify(result,null,2));
  console.log('Multi-case portal passed',checks.length,'loads; only one frame at a time');
}
try {
  await rpc('Page.enable'); await rpc('Runtime.enable'); await rpc('Network.enable');
  await rpc('Network.emulateNetworkConditions', {offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0});
  await rpc('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await rpc('Page.navigate', {url});
  if(url.endsWith('/index.html')){await portalSmoke()}else{
  let initial;
  for (let n = 0; n < 90; n++) {
    await sleep(1000);
    const state = JSON.parse(await evaluate(`JSON.stringify({info:window.viewerInfo?viewerInfo():null,
      error:window.viewerError,status:document.getElementById('status')?.textContent})`));
    if (state.error) throw Error(state.error);
    if (state.info) {initial = state; break;}
    if (n === 89) throw Error('Viewer readiness timeout');
  }
  console.log('Initial', JSON.stringify(initial));
  await screenshot('initial.png');
  const colorChecks = await evaluate(`(()=>{
    const display=document.getElementById('display'),slider=document.getElementById('maskOpacity'),source=document.getElementById('sliceSource');
    if(display.value!=='alpha')throw Error('Color blocks are not the default');
    if(source.value!=='overlay')throw Error('Unexpected initial slice source');
    const before=viewerInfo(),beforeReadouts=[0,1,2].map(axis=>document.getElementById('readout'+axis).textContent),result=[],planes=[];
    for(let axis=0;axis<3;axis++){
      const {h,w,a,b,hu}=planeMasks(axis),selected={},edges={};
      const cx=axis===2?position[1]:position[2],cy=axis===0?position[1]:shape[0]-1-position[0];
      const boundary=(mask,r,c)=>mask[r*w+c]&&(!mask[(r-1)*w+c]||!mask[(r+1)*w+c]||!mask[r*w+c-1]||!mask[r*w+c+1]);
      for(let r=2;r<h-2;r++)for(let c=2;c<w-2;c++){
        if(Math.abs(r-cy)<3 || Math.abs(c-cx)<3)continue;
        const i=r*w+c,key=a[i]?(b[i]?'both':'expert'):(b[i]?'model':'background'),ae=!!boundary(a,r,c),pe=!!boundary(b,r,c);
        const edgeKey=ae?(pe?'both':'expert'):(pe?'model':'background');
        const point={r,c,hu:hu[i],a:!!a[i],b:!!b[i],ae,pe};
        if(!selected[key])selected[key]=point;if(!edges[edgeKey])edges[edgeKey]=point;
      }
      if(Object.keys(selected).length!==4)throw Error('Need all four mask classes in fixture plane '+axis);
      planes.push({axis,selected,edges,context:document.getElementById('canvas'+axis).getContext('2d')});
    }
    const emit=element=>element.dispatchEvent(new Event('input',{bubbles:true}));
    for(const which of ['overlay','expert','prediction'])for(const opacity of [0,.45,1]){
      source.value=which;slider.value=opacity;emit(source);
      for(const plane of planes)for(const [key,p] of Object.entries(plane.selected)){
        const gray=Math.max(0,Math.min(255,(p.hu-(-300-1400/2))*255/1400));
        const fill=which==='expert'?(p.a?[0,230,230]:null):which==='prediction'?(p.b?[255,165,60]:null):
          key==='both'?[70,210,110]:key==='expert'?[0,230,230]:key==='model'?[255,165,60]:null;
        const expected=[0,1,2].map(k=>fill?(1-opacity)*gray+opacity*fill[k]:gray);
        const actual=Array.from(plane.context.getImageData(p.c,p.r,1,1).data).slice(0,3);
        if(actual.some((value,k)=>Math.abs(value-expected[k])>1))throw Error('Incorrect block color: '+JSON.stringify({axis:plane.axis,which,key,opacity,expected,actual}));
        result.push({axis:plane.axis,which,key,opacity,actual});
      }
    }
    display.value='contour';
    for(const which of ['overlay','expert','prediction']){
      source.value=which;emit(display);
      if(source.disabled)throw Error('Contour source disabled');
      for(const plane of planes)for(const [key,p] of Object.entries(plane.edges)){
        const ae=which!=='prediction'&&p.ae,pe=which!=='expert'&&p.pe;
        const gray=Math.max(0,Math.min(255,(p.hu+1000)*255/1400));
        const expected=ae&&pe?[255,255,255]:ae?[0,235,235]:pe?[255,165,60]:[gray,gray,gray];
        const actual=Array.from(plane.context.getImageData(p.c,p.r,1,1).data).slice(0,3);
        if(actual.some((v,k)=>Math.abs(v-expected[k])>1))throw Error('Contour source did not hide other boundary');
      }
    }
    source.value='expert';display.value='error';emit(display);
    if(!source.disabled)throw Error('Error display must explicitly compare both masks');
    for(const plane of planes)for(const [key,p] of Object.entries(plane.selected)){
      const expected=key==='both'?[150,150,150]:key==='model'?[255,40,40]:key==='expert'?[30,130,255]:[0,0,0];
      const actual=Array.from(plane.context.getImageData(p.c,p.r,1,1).data).slice(0,3);
      if(actual.some((v,k)=>v!==expected[k]))throw Error('Error display incorrectly filtered by source');
    }
    display.value='ct';emit(display);if(!source.disabled)throw Error('CT-only source should be disabled');
    display.value='alpha';emit(display);if(source.disabled||source.value!=='expert')throw Error('Source selection not restored');
    if(!document.getElementById('sliceLegend').textContent.includes('完整专家'))throw Error('Single expert legend misleading');
    source.value='prediction';emit(source);if(!document.getElementById('sliceLegend').textContent.includes('完整模型'))throw Error('Single model legend misleading');
    source.value='overlay';slider.value=.45;emit(source);
    if(JSON.stringify(before)!==JSON.stringify(viewerInfo()) || JSON.stringify(beforeReadouts)!==JSON.stringify([0,1,2].map(axis=>document.getElementById('readout'+axis).textContent)))throw Error('Display filtering changed data or metrics');
    if(document.getElementById('mode').value!=='overlay')throw Error('Slice source unexpectedly changed 3D');
    return result;
  })()`);
  const layouts = [];
  for (const [width,height] of [[1366,768],[1920,1080],[1100,650]]) {
    await rpc('Emulation.setDeviceMetricsOverride', {width,height,deviceScaleFactor:1,mobile:false});
    await sleep(1000);
    const layout = JSON.parse(await evaluate(`JSON.stringify({width:innerWidth,height:innerHeight,
      scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,
      panels:['view3d','slice0','slice1','slice2'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();
        return {id,left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height}}),
      canvases:[0,1,2].map(axis=>{const c=document.getElementById('canvas'+axis),r=c.getBoundingClientRect(),b=c.parentElement.getBoundingClientRect();
        return {axis,left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height,
          box:{left:b.left,top:b.top,right:b.right,bottom:b.bottom}}})})`));
    if (layout.scrollWidth > width + 1 || layout.scrollHeight > height + 1) throw Error('Desktop page requires scrolling: '+JSON.stringify(layout));
    for (const p of layout.panels) if(p.height < 100 || p.top < 0 || p.bottom > height || p.right > width) throw Error('Panel outside screen');
    if (!(layout.panels[0].left < layout.panels[1].left && layout.panels[0].top === layout.panels[1].top &&
          layout.panels[2].top > layout.panels[0].top && layout.panels[2].top === layout.panels[3].top)) throw Error('Not a four-panel grid');
    for (const c of layout.canvases) {
      if(c.width < 40 || c.height < 40 || c.left < c.box.left-1 || c.top < c.box.top-1 || c.right > c.box.right+1 || c.bottom > c.box.bottom+1) throw Error('Canvas does not fit panel');
    }
    layouts.push(layout);
    console.log('Layout passed', width,height);
    if(width===1920)await screenshot('desktop1920.png');
  }
  await rpc('Emulation.setDeviceMetricsOverride', {width:450,height:900,deviceScaleFactor:1,mobile:false});
  await sleep(1000);
  const narrow = JSON.parse(await evaluate(`JSON.stringify({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
    scrollHeight:document.documentElement.scrollHeight,
    panels:['view3d','slice0','slice1','slice2'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {left:r.left,top:r.top}})})`));
  if(narrow.scrollWidth > 451 || narrow.scrollHeight <= 900 || !narrow.panels.every((p,i,a)=>!i || (p.left===a[0].left && p.top>a[i-1].top)))throw Error('Narrow responsive layout failed');
  await rpc('Emulation.setDeviceMetricsOverride', {width:1366,height:768,deviceScaleFactor:1,mobile:false});
  await sleep(1000);
  const click = JSON.parse(await evaluate(`const canvas=document.getElementById('canvas0'),rect=canvas.getBoundingClientRect();
    const event=new MouseEvent('click',{bubbles:true,clientX:rect.left+rect.width*.25,clientY:rect.top+rect.height*.75});
    const expected=[Math.floor((event.clientY-rect.top)*canvas.height/rect.height),Math.floor((event.clientX-rect.left)*canvas.width/rect.width)];
    canvas.dispatchEvent(event); JSON.stringify({info:viewerInfo(),expected})`));
  if(click.info.position[1]!==click.expected[0] || click.info.position[2]!==click.expected[1])throw Error('Resized canvas click mapping failed: '+JSON.stringify(click));
  const changed = JSON.parse(await evaluate(`document.getElementById('number0').value=79;
    document.getElementById('number0').dispatchEvent(new Event('input',{bubbles:true})); JSON.stringify(viewerInfo())`));
  if (changed.position[0] !== Math.min(79, changed.shape[0] - 1)) throw Error('Slice input not synchronized');
  await sleep(1500); await screenshot('slice79.png');
  const probes = [];
  for (const coords of [[72,160,187], [79,208,176], [79,253,371]]) {
    const result = JSON.parse(await evaluate(`for(const [axis,value] of ${JSON.stringify(coords)}.entries()) {
      const element=document.getElementById('number'+axis); element.value=value;
      element.dispatchEvent(new Event('input',{bubbles:true})); } JSON.stringify(viewerInfo())`));
    probes.push(result);
  }
  const options = JSON.parse(await evaluate(`document.getElementById('mode').value='expert';
    document.getElementById('mode').dispatchEvent(new Event('input',{bubbles:true}));
    document.getElementById('cut').value='high';
    document.getElementById('cut').dispatchEvent(new Event('input',{bubbles:true}));
    document.getElementById('display').value='error';
    document.getElementById('display').dispatchEvent(new Event('input',{bubbles:true}));
    JSON.stringify(viewerInfo())`));
  await sleep(2500); await screenshot('clipped_error.png');
  const state = JSON.parse(await evaluate(`JSON.stringify({visible:document.getElementById('plot').data.slice(0,2).map(t=>t.visible),
    planeVertices:document.getElementById('plot').data[2].x.length,
    counts:document.getElementById('plot').data.slice(0,2).map(t=>t.i.length)})`));
  if (!state.visible[0] || state.visible[1]) throw Error('Expert/model visibility control failed');
  if (state.planeVertices !== 4) throw Error('Missing synchronized physical plane');
  const variantChecks=await evaluate(`(()=>{
    const selector=document.getElementById('variant');
    if(selector.value!=='original')throw Error('Original prediction must remain default');
    if(!D.cleaned){if(!selector.disabled)throw Error('Missing cleanup not disabled');return {available:false}}
    const changedIndex=rawP.findIndex((v,i)=>v&&!cleanedP[i]);
    if(changedIndex<0)throw Error('No removed voxel to verify');
    const x=changedIndex%shape[2],y=Math.floor(changedIndex/shape[2])%shape[1],z=Math.floor(changedIndex/(shape[1]*shape[2]));
    for(const [axis,value]of [z,y,x].entries()){
      const input=document.getElementById('number'+axis);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));
    }
    const original=viewerInfo(),originalMetrics=document.getElementById('metrics').textContent;
    selector.value='cleaned';selector.dispatchEvent(new Event('input',{bubbles:true}));
    const cleaned=viewerInfo(),cleanedMetrics=document.getElementById('metrics').textContent;
    if(original.prediction!==1||cleaned.prediction!==0||original.expert!==0||cleaned.expert!==0||original.hu!==cleaned.hu)throw Error('Variant voxel mismatch');
    if(P!==cleanedP||meshData.prediction!==cleanedMesh)throw Error('Mask and 3D mesh not synchronized');
    if(!cleanedMetrics.includes(D.metadata.cleanup.metrics.dice.toFixed(4)))throw Error('Cleanup metrics not updated');
    if(!document.getElementById('readout0').textContent.includes('模型 0 像素'))throw Error('Cleared empty slice metrics incorrect');
    selector.value='original';selector.dispatchEvent(new Event('input',{bubbles:true}));
    if(P!==rawP||meshData.prediction!==rawMesh||JSON.stringify(viewerInfo())!==JSON.stringify(original)||document.getElementById('metrics').textContent!==originalMetrics)throw Error('Original not exactly restored');
    selector.value='cleaned';selector.dispatchEvent(new Event('input',{bubbles:true}));
    document.getElementById('mode').value='prediction';document.getElementById('mode').dispatchEvent(new Event('input',{bubbles:true}));
    document.getElementById('cut').value='off';document.getElementById('cut').dispatchEvent(new Event('input',{bubbles:true}));
    document.getElementById('display').value='alpha';document.getElementById('display').dispatchEvent(new Event('input',{bubbles:true}));
    return {available:true,original,cleaned,originalMetrics,cleanedMetrics};
  })()`);
  await sleep(2500);await screenshot('cleaned.png');
  if(variantChecks.available){
    const cleanedState=await evaluate(`({name:document.getElementById('plot').data[1].name,faces:document.getElementById('plot').data[1].i.length,expected:cleanedMesh.faces.length/3})`);
    if(cleanedState.faces!==cleanedState.expected||!cleanedState.name.includes('清理'))throw Error('Rendered cleanup mesh mismatch');
    variantChecks.rendered=cleanedState;
  }
  if (requests.some(s => /^https?:/.test(s))) throw Error('Viewer requested online resources');
  if (errors.length) throw Error('Runtime errors: ' + JSON.stringify(errors));
  const report = {status:'passed_offline_browser_smoke', initial, colorChecks, layouts, narrow, click, changed, probes, options, state, variantChecks, errors, requests};
  fs.writeFileSync(path.join(output, 'smoke.json'), JSON.stringify(report, null, 2));
  console.log('Browser smoke passed', JSON.stringify({probes, state, errors: errors.length}));
  }
} catch (error) {
  fs.writeFileSync(path.join(output, 'smoke_failure.json'), JSON.stringify({error:String(error), errors, requests}, null, 2));
  throw error;
} finally {
  try { await rpc('Browser.close'); } catch {}
  ws.close();
}
