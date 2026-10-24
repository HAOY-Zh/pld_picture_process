// Dedicated browser + synthetic server only. Never operates the user's live tab.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {chromium}=require('C:/Users/17821/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'../..'),folder=path.resolve(process.argv[2]||path.join(root,'data/nodule_integration_checks/browser'));
fs.mkdirSync(folder,{recursive:true});
let server,browser;
(async()=>{
  server=spawn(path.join(root,'.venv-web/Scripts/python.exe'),[path.join(__dirname,'ct_nodule_browser_fixture.py'),'--folder',folder],{cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe']});
  const url=await new Promise((resolve,reject)=>{let output='';server.stdout.on('data',data=>{output+=data;const m=output.match(/TEST_URL=(http:\/\/127\.0\.0\.1:\d+)/);if(m)resolve(m[1]);});server.stderr.on('data',data=>fs.appendFileSync(path.join(folder,'server.log'),data));server.on('exit',code=>reject(Error('Fixture exited '+code)));setTimeout(()=>reject(Error('Fixture startup timed out')),20000).unref();});
  browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--enable-unsafe-swiftshader']});
  const page=await browser.newPage({viewport:{width:1700,height:1150}}),errors=[],computations=[];let mipRequests=0;
  page.on('request',r=>{if(new URL(r.url()).pathname==='/api/nodule/mip')mipRequests++;});
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/(run|ldct\/run|basic\/run|roi\/run|nodule\/run)$/.test(new URL(r.url()).pathname))computations.push(new URL(r.url()).pathname);});
  await page.goto(url);await page.waitForFunction(()=>state?.shape);
  await page.locator('[data-tab="segmentation"]').click();
  const frame=page.frameLocator('#segmentationFrame');
  await frame.locator('#c0').waitFor();
  await page.waitForFunction(()=>$('segmentationFrame').contentWindow.ctWorkspaceView?.ready());
  assert.equal(await page.locator('#noduleRun').isDisabled(),true);
  await page.locator('.nodule-connect summary').click();await page.locator('#noduleFixture').click();
  await page.locator('.nodule-candidate').first().waitFor();
  await page.waitForFunction(()=>$('segmentationFrame').contentWindow.ctWorkspaceNodules?.selected()==='test_00001');
  async function waitMarkers(){await page.waitForFunction(()=>{const f=$('segmentationFrame').contentWindow;return f.document.getElementById('scene').data?.some(t=>t.meta?.ct_nodule_marker);});}
  await waitMarkers();
  const consistency=await page.evaluate(()=>{const f=$('segmentationFrame').contentWindow,p=f.ctWorkspaceNodules.packet(),c=p.candidates[0],s=f.document.getElementById('scene').data.find(t=>t.meta?.ct_nodule_marker);return {indices:f.ctWorkspaceView.currentIndices(),center:c.center_index_zyx,world:c.center_lps_xyz_mm,drawn:[s.x[0],s.y[0],s.z[0]],roi:[$('roiY').value,$('roiHeight').value,$('roiX').value,$('roiWidth').value],box:CTNodule.displayBox(c,state.shape,state.geometry.spacing_zyx_mm,3),test:$('noduleBadge').textContent};});
  assert.deepEqual(consistency.indices,consistency.center.map(Math.round));
  assert.deepEqual(consistency.world,consistency.drawn);
  const b=consistency.box;assert.deepEqual(consistency.roi.map(Number),[b[2],b[3]-b[2],b[4],b[5]-b[4]]);
  assert.match(consistency.test,/测试/);
  await page.locator('.nodule-candidate').nth(1).click();await page.locator('.nodule-candidate').nth(2).click();
  await page.waitForFunction(()=>$('segmentationFrame').contentWindow.ctWorkspaceNodules.selected()==='test_00003');
  await page.waitForFunction(()=>{const f=$('segmentationFrame').contentWindow;return f.document.getElementById('scene').data?.find(t=>t.meta?.ct_nodule_marker)?.marker.size[2]===11;});
  // Exercise Plotly's real event handler, using its customdata candidate identity.
  await page.evaluate(()=>{const f=$('segmentationFrame').contentWindow,scene=f.document.getElementById('scene'),trace=scene.data.find(t=>t.meta?.ct_nodule_marker);scene.emit('plotly_click',{points:[{data:trace,customdata:'test_00002'}]});});
  await page.waitForFunction(()=>document.querySelector('.nodule-candidate[data-candidate-id="test_00002"]').getAttribute('aria-pressed')==='true');
  await page.locator('.nodule-mip summary').click();await page.locator('#noduleMipRun').click();
  await page.waitForFunction(()=>$('noduleMipNote').textContent.includes('Z 层段'));
  assert.match(await page.locator('#noduleMipNote').innerText(),/test_00002/);
  const requestsAfterMip=mipRequests;await page.locator('#noduleMipWindow').selectOption('soft');
  await page.locator('#noduleWindow').selectOption('soft');assert.equal(mipRequests,requestsAfterMip);
  await page.locator('.nodule-mip').screenshot({path:path.join(folder,'local_mip_TEST_ONLY.png')});
  let releaseMip,startedMip;const mipGate=new Promise(r=>releaseMip=r),mipStarted=new Promise(r=>startedMip=r);
  await page.route('**/api/nodule/mip?*',async route=>{const response=await route.fetch();startedMip();await mipGate;await route.fulfill({response});});
  await page.locator('#noduleMipRun').click();await mipStarted;
  await page.locator('.nodule-candidate').nth(0).click();releaseMip();await page.waitForTimeout(300);
  assert.doesNotMatch(await page.locator('#noduleMipNote').innerText(),/test_00002/);
  await page.unroute('**/api/nodule/mip?*');await page.locator('.nodule-candidate').nth(1).click();
  await page.locator('#noduleShowAll').uncheck();
  await page.waitForFunction(()=>{const f=$('segmentationFrame').contentWindow;return f.document.getElementById('scene').data.find(t=>t.meta?.ct_nodule_marker).x.length===1;});
  const gold=()=>page.evaluate(()=>{const data=$('segmentationFrame').contentWindow.document.getElementById('c0').getContext('2d').getImageData(0,0,$('segmentationFrame').contentWindow.document.getElementById('c0').width,$('segmentationFrame').contentWindow.document.getElementById('c0').height).data;let count=0;for(let i=0;i<data.length;i+=4)if(data[i]===244&&data[i+1]===198&&data[i+2]===108)count++;return count;});
  await page.waitForTimeout(150);assert.ok(await gold()>0);
  await frame.locator('#n0').fill('0');await frame.locator('#n0').dispatchEvent('input');
  await page.waitForFunction(()=>index===0&&$('segmentationFrame').contentWindow.ctWorkspaceView.currentIndices()[0]===0);
  await page.waitForTimeout(150);assert.equal(await gold(),0);
  await page.locator('#noduleLocate').click();
  await page.waitForFunction(()=>index===6);
  await page.locator('.nodule-connect summary').click();
  await page.screenshot({path:path.join(folder,'linked_TEST_ONLY.png'),fullPage:true});
  await page.locator('.nodule-layout').screenshot({path:path.join(folder,'linked_panel_TEST_ONLY.png')});
  assert.deepEqual(computations,[]);
  await page.locator('#noduleToRoi').click();await page.locator('#roiBackend').selectOption('CPU_REFERENCE');
  assert.match(await page.locator('#roiSourceNote').innerText(),/test_00002/);
  assert.equal(await page.locator('#roiMode').inputValue(),'range');
  await page.locator('#roiRun').click();await page.waitForFunction(()=>state.roi.result?.candidate_reference?.candidate_id==='test_00002'&&state.phase==='done');
  assert.deepEqual(computations,['/api/roi/run']);
  assert.match(await page.locator('#roiStatsNote').innerText(),/总体标准差/);
  await page.locator('[data-tab="segmentation"]').click();
  // A finished result from an old request must not resurrect after clear.
  let release,intercepted;const gate=new Promise(r=>release=r),started=new Promise(r=>intercepted=r);
  await page.route('**/api/nodule/result?*',async route=>{const response=await route.fetch();intercepted();await gate;await route.fulfill({response});});
  await page.locator('.nodule-connect summary').click();await page.locator('#noduleFixture').click();await started;
  await page.evaluate(async()=>apply(await api('/api/nodule/clear',{case_id:state.case_id,analysis_source_id:state.analysis_source_id,context_id:state.nodule.context.context_id})));
  release();await page.waitForTimeout(1000);assert.equal(await page.locator('.nodule-candidate').count(),0);
  await page.unroute('**/api/nodule/result?*');
  await page.locator('#noduleFixture').click();await page.locator('.nodule-candidate').first().waitFor();await waitMarkers();
  // Import an explicit empty test result through the same file input as real results.
  const empty=await page.evaluate(()=>{const p=structuredClone($('segmentationFrame').contentWindow.ctWorkspaceNodules.packet());p.result_id='test_empty';p.candidates=[];p.candidate_count=0;return p;});
  await page.locator('#noduleFile').setInputFiles({name:'empty_TEST_ONLY.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(empty))});
  await page.waitForFunction(()=>state.nodule.result?.candidate_count===0&&$('noduleCount').textContent.includes('未检出'));
  await page.waitForFunction(()=>!$('segmentationFrame').contentWindow.document.getElementById('scene').data?.some(t=>t.meta?.ct_nodule));
  assert.equal(await page.locator('.nodule-candidate').count(),0);assert.equal(await page.locator('#noduleToRoi').isDisabled(),true);
  // Refill, then reloading CT invalidates candidates and old auto-ROI state.
  await page.locator('#noduleFixture').click();await page.locator('.nodule-candidate').first().waitFor();
  await page.evaluate(()=>{$('path').value=state.source.source;$('confirm').checked=true;$('loadExpert').checked=false;$('load').click();});
  await page.waitForFunction(()=>state.phase==='idle'&&!state.nodule.result&&!$('noduleList').children.length);
  const layouts=[];
  for(const width of [1700,1280,900,390]){
    await page.setViewportSize({width,height:1200});await page.waitForTimeout(100);
    const d=await page.evaluate(()=>({width:innerWidth,content:document.documentElement.scrollWidth}));layouts.push(d);assert.ok(d.content<=d.width+1,JSON.stringify(d));
  }
  assert.deepEqual(errors,[]);
  const result={status:'passed_synthetic_browser_integration',models_executed:0,board_accessed:false,checks:['candidate to three planes and ROI','LPS coordinates with flipped direction and anisotropy','latest rapid selection','3D event back to list','hide other markers','outside slice hides box','ROI record provenance','local MIP and windowing without recomputation','stale MIP response after selection ignored','ROI population std visible','late response after clear ignored','empty result removes overlays','new case invalidates','responsive layout'],layouts,errors};
  fs.writeFileSync(path.join(folder,'validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(server)server.kill();});
