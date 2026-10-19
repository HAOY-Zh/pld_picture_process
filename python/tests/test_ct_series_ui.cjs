/* Load-button interaction tests. No FPGA, browser driver or patient data. */
const assert=require('node:assert/strict');
const test=require('node:test');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../lungseg/workspace/series.js'),'utf8');

function inspected(overrides={}){
  return {path:'/ct',series:[{series_uid:'series-A',slices:342,rows:512,columns:512,
    description:'Low Dose Images',valid:true,requires_compatibility:true,
    spacing_zyx_mm:[1,.78,.78],note:'Geometry checked',...overrides}]};
}
function setup(response=inspected()){
  const elements=new Map(),calls=[],loads=[],messages=[];
  function element(id){
    if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:false,
      hidden:true,dataset:{},textContent:'',listeners:{},children:[],
      addEventListener(name,callback){this.listeners[name]=callback;},
      replaceChildren(){this.children=[];this.value='';},
      append(child){this.children.push(child);},click(){return this.onclick?.();}});
    return elements.get(id);
  }
  element('path').value='/ct';element('loadExpert').checked=true;
  element('load').onclick=async()=>{
    loads.push({path:element('path').value,uid:element('uid').value,
      compatible:element('frameUidCompatible').checked,
      expert:element('loadExpert').checked});return {accepted:true};
  };
  const state={phase:'idle',source:{source:'/retained-case'}};
  const context={state,ACTIVE:['loading','processing','computing','connecting'],
    $:element,URLSearchParams,
    document:{createElement:()=>({value:'',textContent:''})},
    window:{addEventListener(){}},
    message:(...args)=>messages.push(args),
    api:async(route,body)=>{calls.push({route,body});return typeof response==='function'?response(route,body):response;}};
  vm.runInNewContext(source,context,{filename:'series.js'});
  return {element,calls,loads,messages,state,load:()=>element('load').onclick()};
}

test('pasted folder reveals compatibility before any load; explicit confirmation loads all',async()=>{
  const ui=setup();
  await ui.load();
  assert.equal(ui.calls.length,1);assert.equal(ui.calls[0].route,'/api/series/inspect');
  assert.equal(ui.loads.length,0);assert.equal(ui.element('seriesPanel').hidden,false);
  assert.equal(ui.element('frameUidChoice').hidden,false);
  assert.equal(ui.element('load').disabled,true);
  assert.match(ui.element('seriesStatus').textContent,/342/);
  assert.equal(ui.state.source.source,'/retained-case');
  ui.element('frameUidCompatible').checked=true;
  ui.element('frameUidCompatible').onchange();
  assert.equal(ui.element('load').disabled,false);
  await ui.load();
  assert.equal(ui.calls.length,1);assert.equal(ui.loads.length,1);
  assert.equal(ui.loads[0].uid,'series-A');assert.equal(ui.loads[0].compatible,true);
  assert.equal(ui.loads[0].expert,false);assert.equal(ui.element('loadExpert').disabled,true);
});

test('ordinary single series automatically preflights and loads without another click',async()=>{
  const ui=setup(inspected({requires_compatibility:false}));await ui.load();
  assert.equal(ui.calls.length,1);assert.equal(ui.loads.length,1);
  assert.equal(ui.loads[0].compatible,false);assert.equal(ui.loads[0].expert,true);
  assert.equal(ui.element('frameUidChoice').hidden,true);
});

test('multiple series require a selection; never load an arbitrary series',async()=>{
  const data=inspected({requires_compatibility:false});
  data.series.push({...data.series[0],series_uid:'series-B'});
  const ui=setup(data);await ui.load();assert.equal(ui.loads.length,0);
  ui.element('seriesChoice').value='series-B';ui.element('seriesChoice').onchange();
  await ui.load();assert.equal(ui.loads[0].uid,'series-B');
  assert.equal(ui.calls.length,1);
});

test('explicit Series UID still selects the requested series during automatic preflight',async()=>{
  const data=inspected({requires_compatibility:false});data.series.push({...data.series[0],series_uid:'series-B'});
  const ui=setup(data);ui.element('uid').value='series-B';await ui.load();
  assert.equal(ui.loads.length,1);assert.equal(ui.loads[0].uid,'series-B');
});

test('changing the path invalidates the reviewed compatibility choice',async()=>{
  const ui=setup();await ui.load();ui.element('frameUidCompatible').checked=true;
  ui.element('frameUidCompatible').onchange();
  ui.element('path').value='/different-case';ui.element('path').listeners.input();
  assert.equal(ui.element('frameUidCompatible').checked,false);
  assert.equal(ui.element('seriesPanel').hidden,true);assert.equal(ui.element('uid').value,'');
  assert.equal(ui.element('loadExpert').disabled,false);
});

test('a stale preflight response cannot overwrite a newly typed path or load it',async()=>{
  let resolve;const pending=new Promise(r=>{resolve=r;});const ui=setup(()=>pending);
  const loading=ui.load();ui.element('path').value='/new-path';ui.element('path').listeners.input();
  resolve(inspected());await loading;
  assert.equal(ui.element('path').value,'/new-path');assert.equal(ui.loads.length,0);
  assert.equal(ui.element('seriesPanel').hidden,true);
  assert.match(ui.messages.at(-1)[0],/路径已变化/);
});

test('invalid geometry and failed inspection preserve the existing case without loading',async()=>{
  for(const result of [inspected({valid:false,note:'Missing slice'}),()=>{throw Error('No such directory');}]){
    const ui=setup(result);await ui.load();assert.equal(ui.loads.length,0);
    assert.equal(ui.state.source.source,'/retained-case');
    assert.equal(ui.element('inspectSeries').disabled,false);
  }
});

test('NIfTI and NPZ retain their existing HU/mm confirmation workflow',async()=>{
  for(const name of ['/ct.npz','/ct.nii','/ct.nii.gz']){
    const ui=setup();ui.element('path').value=name;await ui.load();
    assert.equal(ui.calls.length,0);assert.equal(ui.loads.length,1);
  }
});

test('a single DICOM file loads as one slice and explains how to load a volume',async()=>{
  const data=inspected({slices:1,requires_compatibility:false});data.path='/one.dcm';
  const ui=setup(data);ui.element('path').value='/one.dcm';await ui.load();
  assert.equal(ui.loads.length,1);assert.match(ui.element('seriesAdvice').textContent,/当前只有一层/);
});

test('busy/double clicks cannot create concurrent preflights or duplicate loads',async()=>{
  let resolve;const pending=new Promise(r=>{resolve=r;});const ui=setup(()=>pending);
  const first=ui.load();await ui.load();assert.equal(ui.calls.length,1);
  resolve(inspected({requires_compatibility:false}));await first;assert.equal(ui.loads.length,1);
  ui.state.phase='computing';await ui.load();assert.equal(ui.loads.length,1);
});
