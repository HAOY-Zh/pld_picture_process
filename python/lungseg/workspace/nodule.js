'use strict';
(() => {
  const G=window.CTNodule,frame=$('segmentationFrame');
  let packet=null,selectedId=null,resultKey=null,contextKey=null,ticket=0,zoomTicket=0,zoomKey=null,zoomImage=null,localError='',selectionRevision=0;
  let mipImage=null,mipTicket=0,mipKey=null,mipBusy=false,mipMessage='';
  let visibleRows=100;
  const selected=()=>packet?.candidates.find(c=>c.candidate_id===selectedId)||null;
  const context=()=>state?.nodule?.context;
  const busy=()=>ACTIVE.includes(state?.phase);
  const padding=()=>Number($('nodulePadding').value);
  const valid=()=>G.matches(packet,state);
  const identity=()=>({case_id:state.case_id,analysis_source_id:state.analysis_source_id,context_id:context().context_id});
  const reference=()=>({context_id:packet.context_id,result_id:packet.result_id,candidate_id:selectedId,padding_mm:padding()});
  const mipSelectionKey=()=>valid()&&selected()?[packet.context_id,packet.result_id,selectedId,padding()].join(':'):null;
  function status(text,error=false){$('noduleStatus').textContent=text;$('noduleStatus').className='helper'+(error?' nodule-error':packet?.is_test_data?' nodule-test-note':'');}
  function send(recenter=false){
    frame.contentWindow?.postMessage({type:'ct-nodule-link',packet:valid()?packet:null,selected_id:selectedId,
      show_all:$('noduleShowAll').checked,recenter,selection_revision:selectionRevision},location.origin);
  }
  function clear(){
    packet=null;selectedId=null;zoomKey=null;zoomImage=null;zoomTicket++;selectionRevision++;
    visibleRows=100;
    $('noduleList').replaceChildren();window.dispatchEvent(new CustomEvent('ct-nodule-roi',{detail:null}));
    send();render();
  }
  function applyRoi(){
    if(!valid()||!selected())return;
    const b=G.displayBox(selected(),state.shape,state.geometry.spacing_zyx_mm,padding());
    window.dispatchEvent(new CustomEvent('ct-nodule-roi',{detail:{...identity(),candidate_reference:reference(),
      bbox_zyx:selected().bbox_zyx.slice(),display_bbox_zyx:b,is_test_data:packet.is_test_data}}));
  }
  function select(id,recenter=true){
    if(!valid())return;const c=packet.candidates.find(c=>c.candidate_id===id);if(!c)return;
    selectedId=id;selectionRevision++;
    if(!Array.from($('noduleList').children).some(b=>b.dataset.candidateId===id))list();
    if(recenter)setIndex(c.center_index_zyx[0]);
    applyRoi();send(recenter);render();fetchZoom();
  }
  function list(){
    const target=$('noduleList');target.replaceChildren();
    const all=packet?.candidates||[],shown=all.slice(0,visibleRows),chosen=selected();
    if(chosen&&!shown.includes(chosen))shown.push(chosen);
    for(const c of shown){
      const button=document.createElement('button');button.type='button';button.className='nodule-candidate';button.dataset.candidateId=c.candidate_id;
      const title=document.createElement('strong');title.textContent=c.candidate_id;
      const score=document.createElement('small');score.textContent=`估计 ${c.estimated_scale_diameter_mm.toLocaleString('zh-CN',{maximumFractionDigits:2})} mm · 分数 ${c.score}`;
      button.append(title,score);button.onclick=()=>select(c.candidate_id);target.append(button);
    }
    if(all.length>visibleRows){const more=document.createElement('button');more.className='button secondary';more.textContent=`再显示 100 项（共 ${all.length} 项）`;more.onclick=()=>{visibleRows+=100;list();render();};target.append(more);}
  }
  function render(){
    const has=!!state?.shape,c=valid()?selected():null;
    for(const id of ['noduleContext','noduleFile','noduleFixture'])$(id).disabled=!has||busy();
    $('noduleClear').disabled=!packet||busy();$('noduleExport').disabled=!packet;
    $('noduleToRoi').disabled=$('noduleLocate').disabled=!c||busy();
    $('nodulePadding').disabled=!c||busy();$('noduleWindow').disabled=!c;
    const key=mipSelectionKey();
    if(key!==mipKey){mipKey=key;mipImage=null;mipTicket++;mipBusy=false;mipMessage='';}
    $('noduleMipRun').disabled=!c||busy()||mipBusy;
    $('noduleMipWindow').disabled=!mipImage;
    $('noduleMipNote').textContent=mipMessage||(c?'点击生成当前来源 HU 的局部 Z 向投影；调窗和浏览切片不会重新计算。':'选择候选后点击生成；投影用于观察层段内高密度结构。');
    drawMip();
    const job=state?.nodule?.job,running=job?.status==='RUNNING',online=packet?.ingestion_kind==='computed_live_fpga';
    const ready=context()?.whole_case_ready;
    $('noduleRun').disabled=!has||busy()||!ready||!state.nodule.service_available||!state.board_connected;
    $('noduleRun').textContent=running?`检测中 · ${job.completed}/${job.total||'…'} patch`:'检测当前 CT · 整层加速＋冻结评分';
    $('noduleCancel').hidden=!running;
    $('noduleManualRoi').disabled=!has||busy();
    $('noduleBadge').textContent=!packet?'尚无候选结果':packet.is_test_data?'联动测试 · 非检测结果':online?(packet.detector_provenance?.ranker_applied?'FPGA 候选 · 冻结跨层评分':'FPGA 历史候选 · 原始分数'):'导入候选 · 本机未执行检测';
    $('noduleBadge').className='chip'+(packet?.is_test_data?' amber':'');
    $('noduleCount').textContent=!packet?'尚无结果':packet.candidate_count?`${packet.is_test_data?'测试候选':'候选'} ${packet.candidate_count} 项 · 保持服务排序`:'未检出满足当前条件的候选'+(packet.is_test_data?'（测试空结果）':'');
    if(packet?.detector_provenance?.additional_score_filter!==undefined)$('noduleCount').textContent+=` · 已应用阈值 ${Number(packet.detector_provenance.additional_score_filter).toFixed(4)}`;
    const jobMessage=running?`${state.analysis_source.label} · ${job.stage==='search'?'生成搜索区域':job.stage==='merge'?'合并三维候选':`实际 FPGA 推理 ${job.completed}/${job.total} patch`}；完成后发布完整候选。`:job?.status==='FAILED'?job.error:job?.status==='CANCELLED'?'检测已停止，未发布部分候选。':null;
    const elapsedNote=job?.operation==='threshold_only_no_inference'?`仅重新筛选 · ${(job.elapsed_ms/1000).toFixed(3)} 秒，未重跑 CNN。 `:online&&job?.status==='COMPLETE'&&job.job_id===packet.result_id&&Number.isFinite(job.elapsed_ms)?`实际检测完成 · ${(job.elapsed_ms/1000).toFixed(2)} 秒 · ${job.completed} patch。 `:'';
    status(localError||jobMessage||(!has?'先加载 CT；候选绑定当前 HU 来源与几何。':packet?.is_test_data?'当前是几何联动测试数据，未执行检测，不代表检出效果。':online?`${elapsedNote}${state.analysis_source.label} · ${packet.detector_provenance?.ranker_applied?'实际 FPGA 整层复用＋冻结跨层评分':'历史 FPGA 原始候选'}；分数非恶性概率，当前输入的检测质量尚未验收。`:packet?'已导入匹配当前上下文的候选；推理来源以导入记录为准。':ready?state.nodule.service_status:'先完成当前所选 CT 的整例肺分割。也可直接框选 ROI，无需先检测。'),!!localError||job?.status==='FAILED');
    for(const button of $('noduleList').querySelectorAll('button')){button.setAttribute('aria-pressed',String(button.dataset.candidateId===selectedId));button.disabled=busy();}
    $('noduleSelected').textContent=c?`${c.candidate_id} · 中心 ZYX [${c.center_index_zyx.join(', ')}] · ${c.score_kind}`:'尚未选择候选。';
    $('noduleRoiTitle').textContent=c?`${c.candidate_id} · 自动 ROI`:'选择候选以自动设置 ROI';
    $('noduleRoiInfo').textContent=c?`${state.analysis_source.label} · 当前轴位 ${index}（源 ${state.source.source_slice_indices[index]} 层） · ${packet.is_test_data?'测试候选':online?'FPGA 实验候选':'导入候选'}`:'列表、三个切面、肺三维和 ROI 使用同一个候选编号。';
    if(c){const b=G.displayBox(c,state.shape,state.geometry.spacing_zyx_mm,padding());$('noduleBounds').textContent=`检测原框 [${c.bbox_zyx.join(', ')}]；观察 ROI ${b[5]-b[4]} × ${b[3]-b[2]} 像素，层段 [${b[0]}, ${b[1]})。${index<c.bbox_zyx[0]||index>=c.bbox_zyx[1]?'当前层在检测框范围外。':''}`;}
    else $('noduleBounds').textContent='金色为检测原框；虚线为观察用扩展框。';
    drawZoom();
  }
  async function refresh(){
    if(contextKey!==(context()?.context_id||null)){contextKey=context()?.context_id||null;localError='';}
    const next=state?.nodule?.result,key=next?[next.context_id,next.result_id].join(':'):null;
    if(key!==resultKey){resultKey=key;const request=++ticket;clear();localError='';
      if(next){try{
        const p=await api('/api/nodule/result?'+new URLSearchParams({case:next.case_id,source:next.analysis_source_id,context:next.context_id,result:next.result_id}));
        if(request!==ticket||key!==resultKey||!G.matches(p,state))return;
        packet=p;list();if(p.candidates.length)select(p.candidates[0].candidate_id);else{send();render();}
      }catch(e){if(request===ticket){localError=e.message;resultKey=null;render();}}}
    }
    if(packet&&!valid()){clear();resultKey=null;}
    render();fetchZoom();
  }
  async function fetchZoom(){
    if(!selected()||!valid()||$('tab-segmentation').hidden)return;
    const key=[state.case_id,packet.context_id,index].join(':');if(key===zoomKey)return;
    zoomKey=key;zoomImage=null;const current=++zoomTicket;drawZoom();
    try{const p=await api('/api/plane?'+new URLSearchParams({case:state.case_id,axis:0,index,stage:'body'}));
      if(current!==zoomTicket||key!==zoomKey||!valid())return;p.values=new Float32Array(bytes(p.hu).buffer);zoomImage=p;drawZoom();
    }catch(e){if(current===zoomTicket){zoomKey=null;localError=e.message;render();}}
  }
  function drawZoom(){
    const canvas=$('noduleZoom'),ctx=canvas.getContext('2d');ctx.fillStyle='#09141e';ctx.fillRect(0,0,canvas.width,canvas.height);
    const c=valid()?selected():null;
    const text=value=>{ctx.fillStyle='#c8d9df';ctx.font='14px sans-serif';ctx.textAlign='center';ctx.fillText(value,canvas.width/2,canvas.height/2);};
    if(!c)return text('选择候选后显示原生 HU 放大');
    const b=G.displayBox(c,state.shape,state.geometry.spacing_zyx_mm,padding());if(index<b[0]||index>=b[1])return text('当前层不在观察 ROI 范围内');
    if(!zoomImage)return text('读取当前切片…');
    const [h,w]=zoomImage.shape,[, ,y0,y1,x0,x1]=b,cw=x1-x0,ch=y1-y0,off=document.createElement('canvas');off.width=cw;off.height=ch;
    const oc=off.getContext('2d'),im=oc.createImageData(cw,ch),[wl,ww]=$('noduleWindow').value==='soft'?[40,400]:[-600,1500];
    for(let y=0;y<ch;y++)for(let x=0;x<cw;x++){const v=G.clamp(Math.round((zoomImage.values[(y+y0)*w+x+x0]-(wl-ww/2))/ww*255),0,255);im.data.set([v,v,v,255],(y*cw+x)*4);}
    oc.putImageData(im,0,0);const [,sy,sx]=state.geometry.spacing_zyx_mm,scale=Math.min((canvas.width-24)/(cw*sx),(canvas.height-24)/(ch*sy)),dw=cw*sx*scale,dh=ch*sy*scale,dx=(canvas.width-dw)/2,dy=(canvas.height-dh)/2;
    ctx.imageSmoothingEnabled=false;ctx.drawImage(off,dx,dy,dw,dh);ctx.strokeStyle='#ce7ed3';ctx.lineWidth=1;ctx.setLineDash([5,4]);ctx.strokeRect(dx,dy,dw,dh);ctx.setLineDash([]);
    if(c.bbox_zyx[0]<=index&&index<c.bbox_zyx[1]){const raw=G.displayBox(c,state.shape,state.geometry.spacing_zyx_mm,0);ctx.strokeStyle='#f4c66c';ctx.lineWidth=2;ctx.strokeRect(dx+(raw[4]-x0)/cw*dw,dy+(raw[2]-y0)/ch*dh,(raw[5]-raw[4])/cw*dw,(raw[3]-raw[2])/ch*dh);}
  }
  function drawMip(){
    const canvas=$('noduleMip'),ctx=canvas.getContext('2d');ctx.fillStyle='#09141e';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!mipImage){ctx.fillStyle='#c8d9df';ctx.font='14px sans-serif';ctx.textAlign='center';ctx.fillText(mipBusy?'正在生成局部 MIP…':'点击按钮生成局部 MIP',canvas.width/2,canvas.height/2);return;}
    const [h,w]=mipImage.shape_yx,[sy,sx]=mipImage.spacing_yx_mm;
    const off=document.createElement('canvas');off.width=w;off.height=h;
    const oc=off.getContext('2d'),im=oc.createImageData(w,h),[wl,ww]=$('noduleMipWindow').value==='soft'?[40,400]:[-600,1500];
    for(let i=0;i<w*h;i++){const v=G.clamp(Math.round((mipImage.values[i]-(wl-ww/2))/ww*255),0,255);im.data.set([v,v,v,255],i*4);}
    oc.putImageData(im,0,0);const scale=Math.min((canvas.width-24)/(w*sx),(canvas.height-24)/(h*sy)),dw=w*sx*scale,dh=h*sy*scale;
    ctx.imageSmoothingEnabled=false;ctx.drawImage(off,(canvas.width-dw)/2,(canvas.height-dh)/2,dw,dh);
  }
  $('noduleMipRun').onclick=async()=>{
    if(!valid()||!selected()||busy())return;
    const key=mipSelectionKey(),current=++mipTicket;
    mipBusy=true;mipMessage='正在生成局部 MIP…';render();
    try{
      const p=await api('/api/nodule/mip?'+new URLSearchParams({case:packet.case_id,source:packet.analysis_source_id,
        context:packet.context_id,result:packet.result_id,candidate:selectedId,padding:padding()}));
      if(current!==mipTicket||key!==mipSelectionKey())return;
      if(p.backend!=='CPU_DISPLAY_ONLY'||p.context_id!==packet.context_id||p.result_id!==packet.result_id||p.candidate_id!==selectedId)throw Error('MIP 来源与当前候选不一致');
      p.values=new Float32Array(bytes(p.hu).buffer);mipImage=p;
      mipMessage=`${state.analysis_source.label} · ${selectedId} · Z 层段 [${p.bbox_zyx[0]}, ${p.bbox_zyx[1]}) · 电脑显示投影，非病灶分割${p.is_test_data?' · 联动测试候选':''}。`;
    }catch(e){if(current===mipTicket)mipMessage='局部 MIP 生成失败：'+e.message;}
    finally{if(current===mipTicket){mipBusy=false;render();}}
  };
  $('noduleMipWindow').onchange=drawMip;
  async function action(name,extra={}){
    try{localError='';apply(await api('/api/nodule/'+name,{...identity(),...extra}));}
    catch(e){localError=e.message;render();}
  }
  function download(value,name){const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('noduleFixture').onclick=()=>action('fixture');$('noduleClear').onclick=()=>action('clear');
  $('noduleRun').onclick=()=>action('run',{scoring_mode:'frozen_cross_layer'});
  $('noduleCancel').onclick=()=>action('cancel',{job_id:state.nodule.job.job_id});
  $('noduleFile').onchange=async()=>{try{const file=$('noduleFile').files[0];if(!file)return;if(file.size>8*1024*1024)throw Error('候选 JSON 超过 8MB');const p=JSON.parse(await file.text());await action('import',{packet:p});}catch(e){localError=e.message;render();}finally{$('noduleFile').value='';}};
  $('noduleContext').onclick=()=>download({schema:'ct_nodule_context_v1',context:context()},'ct-nodule-context.json');
  $('noduleExport').onclick=()=>{if(valid())download(packet,'ct-nodule-'+packet.result_id+'.json');};
  $('noduleShowAll').onchange=()=>send();$('noduleLocate').onclick=()=>select(selectedId);
  $('nodulePadding').onchange=()=>{applyRoi();render();};$('noduleWindow').onchange=drawZoom;
  $('noduleToRoi').onclick=()=>{applyRoi();showTab('roi');};
  $('noduleManualRoi').onclick=()=>showTab('roi');
  frame.addEventListener('load',()=>send(true));
  window.addEventListener('message',e=>{
    if(e.origin!==location.origin||e.source!==frame.contentWindow)return;
    const m=e.data;
    if(m?.type==='ct-nodule-viewer-ready'){send(true);return;}
    if(!valid()||!m||['case_id','analysis_source_id','context_id','result_id'].some(k=>m[k]!==packet[k]))return;
    if(m.type==='ct-nodule-select')select(m.candidate_id);
    if(m.type==='ct-nodule-browse'&&Number.isInteger(m.index)&&m.index>=0&&m.index<state.shape[0]&&m.index!==index)setIndex(m.index);
  });
  window.addEventListener('ct-state-change',refresh);window.addEventListener('ct-slice-change',()=>{render();fetchZoom();});
  window.addEventListener('ct-tab-change',()=>{render();fetchZoom();if(!$('tab-segmentation').hidden)send(false);});
  refresh();
})();
