'use strict';
(() => {
  let context=null, contextSeries=null, selected=null, latest=null, image=null, imageKey=null, imageTicket=0;
  let payload=null, payloadKey=null, payloadTicket=0, drag=null, historyKey=null, localError='';
  const fields=['roiX','roiY','roiWidth','roiHeight','roiLow','roiHigh','roiUseHu','roiUseLung','roiMaskStage','roiMode','roiZ0','roiZ1'];
  const windows={lung:[-600,1500],soft:[40,400],bone:[400,1800]};
  const current=()=>state?.roi?.history?.find(r=>r.result_id===selected)||null;
  const visible=()=>!$('tab-roi').hidden;
  const number=id=>{
    const el=$(id),value=Number(el.value);
    if(el.value.trim()===''||!Number.isInteger(value)||!el.checkValidity())throw Error('请输入有效的 '+el.parentElement.textContent.trim());
    return value;
  };
  function parameters(){
    const x=number('roiX'),y=number('roiY'),w=number('roiWidth'),h=number('roiHeight');
    if(!state?.shape||x+w>state.shape[2]||y+h>state.shape[1])throw Error('矩形超出图像范围，请调整左侧坐标或宽高。');
    const range=$('roiMode').value==='range',z0=range?number('roiZ0'):index,z1=range?number('roiZ1')+1:index+1;
    if(z0<0||z1<=z0||z1>state.shape[0])throw Error('起止层须在已载入范围内，结束层不能早于起始层。');
    const useHu=$('roiUseHu').checked,limits=useHu?[number('roiLow'),number('roiHigh')]:null;
    if(limits&&limits[0]>limits[1])throw Error('HU 下限不能大于上限。');
    return {roi_yx:[y,y+h,x,x+w],use_hu:useHu,hu_range:limits,use_lung_mask:$('roiUseLung').checked,
      mask_stage:$('roiMaskStage').value,mode:range?'range':'slice',z_bounds:[z0,z1]};
  }
  function matches(record){
    if(!record||record.case_id!==state?.case_id||record.analysis_source_id!==state.analysis_source_id)return false;
    try{
      const p=parameters();
      return ['roi_yx','use_hu','hu_range','use_lung_mask','mode','z_bounds'].every(k=>JSON.stringify(p[k])===JSON.stringify(record[k]))
        &&(!p.use_lung_mask||p.mask_stage===record.lung_mask_stage);
    }catch{return false;}
  }
  function bounds(roi){
    const [y0,y1,x0,x1]=roi;
    $('roiX').value=x0;$('roiY').value=y0;$('roiWidth').value=x1-x0;$('roiHeight').value=y1-y0;
  }
  function restore(record){
    bounds(record.roi_yx);$('roiUseHu').checked=record.use_hu;
    if(record.hu_range)[$('roiLow').value,$('roiHigh').value]=record.hu_range;
    $('roiUseLung').checked=record.use_lung_mask;$('roiMaskStage').value=record.lung_mask_stage||'body';
    $('roiMode').value=record.mode;$('roiZ0').value=record.z_bounds[0];$('roiZ1').value=record.z_bounds[1]-1;
  }
  function fit(canvas,h,w){
    const [sy,sx]=state?.geometry?.spacing_zyx_mm?.slice(1)||[1,1];
    const scale=Math.min(canvas.width/(w*sx),canvas.height/(h*sy)),dw=w*sx*scale,dh=h*sy*scale;
    return {x:(canvas.width-dw)/2,y:(canvas.height-dh)/2,w:dw,h:dh};
  }
  function pointer(event,clamp=false){
    const canvas=$('roiCanvas'),rect=canvas.getBoundingClientRect(),[h,w]=image.shape,box=fit(canvas,h,w);
    const x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;
    if(!clamp&&(x<box.x||y<box.y||x>box.x+box.w||y>box.y+box.h))return null;
    return [Math.max(0,Math.min(w-1,Math.floor((x-box.x)/box.w*w))),Math.max(0,Math.min(h-1,Math.floor((y-box.y)/box.h*h)))];
  }
  function dragTo(point){
    const [x0,y0]=drag.start,[x1,y1]=point;
    bounds([Math.min(y0,y1),Math.max(y0,y1)+1,Math.min(x0,x1),Math.max(x0,x1)+1]);changed();
  }
  function draw(){
    const full=$('roiCanvas'),zoom=$('roiZoom');
    for(const canvas of [full,zoom]){const ctx=canvas.getContext('2d');ctx.fillStyle='#09141e';ctx.fillRect(0,0,canvas.width,canvas.height);}
    if(!image){const ctx=full.getContext('2d');ctx.fillStyle='#95b0b9';ctx.font='16px sans-serif';ctx.textAlign='center';ctx.fillText(state?.shape?'正在读取当前切片…':'载入 CT 后拖动框选',256,256);return;}
    const [h,w]=image.shape,[wl,ww]=windows[$('roiWindow').value],off=document.createElement('canvas');off.width=w;off.height=h;
    const ctx=off.getContext('2d'),rgba=ctx.createImageData(w,h),record=current();
    const mask=$('roiOverlay').checked&&matches(record)&&payload?.known?payload.pixels:null;
    for(let i=0;i<h*w;i++){
      const v=Math.max(0,Math.min(255,Math.round((image.values[i]-(wl-ww/2))/ww*255)));
      rgba.data.set(mask?.[i]?[Math.round(v*.6),Math.round(v*.6+80),Math.round(v*.6+77),255]:[v,v,v,255],i*4);
    }
    ctx.putImageData(rgba,0,0);const fc=full.getContext('2d'),box=fit(full,h,w);fc.imageSmoothingEnabled=false;fc.drawImage(off,box.x,box.y,box.w,box.h);
    try{
      const p=parameters(),[y0,y1,x0,x1]=p.roi_yx,cw=x1-x0,ch=y1-y0;
      fc.strokeStyle='#f4c66c';fc.lineWidth=2;fc.strokeRect(box.x+x0/w*box.w,box.y+y0/h*box.h,cw/w*box.w,ch/h*box.h);
      const zc=zoom.getContext('2d');zc.imageSmoothingEnabled=false;
      if(cw&&ch){const zb=fit(zoom,ch,cw);zc.drawImage(off,x0,y0,cw,ch,zb.x,zb.y,zb.w,zb.h);}
      $('roiViewBadge').textContent=cw&&ch?`${cw} × ${ch} 原生像素`:'空矩形';
    }catch{$('roiViewBadge').textContent='请检查框选参数';}
  }
  async function fetchImage(){
    if(!visible()||!state?.shape)return;
    const key=[state.case_id,index].join(':');if(key===imageKey)return;
    imageKey=key;image=null;const ticket=++imageTicket,caseId=state.case_id,z=index;draw();
    try{
      const p=await api('/api/plane?'+new URLSearchParams({case:caseId,axis:0,index:z,stage:'body'}));
      if(ticket!==imageTicket||caseId!==state.case_id||z!==index)return;
      p.values=new Float32Array(bytes(p.hu).buffer);image=p;draw();
    }catch(e){if(ticket===imageTicket){imageKey=null;localError=e.message;refreshControls();}}
  }
  async function fetchMask(){
    if(!visible()||!current())return;
    const key=[state.case_id,selected,index].join(':');if(key===payloadKey)return;
    payloadKey=key;payload=null;const ticket=++payloadTicket;
    try{
      const p=await api('/api/roi/image?'+new URLSearchParams({case:state.case_id,source:state.analysis_source_id,result:selected,index}));
      if(ticket!==payloadTicket||key!==[state.case_id,selected,index].join(':'))return;
      payload=p;p.pixels=p.mask?bytes(p.mask.pixels):null;draw();metrics();
    }catch(e){if(ticket===payloadTicket){payloadKey=null;localError=e.message;refreshControls();}}
  }
  function metrics(){
    const r=current(),row=payload?.result_id===selected&&payload?.index===index?payload.slice_result:null;
    const fixed=(n,d=2)=>Number.isFinite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
    $('roiCount').textContent=row?fixed(row.valid_count,0):'—';$('roiArea').textContent=fixed(row?.area_mm2);
    $('roiMean').textContent=fixed(r?.hu_statistics?.mean);$('roiVolume').textContent=fixed(r?.region_volume_ml,4);
    $('roiExport').disabled=!r;
    $('roiResultNote').textContent=!r?'设置矩形和条件后点击分析，结果独立保存。':
      `${matches(r)?'已保存结果。':'参数／当前层已变，以下为上次保存结果。'} ${r.analysis_source.label} · ${r.processed_slices}/${r.requested_slices} 层 · ${r.status==='SUCCESS'?'完成':r.status==='CANCELLED'?'已停止，保留已完成层':'失败，保留已有输出'}。`;
    $('roiStatsNote').textContent=!r?'单层只显示面积；连续层段显示实际已计算区域体积，不代表精确病灶体积。':
      `已计算区域 HU ${fixed(r.hu_statistics.min)}～${fixed(r.hu_statistics.max)}；${r.physical_geometry_verified?'已核对毫米几何':'几何不足，不显示物理量'}。${!row?'当前层未计算，不能按零计数。 ':''}${r.mode==='slice'?'单层不报告体积。':'体积仅汇总实际已完成层。'} 电脑参考 ${fixed(r.timing.cpu_reference_ms)} ms；不是 FPGA 实测。`;
  }
  function refreshControls(){
    const busy=ACTIVE.includes(state?.phase),has=!!state?.shape,range=$('roiMode').value==='range';
    for(const id of fields)$(id).disabled=!has||busy;
    $('roiLow').disabled=$('roiHigh').disabled=!has||busy||!$('roiUseHu').checked;
    $('roiMaskStage').disabled=!has||busy||!$('roiUseLung').checked;
    $('roiMaskStage').querySelector('[value="cleaned"]').disabled=!state?.roi?.cleaned_mask_available;
    $('roiRangeFields').hidden=!range;$('roiSlice').disabled=$('roiFull').disabled=$('roiToSeg').disabled=!has||busy;
    $('roiRun').disabled=!has||busy;$('roiRun').textContent=range?'分析连续层段':'分析当前切片';
    $('roiCancel').hidden=!state?.roi?.job?.running;
    if(document.activeElement!==$('roiSlice'))$('roiSlice').value=index;
    $('roiSlice').max=has?state.shape[0]-1:0;
    const info=state?.roi?.job;
    $('roiProgress').textContent=localError||(!has?'等待载入影像。':info?.running?`电脑正在分析 ${info.completed} / ${info.total} 层…`:'使用当前分析来源 HU；点击分析后才计算。');
    let missing=false;
    if(has){
      try{const p=parameters(),known=new Set(state.roi.mask_known_indices);missing=Array.from({length:p.z_bounds[1]-p.z_bounds[0]},(_,n)=>p.z_bounds[0]+n).some(z=>!known.has(z));}
      catch{$('roiRun').disabled=true;}
    }
    if($('roiUseLung').checked&&(missing||$('roiMaskStage').value==='cleaned'&&!state?.roi?.cleaned_mask_available))$('roiRun').disabled=true;
    $('roiMaskNote').textContent=!$('roiUseLung').checked?'未启用肺区域限制，可直接分析框内区域；无需先分割。':missing?'所选范围有未分割层。可关闭肺区域限制直接分析，或先完成当前来源肺分割。':'使用当前来源的预测肺掩膜；不会使用专家标注或其他来源结果。';
    $('roiSourceNote').textContent=has?`当前分析输入：${state.analysis_source?.label||'原始 CT'}。框选和结果绑定这份 HU；切换来源后恢复各自记录。`:'载入一次 CT，即可在此分析；不要求先做肺分割。';
    $('roiSliceLabel').textContent=has?`轴位 ${index} · 源序列第 ${state.source.source_slice_indices[index]} 层`:'等待影像';
  }
  function history(){
    const rows=state?.roi?.history||[],key=JSON.stringify([selected,rows.map(r=>r.result_id)]);if(key===historyKey)return;historyKey=key;
    const body=$('roiHistory');body.replaceChildren();
    for(const r of [...rows].reverse()){
      const tr=document.createElement('tr');if(r.result_id===selected)tr.className='roi-history-current';
      const [y0,y1,x0,x1]=r.roi_yx;
      for(const value of [`${r.z_bounds[0]}～${r.z_bounds[1]-1}`,`${x1-x0}×${y1-y0}${r.use_hu?' · HU '+r.hu_range.join('～'):''}${r.use_lung_mask?' · 肺内':''}`,`${r.processed_slices}/${r.requested_slices}${r.complete?'':' · 未完成'}`,r.valid_count.toLocaleString(),r.timing.cpu_reference_ms.toFixed(2)+' ms'])textCell(tr,value);
      const td=document.createElement('td'),button=document.createElement('button');button.className='text-button';button.textContent='查看';
      button.onclick=()=>{selected=r.result_id;payload=null;payloadKey=null;payloadTicket++;restore(r);if(index<r.z_bounds[0]||index>=r.z_bounds[1])setIndex(r.z_bounds[0]);refresh();};td.append(button);tr.append(td);body.append(tr);
    }
    if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=6;td.textContent='暂无分析记录';tr.append(td);body.append(tr);}
  }
  function changed(){localError='';refreshControls();draw();metrics();}
  function refresh(){
    if(!state?.roi)return;
    const key=[state.series_id,state.case_id,state.analysis_source_id].join(':');
    if(key!==context){const sameSeries=!!contextSeries&&contextSeries===state.series_id;contextSeries=state.series_id;context=key;image=null;imageKey=null;imageTicket++;payload=null;payloadKey=null;payloadTicket++;selected=null;latest=null;localError='';
      if(state.shape&&!sameSeries){bounds([0,state.shape[1],0,state.shape[2]]);$('roiZ0').value=0;$('roiZ1').value=state.shape[0]-1;$('roiUseHu').checked=false;$('roiUseLung').checked=false;$('roiMode').value='slice';}
    }
    if((state.roi.result?.result_id||null)!==latest){latest=state.roi.result?.result_id||null;selected=latest;if(current())restore(current());payload=null;payloadKey=null;payloadTicket++;}
    refreshControls();history();metrics();fetchImage();fetchMask();
  }
  const canvas=$('roiCanvas');
  canvas.addEventListener('pointerdown',event=>{if(!image||ACTIVE.includes(state?.phase)||event.button!==0)return;const p=pointer(event);if(!p)return;drag={start:p,previous:[number('roiY'),number('roiY')+number('roiHeight'),number('roiX'),number('roiX')+number('roiWidth')]};canvas.setPointerCapture(event.pointerId);dragTo(p);});
  canvas.addEventListener('pointermove',event=>{if(drag)dragTo(pointer(event,true));});
  canvas.addEventListener('pointerup',event=>{if(drag){dragTo(pointer(event,true));drag=null;canvas.releasePointerCapture(event.pointerId);}});
  canvas.addEventListener('pointercancel',()=>{if(drag){bounds(drag.previous);drag=null;changed();}});
  for(const id of fields)$(id).addEventListener('input',changed);
  $('roiSlice').onchange=()=>{setIndex($('roiSlice').value);refresh();};
  $('roiFull').onclick=()=>{bounds([0,state.shape[1],0,state.shape[2]]);changed();};
  $('roiWindow').onchange=$('roiOverlay').onchange=draw;
  $('roiToSeg').onclick=()=>{$('syncSlice').click();};
  $('roiRun').onclick=async()=>{try{
    const p=parameters();localError='';apply(await api('/api/roi/run',{...p,case_id:state.case_id,analysis_source_id:state.analysis_source_id,index,backend:'CPU_REFERENCE'}));
  }catch(e){localError=e.message;refreshControls();message(e.message,true);}};
  $('roiCancel').onclick=async()=>{try{apply(await api('/api/roi/cancel',{case_id:state.case_id,analysis_source_id:state.analysis_source_id,result_id:state.roi.job.result_id}));}catch(e){localError=e.message;refreshControls();}};
  $('roiExport').onclick=async()=>{try{
    const value=await api('/api/roi/export?'+new URLSearchParams({case:state.case_id,source:state.analysis_source_id,result:selected}));
    const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='ct-roi-'+selected.slice(0,8)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){localError=e.message;refreshControls();}};
  window.addEventListener('ct-state-change',refresh);window.addEventListener('ct-slice-change',refresh);window.addEventListener('ct-tab-change',()=>{refresh();draw();});
  draw();if(state)refresh();
})();
