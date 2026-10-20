'use strict';
let ldctPayload=null,ldctTicket=0,ldctViewKey=null;
const ldctBrowserTimings=new Map();
const ldctWindows={comparison:[-300,1400],lung:[-600,1500],soft:[40,400]};
function ldctWindowControls(){
  const toolbar=document.querySelector('.ldct-toolbar'),label=document.createElement('label'),select=document.createElement('select');
  label.htmlFor='ldctWindowPreset';label.textContent='显示窗';select.id='ldctWindowPreset';
  for(const [value,text] of [['comparison','对比窗 · 与此前结果一致'],['lung','肺窗 · 观察肺部'],['soft','软组织窗 · 观察软组织'],['custom','自定义']]){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
  label.append(select);toolbar.insertBefore(label,$('ldctWl').parentElement);
  const note=document.createElement('p');note.id='ldctWindowNote';note.className='ldct-window-note';note.setAttribute('aria-live','polite');toolbar.after(note);
  select.onchange=()=>{const preset=ldctWindows[select.value];if(preset){[$('ldctWl').value,$('ldctWw').value]=preset;ldctDraw();}};
  select.value='comparison';[$('ldctWl').value,$('ldctWw').value]=ldctWindows.comparison;
}
function ldctDraw(){
  const wl=Number($('ldctWl').value),ww=Number($('ldctWw').value);
  if(!Number.isFinite(wl)||!Number.isFinite(ww)||ww<=0){if($('ldctWindowNote'))$('ldctWindowNote').textContent='请输入有效窗位和大于 0 的窗宽。';return;}
  if($('ldctWindowNote'))$('ldctWindowNote').textContent=`两图同步显示 ${wl-ww/2}～${wl+ww/2} HU；高于上限显示为白色。调窗只改变显示，不改变 HU 或去噪结果。`;
  for(const [id,key] of [['ldctOriginal','original'],['ldctEnhanced','enhanced']]){
    const canvas=$(id),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
    if(!ldctPayload?.[key]){ctx.fillStyle='#8aa5b0';ctx.font='16px sans-serif';ctx.textAlign='center';ctx.fillText(key==='original'?'载入 CT 后显示原图':'等待当前切片增强',256,256);continue;}
    const packed=ldctPayload[key],values=new Int16Array(bytes(packed.pixels).buffer),[h,w]=packed.shape;
    canvas.width=w;canvas.height=h;const image=ctx.createImageData(w,h);
    for(let i=0;i<values.length;i++){const v=Math.max(0,Math.min(255,Math.round((values[i]-(wl-ww/2))/ww*255)));image.data.set([v,v,v,255],i*4);}
    ctx.putImageData(image,0,0);
  }
}
async function ldctFetch(){
  if(!state?.series_id)return;
  const row=state.ldct?.results?.find(r=>r.index===index),key=[state.series_id,state.ldct?.group_id,index,row?.result_id].join(':');
  if(key===ldctViewKey)return;
  ldctViewKey=key;ldctPayload=null;ldctDraw();
  const ticket=++ldctTicket,series=state.series_id,group=state.ldct.group_id;
  try{const value=await api('/api/ldct/image?'+new URLSearchParams({series,group:group||'',index}));
    if(ticket!==ldctTicket||series!==state.series_id||group!==state.ldct.group_id||key!==ldctViewKey)return;
    ldctPayload=value;ldctDraw();const m=value.metadata;
    if(!m){$('ldctResultBadge').textContent='当前层尚未增强';$('ldctResultInfo').textContent='左侧为原始 CT。可先预览当前层，或增强全部已载入切片后继续分析。';return;}
    const hw=m.hardware_receipt;
    $('ldctResultBadge').textContent=m.backend==='FPGA_LDCT'?'实际 FPGA 输出':'电脑整数参考';
    $('ldctResultInfo').textContent=`源序列第 ${m.source_slice_index} 层 · ${m.method} · ${m.backend==='FPGA_LDCT'?'DMA/PL/拼图':'电脑整数参考'} ${m.processing_ms.toFixed(2)} ms${hw?` · 往返 ${hw.roundtrip_ms.toFixed(2)} ms · bit切换 ${hw.stats.overlay_switch_ms.toFixed(2)} ms · 实际 ${hw.stats.actual_pl_mhz_after} MHz`:''} · HU/坐标保持一致；没有配对正常剂量参考，不显示质量分数。`;
  }catch(e){message(e.message,true);}
}
function ldctRefresh(){
  if(!state)return;const info=state.ldct||{},busy=ACTIVE.includes(state.phase);
  const source=state.analysis_source||{label:'原始 CT'},choices=state.analysis_sources||[];
  const select=$('analysisSource'),selected=state.analysis_source_id||'original';
  if(select.dataset.choices!==JSON.stringify(choices)){select.replaceChildren();for(const choice of choices){const option=document.createElement('option');option.value=choice.source_id;option.textContent=choice.label+(choice.source_id==='original'?'':' · '+choice.source_id.slice(-5));select.append(option);}select.dataset.choices=JSON.stringify(choices);}
  select.value=selected;select.disabled=!state.shape||busy;
  $('analysisSourceBadge').textContent=state.shape?source.label:'等待载入';
  const segmentationBadge=document.querySelector('#tab-segmentation .chip');
  segmentationBadge.textContent=state.desktop_only?'MPR 浏览 · 分割需连接 FPGA':state.board_connected?'FPGA 分割 · 使用当前分析输入':'MPR 浏览 · FPGA 未连接';
  segmentationBadge.className='chip '+(state.board_connected?'teal':'amber');
  document.querySelector('#imagePair .image-title strong').textContent=selected==='original'?'原始 CT · 固定显示参考':'LDCT 增强 CT · 当前分析输入';
  document.querySelector('.analysis-source-bar').dataset.enhanced=String(selected!=='original');
  $('analysisSourceNote').textContent=selected==='original'?'当前基础处理、分割、ROI 与 MPR 使用原始 CT；增强后可切换。':'当前基础处理、分割、ROI 与 MPR 使用这份增强 HU；切回原图可恢复原来的分析结果。';
  const fpga=info.fpga_configured&&state.board_connected;
  $('ldctBackend').querySelector('[value="FPGA_LDCT"]').disabled=!fpga;
  $('ldctBackend').disabled=busy;$('ldctMethod').disabled=busy;
  const hardware=$('ldctBackend').value==='FPGA_LDCT';
  $('ldctBackendBadge').textContent=hardware?'FPGA · 实际输出':'电脑整数参考';
  $('ldctBackendNote').textContent=hardware?'仅调用匹配的统一 FPGA 服务；失败后保留错误，不自动改用电脑。':'使用同一组量化权重验证增强流程；本次时间属于电脑参考。';
  $('ldctPreview').disabled=$('ldctRunVolume').disabled=!state.shape||!info.available||busy||(hardware&&!fpga);
  $('ldctCancel').hidden=!info.running;$('ldctSlice').disabled=!state.shape;
  $('ldctSlice').max=state.shape?state.shape[0]-1:0;if(document.activeElement!==$('ldctSlice'))$('ldctSlice').value=index;
  $('ldctProgress').textContent=state.shape?`增强完成 ${info.completed_slices||0} / ${info.total_slices||0} 层`:'尚未载入 CT';
  const enhanced=choices.filter(c=>c.source_id!=='original'&&c.group_id===info.group_id).at(-1);
  $('ldctUse').disabled=!enhanced||busy;$('ldctUse').dataset.source=enhanced?.source_id||'';
  $('ldctModelInfo').textContent=info.identity?`模型 ${info.identity.model_id.slice(0,16)}…；独立 FPGA 发布版 ${info.identity.clock_mhz} MHz。统一服务实板验收待完成。`:'未配置 LDCT 发布包；基础处理仍可使用。';
  const history=$('ldctHistory');history.replaceChildren();const title=document.createElement('h3');title.textContent='本次增强记录';history.append(title);
  const rows=info.results||[],total=rows.reduce((sum,r)=>sum+r.processing_ms,0);
  $('ldctPerformance').textContent=rows.length?`当前分析输入：${source.label}。当前增强批次 ${rows.length} 层；${rows[0].backend==='FPGA_LDCT'?'实际 FPGA DMA/PL/拼图':'电脑整数参考'}平均 ${(total/rows.length).toFixed(2)} ms/层，逐层运算合计 ${total.toFixed(2)} ms。`:'尚未执行增强；当前分析输入：'+source.label+'。';
  for(const r of [...info.results||[]].reverse()){const line=document.createElement('div');line.className='ldct-history-row';for(const text of [`第 ${r.source_slice_index} 层 · ${r.method}`,r.backend==='FPGA_LDCT'?'实际 FPGA':'电脑整数参考',r.processing_ms.toFixed(2)+' ms']){const span=document.createElement('span');span.textContent=text;line.append(span);}history.append(line);}
  ldctFetch();
}
async function ldctStart(mode){try{await command('/api/ldct/run',{case_id:state.case_id,index,mode,backend:$('ldctBackend').value,method:$('ldctMethod').value});}catch(e){message(e.message,true);}}
$('ldctPreview').onclick=()=>ldctStart('slice');$('ldctRunVolume').onclick=()=>ldctStart('volume');
$('ldctCancel').onclick=()=>command('/api/ldct/cancel',{});
$('ldctUse').onclick=async()=>{await command('/api/analysis-source',{case_id:state.case_id,source_id:$('ldctUse').dataset.source});showTab('basic');};
$('analysisSource').onchange=()=>command('/api/analysis-source',{case_id:state.case_id,source_id:$('analysisSource').value});
$('ldctSlice').onchange=()=>setIndex($('ldctSlice').value);
ldctWindowControls();
for(const id of ['ldctWl','ldctWw'])$(id).oninput=()=>{$('ldctWindowPreset').value='custom';ldctDraw();};
for(const id of ['ldctBackend','ldctMethod'])$(id).onchange=ldctRefresh;
$('flowLoad').onclick=()=>{$('path').focus();};$('flowEnhance').onclick=()=>showTab('ldct');$('flowAnalyze').onclick=()=>showTab('basic');$('flowExport').onclick=()=>showTab('performance');
window.addEventListener('ct-state-change',ldctRefresh);window.addEventListener('ct-slice-change',ldctRefresh);
ldctDraw();if(state)ldctRefresh();
