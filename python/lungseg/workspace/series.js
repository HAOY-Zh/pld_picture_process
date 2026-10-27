'use strict';
async function uploadSeriesFile(task,index,file,onRetry,fetcher=fetch){
  // Keep one bounded file in memory: avoid a disk-backed File upload failing
  // after the browser has already started sending the request.
  let payload;
  try{payload=await file.arrayBuffer();}
  catch(e){throw Error(`无法读取本机文件“${file.name}”：${e.message}。可改用 CT 路径直接读取。`);}
  const query=new URLSearchParams({import:task.import_id,index,name:file.webkitRelativePath||file.name});
  for(let attempt=0;attempt<3;attempt++){
    const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),120000);
    try{
      const response=await fetcher('/api/import/file?'+query,{method:'POST',headers:{'Content-Type':'application/dicom','X-UI-Token':document.querySelector('meta[name="ui-token"]').content},body:payload,signal:abort.signal});
      const value=await response.json();
      if(!response.ok){const error=Error(value.error||'文件传输失败');error.uploadRejected=response.status!==408;throw error;}
      return value;
    }catch(e){
      if(e.uploadRejected)throw e;
      if(attempt===2)throw Error(`第 ${index+1} 个文件“${file.name}”传输中断，已安全重试 2 次。可重新选择文件夹，或改用 CT 路径直接读取。`);
      onRetry(attempt+1);
    }finally{clearTimeout(timer);}
  }
}
let seriesInspection=null,seriesImportBusy=false;
function seriesSelected(){return seriesInspection?.series.find(row=>row.series_uid===$('seriesChoice').value);}
function seriesControls(){
  const busy=seriesImportBusy||ACTIVE.includes(state?.phase),row=seriesSelected();
  $('chooseFolder').disabled=$('inspectSeries').disabled=busy;
  $('seriesChoice').disabled=$('frameUidCompatible').disabled=busy;
  $('load').disabled=busy||!!seriesInspection&&(!row?.valid||row.requires_compatibility&&!$('frameUidCompatible').checked);
  if(row&&state?.source?.source===seriesInspection.path)$('caseLabel').textContent=row.description;
}
function seriesChoiceChanged(){
  const row=seriesSelected();$('uid').value=row?.series_uid||'';
  $('frameUidCompatible').checked=false;$('loadExpert').disabled=false;
  $('frameUidChoice').hidden=!row?.requires_compatibility;
  $('seriesPanel').dataset.invalid=String(!row?.valid);
  $('seriesAdvice').textContent=row?`${row.slices} 层 · ${row.rows} × ${row.columns}${row.spacing_zyx_mm?' · 间距 '+row.spacing_zyx_mm.map(v=>v.toFixed(3)).join(' / ')+' mm':''}。${row.note}${row.slices===1?'。当前只有一层；整组分析和三维请选包含连续切片的文件夹。':''}`:'目录包含多组 CT，请选择需要分析的一组。';
  seriesControls();
}
function showSeriesInspection(value){
  seriesInspection=value;$('path').value=value.path;$('seriesPanel').hidden=false;
  const select=$('seriesChoice');select.replaceChildren();
  if(value.series.length!==1){const option=document.createElement('option');option.value='';option.textContent=value.series.length?'请选择一个 CT 序列':'未找到 CT 序列';select.append(option);}
  value.series.forEach((row,i)=>{const option=document.createElement('option');option.value=row.series_uid;option.textContent=`序列 ${i+1} · ${row.description} · ${row.slices} 层${row.valid?'':' · 检查未通过'}`;select.append(option);});
  if(value.series.length===1)select.value=value.series[0].series_uid;
  $('seriesStatus').textContent=`发现 ${value.series.length} 组 CT；选择序列后点击“加载影像”。`;
  seriesChoiceChanged();
}
$('seriesChoice').onchange=seriesChoiceChanged;
$('frameUidCompatible').onchange=()=>{
  if($('frameUidCompatible').checked)$('loadExpert').checked=false;
  $('loadExpert').disabled=$('frameUidCompatible').checked;seriesControls();
};
$('path').addEventListener('input',()=>{seriesInspection=null;$('seriesPanel').hidden=true;$('uid').value='';$('frameUidCompatible').checked=false;$('loadExpert').disabled=false;seriesControls();});
$('inspectSeries').onclick=async()=>{
  seriesImportBusy=true;seriesControls();$('seriesStatus').textContent='正在读取 DICOM 头信息并检查序列…';
  try{showSeriesInspection(await api('/api/series/inspect',{path:$('path').value}));}
  catch(e){$('seriesStatus').textContent=e.message;message(e.message,true);}
  finally{seriesImportBusy=false;seriesControls();}
};
$('chooseFolder').onclick=()=>$('folderFiles').click();
$('folderFiles').onchange=async()=>{
  const files=[...$('folderFiles').files].filter(file=>/\.(dcm|dicom)$/i.test(file.name)||!file.name.includes('.'));
  if(!files.length){$('seriesStatus').textContent='文件夹中没有可导入的 DICOM 文件（.dcm / .dicom 或无扩展名）。';return;}
  seriesImportBusy=true;seriesControls();
  try{
    const task=await api('/api/import/start',{file_count:files.length,total_bytes:files.reduce((sum,file)=>sum+file.size,0)});
    for(let i=0;i<files.length;i++){
      if(files[i].size>task.max_file_bytes)throw Error('单个文件超过 64MB 限制');
      $('seriesStatus').textContent=`正在导入 ${i+1} / ${files.length} 个文件，请保持页面打开…`;
      await uploadSeriesFile(task,i,files[i],retry=>{
        $('seriesStatus').textContent=`第 ${i+1} / ${files.length} 个文件传输中断，正在安全重试 ${retry} / 2…`;
      });
    }
    showSeriesInspection(await api('/api/import/finish',{import_id:task.import_id}));
  }catch(e){
    const detail=e.message==='Failed to fetch'?'浏览器与本机网页服务的连接中断。请刷新网页后重试，或填写 CT 路径直接读取。':e.message;
    $('seriesStatus').textContent='导入未完成：'+detail;message(detail,true);
  }
  finally{seriesImportBusy=false;$('folderFiles').value='';seriesControls();}
};
const seriesOriginalLoad=$('load').onclick;
$('load').onclick=async()=>{
  if(seriesImportBusy||ACTIVE.includes(state?.phase))return;
  const path=$('path').value.trim();
  if(!path){message('请选择 CT 文件夹，或填写 CT 路径。',true);return;}
  // NIfTI/NPZ retain their explicit HU/mm loader. Every DICOM path, including
  // a pasted folder loaded without the separate Inspect button, is preflighted.
  if(/\.(npz|nii|nii\.gz)["']?$/i.test(path))return seriesOriginalLoad();
  seriesImportBusy=true;seriesControls();
  try{
    if(!seriesInspection||seriesInspection.path!==path){
      const requestedUid=$('uid').value.trim();
      $('seriesStatus').textContent='正在检查切片数量、序列和空间位置…';
      const inspection=await api('/api/series/inspect',{path});
      if($('path').value.trim()!==path){message('检查期间路径已变化，请重新点击加载影像。');return;}
      showSeriesInspection(inspection);
      if(requestedUid&&inspection.series.some(row=>row.series_uid===requestedUid)){
        $('seriesChoice').value=requestedUid;seriesChoiceChanged();
      }
    }
    const row=seriesSelected();
    if(!row?.valid){
      const advice=row?.note||'目录中有多组 CT，请先选择需要加载的一组。';
      $('seriesStatus').textContent=advice;message(advice,!!row);return;
    }
    if(row.requires_compatibility&&!$('frameUidCompatible').checked){
      const advice=`已检查 ${row.slices} 层，几何一致。请确认下方“兼容匿名化数据的坐标系标识差异”，再点击加载影像。`;
      $('seriesStatus').textContent=advice;message(advice);return;
    }
    $('seriesStatus').textContent=`已检查 ${row.slices} 层，正在加载影像…`;
    return await seriesOriginalLoad();
  }catch(e){$('seriesStatus').textContent='序列检查未完成：'+e.message;message(e.message,true);}
  finally{seriesImportBusy=false;seriesControls();}
};
window.addEventListener('ct-state-change',seriesControls);
$('seriesStatus').textContent='整组导入请选择 DICOM 文件夹；加载时自动检查序列，读取层号留空表示全部。';
seriesControls();
