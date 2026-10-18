'use strict';
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
  $('seriesAdvice').textContent=row?`${row.slices} 层 · ${row.rows} × ${row.columns}${row.spacing_zyx_mm?' · 间距 '+row.spacing_zyx_mm.map(v=>v.toFixed(3)).join(' / ')+' mm':''}。${row.note}`:'目录包含多组 CT，请选择需要分析的一组。';
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
      const query=new URLSearchParams({import:task.import_id,index:i,name:files[i].webkitRelativePath||files[i].name});
      const response=await fetch('/api/import/file?'+query,{method:'POST',headers:{'Content-Type':'application/dicom','X-UI-Token':document.querySelector('meta[name="ui-token"]').content},body:files[i]});
      const value=await response.json();if(!response.ok)throw Error(value.error||'文件传输失败');
    }
    showSeriesInspection(await api('/api/import/finish',{import_id:task.import_id}));
  }catch(e){$('seriesStatus').textContent='导入未完成：'+e.message;message(e.message,true);}
  finally{seriesImportBusy=false;$('folderFiles').value='';seriesControls();}
};
const seriesOriginalLoad=$('load').onclick;
$('load').onclick=()=>{const row=seriesSelected();if(seriesImportBusy||seriesInspection&&(!row?.valid||row.requires_compatibility&&!$('frameUidCompatible').checked))return;return seriesOriginalLoad();};
window.addEventListener('ct-state-change',seriesControls);
seriesControls();
