'use strict';
// Embedded workbench extension. The standalone viewer remains unchanged.
(() => {
  const G=window.CTNodule;
  let packet=null,selectedId=null,showAll=true,pending=null,appliedRevision=null,applying=false;
  let markerKey='',markerBusy=false,markerDirty=false;
  const valid=()=>G.matches(packet,state);
  const selected=()=>valid()?packet.candidates.find(c=>c.candidate_id===selectedId):null;
  const emit=(type,extra)=>{if(valid())parent.postMessage({type,...extra,case_id:packet.case_id,
    analysis_source_id:packet.analysis_source_id,context_id:packet.context_id,result_id:packet.result_id},location.origin);};
  function receive(){
    if(!pending)return;
    if(pending.packet&&!G.matches(pending.packet,state))return;
    const m=pending;pending=null;packet=m.packet;selectedId=m.selected_id;showAll=m.show_all!==false;
    const c=selected();
    if(c&&m.recenter&&m.selection_revision!==appliedRevision){
      appliedRevision=m.selection_revision;applying=true;
      indices=c.center_index_zyx.map((v,a)=>G.clamp(Math.round(v),0,state.shape[a]-1));
      for(let a=0;a<3;a++){$('s'+a).value=$('n'+a).value=indices[a];fetchPlane(a);}
      updatePlane3D();applying=false;
    }
    planes.forEach((_,a)=>draw(a));updateMarkers();
  }
  window.addEventListener('message',e=>{
    if(e.origin!==location.origin||e.source!==parent||e.data?.type!=='ct-nodule-link')return;
    pending=e.data;receive();
  });
  const originalApply=apply;
  apply=s=>{
    const changedCase=s.case_id!==state?.case_id;
    originalApply(s);
    if(s.nodule?.job?.status==='RUNNING')message(`正在 FPGA 检测结节候选 · ${s.nodule.job.completed}/${s.nodule.job.total||'…'} patch`);
    if(packet&&!valid()){packet=null;selectedId=null;markerKey='';planes.forEach((_,a)=>draw(a));}
    if(changedCase&&selected()){
      pending={packet,selected_id:selectedId,show_all:showAll,recenter:true,selection_revision:appliedRevision??0};
      appliedRevision=null;
    }
    receive();updateMarkers();
  };
  const originalDraw=draw;
  draw=axis=>{
    originalDraw(axis);const c=selected(),p=planes[axis];if(!c||!p?.box||p.index!==indices[axis])return;
    const b=G.section(c,axis,indices[axis],state.shape);if(!b)return;
    const [y0,y1,x0,x1]=b,[h,w]=p.shape,{dx,dy,dw,dh,flip}=p.box,ctx=$('c'+axis).getContext('2d');
    ctx.strokeStyle='#f4c66c';ctx.lineWidth=2;ctx.setLineDash([]);
    ctx.strokeRect(dx+x0/w*dw,dy+(flip?h-y1:y0)/h*dh,(x1-x0)/w*dw,(y1-y0)/h*dh);
    ctx.font='12px sans-serif';ctx.fillStyle='#f4c66c';ctx.textAlign='left';ctx.fillText(c.candidate_id,dx+5,dy+16);
  };
  const originalPlane=updatePlane3D;
  updatePlane3D=()=>{originalPlane();if(!applying)emit('ct-nodule-browse',{index:indices[0]});};
  const originalSurface=refreshSurface;
  refreshSurface=async()=>{await originalSurface();updateMarkers();};
  function traces(){
    const c=selected();if(!valid()||!c)return [];
    const candidates=showAll?packet.candidates:[c];
    const points=candidates.map(p=>G.world(p.center_index_zyx,state.geometry));
    const centers={type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
      customdata:candidates.map(p=>p.candidate_id),text:candidates.map(p=>p.candidate_id),
      marker:{size:candidates.map(p=>p.candidate_id===selectedId?11:5),symbol:candidates.map(p=>p.candidate_id===selectedId?'diamond':'circle'),
        color:candidates.map(p=>p.candidate_id===selectedId?'#f4c66c':'#9de3e2'),line:{color:'#182634',width:2},opacity:1},
      meta:{ct_nodule:true,ct_nodule_marker:true},name:packet.is_test_data?'测试候选（非检出结果）':packet.ingestion_kind==='computed_live_fpga'?'FPGA 实验候选中心':'导入候选中心',
      hovertemplate:'%{text}<br>LPS (%{x:.2f}, %{y:.2f}, %{z:.2f}) mm<extra></extra>'};
    const edges=G.edges(c,state.geometry,state.shape),box={type:'scatter3d',mode:'lines',
      x:edges.map(p=>p?p[0]:null),y:edges.map(p=>p?p[1]:null),z:edges.map(p=>p?p[2]:null),line:{color:'#f4c66c',width:4},
      meta:{ct_nodule:true},hoverinfo:'skip',name:'候选包围框（非精确分割）',showlegend:false};
    return [centers,box];
  }
  async function updateMarkers(){
    if(markerBusy){markerDirty=true;return;}
    const scene=$('scene');if(!sceneReady||!scene.data)return;
    const key=[surfaceKey,valid()?packet.context_id:'',valid()?packet.result_id:'',selectedId,showAll].join(':');
    const expected=selected()?2:0,existing=scene.data.filter(t=>t.meta?.ct_nodule).length;
    if(key===markerKey&&expected===existing)return;
    markerBusy=true;
    try{
      const base=scene.data.filter(t=>!t.meta?.ct_nodule).map((t,i)=>{
        if(i!==0||t.type!=='mesh3d')return t;
        const original=t.meta?.ct_nodule_original_opacity??t.opacity??1;
        return {...t,opacity:selected() ? .28 : original,meta:{...t.meta,ct_nodule_original_opacity:original}};
      });
      await Plotly.react(scene,[...base,...traces()],scene.layout,{responsive:true,displaylogo:false});
      markerKey=key;
      if(scene.__ctNoduleClick)scene.removeListener('plotly_click',scene.__ctNoduleClick);
      scene.__ctNoduleClick=event=>{const p=event.points?.find(p=>p.data?.meta?.ct_nodule_marker);if(p&&packet?.candidates.some(c=>c.candidate_id===p.customdata))emit('ct-nodule-select',{candidate_id:p.customdata});};
      scene.on('plotly_click',scene.__ctNoduleClick);
    }catch(e){message('候选三维标记更新失败：'+e.message,true);markerKey='';}
    finally{markerBusy=false;if(markerDirty){markerDirty=false;updateMarkers();}}
  }
  window.ctWorkspaceNodules={selected:()=>selectedId,packet:()=>packet,refresh:receive};
  parent.postMessage({type:'ct-nodule-viewer-ready'},location.origin);
})();
