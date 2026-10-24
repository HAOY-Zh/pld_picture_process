'use strict';
// Geometry shared by the workbench and embedded MPR. Bounds are half open.
(() => {
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  function displayBox(c,shape,spacing,padding=0){
    return c.bbox_zyx.map((v,k)=>{const a=Math.floor(k/2),extra=Math.ceil(padding/spacing[a]);return clamp(v+(k%2?extra:-extra),0,shape[a]);});
  }
  function world(index,geometry){return geometry.origin_lps_mm.map((o,a)=>o+geometry.basis_lps_zyx[a].reduce((s,v,k)=>s+v*index[k],0));}
  function section(c,axis,index,shape){
    const b=displayBox(c,shape,[1,1,1]);if(index<b[axis*2]||index>=b[axis*2+1])return null;
    return [0,1,2].filter(a=>a!==axis).flatMap(a=>[b[2*a],b[2*a+1]]);
  }
  function edges(c,geometry,shape){
    const b=displayBox(c,shape,[1,1,1]),points=[];
    for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)points.push(world([b[z]-.5,b[2+y]-.5,b[4+x]-.5],geometry));
    const lines=[];for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))lines.push(points[i],points[i|bit],null);
    return lines;
  }
  function matches(packet,s){return !!packet&&!!s?.nodule?.result&&packet.case_id===s.case_id&&packet.analysis_source_id===s.analysis_source_id&&['case_id','analysis_source_id','context_id','result_id'].every(k=>packet[k]===s.nodule.result[k])&&packet.context_id===s.nodule.context?.context_id;}
  window.CTNodule={clamp,displayBox,world,section,edges,matches};
})();
