// Exercise production upload helper with lost replies and unreadable local files.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../lungseg/workspace/series.js'),'utf8').split('let seriesInspection=')[0];
const context={URLSearchParams,AbortController,setTimeout,clearTimeout,fetch:()=>{},document:{querySelector:()=>({content:'synthetic-ui-test'})}};
vm.createContext(context);vm.runInContext(source,context);
(async()=>{
 const buffer=new ArrayBuffer(16),file={name:'slice.dcm',webkitRelativePath:'CT/slice.dcm',arrayBuffer:async()=>buffer};
 let bodies=[],retries=[];
 await context.uploadSeriesFile({import_id:'test'},251,file,n=>retries.push(n),async(url,options)=>{
   bodies.push(options.body);if(bodies.length===1)throw new TypeError('Failed to fetch');
   return{ok:true,status:200,json:async()=>({completed_files:252,already_received:true})};
 });
 assert.equal(bodies.length,2);assert.equal(bodies[0],buffer);assert.equal(bodies[1],buffer);assert.deepEqual(retries,[1]);
 let rejected=0;
 await assert.rejects(()=>context.uploadSeriesFile({import_id:'test'},0,file,()=>{},async()=>{
   rejected++;return{ok:false,status:400,json:async()=>({error:'invalid DICOM'})};
 }),/invalid DICOM/);assert.equal(rejected,1);
 let unreadableRequests=0;
 await assert.rejects(()=>context.uploadSeriesFile({import_id:'test'},0,{name:'locked.dcm',arrayBuffer:async()=>{throw Error('unreadable');}},()=>{},async()=>{unreadableRequests++;}),/locked.dcm/);
 assert.equal(unreadableRequests,0);
 let failed=0;
 await assert.rejects(()=>context.uploadSeriesFile({import_id:'test'},251,file,()=>{},async()=>{failed++;throw new TypeError('Failed to fetch');}),/252.*slice.dcm/);
 assert.equal(failed,3);console.log('PASS: lost reply retry, immutable byte body, no invalid-file retry, local read error, bounded retries.');
})().catch(e=>{console.error(e);process.exitCode=1;});
