"""Explicit local restoration from saved immutable results; never rerun a model."""
import copy,json
from pathlib import Path
import numpy as np
from ldct_workspace import SESSION_FIELDS,digest
from lung_live_data import read_ct
from lung_live_web import BIT_SHA,MODEL_SHA
from lung_volume_viewer import mesh


def restore_nodule_result(c,state):
    """Restore only the explicitly selected completed live result, in its old context."""
    from nodule_workspace import digest,identifier,validate_packet
    selected=state.get('nodule',{}).get('result')
    if not selected:return
    result_id=identifier(selected.get('result_id'),'result_id')
    context=c.nodule_context()
    folder=c.case_folder/'nodule'/result_id
    packet=json.loads((folder/'prediction.json').read_text(encoding='utf8'))
    job=json.loads((folder/'job.json').read_text(encoding='utf8'))
    if (packet.get('ingestion_kind')!='computed_live_fpga' or packet.get('is_test_data') is not False
            or job.get('status')!='COMPLETE' or job.get('job_id')!=result_id
            or job.get('case_id')!=context['case_id']
            or job.get('analysis_source_id')!=context['analysis_source_id']
            or job.get('candidate_count')!=packet.get('candidate_count')):
        raise ValueError('不能恢复未完成或来源不匹配的在线候选')
    for key in ('case_id','analysis_source_id','context_id','result_id','candidate_count','is_test_data','ingestion_kind'):
        if packet.get(key)!=selected.get(key):raise ValueError('保存的候选摘要与结果不一致：'+key)
    restored=validate_packet(packet,context,allow_enhanced=True)
    restored['ingestion_kind']='computed_live_fpga'
    c.nodule_result=restored
    c.nodule_packet_hashes[(context['context_id'],result_id)]=digest(restored)
    c.nodule_job=job


def restore_session(c,state_file):
    state=json.loads(Path(state_file).read_text(encoding='utf8'))
    if state.get('phase') in ('loading','connecting','computing','processing','reconstructing'):
        raise ValueError('不能恢复未完成任务；原网页保留')
    if state.get('expert_present'):raise ValueError('带专家标注的会话暂不自动恢复；原网页保留')
    choices={v['source_id']:copy.deepcopy(v) for v in state['analysis_sources']}
    original=choices['original'];base=c.output/original['case_id']
    details=json.loads((base/'input.json').read_text(encoding='utf8'))
    indices=details['source_slice_indices'];selection='' if not details.get('partial_input') else f'{indices[0]}-{indices[-1]}'
    hu,geom,fresh=read_ct(details['source'],confirm_hu=True,series_uid=details.get('series_uid',''),slices=selection,
        allow_inconsistent_frame_uids=details.get('frame_uid_exception_applied')is True)
    if digest(hu)!=original['hu_sha256'] or fresh['geometry']!=details['geometry']:
        raise ValueError('恢复源 CT 哈希或几何发生变化')
    c.series_id=state['series_id'];c.original_hu=hu;c.original_details=copy.deepcopy(details)
    c.source_choices=choices;c.ldct_groups={};c.ldct_group=state['ldct']['group_id']
    shared_arrays={};allowed_groups={v.get('group_id') for v in choices.values()}|{c.ldct_group}
    for p in sorted((base/'ldct').glob('*.json')):
        meta=json.loads(p.read_text(encoding='utf8'))
        if meta.get('series_id')!=c.series_id or meta.get('group_id')not in allowed_groups:continue
        z=meta['index']
        if meta['model_id']!=c.ldct_identity['model_id'] or not 0<=z<len(hu):raise ValueError('恢复增强模型/层号不匹配')
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:value=a['hu']
        if value.dtype!=np.int16 or value.shape!=hu.shape[1:] or digest(value)!=meta['output_sha256']:
            raise ValueError('恢复增强数组哈希不一致')
        key=meta['output_sha256'];value=shared_arrays.setdefault(key,value);value.flags.writeable=False
        c.ldct_groups.setdefault(meta['group_id'],{})[z]=dict(hu=value,metadata=meta)
    hu_cache={original['hu_sha256']:hu};c.analysis_sessions={};c.roi_sessions={}
    for source_id,descriptor in choices.items():
        folder=c.output/descriptor['case_id'];source_details=json.loads((folder/'input.json').read_text(encoding='utf8'))
        key=descriptor['hu_sha256']
        if key not in hu_cache:
            group=c.ldct_groups[descriptor['group_id']]
            if len(group)!=len(hu):raise ValueError('完整增强来源缺少切片')
            volume=np.stack([group[z]['hu'] for z in range(len(hu))]).astype(np.float32)
            if digest(volume)!=key:raise ValueError('完整增强体数据哈希不一致')
            hu_cache[key]=volume
        volume=hu_cache[key]
        if source_details['hu_sha256']!=key or source_details['geometry']!=geom:raise ValueError('恢复来源身份不一致')
        status=copy.deepcopy(c.status);status.update(case_id=descriptor['case_id'],phase='idle',error=None,
            completed=0,total=0,elapsed_ms=None,latest=None)
        session=dict(hu=volume,raw=np.zeros(volume.shape,bool),body=np.zeros(volume.shape,bool),cleaned=None,
            known=np.zeros(len(volume),bool),geom=geom,details=source_details,case_folder=folder,
            expert=None,expert_info=None,scores={n:{} for n in ('raw','body','cleaned')},surfaces={},receipts={},
            basic_result=None,basic_image=None,threshold_result=None,threshold_image=None,history=[],status=status)
        records=sorted(folder.glob('*_volume.json'),key=lambda p:p.stat().st_mtime)
        if source_id==state['analysis_source_id'] and state.get('volume_job_id'):
            records=[folder/(state['volume_job_id']+'_volume.json')]
        if records:
            meta=json.loads(records[-1].read_text(encoding='utf8'))
            if (meta['source']['hu_sha256']!=key or meta['model_sha256']!=MODEL_SHA or not set(meta.get('bit_sha256s') or [meta['bit_sha256']]).issubset({BIT_SHA, getattr(c.client,'segmentation_bit_sha256',BIT_SHA)})
                    or meta['input_geometry']!=geom):raise ValueError('恢复分割结果的来源/模型不一致')
            job=meta['job_id'];plan=meta['reconstruction'];lo,hi=plan['local_start'],plan['local_stop']
            with np.load(folder/(job+'_masks.npz'),allow_pickle=False) as masks:
                for name in ('raw','body','cleaned'):
                    if name not in masks.files:continue
                    data=masks[name]
                    if data.dtype!=np.bool_ or data.shape!=(hi-lo,*volume.shape[1:]):raise ValueError('恢复掩膜类型/形状不一致')
                    if name=='cleaned':session[name]=np.zeros(volume.shape,bool)
                    session[name][lo:hi]=data
            session['receipts']={int(z):rid for z,rid in meta['slice_receipts'].items()}
            session['known'][list(session['receipts'])]=True
            for z,rid in session['receipts'].items():
                receipt=json.loads((folder/rid/'receipt.json').read_text(encoding='utf8'))
                if receipt['source_hu_sha256']!=key or receipt['case_id']!=descriptor['case_id']:
                    raise ValueError('恢复分割回执与来源不匹配')
            for name in ('body','cleaned'):
                if session[name]is not None:session['surfaces'][name]=mesh(session[name][lo:hi],plan['geometry'])
            status.update(phase='done',completed=int(session['known'].sum()),total=len(volume),
                volume_ready=True,full_volume_ready=meta['volume_scope']=='input_volume' and bool(session['known'].all()),
                volume_scope=meta['volume_scope'],reconstruction=plan,volume_job_id=job,
                region_volume_ml=meta['region_volume_ml'],elapsed_ms=meta['total_elapsed_ms'])
            last=max(session['receipts']);status['active_slice']=last
            status['latest']=json.loads((folder/session['receipts'][last]/'receipt.json').read_text(encoding='utf8'))
        basic_paths=sorted(folder.glob('**/*_basic.json'),key=lambda p:p.stat().st_mtime)
        for p in basic_paths:
            meta=json.loads(p.read_text(encoding='utf8'))
            if meta.get('case_id')!=descriptor['case_id']:raise ValueError('恢复基础结果病例不一致')
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:image=a['image']
            if digest(image)!=meta['output_sha256']:raise ValueError('恢复基础图像哈希不一致')
            session['history'].append(meta);session.update(basic_result=meta,basic_image=image)
            if meta['parameters']['operation']=='threshold':session.update(threshold_result=meta,threshold_image=image.copy())
        roi_records={}
        for p in sorted((folder/'roi').glob('*/result.json'),key=lambda p:p.stat().st_mtime):
            row=json.loads(p.read_text(encoding='utf8'))
            if row['case_id']!=descriptor['case_id'] or row['source_hu_sha256']!=key:raise ValueError('恢复ROI来源不一致')
            roi_records[row['result_id']]=row
        selected=list(roi_records)[-1] if roi_records else None
        if source_id==state['analysis_source_id']:selected=(state['roi']['result'] or {}).get('result_id')
        result=roi_records.get(selected)
        roi_job=dict(result_id=selected,running=False,status=result['status'],completed=result['processed_slices'],total=result['requested_slices']) if result else None
        c.roi_sessions[source_id]=(result,roi_records,roi_job)
        c.analysis_sessions[source_id]=session
    selected=state['analysis_source_id'];session=c.analysis_sessions[selected]
    for name in SESSION_FIELDS:setattr(c,name,session[name])
    c.status=session['status'];c.analysis_source_id=selected
    c.roi_result,c.roi_records,c.roi_job=c.roi_sessions[selected]
    # Preserve the current source's selected result IDs and visible counters.
    for key in ('revision','completed','total','latest','active_slice','elapsed_ms'):
        if key in state:c.status[key]=copy.deepcopy(state[key])
    c.status.update(phase='done' if c.known.any() else 'idle',error=None,board_connected=False)
    if c.client is not None:
        # Saved result verification is independent of a temporary LAN outage.
        # Keep actual connection status false; never substitute CPU inference.
        from urllib.error import HTTPError, URLError
        try:
            c.client.health()
        except HTTPError:
            raise  # Authentication/identity rejection must remain explicit.
        except (TimeoutError, ConnectionError, URLError, OSError) as exc:
            c.status['board_connection_error']=type(exc).__name__+': '+str(exc)
        else:
            c.status['board_connected']=True
    old_context=state.get('nodule',{}).get('context')
    if old_context:
        from nodule_workspace import digest as context_digest
        for generation in range(1024):
            signature=context_digest(dict(receipts=c.receipts,known=np.flatnonzero(c.known).tolist(),whole=bool(c.status.get('full_volume_ready')),generation=generation))
            if signature==old_context['segmentation_signature']:c.nodule_generation=generation;break
        else:raise ValueError('无法核对原候选上下文版本')
    restore_nodule_result(c,state)
    print('RESTORED_LOCAL_SESSION: original/enhanced HU, saved FPGA masks/receipts, ROI, candidates and basic records; no inference rerun',flush=True)
