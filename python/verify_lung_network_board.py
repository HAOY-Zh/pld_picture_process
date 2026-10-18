"""Audit and package a routed dual-engine overlay, never replace proven artifacts."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from lung_unet import file_sha256
from lung_segmentation import write_json
from verify_lung_axil import check_hashes,timing
from verify_lung_axil_board import utilization
from verify_lung_axil_resetfix import check_reset


def check_hardware(hardware):
    modules={m.get('INSTANCE'):m for m in hardware.findall('.//MODULE')}
    reset=modules['proc_sys_reset_0']
    links=reset.find('./PORTS/PORT[@NAME="peripheral_aresetn"]/CONNECTIONS')
    expected={('lung_conv8_axil_0','rst_n'),('lung_tensor_axil_0','rst_n'),
              ('control_interconnect','aresetn'),('memory_interconnect','aresetn')}
    if {(c.get('INSTANCE'),c.get('PORT')) for c in links}!=expected or len(links)!=4:
        raise ValueError('Both engines and interconnects must share reset')
    # The existing checker expects exactly the old three consumers. Validate
    # the actual new four-consumer set ABOVE, then reuse its polarity/constant checks.
    reduced=copy.deepcopy(hardware)
    reduced_links=reduced.find('.//MODULE[@INSTANCE="proc_sys_reset_0"]/PORTS/PORT[@NAME="peripheral_aresetn"]/CONNECTIONS')
    for link in list(reduced_links):
        if link.get('INSTANCE')=='lung_tensor_axil_0': reduced_links.remove(link)
    check_reset(reduced)
    ps=modules['processing_system7_0'];params={p.get('NAME'):p.get('VALUE') for p in ps.findall('./PARAMETERS/PARAMETER')}
    if abs(float(params['PCW_FPGA0_PERIPHERAL_FREQMHZ'])-100)>.01 or int(params['PCW_DDR_RAM_HIGHADDR'],0)!=0x1fffffff:
        raise ValueError('Wrong PS clock/DDR configuration')
    ranges=hardware.findall('.//MEMRANGE')
    for name,kind,base in [('lung_conv8_axil_0','lung_conv8_axil',0x43c00000),
                           ('lung_tensor_axil_0','lung_tensor_axil',0x43c10000)]:
        module=modules[name]
        if module.get('MODTYPE')!=kind: raise ValueError('Wrong engine VLNV/type')
        if not any(r.get('INSTANCE')==name and int(r.get('BASEVALUE','0'),0)==base and
                   int(r.get('HIGHVALUE','0'),0)==base+1023 for r in ranges):
            raise ValueError('Wrong AXI-Lite aperture')
        if not any(r.get('INSTANCE')=='processing_system7_0' and int(r.get('BASEVALUE','0'),0)==0 and
                   int(r.get('HIGHVALUE','0'),0)==0x1fffffff for r in module.findall('.//MEMRANGE')):
            raise ValueError('Engine DDR address map missing')
    for name,pin in [('lung_conv8_axil_0','clk'),('lung_tensor_axil_0','clk'),
                     ('control_interconnect','aclk'),('memory_interconnect','aclk'),('proc_sys_reset_0','slowest_sync_clk')]:
        port=modules[name].find('./PORTS/PORT[@NAME="'+pin+'"]')
        sources={(p.get('INSTANCE'),p.get('PORT')) for p in port.findall('./CONNECTIONS/CONNECTION')}
        if sources!={('processing_system7_0','FCLK_CLK0')} or port.get('CLKFREQUENCY')!='100000000':
            raise ValueError('Unshared/wrong clock: '+name)
    return dict(clock_mhz=100,auxiliary_reset_inactive=True,shared_reset_consumers=4,
                convolution_base='0x43c00000',tensor_base='0x43c10000',ddr_bytes=512*1024*1024)


def audit(root):
    root=Path(root).resolve();workspace=Path(__file__).resolve().parents[1]
    target=root/'lung_network'
    if (root/'verification.json').exists() or target.exists(): raise ValueError('Preserve existing proof/package')
    core_path=workspace/'data/lung_tensor_20261015/verification.json'
    core=json.loads(core_path.read_text(encoding='utf-8'))
    if core['status']!='passed_tensor_axi_operators_not_board_tested': raise ValueError('Tensor core proof absent')
    check_hashes(core['source_sha256']);check_hashes(core['evidence_sha256']);check_hashes(core['original_vector_sources_sha256'])
    vectors_path=workspace/'data/lung_tensor_20261015/vectors/vectors.json'
    vectors=json.loads(vectors_path.read_text(encoding='utf-8'))
    check_hashes({str(vectors_path.parent/n):h for n,h in vectors['files_sha256'].items()})
    old_path=workspace/'data/lung_axil_20261015/verification.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));check_hashes(old['source_sha256']);check_hashes(old['evidence_sha256'])
    bit=root/'deploy/lung_network.bit';hwh=bit.with_suffix('.hwh')
    if bit.stat().st_size<1000000: raise ValueError('Bitstream missing')
    hardware=check_hardware(ET.parse(hwh).getroot())
    resources=utilization(root/'utilization_routed.rpt');wns=timing(root/'timing_routed.rpt')
    raw=(root/'timing_routed.rpt').read_text(encoding='utf-8')
    match=re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)',raw,re.S)
    if not match or float(match[1].split()[4])<0: raise ValueError('Hold timing failed')
    whs=float(match[1].split()[4])
    drc=(root/'drc.rpt').read_text(encoding='utf-8')
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b',drc): raise ValueError('DRC error/critical warning')
    imported={}
    for source in (workspace/'fpga').glob('lung_*.v'):
        for packaged in (root/'ip_repo').rglob(source.name):
            if file_sha256(source)!=file_sha256(packaged): raise ValueError('Packaged RTL source differs')
            imported[str(packaged)]=file_sha256(packaged)
    # IP packager includes only reachable dependencies, not all project files.
    required={
        'lung_conv8_axil':{'lung_requant.v','lung_mac_tile8.v','lung_mac_tile8_reuse.v','lung_feature_window.v',
                          'lung_conv8_window.v','lung_feature_cache.v','lung_axi_read64.v','lung_conv8_cached_axi.v',
                          'lung_axi_write64.v','lung_output_writer.v','lung_conv8_memory_axi.v','lung_axil_control.v','lung_conv8_axil.v'},
        'lung_tensor_axil':{'lung_requant.v','lung_axi_read64.v','lung_axi_write64.v',
                           'lung_tensor_engine.v','lung_tensor_control.v','lung_tensor_axil.v'}}
    for engine,names in required.items():
        copies=list((root/'ip_repo'/engine).rglob('lung_*.v'))
        if {p.name for p in copies}!=names or len(copies)!=len(names):
            raise ValueError('Incomplete/ambiguous packaged dependencies: '+engine)
    source_manifest=workspace/'data/lung_model_layers_20261015/lung_model_layers/layers.json'
    model=json.loads(source_manifest.read_text(encoding='utf-8'))
    check_hashes(model['source_sha256']);check_hashes({str(source_manifest.parent/n):h for n,h in model['files_sha256'].items()})
    target.mkdir()
    for source in [bit,hwh]: shutil.copy2(source,target/source.name)
    for name in ['lung_axil.py','lung_axil_smoke.py','lung_model_layers_test.py',
                 'lung_tensor.py','lung_tensor_smoke.py','lung_network_smoke.py']:
        shutil.copy2(workspace/'pynq'/name,target/name)
    shutil.copy2(source_manifest.parent/'vectors.npz',target/'vectors.npz')
    shutil.copy2(vectors_path.parent/'vectors.npz',target/'tensor_vectors.npz')
    # Rebind a NEW copy of the real-layer fixture manifest to the NEW overlay.
    # The original fixture manifest and all codes/weights remain unchanged.
    model.update(bit_sha256=file_sha256(bit),hwh_sha256=file_sha256(hwh),
                 original_fixture_manifest_sha256=file_sha256(source_manifest))
    write_json(target/'layers.json',model)
    manifest=dict(schema='lung_dual_engine_smoke_v1',tensor_cases=vectors['cases'],hardware=hardware,
                  tensor_rtl_proof_sha256=file_sha256(core_path),original_real_layer_manifest_sha256=file_sha256(source_manifest),
                  files_sha256={p.name:file_sha256(p) for p in target.iterdir()},
                  warning='Convolution and tensor regressions only; NOT continuously scheduled whole-network inference')
    write_json(target/'network_manifest.json',manifest)
    evidence=[root/n for n in ['preparation.json','ps_settings.tcl','utilization_routed.rpt','timing_routed.rpt','configuration_exceptions.rpt','drc.rpt']]
    result=dict(status='built_routed_dual_engine_regression_not_board_tested',hardware=hardware,utilization=resources,
                routed_wns_ns=wns,routed_whs_ns=whs,tensor_proof_sha256=file_sha256(core_path),
                original_convolution_proof_sha256=file_sha256(old_path),packaged_source_sha256=imported,
                package_sha256=file_sha256(target/'network_manifest.json'),package_files_sha256=manifest['files_sha256'],
                evidence_sha256={str(p):file_sha256(p) for p in evidence},verifier_sha256=file_sha256(__file__),
                bitstream_generated=True,board_tested=False,continuous_network=False)
    write_json(root/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',default='data/lung_network_board_20261015')
    audit(parser.parse_args().board)
