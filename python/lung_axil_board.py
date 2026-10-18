"""Prepare a separate single-block board smoke project, never edit old projects."""
import argparse
import json
from pathlib import Path

from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256


def tcl_string(value):
    value = str(value)
    for old, new in [('\\', '\\\\'), ('"', '\\"'), ('$', '\\$'), ('[', '\\['), (']', '\\]'), ('\n', '\\n'), ('\r', '\\r')]:
        value = value.replace(old, new)
    return '"' + value + '"'


def prepare(outdir):
    workspace = Path(__file__).resolve().parents[1]
    source = workspace/'fpga/pynq_z2_closing_axis/pynq_z2_closing_axis.srcs/sources_1/bd/design_1/ip/design_1_processing_system7_0_0/design_1_processing_system7_0_0.xci'
    document = json.loads(source.read_text(encoding='utf-8'))
    if document['ip_inst']['component_reference'] != 'xilinx.com:ip:processing_system7:5.5':
        raise ValueError('Unexpected source PS IP')
    parameters = document['ip_inst']['parameters']['component_parameters']
    config = {key:entries[0]['value'] for key, entries in parameters.items()
              if key.startswith('PCW_') and entries[0].get('resolve_type') == 'user' and entries[0].get('enabled', True)}
    if int(config['PCW_DDR_RAM_HIGHADDR'], 16) != 0x1FFFFFFF:
        raise ValueError('Expected known PYNQ-Z2 512 MiB DDR geometry')
    target = Path(outdir).resolve()
    if workspace/'data' not in target.parents:
        raise ValueError('Board output must be a new subdirectory of workspace/data')
    target = fresh_output(target)
    # Generated build artifact, not a rewrite of the old PS/XCI or any source.
    body = 'set ps_config [list \\\n' + ''.join('    CONFIG.'+key+' '+tcl_string(value)+' \\\n' for key, value in sorted(config.items())) + ']\n'
    (target/'ps_settings.tcl').write_text(body, encoding='utf-8')
    report = dict(status='prepared_single_block_board_smoke', source=str(source), source_sha256=file_sha256(source),
        ps_parameter_count=len(config), ddr_bytes=512*1024*1024, ps_settings_sha256=file_sha256(target/'ps_settings.tcl'),
        generator_sha256=file_sha256(__file__), warning='Not full U-Net; existing Linux SD boot retained; do not use as FSBL/BOOT replacement')
    write_json(target/'preparation.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    prepare(parser.parse_args().outdir)
