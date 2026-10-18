"""Board audit adapter; preserve the already hashed core verifier unchanged.

Vivado's routed utilization table omits the optional LUT footnote asterisk.
Only that report parser is adapted; all original board checks still run.
"""
import argparse
from pathlib import Path
import re

import verify_lung_axil as core_audit


def parse_utilization(raw):
    result = {}
    for label, limit in [('Slice LUTs*', 53200), ('Slice Registers', 106400),
                         ('DSPs', 220), ('Block RAM Tile', 140)]:
        pattern = r'Slice LUTs\*?' if label == 'Slice LUTs*' else re.escape(label)
        matches = re.findall(r'\|\s*' + pattern + r'\s*\|\s*([\d.]+)\s*\|', raw)
        # Vivado repeats the register total in Slice Logic Distribution.
        if not matches or len(set(map(float, matches))) != 1:
            raise ValueError('Missing or ambiguous resource: ' + label)
        value = float(matches[0])
        if not 0 <= value <= limit:
            raise ValueError('Resource budget exceeded: ' + label)
        result[label] = value
    return result


def utilization(path):
    return parse_utilization(Path(path).read_text(encoding='utf-8'))


def audit_board(root, core_root):
    original = core_audit.utilization
    core_audit.utilization = utilization
    try:
        result = core_audit.audit_board(root, core_root)
    finally:
        core_audit.utilization = original
    # Bind this compatibility adapter as well as the original frozen verifier.
    adapter = Path(__file__).resolve()
    result['board_auditor_sha256'] = core_audit.file_sha256(adapter)
    for target in [Path(root)/'verification.json', Path(root)/'deploy/package.json']:
        core_audit.write_json(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', required=True)
    parser.add_argument('--root', default='data/lung_axil_20261015')
    args = parser.parse_args()
    audit_board(args.board, args.root)
