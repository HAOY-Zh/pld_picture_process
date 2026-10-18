from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_lung_axil_resetfix import check_reset


def hardware():
    root = ET.Element('SYSTEM')
    for instance, value in [('const_one', 1), ('const_zero', 0)]:
        module = ET.SubElement(root, 'MODULE', INSTANCE=instance, MODTYPE='xlconstant')
        params = ET.SubElement(module, 'PARAMETERS')
        for name, val in [('CONST_VAL', value), ('CONST_WIDTH', 1)]:
            ET.SubElement(params, 'PARAMETER', NAME=name, VALUE=str(val))
    reset = ET.SubElement(root, 'MODULE', INSTANCE='proc_sys_reset_0')
    params = ET.SubElement(reset, 'PARAMETERS')
    for name in ['C_AUX_RESET_HIGH', 'C_EXT_RESET_HIGH']:
        ET.SubElement(params, 'PARAMETER', NAME=name, VALUE='0')
    ports = ET.SubElement(reset, 'PORTS')
    for name, sources in [
        ('aux_reset_in', [('const_one', 'dout')]), ('dcm_locked', [('const_one', 'dout')]),
        ('mb_debug_sys_rst', [('const_zero', 'dout')]),
        ('ext_reset_in', [('processing_system7_0', 'FCLK_RESET0_N')]),
        ('peripheral_aresetn', [('lung_conv8_axil_0', 'rst_n'),
                              ('control_interconnect', 'aresetn'), ('memory_interconnect', 'aresetn')])]:
        port = ET.SubElement(ports, 'PORT', NAME=name)
        connections = ET.SubElement(port, 'CONNECTIONS')
        for instance, pin in sources:
            ET.SubElement(connections, 'CONNECTION', INSTANCE=instance, PORT=pin)
    return root


class BoardResetTests(unittest.TestCase):
    def test_inactive_auxiliary_accepted(self):
        self.assertTrue(check_reset(hardware())['shared_reset'])

    def test_original_stuck_reset_rejected(self):
        root = hardware()
        root.find('.//PORT[@NAME="aux_reset_in"]/CONNECTIONS/CONNECTION').set('INSTANCE', 'const_zero')
        with self.assertRaisesRegex(ValueError, 'wrong constant'):
            check_reset(root)

    def test_wrong_polarity_rejected(self):
        root = hardware()
        root.find('.//PARAMETER[@NAME="C_AUX_RESET_HIGH"]').set('VALUE', '1')
        with self.assertRaisesRegex(ValueError, 'low-active'):
            check_reset(root)

    def test_wrong_constant_value_not_hidden_by_name(self):
        root = hardware()
        root.find('.//MODULE[@INSTANCE="const_one"]//PARAMETER[@NAME="CONST_VAL"]').set('VALUE', '0')
        with self.assertRaisesRegex(ValueError, 'wrong constant'):
            check_reset(root)

    def test_missing_shared_reset_rejected(self):
        root = hardware()
        items = root.find('.//PORT[@NAME="peripheral_aresetn"]/CONNECTIONS')
        items.remove(items[-1])
        with self.assertRaisesRegex(ValueError, 'not shared'):
            check_reset(root)


if __name__ == '__main__':
    unittest.main()
