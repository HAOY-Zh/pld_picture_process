from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_window_step_project import WORKSPACE, burst_top, project_script
from lung_tensor_burst_prepare import engine_source


class SourceOnlyProjectTests(unittest.TestCase):
    def test_no_hardware_execution_commands(self):
        text=project_script((WORKSPACE/'fpga/build_lung_network_board.tcl').read_text(encoding='utf-8'))
        for command in ['launch_runs','wait_on_run','synth_design','place_design','route_design','write_bitstream','close_project']:
            self.assertNotIn(command,text)
        self.assertIn('if {$argc != 1}',text)
        self.assertIn('source [file join $task_out validate_bd.tcl]',text)
        self.assertIn('[file join $task_out rtl $f]',text)

    def test_staged_tensor_matches_physically_tested_burst_rtl(self):
        old=WORKSPACE/'data/lung_tensor_burst_20261016_v3/rtl'
        top=burst_top((WORKSPACE/'fpga/lung_tensor_axil.v').read_text(encoding='utf-8'))
        self.assertEqual(top,(old/'lung_tensor_axil.v').read_text(encoding='utf-8'))
        engine=engine_source((WORKSPACE/'fpga/lung_tensor_word_engine.v').read_text(encoding='utf-8'))
        self.assertEqual(engine,(old/'lung_tensor_burst_engine.v').read_text(encoding='utf-8'))

    def test_bd_checker_covers_clocks_reset_interfaces_and_ranges(self):
        text=(WORKSPACE/'fpga/validate_lung_window_step.tcl').read_text(encoding='utf-8')
        for required in ['FCLK_RESET0_N','FCLK_CLK0','aux_reset_in','C_AUX_RESET_HIGH',
                         'S_AXI_HP0_ACLK','M_AXI_GP0_ACLK','0x43c00000','0x43c10000','0x20000000',
                         'validate_bd_design','control_interconnect/S00_AXI','memory_interconnect/M00_AXI']:
            self.assertIn(required,text)
        for command in ['launch_runs','synth_design','place_design','route_design','write_bitstream']:
            self.assertFalse(re.search(r'^\s*'+command+r'\b',text,re.M))

    def test_tcl_files_complete_without_starting_vivado(self):
        import tkinter
        interpreter=tkinter.Tcl()
        files=[WORKSPACE/'fpga/validate_lung_window_step.tcl',WORKSPACE/'fpga/lung_axil_config_timing.xdc',
               WORKSPACE/'fpga/resume_lung_window_step.tcl']
        texts=[p.read_text(encoding='utf-8') for p in files]
        texts.append(project_script((WORKSPACE/'fpga/build_lung_network_board.tcl').read_text(encoding='utf-8')))
        for text in texts: self.assertEqual(interpreter.call('info','complete',text),1)

    def test_authorized_resume_gates_bit_after_route_and_timing(self):
        text=(WORKSPACE/'fpga/resume_lung_window_step.tcl').read_text(encoding='utf-8')
        self.assertLess(text.index('to_step route_design'),text.index('Routed setup/hold failed'))
        self.assertLess(text.index('Routed setup/hold failed'),text.index('to_step write_bitstream'))
        self.assertIn('source [file join $task_root fpga validate_lung_window_step.tcl]',text)
        self.assertIn('if {[get_property PROGRESS $r] ne "100%"} { reset_runs $r }',text)
        self.assertIn('interrupted_logs',text)
        self.assertNotIn('file delete',text)
        self.assertNotIn('file copy -force',text)


if __name__=='__main__': unittest.main()
