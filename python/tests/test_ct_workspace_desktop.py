"""Desktop mode browses CT and rejects removed software execution paths."""
import base64
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_workspace_web import WorkspaceController


def wait(controller):
    deadline = time.monotonic() + 15
    while controller.snapshot()['phase'] in ('loading', 'processing'):
        if time.monotonic() > deadline:
            raise AssertionError('Desktop task timeout')
        time.sleep(.01)
    return controller.snapshot()


class DesktopTests(unittest.TestCase):
    def test_load_without_model_credentials_and_reject_reference_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'synthetic.npz'
            np.savez(path, hu=np.full((3, 32, 32), -800, np.int16),
                     spacing_zyx_mm=[2., 1., 1.], origin_lps_mm=[0., 0., 0.],
                     orientation=[1., 0., 0., 0., 1., 0.])
            controller = WorkspaceController(None, {}, root/'results')
            self.assertTrue(controller.snapshot()['desktop_only'])
            controller.submit('load', dict(path=str(path), confirm_hu=True, load_expert=False))
            state = wait(controller)
            self.assertEqual(state['phase'], 'idle')
            for backend in ('PC_REFERENCE','FPGA_BASIC'):
                with self.assertRaises(ValueError):
                    controller.submit('basic',dict(case_id=state['case_id'],index=1,
                        backend=backend,operation='threshold',lower=-1000,upper=-400))
            state=controller.snapshot()
            self.assertEqual(state['phase'],'idle')
            self.assertIsNone(state['basic'])
            self.assertIsNone(state['latest'])
            self.assertEqual(state['known_slices'],0)


    def test_hardware_actions_are_rejected_and_never_report_connected(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = WorkspaceController(None, {}, Path(tmp)/'results')
            for action in ('connect', 'connect_fast', 'run'):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, '--config'):
                    controller.submit(action, {})
            self.assertFalse(controller.snapshot()['board_connected'])
            self.assertFalse(controller.snapshot()['fast_platform_available'])
            self.assertFalse(controller.snapshot()['basic_pl_configured'])


if __name__ == '__main__':
    unittest.main()
