import tempfile
from pathlib import Path
import unittest
from lung_nodule_context_eval_worker import active_timing, commit_case


class ContextWorkerTests(unittest.TestCase):
    def test_only_unfinished_declared_timing_ct_pauses(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            plan = {'timing_uids':['timed']}
            regular = out/'variant/cases/regular'
            regular.mkdir(parents=True)
            self.assertFalse(active_timing(out, 'variant', plan))
            timed = out/'variant/cases/timed'
            timed.mkdir()
            self.assertTrue(active_timing(out, 'variant', plan))
            (timed/'complete.json').write_text('{}')
            self.assertFalse(active_timing(out, 'variant', plan))

    def test_atomic_publish_never_replaces_main_partial_case(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            main, staging = out/'main', out/'staging'
            main.mkdir()
            staging.mkdir()
            (staging/'complete.json').write_text('{}')
            self.assertFalse(commit_case(staging, main))
            self.assertFalse((main/'complete.json').exists())
            self.assertTrue((staging/'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
