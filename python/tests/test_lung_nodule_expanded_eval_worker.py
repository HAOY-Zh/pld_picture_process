import tempfile
from pathlib import Path
import unittest
from lung_nodule_expanded_eval_worker import commit_case


class WorkerCommitTests(unittest.TestCase):
    def test_main_process_empty_directory_already_owns_case(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stage, main = root/'stage', root/'main'
            stage.mkdir()
            main.mkdir()
            (stage/'complete.json').write_text('worker')
            self.assertFalse(commit_case(stage, main))
            self.assertEqual(list(main.iterdir()), [])
            self.assertTrue((stage/'complete.json').exists())

    def test_complete_case_commit_and_existing_case_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first, final = root/'first', root/'final'
            first.mkdir()
            (first/'complete.json').write_text('first')
            self.assertTrue(commit_case(first, final))
            second = root/'second'
            second.mkdir()
            (second/'complete.json').write_text('second')
            self.assertFalse(commit_case(second, final))
            self.assertEqual((final/'complete.json').read_text(), 'first')
            self.assertTrue(second.exists())


if __name__ == '__main__':
    unittest.main()
