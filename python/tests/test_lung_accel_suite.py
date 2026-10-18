"""Healthy handover and retained-owner safeguards for the board-only suite."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
import lung_accel_suite as suite


class SuiteHandoverTests(unittest.TestCase):
    def service(self):
        service=Mock();service.failed=False;service.hardware.fault=None
        service.gate.acquire.return_value=True
        return service

    def test_healthy_owner_released_before_server_and_lock_close(self):
        service=self.service();events=Mock()
        for label,m in [('hardware',service.hardware.release_idle),('shutdown',service.server.shutdown),
                        ('server_close',service.server.server_close),('join',service.thread.join),
                        ('lock_close',service.hardware.lock_file.close),('gate_release',service.gate.release)]:
            events.attach_mock(m,label)
        suite.release_web(service)
        self.assertEqual([x[0] for x in events.mock_calls],
                         ['hardware','shutdown','server_close','join','lock_close','gate_release'])

    def test_busy_or_faulted_service_never_releases_hardware(self):
        for busy,fault in [(True,False),(False,True)]:
            service=self.service();service.gate.acquire.return_value=not busy;service.hardware.fault=fault
            with self.assertRaises(RuntimeError):suite.release_web(service)
            service.hardware.release_idle.assert_not_called();service.server.shutdown.assert_not_called()
            self.assertEqual(service.gate.release.call_count,0 if busy else 1)

    def test_release_failure_preserves_server_and_hardware_owner(self):
        service=self.service();service.hardware.release_idle.side_effect=RuntimeError('engine not idle')
        with self.assertRaises(RuntimeError):suite.release_web(service)
        service.server.shutdown.assert_not_called();service.hardware.lock_file.close.assert_not_called()
        service.gate.release.assert_called_once()

    def test_incomplete_suite_owner_rejects_retry_before_any_new_board_access(self):
        owner=object();suite._RETAINED_LOCKS.append(owner)
        try:
            with patch.object(suite,'validate',return_value={}),patch.object(suite.platform,'machine',return_value='armv7l'):
                with self.assertRaisesRegex(RuntimeError,'retained'):
                    suite.run(argparse.Namespace(runs=20,warmup=3,timeout=30))
        finally:suite._RETAINED_LOCKS.remove(owner)


class PortableManifestTests(unittest.TestCase):
    def test_windows_and_posix_keys_have_identical_posix_components(self):
        self.assertEqual(suite.relative_parts(r'original\clock_manifest.json'),
                         ('original','clock_manifest.json'))
        self.assertEqual(suite.relative_parts('pruned/clock_manifest.json'),
                         ('pruned','clock_manifest.json'))
        for invalid in ['../file','original/../../file','/etc/passwd',r'C:\file',r'\\server\share\file','', '.', 'a\x00b']:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):suite.relative_parts(invalid)

    def test_both_manifest_formats_validate_real_files_and_tampering_still_fails(self):
        for separator in ['/', '\\']:
            with self.subTest(separator=separator),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);tracked={}
                for label in ['original','pruned']:
                    folder=root/label;folder.mkdir();bit=folder/'lung_network.bit';bit.write_bytes(b'identical fixture bit')
                    digest=suite.sha(bit)
                    (folder/'clock_manifest.json').write_text(json.dumps({'routed_design':{'bit_sha256':digest}}),encoding='utf-8')
                    (folder/'fp32_manifest.json').write_text(json.dumps({'model_label':label,'schema':'lung_fp32_threeway_v2'}),encoding='utf-8')
                    tracked.update({label+separator+p.name:suite.sha(p) for p in folder.iterdir()})
                (root/'suite_manifest.json').write_text(json.dumps(dict(schema='lung_all_acceleration_suite_v1',
                    models=['original','pruned'],bit_sha256=digest,files_sha256=tracked)),encoding='utf-8')
                self.assertEqual(suite.validate(root)['models'],['original','pruned'])
                (root/'original/lung_network.bit').write_bytes(b'changed fixture bit')
                with self.assertRaisesRegex(ValueError,'Changed suite file'):suite.validate(root)


if __name__=='__main__':unittest.main()
