"""Ownership and deployment failure checks without PYNQ or systemd mutations."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ct_studio_boot as boot
import install_ct_studio_boot as install
import start_ct_studio_demo as demo
from urllib.error import URLError, HTTPError


class BootTests(unittest.TestCase):
    def owner(self):
        service = Mock()
        service.failed = service.hardware.fault = None
        service.gate.acquire.return_value = True
        service.thread.is_alive.return_value = False
        return service

    def test_busy_owner_is_not_stopped_or_released(self):
        owner = self.owner()
        owner.gate.acquire.return_value = False
        self.assertFalse(boot.stop_idle(owner))
        owner.server.shutdown.assert_not_called()
        owner.hardware.release_idle.assert_not_called()

    def test_fault_or_non_idle_dma_keeps_hardware_and_lock(self):
        for fault in ('hardware', 'service', 'dma'):
            owner = self.owner()
            if fault == 'hardware': owner.hardware.fault = 'test fault'
            elif fault == 'service': owner.failed = 'test failure'
            else: owner.hardware._dma_status.side_effect = RuntimeError('DMA active')
            with self.assertRaises(RuntimeError): boot.stop_idle(owner)
            owner.hardware.release_idle.assert_not_called()
            owner.hardware.lock_file.close.assert_not_called()
            owner.gate.release.assert_called_once()

    def test_healthy_idle_stop_releases_owner_after_server_stops(self):
        owner = self.owner()
        self.assertTrue(boot.stop_idle(owner))
        names = [str(call).split('(')[0] for call in owner.mock_calls]
        self.assertLess(names.index('call.hardware._dma_status'), names.index('call.server.shutdown'))
        self.assertLess(names.index('call.server.shutdown'), names.index('call.hardware.release_idle'))
        self.assertLess(names.index('call.hardware.release_idle'), names.index('call.hardware.lock_file.close'))

    def test_wrong_pinned_package_rejected_before_importing_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = json.dumps(dict(basic_transport='ct_basic_zlib_cache_v1', ldct={'clock_mhz':100})).encode()
            (folder/'platform_manifest.json').write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(boot.validate_identity(folder,digest)['ldct']['clock_mhz'],100)
            with self.assertRaises(ValueError): boot.validate_identity(folder,'0'*64)

    def test_installer_preserves_existing_different_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'existing'
            path.write_bytes(b'old')
            with self.assertRaises(FileExistsError): install.write_once_or_same(path,b'new',0o644)
            self.assertEqual(path.read_bytes(),b'old')

    def test_unit_has_no_restart_and_waits_for_pynq_boot(self):
        unit = install.unit_text(PurePosixPath('/home/xilinx/ct_test/accepted'), 'a'*64)
        self.assertIn('After=network.target bootpy.service',unit)
        self.assertIn('Restart=no',unit)
        self.assertIn('SendSIGKILL=no',unit)
        self.assertIn('Type=notify',unit)
        with self.assertRaises(ValueError): install.unit_text(PurePosixPath('/home/invalid space'), 'a'*64)

    def test_pc_waits_only_for_connection_then_health_passes(self):
        client = Mock()
        client.health.side_effect = [URLError('booting'), {'ready':True}]
        with patch.object(demo.time,'sleep') as sleep:
            self.assertEqual(demo.wait_board(client,30),{'ready':True})
        sleep.assert_called_once()
        client.call.assert_not_called()

    def test_pc_does_not_retry_reported_fault_wrong_identity_or_http_error(self):
        for exc in (RuntimeError('fault'),ValueError('identity differs'),HTTPError('http://board',403,'forbidden',{},None)):
            client = Mock(); client.health.side_effect = exc
            with patch.object(demo.time,'sleep') as sleep:
                with self.assertRaises(type(exc)): demo.wait_board(client,300)
            client.health.assert_called_once();sleep.assert_not_called()

    def test_pc_check_only_does_not_wait_on_unreachable_board(self):
        client = Mock();client.health.side_effect = URLError('offline')
        with patch.object(demo.time,'sleep') as sleep:
            with self.assertRaises(URLError):demo.wait_board(client,0)
        sleep.assert_not_called()


if __name__ == '__main__': unittest.main()
