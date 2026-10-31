"""Behavior tests for non-destructive launcher decisions, no real board involved."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch,Mock
from urllib.error import HTTPError,URLError
from types import SimpleNamespace

SRC=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bootstrap',SRC/'desktop_bootstrap.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class LauncherTests(unittest.TestCase):
    def pair(self,**changed):
        legacy=dict(platform_manifest_sha256='old',busy=False,ready=True,error=None)
        identity=dict(service_bundle_sha256='bundle',shared_nodule={'model':'original'},
                      shared_batch={'max_patches':8},busy=False,ready=True,error=None)
        identity.update(changed)
        client=Mock();client.manifest={'platform_manifest_sha256':'old'};client.call.return_value=legacy
        shared=Mock();shared.bundle={k:identity[k] for k in ['service_bundle_sha256','shared_nodule','shared_batch']}
        shared.call.return_value=identity
        return client,shared

    def test_matching_ready_only_reads_identities(self):
        client,shared=self.pair()
        self.assertEqual(b.ready_once(client,shared),(True,'ready'))
        client.call.assert_called_once_with('/v1/platform',timeout=4)
        shared.call.assert_called_once_with('/v1/shared/platform',timeout=4)
        client.health.assert_called_once();shared.health.assert_called_once()

    def test_mismatched_service_is_rejected_before_connect(self):
        client,shared=self.pair();shared.call.return_value['service_bundle_sha256']='different'
        with self.assertRaises(b.LaunchError) as got:b.ready_once(client,shared)
        self.assertEqual(got.exception.code,'IDENTITY')
        client.health.assert_not_called()

    def test_busy_does_not_issue_computation_or_cancel(self):
        client,shared=self.pair(busy=True)
        self.assertEqual(b.ready_once(client,shared),(False,'busy'))
        client.health.assert_not_called();shared.health.assert_not_called()

    def test_board_fault_is_not_retried(self):
        client,shared=self.pair(error='fault')
        with self.assertRaises(b.LaunchError) as got:b.wait_ready(client,shared,75)
        self.assertEqual(got.exception.code,'BOARD_FAULT')
        self.assertEqual(client.call.call_count,1)

    def test_forbidden_becomes_chinese_pairing_message(self):
        client,shared=self.pair();client.call.side_effect=HTTPError('url',403,'Forbidden',None,None)
        with self.assertRaises(b.LaunchError) as got:b.wait_ready(client,shared,75)
        self.assertEqual(got.exception.code,'AUTH');self.assertIn('凭据',str(got.exception))
        self.assertEqual(client.call.call_count,1)

    def test_network_failure_is_bounded_and_no_reinstall(self):
        client,shared=self.pair();client.call.side_effect=URLError('unreachable')
        with patch.object(b,'emit'):
            with self.assertRaises(b.LaunchError) as got:
                b.wait_ready(client,shared,0,clock=lambda:1,sleep=Mock())
        self.assertEqual(got.exception.code,'NETWORK')

    def test_busy_existing_session_is_preserved(self):
        client,shared=self.pair(busy=True)
        b.wait_ready(client,shared,0,existing=('valid existing web',))
        client.health.assert_not_called()

    def test_token_redaction(self):
        with patch.object(b,'TOKEN','a-secret-token'):
            self.assertNotIn('a-secret-token',b.safe('failure a-secret-token'))

    def test_unrelated_port_is_skipped_and_never_closed(self):
        module=SimpleNamespace(web_info=Mock(side_effect=[ValueError('other service'),None]))
        with patch.dict('sys.modules',{'start_ct_studio_demo':module}),patch.object(b,'ROOT',Path('X:/not-installed')):
            self.assertEqual(b.choose_port({}),(8100,None))
        self.assertEqual(module.web_info.call_count,2)

if __name__=='__main__':unittest.main(verbosity=2)
