"""Launcher safety/ownership checks; no real board, process or browser used."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from urllib.error import URLError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import start_ct_studio_demo as demo


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'ldct':{'clock_mhz':100.,'model_id':'accepted-model'}}
        self.state=dict(desktop_only=False,fast_platform_active=True,board_connected=True,
                        phase='idle',case_id='preserved-case',
                        ldct=dict(identity=self.manifest['ldct'],fpga_configured=True,running=False))
        self.info=(Mock(),{'X-UI-Token':'TEST UI TOKEN'},self.state)
        self.client=Mock()
        self.args=SimpleNamespace(config=Path('TEST-config.json'),port=8095,
                                  check_only=False,no_browser=False)

    def test_existing_session_reused_without_reloading_or_spawning(self):
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',return_value=self.info),\
             patch.object(demo,'spawn_web') as spawn,patch.object(demo.webbrowser,'open') as browser:
            self.assertEqual(demo.launch(self.args),0)
        spawn.assert_not_called();self.info[0].open.assert_not_called()
        self.client.health.assert_called_once();browser.assert_called_once_with('http://127.0.0.1:8095/')
        self.assertEqual(self.state['case_id'],'preserved-case')

    def test_fault_or_busy_never_starts_a_server_or_submits_a_command(self):
        self.client.health.side_effect=RuntimeError('TEST hardware busy/faulted')
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',return_value=self.info),\
             patch.object(demo,'spawn_web') as spawn,patch.object(demo.webbrowser,'open'):
            self.assertEqual(demo.launch(self.args),1)
        spawn.assert_not_called();self.info[0].open.assert_not_called()

    def test_check_only_does_not_connect_launch_or_open_browser(self):
        self.args.check_only=True
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',return_value=None),\
             patch.object(demo,'spawn_web') as spawn,patch.object(demo.webbrowser,'open') as browser:
            self.assertEqual(demo.launch(self.args),0)
        spawn.assert_not_called();browser.assert_not_called()

    def test_accepted_busy_service_opens_existing_progress_without_reconnecting(self):
        self.manifest.update(platform_manifest_sha256='accepted-package',basic={'window':'accepted-bit'})
        self.client.health.side_effect=RuntimeError('TEST busy')
        self.client.call.return_value=dict(schema='ct_platform_identity_v1',ready=True,busy=True,error=None,
            platform_manifest_sha256='accepted-package',ldct=self.manifest['ldct'],basic=self.manifest['basic'],
            integration_board_acceptance='startup-selftests-passed')
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',return_value=self.info),\
             patch.object(demo,'spawn_web') as spawn,patch.object(demo.webbrowser,'open'):
            self.assertEqual(demo.launch(self.args),0)
        spawn.assert_not_called();self.info[0].open.assert_not_called()

    def test_new_server_started_once_then_connected_without_stopping_it(self):
        process=Mock();process.poll.return_value=None
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',side_effect=[None,None,self.info]),\
             patch.object(demo,'spawn_web',return_value=(process,Path('TEST-logs'))) as spawn,\
             patch.object(demo,'connect_web') as connect,patch.object(demo.time,'sleep'),\
             patch.object(demo.webbrowser,'open'):
            self.assertEqual(demo.launch(self.args),0)
        spawn.assert_called_once();connect.assert_called_once()
        process.terminate.assert_not_called();process.kill.assert_not_called()

    def test_failed_server_is_reported_without_respawn_or_kill(self):
        process=Mock();process.poll.return_value=1
        with patch.object(demo,'configuration',return_value=(self.client,self.manifest)),\
             patch.object(demo,'web_info',return_value=None),\
             patch.object(demo,'spawn_web',return_value=(process,Path('TEST-logs'))) as spawn:
            with self.assertRaisesRegex(RuntimeError,'日志'):demo.launch(self.args)
        spawn.assert_called_once();process.kill.assert_not_called()

    def test_connect_keeps_existing_computation_untouched(self):
        self.state.update(board_connected=False,phase='computing')
        demo.connect_web('http://127.0.0.1:8095',self.info,self.manifest)
        self.info[0].open.assert_not_called()

    def test_other_or_desktop_server_is_rejected_instead_of_replaced(self):
        for wrong in ('<html>Unrelated server</html>',None):
            responses=[io.BytesIO((wrong or '<meta name="ui-token" content="TEST">').encode())]
            if wrong is None:responses.append(io.BytesIO(json.dumps(dict(self.state,desktop_only=True)).encode()))
            opener=Mock();opener.open.side_effect=responses
            with patch.object(demo.request,'build_opener',return_value=opener):
                with self.assertRaises(ValueError):demo.web_info('http://127.0.0.1:8095',self.manifest)

    def test_only_connection_refused_means_no_server(self):
        opener=Mock();opener.open.side_effect=URLError(ConnectionRefusedError())
        with patch.object(demo.request,'build_opener',return_value=opener):
            self.assertIsNone(demo.web_info('http://127.0.0.1:8095',self.manifest))
        opener.open.side_effect=URLError(TimeoutError())
        with patch.object(demo.request,'build_opener',return_value=opener):
            with self.assertRaises(RuntimeError):demo.web_info('http://127.0.0.1:8095',self.manifest)

    def test_wrong_configuration_cannot_start_in_desktop_or_classic_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'model.json').write_bytes(b'TEST model')
            (root/'token.txt').write_text('TEST SECRET TOKEN')
            config=dict(model='model.json',token_file='token.txt',platform_manifest='manifest.json',
                        board_url='http://127.0.0.1:8765')
            (root/'config.json').write_text(json.dumps(config))
            (root/'manifest.json').write_text(json.dumps(self.manifest))
            with patch.object(demo,'MODEL_SHA',demo.hashlib.sha256(b'TEST model').hexdigest()):
                with self.assertRaisesRegex(ValueError,'共享基础'):demo.configuration(root/'config.json')


class DesktopLauncherTests(unittest.TestCase):
    def setUp(self):
        from roi_workspace import reference_identity
        self.args=SimpleNamespace(port=8096,check_only=False,no_browser=True)
        self.info=(Mock(),{},dict(desktop_only=True,roi={'identity':reference_identity()},case_id='keep-case'))

    def test_existing_pc_session_reused_without_board_credentials_or_commands(self):
        with patch.object(demo,'read_web_info',return_value=self.info),patch.object(demo,'configuration') as config,\
             patch.object(demo,'spawn_web') as spawn,patch.object(demo,'connect_web') as connect:
            self.assertEqual(demo.launch_desktop(self.args),0)
        config.assert_not_called();spawn.assert_not_called();connect.assert_not_called()
        self.assertEqual(self.info[2]['case_id'],'keep-case')

    def test_cold_pc_start_uses_desktop_flag_and_never_connects_board(self):
        process=Mock();process.poll.return_value=None
        with patch.object(demo,'read_web_info',side_effect=[None,None,self.info]),\
             patch.object(demo,'spawn_web',return_value=(process,Path('TEST-logs'))) as spawn,\
             patch.object(demo,'connect_web') as connect,patch.object(demo.time,'sleep'):
            self.assertEqual(demo.launch_desktop(self.args),0)
        spawn.assert_called_once_with(None,8096,desktop=True);connect.assert_not_called()
        process.kill.assert_not_called()

    def test_other_port_owner_preserved(self):
        self.info[2]['desktop_only']=False
        with patch.object(demo,'read_web_info',return_value=self.info),patch.object(demo,'spawn_web') as spawn:
            with self.assertRaises(ValueError):demo.launch_desktop(self.args)
        spawn.assert_not_called()

    def test_pc_check_only_never_starts_a_server(self):
        self.args.check_only=True
        with patch.object(demo,'read_web_info',return_value=None),patch.object(demo,'spawn_web') as spawn:
            self.assertEqual(demo.launch_desktop(self.args),0)
        spawn.assert_not_called()


if __name__=='__main__':unittest.main()
