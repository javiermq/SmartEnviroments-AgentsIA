import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smart_home_agent.user_sessions import UserSessions
from smart_home_agent.recognized_conversation import accepted_identity, circle
from smart_home_agent.speech_text import for_speech
from smart_home_agent.human_console_bridge import make_handler

T0 = '2026-09-16T00:02:00+02:00'


class SessionTests(unittest.TestCase):
    def test_http_endpoint_keeps_users_separate_and_rejects_bad_payload(self):
        from http.server import ThreadingHTTPServer
        import threading
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        handler = make_handler(Mock(), Mock(), Mock(), Mock(), 'http://unused', False, False, False)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        endpoint = f'http://127.0.0.1:{server.server_port}/conversation'
        def post(payload):
            with urlopen(Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})) as response:
                return json.load(response)
        try:
            with patch('smart_home_agent.conversation_backend.ConversationBackend._request_ollama',
                       return_value={'content': 'Hola.'}):
                for name in ['javi', 'mariola', 'javi']:
                    result = post(dict(user=name, text='Hola', t0='2026-09-16T18:56:00+02:00'))
                    self.assertEqual(result['user'], name)
                    self.assertEqual(result['text'], 'Hola.')
            with self.assertRaises(HTTPError) as error:
                post([])
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_separate_sensor_totals_and_histories_when_switching_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, steps in [('javi', 3), ('mariola', 8)]:
                folder = root / name
                folder.mkdir()
                (folder / 'user_context.json').write_text(json.dumps({'nombre': name}))
                (folder / 'agent_style.txt').write_text('Breve')
                (folder / 'simulated_sensor_data_test.tsv').write_text(
                    'timestamp\tsteps\tdistance_m\tsleep\n'
                    f'2026-09-16T00:00:00+02:00\t{steps}\t0\t0\n'
                    f'2026-09-16T00:01:00+02:00\t{steps}\t0\t0\n')
            service = UserSessions('http://unused', root)
            def ask(user):
                return service.respond(dict(user=user, text='Pasos hoy', t0=T0))
            self.assertIn('6.', ask('javi')['text'])
            self.assertIn('16.', ask('mariola')['text'])
            self.assertIn('6.', ask('Javi')['text'])
            javi = service.sessions[('reachy', 'javi', T0, 'qwen3:4b-instruct-2507-q4_K_M')][0]
            mariola = service.sessions[('reachy', 'mariola', T0, 'qwen3:4b-instruct-2507-q4_K_M')][0]
            self.assertEqual(sum(m['role'] == 'user' for m in javi.messages), 2)
            self.assertEqual(sum(m['role'] == 'user' for m in mariola.messages), 1)
            self.assertNotIn('mariola', javi.messages[0]['content'])
            with self.assertRaises(ValueError):
                ask('../mariola')

    def test_identity_rejects_unknown_nonfinite_and_insufficient_votes(self):
        candidate = dict(user='Javi', distance=0.2, threshold=0.3, votes=2)
        result = dict(user='Javi', status='matched', candidates=[candidate])
        self.assertEqual(accepted_identity(result), 'javi')
        for changes in [dict(votes=1), dict(distance=0.4), dict(distance=float('nan'))]:
            self.assertIsNone(accepted_identity({**result, 'candidates': [{**candidate, **changes}]}))
        self.assertIsNone(accepted_identity({'user': 'unknow', 'status': 'no_match'}))

    def test_tts_time_uses_minutes_without_iso_offset(self):
        self.assertEqual(for_speech('A las 18:56.'), 'A las dieciocho horas y cincuenta y seis minutos.')
        self.assertEqual(for_speech('2026-09-16T00:00:00+02:00'), 'cero horas y cero minutos')


class MotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_recognition_returns_to_neutral(self):
        interface = SimpleNamespace(robot=Mock(), _idle_gesture=Mock())
        with patch.dict(sys.modules, {'reachy_mini.utils': SimpleNamespace(create_head_pose=lambda **kw: kw)}):
            task = asyncio.create_task(circle(interface))
            await asyncio.sleep(0.06)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        interface.robot.set_target.assert_called()
        interface._idle_gesture.assert_called_once()
