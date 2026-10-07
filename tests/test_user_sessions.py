import asyncio
import json
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

from smart_home_agent.user_sessions import UserSessions
from smart_home_agent.recognized_conversation import (
    accepted_identity, circle, is_goodbye, recognition_chirp, recognition_sound, run_recognized,
)
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
            javi = service.sessions[('reachy', 'javi')].backend
            mariola = service.sessions[('reachy', 'mariola')].backend
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
    async def test_recognition_sound_is_soft_and_stops_before_greeting(self):
        samples = recognition_chirp(24000)
        self.assertEqual(samples.shape, (7200, 1))
        self.assertEqual(samples.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(samples))), 0.06)
        self.assertGreater(float(np.max(np.abs(samples))), 0.01)
        interface = SimpleNamespace(robot=Mock())
        interface.robot.media.get_output_audio_samplerate.return_value = 24000
        task = asyncio.create_task(recognition_sound(interface))
        while not interface.robot.media.push_audio_sample.called:
            await asyncio.sleep(0.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        interface.robot.media.push_audio_sample.assert_called_once()

    async def test_conversation_holds_identity_until_goodbye_then_recognizes_next_user(self):
        import time
        camera_passes = []
        permit_holder = []
        def camera(robot, detector, endpoint, interval, minimum, event, stop, permit):
            permit_holder.append(permit)
            for name in ['Javi', 'mariola']:
                while not permit.is_set() and not stop.is_set():
                    time.sleep(0.005)
                if stop.is_set():
                    return
                camera_passes.append(name)
                event('recognizing', None)
                event('identity', dict(user=name, status='matched', candidates=[
                    dict(user=name, distance=0.2, threshold=0.3, votes=2)]))
                event('finished', None)
                # Simula un evento tardío que no debe borrar la identidad.
                event('finished', None)
                while not permit.is_set() and not stop.is_set():
                    time.sleep(0.005)
            while not permit.is_set() and not stop.is_set():
                time.sleep(0.005)

        utterances = iter([TimeoutError('silencio'), 'Hola', 'Adiós, Reachy.', 'Hola', 'adios'])
        async def listen():
            self.assertFalse(permit_holder[0].is_set())
            item = next(utterances)
            if isinstance(item, Exception):
                raise item
            return item
        async def no_motion(interface):
            await asyncio.Future()
        interface = SimpleNamespace(robot=Mock(), _gesture_task=None,
                                    listen=listen, speak=AsyncMock(),
                                    on_thinking=AsyncMock(), set_response_gesture=Mock(),
                                    _idle_gesture=Mock())
        interface.robot.media.get_output_audio_samplerate.return_value = 24000
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.onnx'
            model.touch()
            args = SimpleNamespace(detector_model=str(model), vision_url='unused',
                                   conversation_url='unused', t0=T0, session_id='robot')
            backends = []
            def remote(endpoint, t0, user, model_name, session):
                backend = SimpleNamespace(respond=Mock(return_value='Respuesta'), last_gesture=None,
                                          activate=Mock(), deactivate=Mock(),
                                          poll=Mock(return_value={'action': 'silent'}), accept=Mock())
                backends.append((user, backend))
                return backend
            with patch.dict(sys.modules, {'cv2': SimpleNamespace(FaceDetectorYN=Mock())}), \
                 patch('smart_home_agent.recognized_conversation.run_camera', camera), \
                 patch('smart_home_agent.recognized_conversation.circle', no_motion), \
                 patch('smart_home_agent.recognized_conversation.RemoteConversation', remote):
                with self.assertRaisesRegex(RuntimeError, 'cámara'):
                    await asyncio.wait_for(run_recognized(
                        interface, SimpleNamespace(ollama_model='test'), args, None), timeout=3)
        self.assertEqual(camera_passes, ['Javi', 'mariola'])
        self.assertEqual([user for user, _ in backends], ['javi', 'mariola'])
        for _, backend in backends:
            backend.respond.assert_called_once_with('Hola', None)
        self.assertEqual([call.args[0] for call in interface.speak.await_args_list],
                         ['Hola Javi', 'Respuesta', 'Adiós Javi',
                          'Hola Mariola', 'Respuesta', 'Adiós Mariola'])
        self.assertGreaterEqual(interface._idle_gesture.call_count, 2)

    def test_goodbye_accepts_stt_accents_and_punctuation_without_matching_other_topics(self):
        for text in ['adios', '¡Adiós!', 'Bueno, adiós, Reachy.', 'Gracias, adiós']:
            self.assertTrue(is_goodbye(text))
        for text in ['Hola', 'No quiero decir adiós', 'Qué significa adiós']:
            self.assertFalse(is_goodbye(text))

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
