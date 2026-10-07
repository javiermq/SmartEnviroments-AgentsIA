import json
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from threading import Thread
from urllib.error import HTTPError

from smart_home_agent.conversation_service import make_handler
from smart_home_agent.environment_service import environment_payload, post
from smart_home_agent.human_console_bridge import make_handler as voice_handler
from smart_home_agent.user_sessions import UserSessions
from tests.test_live_environment import wait_worker
from unittest.mock import Mock


class ConversationServiceTests(unittest.TestCase):
    def test_http_environment_voice_decision_and_delivery(self):
        requests = []
        class FakeOllama(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                content = json.dumps({'action': 'speak', 'text': '¿Qué tal va la tarde?'}) if 'format' in payload else 'Hola, Javi.'
                body = json.dumps({'message': {'role': 'assistant', 'content': content}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        ollama = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
        sessions = UserSessions(f'http://127.0.0.1:{ollama.server_port}/api/chat')
        conversation = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(sessions))
        workers = [Thread(target=s.serve_forever, daemon=True) for s in (ollama, conversation)]
        for worker in workers:
            worker.start()
        base = f'http://127.0.0.1:{conversation.server_port}'
        payload = environment_payload('javi', '2026-09-16T18:56:00+02:00')
        try:
            with redirect_stdout(StringIO()):
                self.assertTrue(post(base + '/environment', payload)['accepted'])
                post(base + '/activate', payload)
                state = sessions.sessions[('reachy', 'javi')]
                wait_worker(state)
                self.assertEqual(post(base + '/events', payload)['action'], 'speak')
                answer = post(base + '/conversation', {**payload, 'text': 'Hola'})
                self.assertEqual(answer['text'], 'Hola, Javi.')
                self.assertEqual(post(base + '/events', payload)['action'], 'silent')
                updated = environment_payload('javi', '2026-09-16T18:57:00+02:00')
                post(base + '/environment', updated)
                wait_worker(state)
                event = post(base + '/events', payload)
                self.assertEqual(event['t0'], updated['t0'])
                self.assertTrue(post(base + '/spoken', {**payload, 'revision': event['revision']})['accepted'])
                post(base + '/deactivate', payload)
            context = json.loads(requests[-1]['messages'][0]['content'].split('CONTEXTO_JSON:\n')[1])
            self.assertEqual(context['t0'], updated['t0'])
            self.assertEqual(context['user_turns_last_minute'], 1)
            self.assertEqual(context['environment_row']['room'], updated['row']['room'])
            self.assertIn('activities_last_36h', context)
            self.assertNotIn('tools', requests[-1])
            self.assertEqual(requests[-1]['format']['properties']['action']['enum'], ['silent', 'speak'])
            self.assertEqual(requests[-1]['messages'][-1]['role'], 'system')
        finally:
            for server in (conversation, ollama):
                server.shutdown()
                server.server_close()
            for worker in workers:
                worker.join()

    def test_voice_service_rejects_conversation_routes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), voice_handler(
            Mock(), Mock(), Mock(), Mock(), None, False, False, False, voice_only=True))
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with self.assertRaises(HTTPError) as error:
                post(f'http://127.0.0.1:{server.server_port}/conversation', {'text': 'Hola'})
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
