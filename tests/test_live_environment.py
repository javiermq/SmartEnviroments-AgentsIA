import asyncio
import json
import time
import unittest
from copy import deepcopy
from datetime import timedelta
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from smart_home_agent.context import DATA_DIR, parse_timestamp, system_prompt
from smart_home_agent.conversation_backend import ConversationBackend
from smart_home_agent.environment_service import environment_payload, run
from smart_home_agent.proactive import listen_with_environment
from smart_home_agent.user_sessions import UserSessions

T0 = '2026-09-16T18:56:00+02:00'
MODEL = 'qwen3:4b-instruct-2507-q4_K_M'


def wait_worker(state):
    deadline = time.monotonic() + 3
    while state.worker_running and time.monotonic() < deadline:
        time.sleep(0.005)
    if state.worker_running:
        raise AssertionError('Worker no finalizó')


class LiveEnvironmentTests(unittest.TestCase):
    def payload(self, t0=T0, user='javi'):
        return environment_payload(user, t0)

    def test_same_user_keeps_session_across_time_and_counts_real_voice(self):
        sessions = UserSessions('unused')
        payload = self.payload()
        sessions.environment(payload)
        with patch.object(ConversationBackend, '_request_ollama', return_value={'content': 'Hola.'}):
            sessions.respond({**payload, 'text': 'Hola'})
            state = sessions.sessions[('reachy', 'javi')]
            backend = state.backend
            next_payload = self.payload('2026-09-16T18:57:00+02:00')
            sessions.environment(next_payload)
            # El cliente aún tiene el t0 inicial; no puede retroceder el entorno.
            result = sessions.respond({**payload, 'text': 'Sigo aquí'})
        self.assertIs(state.backend, backend)
        self.assertEqual(len(sessions.sessions), 1)
        self.assertEqual(result['t0'], next_payload['t0'])
        self.assertEqual(sum(m['role'] == 'user' for m in backend.messages), 2)
        self.assertEqual(backend.environment['user_turns_last_minute'], 2)
        with state.lock:
            state.turns = __import__('collections').deque([time.monotonic() - 61])
            self.assertEqual(state.context_with_conversation()['user_turns_last_minute'], 0)

    def test_full_row_and_36h_har_exclude_future(self):
        payload = self.payload()
        context = payload['context']
        self.assertIn('steps', payload['row'])
        self.assertIn('sleep', payload['row'])
        self.assertIn('room', payload['row'])
        self.assertEqual(parse_timestamp(context['t0']) - parse_timestamp(context['history_start']),
                         timedelta(hours=36))
        for activity in context['activities_last_36h']:
            self.assertLessEqual(parse_timestamp(activity['end_time']), parse_timestamp(T0))
        self.assertTrue(all('end_time' not in item for item in context['current_activities']))
        prompt = system_prompt(T0, DATA_DIR / 'javi', context)
        self.assertIn('36 horas', prompt)

    def test_environment_is_not_a_user_turn_and_silence_keeps_session_alive(self):
        sessions = UserSessions('unused')
        payload = self.payload()
        sessions.environment(payload)
        with patch.object(ConversationBackend, '_request_ollama',
                          return_value={'content': '{"action":"silent","text":""}'}):
            sessions.activate(payload)
            state = sessions.sessions[('reachy', 'javi')]
            wait_worker(state)
            sessions.environment(self.payload('2026-09-16T18:57:00+02:00'))
            wait_worker(state)
        self.assertTrue(state.active)
        self.assertEqual(state.context['t0'], '2026-09-16T18:57:00+02:00')
        self.assertEqual(len(state.backend.messages), 1)
        self.assertEqual(state.context_with_conversation()['user_turns_last_minute'], 0)
        self.assertEqual(sessions.events(payload)['action'], 'silent')

    def test_speak_is_committed_once_and_other_user_is_isolated(self):
        sessions = UserSessions('unused')
        payload = self.payload()
        sessions.environment(payload)
        with patch.object(ConversationBackend, '_request_ollama',
                          return_value={'content': '{"action":"speak","text":"¿Qué tal va la tarde?"}'}):
            sessions.activate(payload)
            state = sessions.sessions[('reachy', 'javi')]
            wait_worker(state)
            event = sessions.events(payload)
            self.assertEqual(event['action'], 'speak')
            self.assertEqual(len(state.backend.messages), 1)
            self.assertTrue(sessions.spoken({**payload, 'revision': event['revision']})['accepted'])
            self.assertFalse(sessions.spoken({**payload, 'revision': event['revision']})['accepted'])
            self.assertEqual(state.backend.messages[-1]['content'], event['text'])
            mariola = self.payload(user='mariola')
            sessions.environment(mariola)
            self.assertEqual(sessions.events(mariola)['action'], 'silent')
            sessions.activate(mariola)
            wait_worker(sessions.sessions[('reachy', 'mariola')])
            self.assertFalse(state.active)

    def test_new_environment_arrives_while_ollama_is_running(self):
        sessions = UserSessions('unused')
        payload = self.payload()
        sessions.environment(payload)
        started, release = Event(), Event()
        observed = []
        def decide(backend):
            observed.append(backend.t0)
            if len(observed) == 1:
                started.set()
                release.wait(2)
            return {'action': 'speak', 'text': 'Comentario reciente'}
        with patch.object(ConversationBackend, 'evaluate_environment', decide):
            sessions.activate(payload)
            try:
                self.assertTrue(started.wait(1))
                newest = self.payload('2026-09-16T18:58:00+02:00')
                sessions.environment(newest)
                state = sessions.sessions[('reachy', 'javi')]
                self.assertEqual(state.context['t0'], newest['t0'])
                self.assertEqual(sessions.events(payload)['action'], 'silent')
            finally:
                release.set()
            wait_worker(state)
        self.assertEqual(observed, [T0, newest['t0']])
        self.assertEqual(sessions.events(payload)['t0'], newest['t0'])

    def test_rejects_clock_reversal_and_future_har(self):
        sessions = UserSessions('unused')
        payload = self.payload()
        sessions.environment(payload)
        with self.assertRaisesRegex(ValueError, 'retroceder'):
            sessions.environment(self.payload('2026-09-16T18:55:00+02:00'))
        bad = deepcopy(payload)
        bad['context']['activities_last_36h'][0]['end_time'] = '2026-09-17T00:00:00+02:00'
        with self.assertRaisesRegex(ValueError, 'HAR futuro'):
            sessions.environment(bad)

    def test_daemon_sends_initial_and_next_minute_for_both_users(self):
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.side_effect = [False, True]
        with patch('smart_home_agent.environment_service.post') as send:
            run('http://unused/conversation', ['javi', 'mariola'], T0, stop=stop)
        sent = [call.args[1] for call in send.call_args_list]
        self.assertEqual([p['user'] for p in sent], ['javi', 'mariola', 'javi', 'mariola'])
        self.assertEqual(sent[2]['t0'], '2026-09-16T18:57:00+02:00')

    def test_json_decision_request_uses_system_event(self):
        backend = ConversationBackend(T0)
        backend.update_environment(self.payload()['context'])
        captured = []
        def request(decision=False):
            captured.extend(deepcopy(backend.messages))
            self.assertTrue(decision)
            return {'content': '{"action":"silent","text":""}'}
        backend._request_ollama = request
        self.assertEqual(backend.evaluate_environment()['action'], 'silent')
        self.assertTrue(all(m['role'] != 'user' for m in captured))
        self.assertIn('EVENTO ENVIRONMENT', captured[-1]['content'])


class ProactiveDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_environment_speaks_during_silence_and_then_listens(self):
        waiting = asyncio.Event()
        calls = 0
        async def listen():
            nonlocal calls
            calls += 1
            if calls == 1:
                await waiting.wait()
            return 'Respuesta del usuario'
        interface = SimpleNamespace(listen=listen, speak=AsyncMock(),
                                    user_speaking=False, set_response_gesture=Mock())
        backend = SimpleNamespace(accept=Mock(return_value=True))
        with patch('smart_home_agent.proactive.wait_for_environment',
                   AsyncMock(return_value={'action': 'speak', 'text': 'Hola', 'revision': 1})):
            self.assertEqual(await listen_with_environment(interface, backend), 'Respuesta del usuario')
        interface.speak.assert_awaited_once_with('Hola')
