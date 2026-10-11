import io
import json
import unittest
from unittest.mock import patch
from smart_home_agent.two_agent_backend import TwoAgentBackend
from smart_home_agent.sensor_queries import query_window


class TwoAgentTests(unittest.TestCase):
    def test_no_tools_and_history(self):
        backend = TwoAgentBackend()
        with patch.object(backend, '_request', side_effect=[{'content': 'Saludo sin datos necesarios'}, {'content': 'Hola.'}]) as request:
            self.assertEqual(backend.respond('Hola'), 'Hola.')
        self.assertEqual([c.args[0] for c in request.call_args_list], ['A', 'B'])
        self.assertEqual(request.call_args_list[1].args[2], [])
        self.assertEqual(len(backend.history), 2)

    def test_multiple_queries_and_handoff(self):
        backend = TwoAgentBackend()
        calls = [{'function': {'name': 'query_window', 'arguments': {'sensor_type': kind, 'window_minutes': 60}}}
                 for kind in ['watch.steps', 'watch.sleep']]
        events = []
        with patch.object(backend, '_request', side_effect=[{'content': '', 'tool_calls': calls}, {'content': 'Resumen'}, {'content': 'Respuesta'}]) as request:
            backend.respond('Pasos y sueño en la última hora', lambda e, d: events.append((e, d)))
        handoff = next(d for e, d in events if e == 'agent_handoff')
        self.assertEqual(len(handoff['verified_results']), 2)
        self.assertEqual(handoff['verified_results'][0]['t0'], backend.t0)
        self.assertIn('verified_results', request.call_args_list[-1].args[1][0]['content'])

    def test_future_query_rejected(self):
        backend = TwoAgentBackend()
        call = {'function': {'name': 'query_window', 'arguments': {'sensor_type': 'watch.steps', 'window_minutes': 60, 't0': '2026-09-17T12:00:00+02:00'}}}
        events = []
        with patch.object(backend, '_request', side_effect=[{'tool_calls': [call]}, {'content': 'Sin datos'}, {'content': 'No disponible'}]):
            backend.respond('Mañana', lambda e, d: events.append(e))
        self.assertIn('tool_error', events)
        self.assertNotIn('tool_result', events)

    def test_verbose_http_payloads_and_responses(self):
        backend = TwoAgentBackend()
        events = []
        raw = {'message': {'role': 'assistant', 'content': 'Resumen'}, 'done': True}
        with patch('smart_home_agent.two_agent_backend.urlopen', return_value=io.BytesIO(json.dumps(raw).encode())):
            backend._request('A', [{'role': 'user', 'content': 'Hola'}], [], lambda e, d: events.append((e, d)))
        self.assertEqual([e for e, d in events], ['agent_request', 'agent_response'])
        self.assertEqual(events[0][1]['payload']['messages'][0]['content'], 'Hola')
        self.assertEqual(events[1][1]['response'], raw)

    def test_window_and_invalid_sizes(self):
        result = query_window('watch.steps', 40, t0='2026-09-16T09:40:00+02:00')
        self.assertEqual(result['samples'], 40)
        self.assertTrue(result['time_init'].startswith('2026-09-16T09:00:00'))
        for size in [0, -1, float('inf'), True, 0.5]:
            with self.assertRaises(ValueError):
                query_window('watch.steps', size)


if __name__ == '__main__':
    unittest.main()
