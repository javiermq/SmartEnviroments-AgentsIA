import io
import json
import unittest
from unittest.mock import Mock, patch

from smart_home_agent.user_sessions import RemoteConversation, UserSessions
from smart_home_agent.verified_metrics import requested_interval


class RemoteTraceTests(unittest.TestCase):
    def test_bridge_prints_python_calls_without_trace_flag(self):
        from contextlib import redirect_stdout
        from http.server import ThreadingHTTPServer
        from threading import Thread
        from urllib.request import Request, urlopen
        from smart_home_agent.human_console_bridge import make_handler

        handler = make_handler(Mock(), Mock(), Mock(), Mock(), 'unused', False, False, False)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        payload = dict(user='javi', text='Cuánto he dormido de 00:00 a 01:00',
                       t0='2026-09-16T18:56:00+02:00')
        output = io.StringIO()
        try:
            with redirect_stdout(output), patch('smart_home_agent.verified_metrics.query_aggregation',
                                                return_value={'samples': 60, 'result': 45}):
                with urlopen(Request(f'http://127.0.0.1:{server.server_port}/conversation',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})) as response:
                    self.assertEqual(response.status, 200)
            self.assertIn('PYTHON llamada:', output.getvalue())
            self.assertIn('query_aggregation', output.getvalue())
            self.assertIn('PYTHON resultado:', output.getvalue())
            self.assertIn('"result": 45', output.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_python_query_is_executed_and_trace_returned(self):
        service = UserSessions('unused')
        trace = Mock()
        with patch('smart_home_agent.verified_metrics.query_aggregation',
                   return_value={'samples': 60, 'result': 45}) as query:
            result = service.respond(dict(user='javi', text='Cuánto he dormido de 00:00 a 01:00',
                                          t0='2026-09-16T18:56:00+02:00'), trace)
        query.assert_called_once()
        self.assertEqual([item['event'] for item in result['trace']], ['tool_call', 'tool_result'])
        self.assertEqual(result['trace'][0]['data']['name'], 'query_aggregation')
        self.assertEqual(result['trace'][1]['data']['result'], 45)
        self.assertEqual(trace.call_count, 2)

    def test_reachy_receives_python_trace_events(self):
        remote = RemoteConversation('http://unused', 'unused', 'javi', 'model')
        trace = Mock()
        response = dict(text='Respuesta', trace=[dict(event='tool_call', data={'name': 'query_aggregation'}),
                                                 dict(event='tool_result', data={'result': 45})])
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            self.assertEqual(remote.respond('Sueño', trace), 'Respuesta')
        self.assertEqual([call.args[0] for call in trace.call_args_list],
                         ['tool_call', 'tool_result', 'remote_response'])

    def test_ambiguous_stt_night_does_not_default_to_today(self):
        with self.assertRaisesRegex(ValueError, 'anoche'):
            requested_interval('¿Cuánto dormía noche?', '2026-09-16T18:56:00+02:00')
