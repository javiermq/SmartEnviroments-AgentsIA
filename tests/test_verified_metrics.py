import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from smart_home_agent.context import build_context
from smart_home_agent.conversation_backend import ConversationBackend
from smart_home_agent.conversation_demo import _run_turn, _system_prompt
from smart_home_agent.verified_metrics import verified_answer


T0 = '2026-09-16T18:56:00+02:00'


class VerifiedMetricsTests(unittest.TestCase):
    def test_snapshot_cannot_be_mistaken_for_daily_counters(self):
        context = build_context(T0)
        for name in ('steps', 'distance_m', 'sleep'):
            self.assertNotIn(name, context['sensors_at_t0'])
        sample = context['watch_minute_sample_NOT_TOTALS']
        self.assertEqual(sample['steps_in_this_minute'], 5)
        self.assertFalse(sample['is_daily_total'])

    def test_both_totals_require_real_queries_not_llm_or_history(self):
        backend = ConversationBackend(T0)
        backend.messages.append({'role': 'assistant', 'content': 'Has dado 5 pasos y dormido 4 minutos.'})
        with patch.object(backend, '_request_ollama', side_effect=AssertionError('No usar LLM para cifras')):
            answer = backend.respond('¿Cuántos pasos he dado y cuánto he dormido hoy?')
        self.assertIn('10009', answer)
        self.assertIn('7 h y 0 min', answer)
        results = [json.loads(m['content']) for m in backend.messages if m['role'] == 'tool']
        self.assertEqual([r['result'] for r in results], [10009, 420])
        self.assertEqual(backend.messages[-1]['content'], answer)

    def test_explicit_interval_and_midnight(self):
        result = verified_answer('Pasos entre 09:00 y 09:40', T0, [])
        self.assertIn('4855', result)
        result = verified_answer('Cuánto he dormido entre 2026-09-15T22:15:00+02:00 y 2026-09-16T07:00:00+02:00', T0, [])
        self.assertIn('8 h y 45 min', result)

    def test_yesterday_does_not_use_today(self):
        messages = []
        verified_answer('Pasos ayer', T0, messages)
        result = json.loads(messages[-1]['content'])
        self.assertEqual(result['time_init'], '2026-09-15T00:00:00+02:00')
        self.assertEqual(result['time_end'], '2026-09-16T00:00:00+02:00')

    def test_ambiguous_interval_does_not_default_to_today(self):
        for question in ('Cuánto he dormido anoche', 'Pasos esta semana', 'Pasos desde las 10', 'Pasos mañana', 'Pasos el lunes', 'Cuánto dormí hace dos días'):
            messages = []
            with patch('smart_home_agent.verified_metrics.query_aggregation') as query:
                answer = verified_answer(question, T0, messages)
            query.assert_not_called()
            self.assertTrue(answer)
            self.assertEqual(messages, [])

    def test_missing_and_partial_data_do_not_become_zero_or_complete_total(self):
        answer = verified_answer('Pasos hoy', '2026-09-18T18:56:00+02:00', [])
        self.assertIn('No puedo verificar', answer)
        with patch('smart_home_agent.verified_metrics.query_aggregation', return_value={'samples': 1, 'result': 5}):
            answer = verified_answer('Pasos hoy', T0, [])
        self.assertIn('No puedo verificar', answer)
        self.assertNotIn('Pasos registrados: 5', answer)

    def test_unsolicited_unverified_measurement_is_not_saved(self):
        backend = ConversationBackend(T0)
        backend._request_ollama = lambda: {'content': 'Has dado cinco pasos y dormido una hora.'}
        answer = backend.respond('Hola')
        self.assertNotIn('cinco', answer)
        self.assertEqual(answer, backend.messages[-1]['content'])

    def test_demo_executes_and_traces_tools_without_model(self):
        messages = [{'role': 'system', 'content': _system_prompt(T0)}]
        out, err = io.StringIO(), io.StringIO()
        with patch('smart_home_agent.conversation_demo._request_ollama', side_effect=AssertionError('No LLM')):
            with redirect_stdout(out), redirect_stderr(err):
                _run_turn(messages, 'Cuánto he dormido hoy', T0)
        self.assertIn('7 h y 0 min', out.getvalue())
        self.assertIn('tool_call', err.getvalue())
        self.assertIn('tool_result', err.getvalue())

    def test_ordinary_conversation_still_uses_model(self):
        backend = ConversationBackend(T0)
        with patch.object(backend, '_request_ollama', return_value={'content': 'Hola, ¿qué tal?'}) as request:
            self.assertEqual(backend.respond('Hola'), 'Hola, ¿qué tal?')
        request.assert_called_once()

    def test_general_sleep_advice_without_numbers_is_preserved(self):
        backend = ConversationBackend(T0)
        with patch.object(backend, '_request_ollama', return_value={'content': 'Para dormir mejor, procura mantener horarios regulares.'}):
            self.assertIn('horarios regulares', backend.respond('Consejos para dormir mejor'))

    def test_next_step_is_not_a_watch_measurement(self):
        backend = ConversationBackend(T0)
        content = 'Estás cocinando. Llevas 16 minutos. ¿Cuál es el siguiente paso?'
        with patch.object(backend, '_request_ollama', return_value={'content': content}):
            self.assertEqual(backend.respond('¿Qué actividad estoy haciendo?'), content)
