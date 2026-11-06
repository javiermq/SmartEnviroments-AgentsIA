import json
import unittest

from tests.test_historical_har import Fixture
from smart_home_agent.historical_agent import HistoricalAgent
from smart_home_agent.historical_har import WATCH, iso, stamp
from smart_home_agent.metrics_har_agent import schemas
from smart_home_agent.query_time import resolve_window, select_relative_window, reference_time


class TimeContextTests(unittest.TestCase):
    def test_question_record_utc_is_converted_and_explicit_t0_has_priority(self):
        cutoff, source = reference_time(None, '2026-10-31T23:10:45Z', 0)
        self.assertEqual(iso(cutoff), '2026-11-01T00:10:00+01:00')
        self.assertEqual(source, 'question_record')
        override, source = reference_time('2026-10-31T20:00:00+01:00', '2026-10-31T23:10:45Z', 0)
        self.assertEqual(iso(override), '2026-10-31T20:00:00+01:00')
        self.assertEqual(source, 'explicit_t0')
    def test_today_yesterday_and_rolling_across_dst(self):
        ref = stamp('2026-10-26T08:00:00+01:00')
        yesterday = resolve_window('pasos ayer', ref)
        self.assertEqual((yesterday.end-yesterday.start)//60, 1500)
        rolling = resolve_window('pasos ultimas 24 horas', ref)
        self.assertEqual((rolling.end-rolling.start)//60, 1440)
        spring = resolve_window('ayer', stamp('2026-03-30T08:00:00+02:00'))
        self.assertEqual((spring.end-spring.start)//60, 1380)

    def test_fixed_t0_duration_and_spelled_minutes(self):
        ref = stamp('2026-11-03T12:06:00+01:00')
        bound = resolve_window('pasos', ref, window_minutes=4320)
        self.assertEqual(iso(bound.start), '2026-10-31T12:06:00+01:00')
        spelled = resolve_window('pulso en los ultimos diez minutos', ref)
        self.assertEqual(spelled.end-spelled.start, 600)
        explicit = resolve_window('pasos', ref, start='2026-11-02T10:00:00+01:00', end='2026-11-02T11:00:00+01:00')
        self.assertLess(explicit.end, explicit.cutoff)

    def test_unknown_dates_future_and_invalid_context_are_not_silently_today(self):
        ref = stamp('2026-11-03T12:06:00+01:00')
        for question in ('pasos del 31/10', 'pasos del 31 de octubre', 'pasos que dare hoy', 'pasos manana', 'desde las 18:00'):
            with self.subTest(question=question), self.assertRaises(ValueError):
                resolve_window(question, ref)
        for kwargs in ({'window_minutes': True}, {'window': 'ayer', 'window_minutes': 10},
                       {'start': '2026-11-03T10:00:00+01:00'}):
            with self.assertRaises(ValueError):
                resolve_window('pasos', ref, **kwargs)

    def test_morning_is_bounded_and_unsupported_period_is_not_today(self):
        ref = stamp('2026-11-03T18:00:00+01:00')
        bound = resolve_window('pasos esta mañana', ref)
        self.assertEqual(iso(bound.end), '2026-11-03T12:00:00+01:00')
        self.assertEqual(bound.end-bound.start, 720*60)
        with self.assertRaises(ValueError):
            resolve_window('pasos esta tarde', ref)

    def test_schema_has_no_dates_or_paths(self):
        properties = schemas()[0]['function']['parameters']['properties']
        self.assertEqual(set(properties), {'target', 'operation', 'trend_minutes', 'reason',
                                           'period', 'minutes', 'day_offset', 'anchor_offset_minutes'})
        self.assertFalse(set(properties) & {'t0', 'start', 'end', 'window', 'path'})

    def test_model_selects_relative_days_and_offsets_without_moving_cutoff(self):
        ref = stamp('2026-10-26T08:00:00+01:00')
        base = resolve_window('', ref, window_minutes=60)
        yesterday = select_relative_window(base, 'ayer')
        self.assertEqual((yesterday.end-yesterday.start)//60, 1500)
        self.assertEqual(yesterday.cutoff, ref)
        back = select_relative_window(base, 'ultimos_minutos', minutes=90, anchor_offset_minutes=-120)
        self.assertEqual(back.end, ref-120*60)
        self.assertEqual(back.end-back.start, 90*60)
        day = select_relative_window(base, 'dia_relativo', day_offset=-2)
        self.assertEqual(iso(day.start)[:10], '2026-10-24')
        for kwargs in ({'anchor_offset_minutes': 1}, {'day_offset': 1}):
            with self.assertRaises(ValueError):
                select_relative_window(base, **kwargs)


class BoundAgentTests(Fixture):
    def test_humidity_median_minus_current_sign_sparse_values_and_future(self):
        h = self.write([{'bano_humedad7_pct': '50'}, {}, {'bano_humedad7_pct': '60'}, {},
                        {'bano_humedad7_pct': '80'}, {'bano_humedad7_pct': '999'}])
        r = h.desviacion_mediana_movil('bano_humedad7_pct', t0=iso(self.start+300))
        self.assertEqual(r['window_minutes'], 120)
        self.assertEqual((r['median'], r['current_value'], r['median_minus_current']), (60, 80, -20))
        self.assertEqual(r['observed_values'], 3)
        self.assertTrue(r['includes_current_in_median'])
        self.assertEqual(r['current_at'], iso(self.start+240))
        self.assertEqual(r['available_export_minutes'], 5)
        self.assertEqual(r['units']['difference'], 'percentage_points')

    def test_tv_median_uses_24_hours_and_partial_history_without_coverage_gate(self):
        h = self.write([{'salon_consumo14_w': '100', 'bano_humedad7_pct': '100'}]*200+
                       [{'salon_consumo14_w': '1', 'bano_humedad7_pct': '1'}]*200)
        power = h.desviacion_mediana_movil('salon_consumo14_w')
        humidity = h.desviacion_mediana_movil('bano_humedad7_pct')
        self.assertEqual(power['window_minutes'], 1440)
        self.assertEqual(power['observed_values'], 400)
        self.assertEqual(power['median_minus_current'], 49.5)
        self.assertEqual(power['units']['difference'], 'W')
        self.assertEqual(humidity['observed_values'], 120)
        self.assertEqual(humidity['median_minus_current'], 0)

    def test_mobile_median_single_value_and_no_values(self):
        h = self.write([{}, {'bano_humedad7_pct': '80'}])
        r = h.desviacion_mediana_movil('bano_humedad7_pct')
        self.assertEqual((r['observed_values'], r['median_minus_current']), (1, 0))
        r = h.desviacion_mediana_movil('salon_consumo14_w')
        self.assertEqual(r['status'], 'no_observations')
        self.assertIsNone(r['median_minus_current'])

    def test_median_feature_tool_returns_calculated_not_model_arithmetic(self):
        self.write([{'bano_humedad7_pct': '50'}, {'bano_humedad7_pct': '70'}])
        a = self.agent(self.call('humedad', 'mediana'), final='diferencia inventada 999')
        answer = a.respond_record({'question': 'humedad respecto a la mediana',
                                   'timestamp': iso(self.start+120)})
        self.assertIn('-10.000', answer)
        self.assertEqual(a.last_result['facts']['median'], 60)
        self.assertNotIn('999', answer)

    def test_record_time_is_default_reference_and_qwen_selects_yesterday(self):
        self.write([{WATCH+'steps': '2'}]*10+[{WATCH+'steps': '900'}]*10)
        a = self.agent(self.call('pasos', 'total', period='ayer'))
        answer = a.respond_record(
            {'question': 'pasos ayer', 'timestamp': '2026-11-01T00:10:45+01:00'})
        self.assertTrue(answer.startswith('20 pasos'))
        self.assertEqual(a.last_result['reference_source'], 'question_record')
        self.assertEqual(a.last_result['query_context']['t0'], '2026-11-01T00:10:00+01:00')
        self.assertEqual(a.last_result['query_context']['end'], '2026-11-01T00:00:00+01:00')

    def test_model_may_choose_ninety_minutes_over_default_window(self):
        self.write([{WATCH+'steps': '2'}]*120)
        a = self.agent(self.call('pasos', 'total', period='ultimos_minutos', minutes=90))
        answer = a.respond('pasos ultimos 90 minutos', iso(self.start+7200), window_minutes=10)
        self.assertTrue(answer.startswith('180 pasos'))
        self.assertEqual(a.last_result['query_context']['minutes'], 90)

    def test_humidity_difference_uses_real_reference_outside_sparse_window(self):
        h = self.write([{'bano_humedad7_pct': '60.7'}, {}, {}, {}, {'bano_humedad7_pct': '82.64'}])
        r = h.analizar_cambios('bano_humedad7_pct', start=iso(self.start+240), end=iso(self.start+300), t0=iso(self.start+300))
        change = r['latest_change']
        self.assertAlmostEqual(change['difference'], 21.94)
        self.assertEqual(change['elapsed_minutes'], 4)
        self.assertTrue(change['reference_before_window'])
        self.assertEqual(r['units']['difference'], 'percentage_points')
        a = self.agent(self.call('humedad', 'cambios', period='ultimos_minutos', minutes=1))
        answer = a.respond('diferencia humedad', iso(self.start+300))
        self.assertIn('+21.940', answer)

    def test_power_change_and_missing_reference_are_not_tv_labels(self):
        h = self.write([{'salon_consumo14_w': '1'}, {'salon_consumo14_w': '50'}, {'salon_consumo14_w': '1'}])
        r = h.analizar_cambios('salon_consumo14_w', **self.interval(3))
        self.assertEqual(r['latest_change']['difference'], -49)
        self.assertEqual(r['largest_rise']['difference'], 49)
        self.assertEqual(r['units']['difference'], 'W')
        h = self.write([{}, {'bano_humedad7_pct': '80'}])
        r = h.analizar_cambios('bano_humedad7_pct', **self.interval(2))
        self.assertIsNone(r['latest_change'])
        self.assertEqual(r['status'], 'insufficient_reference')

    def agent(self, first, final='99999 pasos inventados'):
        responses = iter([first, {'content': final}])
        return HistoricalAgent(self.directory, transport=lambda _: next(responses), audit=self.directory/'audit.jsonl')

    @staticmethod
    def call(target, operation, **extra):
        return {'content': '', 'tool_calls': [{'function': {'name': 'consultar_metricas_har',
                'arguments': {'target': target, 'operation': operation, 'reason': 'consulta', **extra}}}]}

    def test_calculated_total_cannot_be_replaced_by_model_mean_or_final(self):
        self.write([{WATCH+'steps': '2'}, {WATCH+'steps': '900'}])
        a = self.agent(self.call('pasos', 'total'))
        answer = a.respond('pasos', iso(self.start+60), window_minutes=1)
        self.assertTrue(answer.startswith('2 pasos observados'))
        self.assertNotIn('99999', answer)
        self.assertEqual(a.last_result['facts']['observed_sum'], 2)
        self.assertEqual(a.last_result['query_context']['minutes'], 1)

    def test_model_cannot_override_reference_date(self):
        self.write([{WATCH+'steps': '2'}, {WATCH+'steps': '900'}])
        responses = iter([self.call('pasos', 'total', t0=iso(self.start+120)),
                          self.call('pasos', 'total'), {'content': '999'}])
        a = HistoricalAgent(self.directory, transport=lambda _: next(responses), audit=self.directory/'audit.jsonl')
        answer = a.respond('pasos', iso(self.start+60), window_minutes=1)
        self.assertTrue(answer.startswith('2 pasos'))
        logs = [json.loads(s) for s in (self.directory/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(e['event'] == 'result' and 'error' in e['result'] for e in logs))

    def test_b_prefix_is_not_an_escape_from_missing_tools(self):
        self.write([{}])
        a = HistoricalAgent(self.directory, transport=lambda _: {'content': 'Agente B: 307 minutos.'}, audit=self.directory/'audit.jsonl')
        self.assertIn('No se ha obtenido una consulta válida', a.respond('dormido', iso(self.start+60), window_minutes=1))
        self.assertIsNone(a.last_result)

    def test_bedtime_final_is_read_from_episode_not_generated(self):
        self.write([{WATCH+'bedtime': '1'}]*5+[{WATCH+'bedtime': '0'}]*5)
        a = self.agent(self.call('bedtime', 'levantarse'), final='06:00')
        answer = a.respond('levantado', iso(self.start+600), window_minutes=10)
        self.assertIn('23:55', answer)
        self.assertNotIn('06:00', answer)

    def test_bathroom_uses_room_and_counts_only_supported_minutes(self):
        self.write([{'habitacion': 'WC'}]*4+[{'habitacion': 'Dormitorio'}]*2+[{'habitacion': 'WC'}]*2)
        a = self.agent(self.call('bano', 'intervalos'))
        answer = a.respond('baño', iso(self.start+480), window_minutes=8)
        self.assertIn('75%', answer)
        self.assertEqual(a.last_result['facts']['totals']['rule_support_minutes'], 6)
        self.assertFalse(a.last_result['facts']['episodes'][0]['meets_support_threshold'])

    def test_insufficient_humidity_is_indeterminate_not_false(self):
        self.write([{'bano_humedad7_pct': '80'}]+[{}]*9)
        a = self.agent(self.call('humedad', 'tendencia', trend_minutes=10))
        answer = a.respond('humedad subiendo', iso(self.start+600), window_minutes=10)
        self.assertIn('indeterminada', answer)
        self.assertIn('1/10', answer)

    def test_future_query_is_blocked_before_any_model_request(self):
        self.write([{}])
        def no_model(_):
            raise AssertionError('No debe consultarse el modelo')
        a = HistoricalAgent(self.directory, transport=no_model, audit=self.directory/'audit.jsonl')
        self.assertIn('futuro', a.respond('cuantos pasos dare hoy', iso(self.start+60)))


if __name__ == '__main__':
    unittest.main()
