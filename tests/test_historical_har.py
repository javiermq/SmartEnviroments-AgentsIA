"""Objective tool contracts, independent of semantic activity interpretation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from smart_home_agent.historical_har import History, SIGNALS, WATCH, iso, stamp
from smart_home_agent.historical_agent import HistoricalAgent
from smart_home_agent.har_rules import HarRules


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.start = stamp('2026-10-31T23:50:00+01:00')

    def write(self, observations, start=None):
        start = self.start if start is None else start
        rows = []
        for i, values in enumerate(observations):
            t = start+i*60
            row = dict.fromkeys(['timestamp', 'timestamp_unix', 'watch_id', 'habitacion',
                                 WATCH+'store', WATCH+'revision', *SIGNALS], '')
            row.update(timestamp=iso(t), timestamp_unix=str(t), watch_id='watch_01')
            row.update({WATCH+'store': 'store-a', WATCH+'revision': '1'})
            row.update(values)
            rows.append(row)
        with (self.directory/'2026-10-31.tsv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        return History(self.directory)

    def interval(self, minutes):
        return dict(start=iso(self.start), end=iso(self.start+minutes*60), t0=iso(self.start+minutes*60))


class ObjectiveToolsTests(Fixture):
    def test_missing_zero_steps_hr_power_and_midnight(self):
        h = self.write([{WATCH+'steps': '0', WATCH+'hr_bpm': '60', 'salon_consumo14_w': '100'},
                        {WATCH+'steps': '7', WATCH+'hr_bpm': '80', 'salon_consumo14_w': '200'}, {}]*5)
        result = h.resumir_intervalo(**self.interval(15))
        steps = result['metrics'][WATCH+'steps']
        self.assertEqual((steps['observed_sum'], steps['observed_minutes'], steps['unknown_minutes']), (35, 10, 5))
        self.assertEqual(result['metrics'][WATCH+'hr_bpm']['mean'], 70)
        self.assertEqual(result['metrics'][WATCH+'hr_bpm']['min'], 60)
        self.assertEqual(result['metrics'][WATCH+'hr_bpm']['max'], 80)
        self.assertNotIn('observed_sum', result['metrics']['salon_consumo14_w'])
        self.assertEqual(result['coverage']['rows'], 15)

    def test_reference_and_future(self):
        h = self.write([{WATCH+'steps': '2'}, {WATCH+'steps': '900'}])
        r = h.resumir_intervalo(**self.interval(1))
        self.assertEqual(r['metrics'][WATCH+'steps']['observed_sum'], 2)
        self.assertEqual(h.latest, self.start+120)
        with self.assertRaises(ValueError):
            h.resumir_intervalo(start=iso(self.start), end=iso(self.start+120), t0=iso(self.start+60))

    def test_dst_calendar_vs_rolling(self):
        for day, expected in [('2026-10-25', 1500), ('2026-03-29', 1380)]:
            with self.subTest(day=day):
                first = stamp(day+'T00:00:00'+('+02:00' if '10-25' in day else '+01:00'))
                h = self.write([{}]*expected, start=first)
                self.assertEqual(h.resumir_intervalo(window='ayer')['coverage']['expected_minutes'], expected)
                self.assertEqual(h.resumir_intervalo(window='ultimas_24h')['coverage']['expected_minutes'], 1440)
        with self.assertRaises(ValueError):
            stamp('2026-03-29T02:30:00+01:00')
        self.assertNotEqual(stamp('2026-10-25T02:30:00+02:00'), stamp('2026-10-25T02:30:00+01:00'))

    def test_duplicate_and_revision_require_external_preprocessing(self):
        for revision in ('1', '2'):
            self.write([{}])
            p = self.directory/'2026-10-31.tsv'
            content = p.read_text(encoding='utf-8')
            row = content.splitlines()[1].split('\t')
            columns = content.splitlines()[0].split('\t')
            row[columns.index(WATCH+'revision')] = revision
            p.write_text(content+'\t'.join(row)+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'preprocesamiento externo'):
                History(self.directory)

    def test_snapshot_rejects_partial_and_preserves_source(self):
        h = self.write([{}])
        p = self.directory/'2026-10-31.tsv'
        before = p.read_bytes()
        h.resumir_intervalo(**self.interval(1))
        self.assertEqual(p.read_bytes(), before)
        p.write_bytes(before[:-1])
        with self.assertRaisesRegex(ValueError, 'incompleto'):
            History(self.directory)

    def test_minute_pagination_units_and_allowlist(self):
        h = self.write([{WATCH+'steps': '0'}, {}, {}])
        r = h.consultar_minutos([WATCH+'steps'], limit=1, **self.interval(3))
        self.assertEqual(r['records'][0]['values'][WATCH+'steps'], '0')
        self.assertEqual(r['next_offset'], 1)
        self.assertEqual(r['units'][WATCH+'steps'], 'steps')
        with self.assertRaises(ValueError):
            h.consultar_minutos(['_source'])

    def test_gap_two_three_four_and_observed_nonmatches(self):
        for gap in (2, 3, 4):
            h = self.write([{'habitacion': 'Cocina'}]+[{'habitacion': 'WC'}]*gap+[{'habitacion': 'Cocina'}])
            r = h.buscar_episodios([{'column': 'habitacion', 'op': 'eq', 'value': 'Cocina'}],
                                  merge_gap=3, **self.interval(gap+2))
            self.assertEqual(r['episode_count'], 1 if gap <= 3 else 2)
            if gap <= 3:
                e = r['episodes'][0]
                self.assertEqual(e['matching_minutes'], 2)
                self.assertEqual(e['nonmatching_minutes'], gap)
                self.assertEqual(e['unknown_minutes'], 0)
                self.assertEqual(e['gaps'][0]['kind'], 'nonmatch')

    def test_default_no_merge_and_unknown_gap(self):
        h = self.write([{'habitacion': 'Cocina'}, {}, {'habitacion': 'Cocina'}])
        c = [{'column': 'habitacion', 'op': 'eq', 'value': 'Cocina'}]
        self.assertEqual(h.buscar_episodios(c, **self.interval(3))['episode_count'], 2)
        e = h.buscar_episodios(c, merge_gap=3, **self.interval(3))['episodes'][0]
        self.assertEqual((e['matching_minutes'], e['unknown_minutes']), (2, 1))

    def test_chained_gaps_are_bounded(self):
        h = self.write(([{WATCH+'steps': '0'}, {}, {}])*10)
        r = h.buscar_episodios([{'column': WATCH+'steps', 'op': 'eq', 'value': 0}],
                              merge_gap=3, max_total_gap=3, **self.interval(30))
        self.assertGreater(r['episode_count'], 1)
        self.assertTrue(all(e['unknown_minutes'] <= 3 for e in r['episodes']))

    def test_no_interpretation_of_humidity_bedtime_network_or_missing(self):
        h = self.write([{'bano_humedad7_pct': '85', WATCH+'bedtime': '1', WATCH+'network_evidence': '3'},
                        {'bano_humedad7_pct': '85', WATCH+'network_evidence': '2'}, {}])
        r = h.buscar_episodios([{'column': 'bano_humedad7_pct', 'op': 'gt', 'value': 80}], **self.interval(3))
        self.assertEqual(r['matching_minutes'], 2)
        self.assertEqual(r['unknown_minutes'], 1)
        self.assertNotIn('activity', r['episodes'][0])
        self.assertNotIn('outside_duration_minutes', r)
        self.assertNotIn('ducha', json.dumps(r))
        self.assertIn('Bedtime: modo del SO', ' '.join(r['warnings']))

    def test_three_valued_logic(self):
        h = self.write([{WATCH+'steps': '5'}, {WATCH+'steps': '0'}, {}])
        c = [{'column': WATCH+'steps', 'op': 'eq', 'value': 0},
             {'column': WATCH+'hr_bpm', 'op': 'lt', 'value': 60}]
        all_r = h.buscar_episodios(c, **self.interval(3))
        any_r = h.buscar_episodios(c, mode='any', **self.interval(3))
        self.assertEqual((all_r['nonmatching_minutes'], all_r['unknown_minutes']), (1, 2))
        self.assertEqual((any_r['matching_minutes'], any_r['unknown_minutes']), (1, 2))


class AgentTests(Fixture):
    def test_agent_a_one_measure_and_one_activity_with_b_handoff(self):
        self.write([{'habitacion': 'WC'}])
        h = History(self.directory)
        summary = h.resumir_intervalo(measure='habitacion', **self.interval(1))
        self.assertEqual(summary['metrics'], {})
        summary = h.resumir_intervalo(measure=WATCH+'steps', **self.interval(1))
        self.assertEqual(list(summary['metrics']), [WATCH+'steps'])
        self.assertNotIn('rooms_observed_minutes', summary)
        responses = iter([
            {'tool_calls': [{'function': {'name': 'resumir_intervalo', 'arguments': {'measure': 'habitacion', 'reason': 'contexto', **self.interval(1)}}}]},
            {'tool_calls': [{'function': {'name': 'inferir_actividades', 'arguments': {'activity': 'all', 'reason': 'todo', **self.interval(1)}}}]},
            {'tool_calls': [{'function': {'name': 'inferir_actividades', 'arguments': {'activity': 'bano', 'reason': 'baño', **self.interval(1)}}}]},
            {'tool_calls': [{'function': {'name': 'inferir_actividades', 'arguments': {'activity': 'bedtime', 'reason': 'combinar', **self.interval(1)}}}]},
            {'content': 'Agente B: combinar baño y Bedtime requiere consultas separadas.'},
        ])
        audit = self.directory/'audit.jsonl'
        a = HistoricalAgent(self.directory, transport=lambda _: next(responses), audit=audit)
        a.respond('comparar')
        records = [json.loads(s) for s in audit.read_text(encoding='utf-8').splitlines()]
        results = [e['result'] for e in records if e['event'] == 'result']
        self.assertIn('error', results[1])
        self.assertEqual(results[2]['episodes'][0]['activity'], 'bano')
        self.assertIn('error', results[3])
        b = HistoricalAgent(self.directory, transport=lambda _: {'content': 'Agente B: consultar Bedtime y descanso por separado.'}, audit=self.directory/'b.jsonl')
        self.assertTrue(b.respond('sumar sueño y descanso').startswith('Agente B:'))

    def test_real_dispatch_cutoff_and_audit(self):
        self.write([{WATCH+'steps': '2'}, {WATCH+'steps': '900'}])
        calls = [
            {'name': 'consultar_minutos', 'arguments': {'columns': [WATCH+'steps'], 'reason': 'detalle'}},
            {'name': 'resumir_intervalo', 'arguments': {'reason': 'suma', **self.interval(1)}},
            {'name': 'resumir_intervalo', 'arguments': {'reason': 'futuro', 't0': iso(self.start+120)}},
            {'name': 'resumir_intervalo', 'arguments': {'reason': 'evadir', 't0': None}},
            {'name': '__getattribute__', 'arguments': {'reason': 'ruta'}},
        ]
        def transport(payload):
            if calls:
                return {'content': '', 'tool_calls': [{'function': calls.pop(0)}]}
            return {'content': '2 pasos observados.'}
        audit = self.directory/'audit.jsonl'
        answer = HistoricalAgent(self.directory, transport=transport, audit=audit).respond('pasos', iso(self.start+60))
        self.assertEqual(answer, '2 pasos observados.')
        log = [json.loads(s) for s in audit.read_text(encoding='utf-8').splitlines()]
        results = [e['result'] for e in log if e['event'] == 'result']
        self.assertIn('error', results[0])
        self.assertEqual(results[1]['metrics'][WATCH+'steps']['observed_sum'], 2)
        self.assertTrue(all('error' in e for e in results[2:]))
        self.assertEqual(log[-1]['event'], 'final')

    def test_budget_counts_calls_not_rounds(self):
        self.write([{}])
        def transport(payload):
            return {'content': '', 'tool_calls': [{'function': {'name': 'resumir_intervalo',
                    'arguments': {'reason': 'resumen', **self.interval(1)}}}]*4}
        audit = self.directory/'audit.jsonl'
        answer = HistoricalAgent(self.directory, transport=transport, audit=audit, max_calls=2).respond('resumen')
        self.assertIn('límite', answer)
        self.assertEqual(sum(json.loads(s)['event'] == 'call' for s in audit.read_text(encoding='utf-8').splitlines()), 2)


class HarRuleTests(Fixture):
    def test_exercise_interval_has_observed_boundaries_and_recovery(self):
        low = {WATCH+'steps': '10', WATCH+'hr_bpm': '70'}
        high = {WATCH+'steps': '80', WATCH+'hr_bpm': '120'}
        h = self.write([low]*5+[high]*10+[low]*5)
        r = h.inferir_actividades(activity='ejercicio', **self.interval(20))
        self.assertEqual(r['episode_count'], 1)
        e = r['episodes'][0]
        self.assertEqual((e['start'], e['end'], e['evidence_minutes']),
                         (iso(self.start+300), iso(self.start+900), 10))
        self.assertFalse(e['boundary_censored'])
        trend = h.analizar_tendencia(WATCH+'steps', 5, t0=iso(self.start+1200))
        self.assertEqual(trend['change_of_window_means'], -70)
        # Recovery is supporting context; the rule interval has not been extended.
        self.assertEqual(h.inferir_actividades(activity='ejercicio', **self.interval(20))['episodes'][0]['end'], e['end'])

    def test_exercise_high_stable_levels_do_not_require_rise(self):
        h = self.write([{WATCH+'steps': '80', WATCH+'hr_bpm': '120'}]*6)
        r = h.inferir_actividades(activity='ejercicio', **self.interval(6))
        self.assertEqual(r['totals']['ejercicio']['rule_support_minutes'], 6)
        self.assertEqual(r['episodes'][0]['evidence_minutes'], 6)
        self.assertEqual(h.inferir_actividades(activity='ejercicio', **self.interval(4))['episode_count'], 0)

    def test_exercise_requires_both_levels(self):
        for rows in (
            [{WATCH+'steps': '80'}]*6,
            [{WATCH+'steps': '10', WATCH+'hr_bpm': '130'}]*6,
            [{WATCH+'steps': '80', WATCH+'hr_bpm': '70'}]*6,
        ):
            h = self.write(rows)
            self.assertEqual(h.inferir_actividades(activity='ejercicio', **self.interval(len(rows)))['episode_count'], 0)

    def test_cooking_bathroom_interruption_preserves_duration(self):
        cook = {'habitacion': 'Cocina', 'cocina_vasos13': '1'}
        h = self.write([cook]*8+[{'habitacion': 'WC'}]*2+[cook]*10)
        r = h.inferir_actividades(**self.interval(20))
        e = next(e for e in r['episodes'] if e['activity'] == 'cocinar')
        self.assertEqual((e['duration_minutes'], e['evidence_minutes'], e['interruption_minutes']), (20, 18, 2))
        self.assertEqual(e['gaps'][0]['rooms_observed_minutes'], {'WC': 2})
        self.assertFalse(e['evidence_sample_is_complete'])

    def test_shower_delayed_rise_nested_in_bath_no_future(self):
        h = self.write([{'habitacion': 'Salón', 'bano_humedad7_pct': '50'},
                        {'habitacion': 'WC'}, {'habitacion': 'WC'},
                        {'habitacion': 'Salón', 'bano_humedad7_pct': '65'}])
        self.assertEqual(h.inferir_actividades(activity='ducha', **self.interval(3))['episode_count'], 0)
        r = h.inferir_actividades(**self.interval(4))
        shower = next(e for e in r['episodes'] if e['activity'] == 'ducha')
        bath = next(e for e in r['episodes'] if e['activity'] == 'bano')
        self.assertLessEqual(bath['start'], shower['start'])
        self.assertGreaterEqual(bath['end'], shower['end'])
        self.assertEqual(shower['evidence_sample'][0]['rise_pct_points'], 15)
        self.assertEqual(shower['evidence_sample'][0]['humidity_observed_at'], iso(self.start+180))

    def test_humidity_high_without_rise_or_without_reference(self):
        for observations in (
            [{'bano_humedad7_pct': '85'}, {'habitacion': 'WC', 'bano_humedad7_pct': '85'}],
            [{}, {'habitacion': 'WC', 'bano_humedad7_pct': '85'}],
        ):
            h = self.write(observations)
            self.assertEqual(h.inferir_actividades(activity='ducha', **self.interval(2))['episode_count'], 0)

    def test_shower_delay_and_reference_age_are_bounded(self):
        h = self.write([{'bano_humedad7_pct': '50'}, {'habitacion': 'WC'}]+[{}]*11+[{'bano_humedad7_pct': '80'}])
        self.assertEqual(h.inferir_actividades(activity='ducha', **self.interval(14))['episode_count'], 0)
        h = self.write([{'bano_humedad7_pct': '50'}]+[{}]*61+[{'habitacion': 'WC', 'bano_humedad7_pct': '80'}])
        self.assertEqual(h.inferir_actividades(activity='ducha', **self.interval(63))['episode_count'], 0)

    def test_rest_requires_both_observed_signals_and_sustained_duration(self):
        for rows in ([{WATCH+'steps': '0'}]*12, [{WATCH+'hr_bpm': '60'}]*12):
            h = self.write(rows)
            self.assertEqual(h.inferir_actividades(activity='descansar', **self.interval(12))['episode_count'], 0)
        h = self.write([{WATCH+'steps': '0', WATCH+'hr_bpm': '60'}]*12)
        self.assertEqual(h.inferir_actividades(activity='descansar', **self.interval(9))['episode_count'], 0)
        self.assertEqual(h.inferir_actividades(activity='descansar', **self.interval(12))['totals']['descansar']['rule_support_minutes'], 12)

    def test_bedtime_short_interruption_is_retained_with_support(self):
        rows = [{WATCH+'bedtime': '1'}, {WATCH+'bedtime': '0'}, {WATCH+'bedtime': '1'}]
        h = self.write(rows)
        r = h.inferir_actividades(activity='bedtime', **self.interval(3))
        self.assertEqual(r['episode_count'], 1)
        e = r['episodes'][0]
        self.assertEqual(e['interruption_minutes'], 1)
        self.assertAlmostEqual(e['support_fraction'], 2/3)
        self.assertFalse(e['meets_support_threshold'])

    def test_common_support_fraction_and_five_minute_merging(self):
        for name, positive, negative in (
            ('bano', {'habitacion': 'WC'}, {'habitacion': 'Dormitorio'}),
            ('bedtime', {WATCH+'bedtime': '1'}, {WATCH+'bedtime': '0'}),
            ('cocinar', {'habitacion': 'Cocina', 'cocina_vasos13': '1'}, {'habitacion': 'WC'}),
            ('ejercicio', {WATCH+'steps': '80', WATCH+'hr_bpm': '120'}, {WATCH+'steps': '0', WATCH+'hr_bpm': '70'}),
            ('descansar', {WATCH+'steps': '0', WATCH+'hr_bpm': '60'}, {WATCH+'steps': '80', WATCH+'hr_bpm': '120'}),
        ):
            h = self.write([positive]*4+[negative]+[positive]*5)
            r = h.inferir_actividades(activity=name, **self.interval(10))
            self.assertEqual(r['episode_count'], 1, name)
            self.assertEqual(r['episodes'][0]['support_fraction'], .9)
            self.assertTrue(r['episodes'][0]['meets_support_threshold'])
        h = self.write([{'habitacion': 'WC'}]*4+[{'habitacion': 'Dormitorio'}]*2+[{'habitacion': 'WC'}]*2)
        e = h.inferir_actividades(activity='bano', **self.interval(8))['episodes'][0]
        self.assertEqual((e['duration_minutes'], e['evidence_minutes']), (8, 6))
        self.assertEqual(e['support_fraction'], .75)
        for gap in (5, 6):
            h = self.write([{'habitacion': 'WC'}]+[{}]*gap+[{'habitacion': 'WC'}])
            self.assertEqual(h.inferir_actividades(activity='bano', **self.interval(gap+2))['episode_count'], 1 if gap == 5 else 2)

    def test_exercise_missing_minute_does_not_count_as_support(self):
        hi = {WATCH+'steps': '80', WATCH+'hr_bpm': '130'}
        h = self.write([hi]*3+[{}]+[hi]*3)
        e = h.inferir_actividades(activity='ejercicio', **self.interval(7))['episodes'][0]
        self.assertEqual((e['evidence_minutes'], e['unknown_minutes']), (6, 1))
        self.assertTrue(e['meets_support_threshold'])

    def test_exact_eighty_percent_in_minimum_window_includes_known_margin(self):
        hi = {WATCH+'steps': '80', WATCH+'hr_bpm': '130'}
        low = {WATCH+'steps': '0', WATCH+'hr_bpm': '60'}
        h = self.write([hi]*4+[low])
        e = h.inferir_actividades(activity='ejercicio', **self.interval(5))['episodes'][0]
        self.assertEqual((e['duration_minutes'], e['evidence_minutes'], e['interruption_minutes']), (5, 4, 1))
        self.assertEqual(e['support_fraction'], .8)
        self.assertTrue(e['meets_support_threshold'])
        h = self.write([low]*8+[hi]*2)
        e = h.inferir_actividades(activity='descansar', **self.interval(10))['episodes'][0]
        self.assertEqual((e['duration_minutes'], e['evidence_minutes']), (10, 8))
        self.assertEqual(e['support_fraction'], .8)

    def test_door_only_network_and_missing_never_infer_absence(self):
        h = self.write([{WATCH+'network_evidence': '3'}, {WATCH+'network_evidence': '2'}, {},
                        {'entrada_binario7': '1'}])
        r = h.inferir_actividades(activity='salir', **self.interval(4))
        self.assertEqual(r['episode_count'], 1)
        self.assertEqual(r['episodes'][0]['start'], iso(self.start+180))
        self.assertIsNone(r['outside_duration_minutes'])
        self.assertEqual(r['totals']['salir']['rule_support_minutes'], 1)

    def test_config_changes_are_explicit_and_do_not_change_metrics(self):
        h = self.write([{'habitacion': 'Cocina', 'cocina_frigo_luz24_lx': '15'}])
        before = h.resumir_intervalo(**self.interval(1))
        self.assertEqual(h.inferir_actividades(activity='cocinar', **self.interval(1))['episode_count'], 1)
        h.har_rules = HarRules(fridge_lux=20)
        self.assertEqual(h.inferir_actividades(activity='cocinar', **self.interval(1))['episode_count'], 0)
        self.assertEqual(h.resumir_intervalo(**self.interval(1)), before)
        with self.assertRaises(ValueError):
            HarRules(rest_min_minutes=0)

    def test_qwen_dispatches_separate_rule_tool(self):
        self.write([{'habitacion': 'WC'}])
        responses = iter([
            {'tool_calls': [{'function': {'name': 'resumir_intervalo', 'arguments': {'reason': 'contexto', **self.interval(1)}}}]},
            {'tool_calls': [{'function': {'name': 'inferir_actividades', 'arguments': {'activity': 'bano', 'reason': 'regla', **self.interval(1)}}}]},
            {'content': 'Baño estimado según regla.'},
        ])
        audit = self.directory/'audit.jsonl'
        a = HistoricalAgent(self.directory, transport=lambda _: next(responses), audit=audit)
        self.assertEqual(a.respond('baño'), 'Baño estimado según regla.')
        records = [json.loads(s) for s in audit.read_text(encoding='utf-8').splitlines()]
        result = next(r['result'] for r in records if r['event'] == 'result' and r['name'] == 'inferir_actividades')
        self.assertEqual(result['episodes'][0]['activity'], 'bano')


class TrendTests(Fixture):
    def test_linear_growth_and_mean_change(self):
        h = self.write([{WATCH+'hr_bpm': str(40+2*i)} for i in range(20)])
        r = h.analizar_tendencia(WATCH+'hr_bpm', 10, t0=iso(self.start+1200))
        self.assertEqual(r['current']['linear_slope_per_minute'], 2)
        self.assertEqual(r['current']['change_first_last'], 18)
        self.assertEqual(r['current']['observed_span_minutes'], 9)
        self.assertEqual(r['change_of_window_means'], 20)
        self.assertEqual(r['units']['slope'], 'bpm/min')

    def test_sparse_samples_use_elapsed_times_without_interpolation(self):
        h = self.write([{WATCH+'hr_bpm': '60'}, {}, {}, {}, {WATCH+'hr_bpm': '80'}])
        r = h.analizar_tendencia(WATCH+'hr_bpm', 5, t0=iso(self.start+300), min_coverage=.4)
        self.assertEqual(r['current']['linear_slope_per_minute'], 5)
        self.assertEqual(r['current']['observed_minutes'], 2)
        self.assertEqual(r['current']['unknown_minutes'], 3)
        self.assertIsNone(r['change_of_window_means'])
        low = h.analizar_tendencia(WATCH+'hr_bpm', 5, t0=iso(self.start+300))
        self.assertIsNone(low['current']['linear_slope_per_minute'])

    def test_constant_cadence_zero_mean_and_future_cutoff(self):
        h = self.write([{WATCH+'steps': '0'}]*5+[{WATCH+'steps': '80'}]*5+[{WATCH+'steps': '900'}])
        r = h.analizar_tendencia(WATCH+'steps', 5, t0=iso(self.start+600))
        self.assertEqual(r['current']['linear_slope_per_minute'], 0)
        self.assertEqual(r['change_of_window_means'], 80)
        self.assertEqual(r['units']['slope'], 'steps/min/min')

    def test_decline_dst_actual_elapsed_minutes_and_invalid_parameters(self):
        start = stamp('2026-10-25T02:58:00+02:00')
        h = self.write([{WATCH+'hr_bpm': str(100-2*i)} for i in range(6)], start=start)
        r = h.analizar_tendencia(WATCH+'hr_bpm', 3)
        self.assertEqual(r['current']['linear_slope_per_minute'], -2)
        self.assertEqual(r['change_of_window_means'], -6)
        for kwargs in ({'column': WATCH+'bedtime'}, {'column': WATCH+'hr_bpm', 'window_minutes': 1},
                       {'column': WATCH+'hr_bpm', 'min_coverage': 0}):
            with self.assertRaises(ValueError):
                h.analizar_tendencia(**kwargs)

    def test_trend_tool_executed_by_agent(self):
        self.write([{WATCH+'hr_bpm': str(70+i)} for i in range(10)])
        responses = iter([
            {'tool_calls': [{'function': {'name': 'resumir_intervalo', 'arguments': {'reason': 'contexto', **self.interval(10)}}}]},
            {'tool_calls': [{'function': {'name': 'analizar_tendencia', 'arguments': {'column': WATCH+'hr_bpm', 'window_minutes': 5, 'reason': 'cambio de pulso'}}}]},
            {'content': 'Pendiente observada de 1 bpm/min.'},
        ])
        audit = self.directory/'audit.jsonl'
        a = HistoricalAgent(self.directory, transport=lambda _: next(responses), audit=audit)
        a.respond('tendencia', iso(self.start+600))
        records = [json.loads(s) for s in audit.read_text(encoding='utf-8').splitlines()]
        r = next(e['result'] for e in records if e['event'] == 'result' and e['name'] == 'analizar_tendencia')
        self.assertEqual(r['current']['linear_slope_per_minute'], 1)


class RealExportTests(unittest.TestCase):
    def test_raw_calculation_matches_tools(self):
        directory = Path(__file__).resolve().parents[1]/'dataset_home_real/mariola'
        if not directory.is_dir():
            self.skipTest('Export local privado no disponible')
        h = History(directory)
        a = stamp('2026-10-31T00:00:00+01:00')
        b = stamp('2026-11-01T00:00:00+01:00')
        rows = []
        for p in directory.glob('*.tsv'):
            with p.open(encoding='utf-8-sig') as f:
                rows.extend(r for r in csv.DictReader(f, delimiter='\t') if a <= int(r['timestamp_unix']) < b)
        r = h.resumir_intervalo(start=iso(a), end=iso(b), t0=iso(b))
        values = [float(row[WATCH+'steps']) for row in rows if row[WATCH+'steps']]
        self.assertEqual(r['metrics'][WATCH+'steps']['observed_sum'], sum(values))
        self.assertEqual(r['metrics'][WATCH+'steps']['observed_minutes'], len(values))
        hr = [float(row[WATCH+'hr_bpm']) for row in rows if row[WATCH+'hr_bpm']]
        self.assertAlmostEqual(r['metrics'][WATCH+'hr_bpm']['mean'], sum(hr)/len(hr))
        self.assertEqual(len(rows), r['coverage']['rows'])


if __name__ == '__main__':
    unittest.main()
