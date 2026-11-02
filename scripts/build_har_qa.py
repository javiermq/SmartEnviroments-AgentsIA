"""Build 100 scoped, rule-derived QA references from the private local export.

No LLM calls, human labels, data repair, or device access. Outputs stay in temp_data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from html import escape
import json
from pathlib import Path

from smart_home_agent.har_rules import HarRules, RULE_DESCRIPTIONS
from smart_home_agent.historical_har import History, WATCH, iso, stamp


def time_label(t: int | str) -> str:
    return (iso(t) if isinstance(t, int) else t)[11:16]


def ranges(episodes: list[dict]) -> str:
    return '; '.join(f"{time_label(e['start'])}–{time_label(e['end'])}" for e in episodes)


def build(data: Path, output: Path) -> list[dict]:
    rules = HarRules()
    h = History(data, rules)
    days = sorted({iso(t)[:10] for t in h.rows})
    if len(days) != 3:
        raise ValueError('Este banco requiere exactamente tres días de entrada')
    cases = []
    for day in days:
        day_start = stamp(day+'T00:00:00+01:00')
        # Keep "hoy" within its local day. Completed days stop one minute before midnight.
        ref = min(stamp(day+'T23:59:00+01:00'), max(t+60 for t in h.rows if iso(t)[:10] == day))
        full = dict(start=iso(day_start), end=iso(ref), t0=iso(ref))
        summary = h.resumir_intervalo(**full)

        def episodes_for(name, **interval):
            params = {'activity': name, **(interval or full), 'limit': 100}
            r = h.inferir_actividades(**params)
            es = list(r['episodes'])
            while r['next_offset'] is not None:
                params['offset'] = r['next_offset']
                r = h.inferir_actividades(**params)
                es.extend(r['episodes'])
            return r, es

        activities = {n: episodes_for(n) for n in RULE_DESCRIPTIONS}

        def add(topic, question, answer, facts, tools=None, t0=None, interval=None):
            case_ref = t0 or ref
            case_interval = interval or (day_start, case_ref)
            cases.append({
                'id': f'QA{len(cases)+1:03d}', 'day': day, 'topic': topic,
                'question': question, 't0': iso(case_ref), 'timezone': 'Europe/Madrid',
                'interval': {'start': iso(case_interval[0]), 'end': iso(case_interval[1]), 'closure': '[start,end)'},
                'reference_answer': answer, 'expected_facts': facts,
                'suggested_tools': tools or ['resumir_intervalo', 'inferir_actividades'],
                'reference_kind': 'cálculo determinista y reglas operativas; no anotación humana',
            })

        def total(name):
            return activities[name][0]['totals'][name]['rule_support_minutes']

        def activity_facts(name):
            r, es = activities[name]
            return {'activity': name, 'support_minutes': total(name),
                    'unknown_rule_minutes': r['totals'][name]['unknown_minutes'],
                    'episodes': [{k: e[k] for k in ('start', 'end', 'evidence_minutes', 'duration_minutes',
                                                   'unknown_minutes', 'interruption_minutes')} for e in es]}

        shower = activities['ducha'][1]
        bath = activities['bano'][1]
        bedtime = activities['bedtime'][1]
        exercise = activities['ejercicio'][1]
        kitchen = activities['cocinar'][1]
        rest = activities['descansar'][1]
        doors = activities['salir'][1]

        add('ducha', '¿Cuándo me he duchado hoy?',
            (f'Hay evidencia compatible con ducha en {ranges(shower)}: baño observado y subida de humedad respecto a una referencia previa. '
             'Son minutos de soporte de la regla, no un horario de ducha confirmado.' if shower else
             'Hasta esta hora no hay soporte suficiente para la regla de ducha. No puedo concluir que no te hayas duchado.'),
            activity_facts('ducha'))
        shower_evidence = [v for e in shower for v in e['evidence_sample']]
        humidity_explanation = '; '.join(
            f"humedad de {v['reference_pct']:g}% a las {time_label(v['reference_at'])} a {v['humidity_pct']:g}% a las {time_label(v['humidity_observed_at'])} "
            f"(+{v['rise_pct_points']:g} puntos porcentuales), con baño observado a las {time_label(v['bath_observed_at'])}"
            for v in shower_evidence)
        add('ducha', '¿Qué datos apoyan que me haya duchado hoy?',
            (humidity_explanation+'. Es una hipótesis por regla; el export reducido no permite descartar lecturas retenidas.' if shower_evidence else
             'No hay una combinación suficiente de baño observado, referencia de humedad y subida que respalde la regla. La humedad alta aislada no basta.'),
            {'supporting_samples': shower_evidence})
        add('ducha', '¿Cuántos minutos he estado en la ducha hoy?',
            f'No puedo medir la duración real de la ducha. La regla tiene {total("ducha")} minutos de soporte; ese número no equivale a minutos duchándote.',
            {'actual_shower_duration_minutes': None, 'rule_support_minutes': total('ducha')})

        add('bano', '¿Cuándo he estado en el baño hoy?',
            (f'La habitación se estima como WC/Baño en estos tramos: {ranges(bath)}. Hay {total("bano")} minutos observados; no es una actividad validada.' if bath else
             'No hay minutos con habitación estimada WC/Baño en la ventana. Los huecos impiden afirmar que no hayas ido.'), activity_facts('bano'))
        add('bano', '¿Cuánto tiempo he pasado en el baño hoy?',
            f'Hay {total("bano")} minutos observados con habitación WC/Baño. Es tiempo respaldado por localización estimada; no un total real garantizado ante los huecos.',
            activity_facts('bano'))
        add('bano', '¿Cuál fue mi última visita observada al baño hoy?',
            (f'El último tramo con soporte termina en {ranges(bath[-1:])}. Tiene {bath[-1]["evidence_minutes"]} minutos observados; no permite certificar que fuera tu última visita real.' if bath else
             'No puedo identificar una visita: falta localización compatible con baño en la ventana.'),
            {'last_rule_episode': activity_facts('bano')['episodes'][-1:]})

        bt = summary['metrics'][WATCH+'bedtime']
        add('bedtime', '¿Cuántos minutos he dormido hoy?',
            f'No puedo medir sueño real. Se observó Bedtime activado en {bt.get("active_minutes", 0)} minutos; el modo tiene datos en {bt["observed_minutes"]} '
            f'y falta en {bt["unknown_minutes"]} minutos. Bedtime no demuestra sueño fisiológico.',
            {'actual_sleep_minutes': None, 'bedtime': bt})
        add('bedtime', '¿Entre qué horas estuve en modo Bedtime hoy?',
            (f'Se observó Bedtime activado en {ranges(bedtime)}. Son {total("bedtime")} minutos de soporte del modo, no sueño confirmado.' if bedtime else
             'No hay activaciones de Bedtime observadas en la ventana; los huecos no prueban ausencia de sueño.'), activity_facts('bedtime'))
        add('bedtime', '¿A qué hora me he levantado hoy?',
            (f'No puedo establecer la hora real de levantarte. El último tramo observado de Bedtime termina a las {time_label(bedtime[-1]["end"])}; '
             'ese límite solo describe el modo del dispositivo, no que te hayas despertado o salido de la cama.' if bedtime else
             'No puedo establecer la hora de levantarte con estas señales y sin una referencia de despertar validada.'),
            {'actual_wakeup_time': None, 'last_bedtime_episode_end': bedtime[-1]['end'] if bedtime else None})
        morning_end = min(ref, stamp(day+'T12:00:00+01:00'))
        night = h.resumir_intervalo(start=iso(day_start), end=iso(morning_end), t0=iso(ref))
        night_bt = night['metrics'][WATCH+'bedtime']
        add('bedtime', '¿Cuánto he dormido desde medianoche hasta las doce?',
            f'Entre las 00:00 y {time_label(morning_end)} hay {night_bt.get("active_minutes", 0)} minutos observados de Bedtime=1, '
            f'con {night_bt["unknown_minutes"]} minutos sin valor de ese modo. No es una medición de sueño.',
            {'actual_sleep_minutes': None, 'bedtime': night_bt}, interval=(day_start, morning_end))

        add('ejercicio', '¿A qué hora hice ejercicio hoy?',
            (f'La regla identifica actividad física compatible en {ranges(exercise)}. Requiere al menos {rules.exercise_steps_min:g} pasos/min y '
             f'{rules.exercise_hr_min_bpm:g} bpm durante {rules.exercise_min_minutes} minutos consecutivos. No confirma que fuera deporte.' if exercise else
             'No hay episodios que cumplan la regla de pasos y HR altos sostenidos hasta esta hora. No descarta ejercicio sin pasos o durante huecos.'), activity_facts('ejercicio'))
        add('ejercicio', '¿Cuántos minutos de ejercicio tengo hoy?',
            f'Hay {total("ejercicio")} minutos de soporte de la regla de actividad física compatible. No son minutos de deporte confirmados; los umbrales son provisionales.',
            activity_facts('ejercicio'))
        longest_exercise = max(exercise, key=lambda e: e['evidence_minutes']) if exercise else None
        add('ejercicio', '¿Cuál fue mi intervalo más largo de ejercicio hoy?',
            (f'El mayor soporte de la regla está en {ranges([longest_exercise])}: {longest_exercise["evidence_minutes"]} minutos respaldados '
             f'dentro de {longest_exercise["duration_minutes"]} minutos globales. No se atribuye actividad a los huecos.' if longest_exercise else
             'No puedo seleccionar un intervalo: no hay episodios que cumplan la regla. Esto no prueba ausencia de ejercicio.'),
            {'longest_rule_episode': longest_exercise})

        rows = [(t, r) for t, r in sorted(h.rows.items()) if day_start <= t < ref]
        hr_points = [(t, float(r[WATCH+'hr_bpm'])) for t, r in rows if r[WATCH+'hr_bpm']]
        peak_hr_t, peak_hr = max(hr_points, key=lambda p: p[1]) if hr_points else (None, None)
        trend_ref = min(ref, stamp(exercise[0]['end'])+300) if exercise else min(ref, (peak_hr_t or day_start)+60)
        for column, topic, question in [
            (WATCH+'hr_bpm', 'tendencia_hr', f'¿Cómo cambiaba mi pulso en los diez minutos anteriores a las {time_label(trend_ref)}?'),
            (WATCH+'steps', 'tendencia_pasos', f'¿Cómo cambiaban mis pasos por minuto en los diez minutos anteriores a las {time_label(trend_ref)}?'),
        ]:
            trend = h.analizar_tendencia(column, 10, t0=iso(trend_ref))
            current = trend['current']
            answer = (f'En {time_label(current["start"])}–{time_label(current["end"])}, la pendiente estimada es '
                      f'{current["linear_slope_per_minute"]:.3f} {trend["units"]["slope"]}, con {current["observed_minutes"]}/10 minutos observados. '
                      'Describe el cambio de señal, no identifica por sí solo ejercicio.' if current['status'] == 'ok' else
                      f'No puedo estimar la tendencia con el requisito del 80%: hay {current["observed_minutes"]}/10 minutos observados. No relleno los huecos.')
            add(topic, question, answer, trend, ['resumir_intervalo', 'analizar_tendencia'], t0=trend_ref, interval=(trend_ref-600, trend_ref))

        add('cocinar', '¿Cuándo he estado cocinando hoy?',
            (f'Hay uso de cocina compatible con la regla en {ranges(kitchen)}. Coinciden localización Cocina y alguna señal de frigo/vasos/platos; '
             'no demuestra preparación de alimentos.' if kitchen else
             'No hay coincidencias suficientes entre localización Cocina y sus sensores para respaldar la regla. No puedo afirmar que no cocinaras.'), activity_facts('cocinar'))
        add('cocinar', '¿Cuánto tiempo he cocinado hoy?',
            f'La regla tiene {total("cocinar")} minutos de soporte. Los intervalos globales suman {sum(e["duration_minutes"] for e in kitchen)} minutos, '
            'pero incluyen posibles huecos/interrupciones. Ninguna cifra mide la duración real de preparar comida.', activity_facts('cocinar'))
        longest_kitchen = max(kitchen, key=lambda e: e['duration_minutes']) if kitchen else None
        add('cocinar', '¿Cuál fue el tramo más largo compatible con cocinar y tuvo interrupciones?',
            (f'El tramo es {ranges([longest_kitchen])}: {longest_kitchen["duration_minutes"]} minutos globales, '
             f'{longest_kitchen["evidence_minutes"]} con soporte, {longest_kitchen["unknown_minutes"]} desconocidos y '
             f'{longest_kitchen["interruption_minutes"]} observados sin cumplir la regla.' if longest_kitchen else
             'No hay un tramo con soporte suficiente para esta regla.'), {'longest_rule_episode': longest_kitchen})

        add('descansar', '¿Cuándo he descansado hoy?',
            (f'La regla encuentra reposo compatible en {ranges(rest)}: pasos observados=0 y HR<={rules.rest_hr_max_bpm:g} bpm '
             f'durante al menos {rules.rest_min_minutes} minutos consecutivos. No confirma descanso subjetivo ni sueño.' if rest else
             'No hay episodios con ambas señales y duración suficientes para la regla de descanso. Los datos ausentes no cuentan como inactividad.'), activity_facts('descansar'))
        add('descansar', '¿Cuánto tiempo he estado descansando hoy?',
            f'Hay {total("descansar")} minutos respaldados por la regla de descanso. No sumo huecos como reposo ni interpreto esos minutos como sueño.',
            activity_facts('descansar'))
        overlap = set()
        for t, row in rows:
            if row[WATCH+'bedtime'] != '1':
                continue
            if any(stamp(e['start']) <= t < stamp(e['end']) for e in rest):
                # Restrict to actual rule-support minutes, not bridged unknown minutes.
                if row[WATCH+'steps'] and float(row[WATCH+'steps']) == 0 and row[WATCH+'hr_bpm'] and float(row[WATCH+'hr_bpm']) <= rules.rest_hr_max_bpm:
                    overlap.add(t)
        add('descansar', '¿Puedes sumar lo que he dormido y descansado para saber mi reposo total?',
            f'No como categorías exclusivas: Bedtime no mide sueño y hay {len(overlap)} minutos donde Bedtime=1 coincide con soporte de descanso. '
            'Sumarlos duplicaría esos minutos. Para un total exclusivo hay que acordar una prioridad o una unión de minutos observados.',
            {'bedtime_rest_support_overlap_minutes': len(overlap), 'actual_sleep_plus_rest_total': None})

        add('salir', '¿A qué horas parece que he salido hoy?',
            (f'El sensor de entrada tiene activaciones observadas en {ranges(doors)}. Son compatibles con salida o entrada; '
             'no determinan dirección ni número real de aperturas.' if doors else
             'No hay activaciones de entrada observadas en esta ventana; los vacíos no prueban que no salieras.'), activity_facts('salir'))
        add('salir', '¿Cuánto tiempo he estado fuera de casa hoy?',
            f'No se puede determinar. Hay {total("salir")} minutos con entrada activada, pero no identifican dirección ni duración fuera. '
            'No uso huecos, reposo o red para inventar una salida.', {'outside_duration_minutes': None, 'door_active_minutes': total('salir')})

        def step_answer(metric):
            return (f'Hay {metric.get("observed_sum", 0):g} pasos observados en {metric["observed_minutes"]} minutos; '
                    f'{metric["unknown_minutes"]} minutos no tienen pasos registrados. '+
                    ('La suma cubre todo el intervalo observado.' if metric['unknown_minutes'] == 0 else 'Es una suma parcial, no un total real garantizado.'))

        add('pasos', '¿Cuántos pasos he dado hoy?', step_answer(summary['metrics'][WATCH+'steps']),
            summary['metrics'][WATCH+'steps'], ['resumir_intervalo'])
        add('pasos', '¿Cuántos pasos di esta mañana, desde medianoche hasta las doce?', step_answer(night['metrics'][WATCH+'steps']),
            night['metrics'][WATCH+'steps'], ['resumir_intervalo'], t0=morning_end, interval=(day_start, morning_end))
        afternoon_start = stamp(day+'T12:00:00+01:00')
        afternoon = h.resumir_intervalo(start=iso(afternoon_start), end=iso(ref), t0=iso(ref))
        add('pasos', '¿Cuántos pasos llevo desde las doce hasta ahora?', step_answer(afternoon['metrics'][WATCH+'steps']),
            afternoon['metrics'][WATCH+'steps'], ['resumir_intervalo'], interval=(afternoon_start, ref))
        add('hr', '¿Cuándo tuve el pulso más alto hoy y eso demuestra ejercicio?',
            (f'La mayor última lectura por minuto es {peak_hr:g} bpm, observada por primera vez a las {time_label(peak_hr_t)}. '
             'Una lectura alta aislada no demuestra ejercicio; debe contrastarse con pasos y duración.' if peak_hr is not None else
             'No hay lecturas HR disponibles para identificar un máximo.'),
            {'max_hr_bpm': peak_hr, 'first_max_at': iso(peak_hr_t) if peak_hr_t is not None else None,
             'observed_hr_minutes': len(hr_points)}, ['resumir_intervalo', 'consultar_minutos'])

        # Concrete moments come from actual rule episodes or actual signal peaks.
        candidates = exercise or bath or bedtime
        moment = stamp(candidates[0]['end'])-60 if candidates else day_start
        moment_activities = []
        for name in activities:
            _, es = episodes_for(name, start=iso(day_start), end=iso(moment+60), t0=iso(moment+60))
            for e in es:
                if stamp(e['start']) <= moment < stamp(e['end']):
                    # End minute of the selected episode is supported. Other labels may have gaps:
                    if not any(stamp(g['start']) <= moment < stamp(g['end']) for g in e['gaps']):
                        moment_activities.append(name)
                        break
        add('momento', f'¿Qué estaba haciendo a las {time_label(moment)} de hoy?',
            (f'En ese minuto hay soporte para las etiquetas operativas {", ".join(moment_activities)}. '
             'Describen compatibilidad con reglas, no actividades reales confirmadas.' if moment_activities else
             'No hay una etiqueta con soporte suficiente en ese minuto; no lo interpreto como inactividad.'),
            {'at': iso(moment), 'supported_rule_labels': moment_activities}, t0=moment+60, interval=(moment, moment+60))
        moment2 = stamp(kitchen[0]['start']) if kitchen else (peak_hr_t or day_start)
        raw = h.rows[moment2]
        selected = {k: raw[k] or None for k in ['habitacion', WATCH+'steps', WATCH+'hr_bpm',
                                                'cocina_frigo_luz24_lx', 'cocina_vasos13', 'cocina_platos4']}
        add('momento', f'¿Qué observaciones apoyan una actividad a las {time_label(moment2)} de hoy?',
            'En ese minuto: '+', '.join(f'{k}={v if v is not None else "desconocido"}' for k, v in selected.items())+
            '. Son observaciones; las etiquetas requieren aplicar y explicar la regla.',
            {'at': iso(moment2), 'values': selected, 'source': raw['_source']},
            ['resumir_intervalo', 'consultar_minutos'], t0=moment2+60, interval=(moment2, moment2+60))

        humidity_points = [(t, float(r['bano_humedad7_pct'])) for t, r in rows if r['bano_humedad7_pct']]
        peak_h_t, peak_h = max(humidity_points, key=lambda p: p[1]) if humidity_points else (None, None)
        add('humedad', '¿La humedad máxima de hoy basta para decir que me duché?',
            (f'No. El máximo observado es {peak_h:g}% a las {time_label(peak_h_t)}. Una humedad alta aislada no demuestra ducha; '
             'hay que contrastar subida, referencia previa y baño observado.' if peak_h is not None else
             'No hay humedad observada suficiente. No se puede determinar una ducha a partir de vacíos.'),
            {'max_humidity_pct': peak_h, 'first_max_at': iso(peak_h_t) if peak_h_t is not None else None,
             'high_humidity_alone_proves_shower': False}, ['resumir_intervalo', 'consultar_minutos'])
        humidity_ref = min(ref, stamp(shower_evidence[0]['humidity_observed_at'])+60) if shower_evidence else min(ref, (peak_h_t or day_start)+60)
        trend = h.analizar_tendencia('bano_humedad7_pct', 10, t0=iso(humidity_ref))
        current = trend['current']
        add('humedad', f'¿Se puede calcular la subida de humedad en los diez minutos hasta las {time_label(humidity_ref)} para apoyar una ducha?',
            (f'La pendiente estimada es {current["linear_slope_per_minute"]:.3f} puntos porcentuales/min, con {current["observed_minutes"]}/10 minutos observados. '
             'Complementa la regla, sin confirmar ducha por sí sola.' if current['status'] == 'ok' else
             f'Con el mínimo del 80% no se puede estimar esa pendiente: hay {current["observed_minutes"]}/10 minutos observados. '
             'El sensor registra cambios; no relleno huecos. La comparación de dos lecturas conocidas de la regla es evidencia distinta.'),
            trend, ['resumir_intervalo', 'analizar_tendencia', 'inferir_actividades'], t0=humidity_ref, interval=(humidity_ref-600, humidity_ref))

        cutoff_activity = 'ejercicio' if exercise else 'ducha'
        cutoff_es = exercise if exercise else shower
        early = stamp(cutoff_es[0]['start']) if cutoff_es else day_start+12*3600
        if early <= day_start:
            early = day_start+60
        early_result, early_es = episodes_for(cutoff_activity, start=iso(day_start), end=iso(early), t0=iso(early))
        add('t0', f'Son las {time_label(early)}. ¿Hay evidencia de {"ejercicio" if cutoff_activity == "ejercicio" else "ducha"} hoy hasta ahora?',
            (f'Hasta las {time_label(early)} hay soporte en {ranges(early_es)}. No utilizo observaciones posteriores a esa hora.' if early_es else
             f'Hasta las {time_label(early)} no hay soporte suficiente para esa regla. Una evidencia posterior no puede usarse en esta respuesta.'),
            {'activity': cutoff_activity, 'episode_count': len(early_es), 'episodes': early_es}, t0=early)
        rolling = h.resumir_intervalo(t0=iso(ref), window='ultimas_24h')
        add('pasos', '¿Cuántos pasos di en las últimas 24 horas hasta ahora?', step_answer(rolling['metrics'][WATCH+'steps']),
            rolling['metrics'][WATCH+'steps'], ['resumir_intervalo'], interval=(ref-86400, ref))
        if sum(c['day'] == day for c in cases) != 33:
            raise AssertionError('Se requieren 33 casos por día')

    # One cross-day question uses every exported minute, including daily midnight edges.
    a, b = min(h.rows), h.latest
    rs = h.resumir_intervalo(start=iso(a), end=iso(b), t0=iso(b))
    by_day = []
    for day in days:
        ts = [t for t in h.rows if iso(t)[:10] == day]
        r = h.inferir_actividades(activity='ejercicio', start=iso(min(ts)), end=iso(max(ts)+60), t0=iso(b), limit=100)
        by_day.append({'day': day, 'exercise_rule_support_minutes': r['totals']['ejercicio']['rule_support_minutes']})
    cases.append({'id': 'QA100', 'day': 'tres_dias', 'topic': 'comparacion',
                  'question': '¿En cuál de estos tres días hay más minutos compatibles con ejercicio y cuántos pasos hay registrados en todo el periodo?',
                  't0': iso(b), 'timezone': 'Europe/Madrid',
                  'interval': {'start': iso(a), 'end': iso(b), 'closure': '[start,end)'},
                  'reference_answer': f'El mayor soporte está en {max(by_day, key=lambda v: v["exercise_rule_support_minutes"])["day"]}. '+
                  '; '.join(f"{v['day']}: {v['exercise_rule_support_minutes']} minutos de soporte de ejercicio" for v in by_day)+
                  f'. En todo el periodo hay {rs["metrics"][WATCH+"steps"].get("observed_sum", 0):g} pasos observados, '
                  f'con {rs["metrics"][WATCH+"steps"]["unknown_minutes"]} minutos sin pasos registrados. El último día está incompleto y estas etiquetas no son deporte anotado.',
                  'expected_facts': {'by_day': by_day, 'steps': rs['metrics'][WATCH+'steps']},
                  'suggested_tools': ['resumir_intervalo', 'inferir_actividades'],
                  'reference_kind': 'cálculo determinista y reglas operativas; no anotación humana'})
    if len(cases) != 100 or len({c['id'] for c in cases}) != 100:
        raise AssertionError('Se requieren exactamente 100 referencias únicas')
    metadata = {'count': 100, 'source_snapshot': h.sources, 'rules': asdict(rules),
                'scope': ['ducha', 'bano', 'bedtime', 'ejercicio', 'cocinar', 'descansar', 'salir', 'pasos', 'hr', 'humedad', 'tendencias'],
                'not_human_annotations': True, 'not_qwen_generated_answers': True,
                'daily_cutoffs': {d: next(c['t0'] for c in cases if c['day'] == d) for d in days},
                'notes': ['Referencias para coherencia de consultas y reglas; no medir precisión HAR con este banco.',
                          'Todos los intervalos son semiabiertos. Los horarios sin fecha se refieren al día del caso.',
                          'Los casos retrospectivos pueden usar contexto anterior o posterior al momento, siempre hasta su t0.',
                          'Los días cerrados usan t0=23:59 para que hoy sea inequívoco; QA100 incluye todos los minutos exportados.',
                          'No dar estas respuestas al modelo durante evaluación; comparar hechos, no coincidencia literal.',
                          'No confundir duración de soporte de regla con duración real de sueño, ducha o deporte.']}
    output.mkdir(parents=True, exist_ok=True)
    (output/'referencias_100.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases), encoding='utf-8')
    (output/'metadatos.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['CAREBOT — 100 preguntas y respuestas de referencia',
             'Datos: 31/10, 01/11 y 02/11 de 2026. Zona: Europe/Madrid.',
             'Referencias calculadas con datos y reglas; no anotaciones humanas ni respuestas generadas por Qwen.',
             'Hoy se interpreta con el t0 de cada caso. Sueño real, duración de ducha y tiempo fuera pueden ser indeterminados.',
             'Este banco comprueba el módulo de consultas; no mide precisión HAR.', '']
    for c in cases:
        lines.extend([f"{c['id']}. {c['question']}", f"Referencia temporal: {c['t0']}",
                      f"Respuesta: {c['reference_answer']}", ''])
    (output/'preguntas_y_respuestas_100.txt').write_text('\n'.join(lines), encoding='utf-8')
    sections = []
    for group in [*days, 'tres_dias']:
        sections.append(f'<h2>{escape(group)}</h2>')
        for c in cases:
            if c['day'] != group:
                continue
            sections.append(f'<article id="{c["id"]}"><h3>{c["id"]}. {escape(c["question"])}</h3>'
                            f'<p class="meta">Referencia: {escape(c["t0"])} · Europe/Madrid · {escape(c["topic"])}</p>'
                            f'<p>{escape(c["reference_answer"])}</p>'
                            f'<details><summary>Hechos verificables y ventana</summary><pre>{escape(json.dumps({"interval": c["interval"], "facts": c["expected_facts"]}, ensure_ascii=False, indent=2))}</pre></details></article>')
    html = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>CAREBOT · 100 preguntas y respuestas de referencia</title>
<style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#202a35;background:#f6f8fa}article{background:white;padding:18px 24px;margin:18px 0;border:1px solid #dce3e8;border-radius:12px}h1,h2,h3{line-height:1.3}h3{font-size:19px}.meta{font-size:13px;color:#566574}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}aside{background:#e8f2f0;padding:20px;border-radius:12px}@media print{body{background:white;font-size:11pt}article{break-inside:avoid}details{display:none}}</style>
<h1>CAREBOT · 100 preguntas y respuestas</h1><p>31/10, 01/11 y 02/11 de 2026 · Módulo de consultas históricas de actividades.</p>
<aside>Respuestas calculadas con los TSV y las reglas explícitas, no generadas por Qwen ni anotadas por una persona.
Sirven para comprobar cálculos, selección de herramientas, límites temporales e incertidumbre; no miden precisión HAR.
«Hoy» depende del t0 que acompaña a cada pregunta. Los días cerrados se consultan hasta las 23:59 y el último hasta el final disponible.
Sueño real, duración de ducha y tiempo fuera pueden quedar indeterminados. Los vacíos nunca cuentan como cero.</aside>
'''+''.join(sections)+'</html>'
    (output/'preguntas_y_respuestas_100.html').write_text(html, encoding='utf-8')
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('dataset_home_real/mariola'))
    p.add_argument('--output', type=Path, default=Path('temp_data/har_qa_100'))
    args = p.parse_args()
    cases = build(args.data, args.output)
    print(json.dumps({'count': len(cases), 'by_day': dict(Counter(c['day'] for c in cases)), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
