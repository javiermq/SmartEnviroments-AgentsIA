"""Live, reproducible Qwen evaluation on the final 72h tab-separated file.

Provided clean observations only; extra scenario/confirmation fields are never model input.
Reference results are held outside the model context. No HAR accuracy is inferred.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from smart_home_agent.config import Settings
from smart_home_agent.har_rules import HarRules
from smart_home_agent.historical_agent import HistoricalAgent, LegacyHistoricalAgent
from smart_home_agent.historical_har import History, iso


def cases():
    final = '2026-11-03T12:06:00+01:00'
    d0 = '2026-10-31T23:59:00+01:00'
    d1 = '2026-11-01T23:59:00+01:00'
    d2 = '2026-11-02T23:59:00+01:00'
    result = []
    def add(question, t0, kind, target=None, **params):
        result.append(dict(id=f'Q{len(result)+1:02d}', question=question, t0=t0,
                           kind=kind, target=target, reference_parameters=params))
    add('¿Cuántos pasos hay registrados entre el 31 de octubre a las 12:06 y el 3 de noviembre a las 12:06 de 2026?', final,
        'metric', 'watch_01.steps', start='2026-10-31T12:06:00+01:00', end=final)
    add('¿Cuántos pasos di en las últimas 24 horas?', final, 'metric', 'watch_01.steps', window='ultimas_24h')
    add('¿Cuántos pasos di ayer?', final, 'metric', 'watch_01.steps', window='ayer')
    add('¿Cuántos pasos llevo hoy?', final, 'metric', 'watch_01.steps')
    add('¿Cuántos pasos di hoy desde las 12:06 hasta ahora?', d0, 'metric', 'watch_01.steps',
        start='2026-10-31T12:06:00+01:00', end=d0)
    add('¿Cuántos minutos he dormido hoy?', d1, 'metric', 'watch_01.bedtime')
    add('¿Cuántos minutos he dormido hoy?', d2, 'metric', 'watch_01.bedtime')
    add('¿Cuánto he dormido hoy?', final, 'metric', 'watch_01.bedtime')
    add('¿A qué hora me he levantado hoy?', d1, 'activity', 'bedtime')
    add('¿A qué hora me he levantado hoy?', final, 'activity', 'bedtime')
    add('¿Cuándo he estado en el baño hoy?', d0, 'activity', 'bano')
    add('¿Cuánto tiempo he estado en el baño hoy?', d2, 'activity', 'bano')
    add('¿Cuándo me he duchado hoy?', d0, 'activity', 'ducha')
    add('¿Cuándo me he duchado hoy?', d1, 'activity', 'ducha')
    add('¿Hay evidencia de una ducha hoy? ¿Sobre qué hora?', d2, 'activity', 'ducha')
    add('¿A qué hora hice ejercicio hoy y cuánto duró?', d1, 'activity', 'ejercicio')
    add('¿He hecho ejercicio hoy? Dame los intervalos y su soporte.', final, 'activity', 'ejercicio')
    add('¿Cuándo he estado cocinando hoy?', d1, 'activity', 'cocinar')
    add('¿Cuánto tiempo he estado cocinando hoy?', d2, 'activity', 'cocinar')
    add('¿Cuándo he descansado hoy y cuántos minutos hay registrados?', d1, 'activity', 'descansar')
    add('¿A qué horas se registraron posibles salidas hoy?', d2, 'activity', 'salir')
    add('¿Cuánto tiempo estuve fuera de casa hoy?', d1, 'activity', 'salir')
    add('¿Cómo ha cambiado mi pulso en los últimos diez minutos?', '2026-11-03T10:43:00+01:00',
        'trend', 'watch_01.hr_bpm', window_minutes=10)
    add('¿Está subiendo la humedad del baño? Mira los últimos diez minutos.', '2026-11-02T13:39:00+01:00',
        'trend', 'bano_humedad7_pct', window_minutes=10)
    add('¿Qué intervalos de baño aparecen hoy hasta esta hora?', '2026-11-01T11:30:00+01:00', 'activity', 'bano')
    add('¿Hay evidencia de que me haya duchado hoy hasta ahora?', '2026-11-02T13:28:00+01:00', 'activity', 'ducha')
    add('¿Puedes sumar lo que he dormido y descansado hoy para saber mi reposo total?', d1, 'scope_B')
    add('¿Cuántos pasos daré hoy entre las 15:00 y las 18:00?', final, 'future')
    return result


def reference(history, case):
    p = dict(t0=case['t0'], **case['reference_parameters'])
    if case['kind'] == 'metric':
        return history.resumir_intervalo(measure=case['target'], **p)
    if case['kind'] == 'activity':
        return history.inferir_actividades(activity=case['target'], limit=100, **p)
    if case['kind'] == 'trend':
        return history.analizar_tendencia(column=case['target'], **p)
    return {'expected_behavior': 'delegar composición al agente B' if case['kind'] == 'scope_B'
            else 'no consultar datos posteriores a t0 ni predecir pasos futuros como observados'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('dataset_home_real/all_data_72hours.txt'))
    p.add_argument('--output', type=Path, default=Path('temp_data/eval_72h_clean_2026-11-06'))
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--limit', type=int, default=27)
    p.add_argument('--bound-time', action='store_true', help='Nueva interfaz: contexto temporal fijado por el llamador')
    args = p.parse_args()
    settings = Settings.from_env()
    args.output.mkdir(parents=True, exist_ok=True)
    history = History(args.data)
    source_hash = sha256(args.data.read_bytes()).hexdigest()
    manifest = dict(data=str(args.data.resolve()), source_sha256=source_hash, layer='supplied_clean_columns',
                    rows=len(history.rows), start=iso(min(history.rows)), end=iso(history.latest),
                    model=settings.ollama_model, url=settings.ollama_url, rules=asdict(HarRules()),
                    max_tool_calls=6, annotations_visible_to_model=False,
                    interface='bound_window_v2' if args.bound_time else 'legacy_dates')
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Composition belongs to the later conversation/composition agent, outside this run.
    all_cases = [c for c in cases() if c['kind'] != 'scope_B']
    (args.output/'questions.json').write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in all_cases[args.start-1:args.start-1+args.limit]:
        output = args.output/case['id']
        output.mkdir(exist_ok=True)
        if (output/'result.json').exists():
            print(case['id']+' already saved', flush=True)
            continue
        print(case['id']+' START '+case['question'], flush=True)
        calls = []
        def transport(payload):
            n = len(calls)+1
            (output/f'request_{n}.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            req = Request(settings.ollama_url, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(req, timeout=180) as response:
                raw = json.load(response)
            (output/f'response_{n}.json').write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
            calls.append({k: raw.get(k) for k in ['prompt_eval_count', 'eval_count', 'total_duration', 'done_reason']})
            return raw['message']
        agent_class = HistoricalAgent if args.bound_time else LegacyHistoricalAgent
        agent = agent_class(args.data, transport=transport, audit=output/'audit.jsonl')
        result = dict(case=case, reference=reference(history, case))
        begun = perf_counter()
        try:
            temporal = case['reference_parameters'].copy() if args.bound_time else {}
            result['answer'] = agent.respond(case['question'], case['t0'], **temporal)
            if args.bound_time:
                result['structured_result'] = agent.last_result
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['elapsed_seconds'] = round(perf_counter()-begun, 3)
        result['llm_requests'] = calls
        events = [json.loads(s) for s in (output/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        result['tool_calls'] = [e for e in events if e['event'] == 'call']
        result['tool_errors'] = [e for e in events if e['event'] == 'result' and 'error' in e['result']]
        (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(case['id']+' SAVED '+str(result['elapsed_seconds'])+'s; tools='+str(len(result['tool_calls'])), flush=True)
    if sha256(args.data.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('El archivo fuente cambió durante la evaluación')


if __name__ == '__main__':
    main()
