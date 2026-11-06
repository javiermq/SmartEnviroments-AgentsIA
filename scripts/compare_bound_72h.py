"""Verify and report the revised time-bound metrics agent against the saved baseline."""
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from statistics import mean

BASE = Path('temp_data/eval_72h_clean_2026-11-06')
NEW = Path('temp_data/eval_72h_bound_final_2026-11-06')


def main():
    manifest = json.loads((NEW/'manifest.json').read_text(encoding='utf-8'))
    assert sha256(Path(manifest['data']).read_bytes()).hexdigest() == manifest['source_sha256']
    rows = []
    field_names = ('start', 'end', 'duration_minutes', 'evidence_minutes', 'unknown_minutes',
                   'interruption_minutes', 'support_fraction', 'meets_support_threshold')
    for path in sorted(NEW.glob('Q*/result.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        case, reference = d['case'], d['reference']
        structured = d.get('structured_result')
        checks = {}
        if case['kind'] == 'future':
            checks['future_blocked'] = structured is None and not d['tool_calls'] and 'futuro' in d['answer']
        else:
            expected_target = {'watch_01.steps': 'pasos', 'watch_01.bedtime': 'bedtime',
                               'watch_01.hr_bpm': 'pulso', 'bano_humedad7_pct': 'humedad'}.get(case['target'], case['target'])
            checks['target'] = structured is not None and structured['target'] == expected_target
            checks['interval'] = structured is not None and structured['interval'] == reference['interval']
            checks['canonical_answer'] = structured is not None and d['answer'] == structured['answer']
            facts = structured['facts'] if structured else {}
            if case['kind'] == 'metric':
                checks['facts'] = facts == reference['metrics'][case['target']]
                checks['operation'] = structured is not None and structured['operation'] == 'total'
            elif case['kind'] == 'activity':
                expected_episodes = [{k: e[k] for k in field_names} for e in reference['episodes']]
                checks['episodes'] = facts.get('episodes') == expected_episodes
                checks['totals'] = facts.get('totals') == reference['totals'][case['target']]
                op = 'levantarse' if case['id'] in ('Q09', 'Q10') else 'total' if case['id'] in ('Q12', 'Q19', 'Q22') else 'intervalos'
                checks['operation'] = structured is not None and structured['operation'] == op
            elif case['kind'] == 'trend':
                checks['facts'] = facts == {k: reference[k] for k in ('previous', 'current', 'change_of_window_means', 'units', 'status')}
                checks['operation'] = structured is not None and structured['operation'] == 'tendencia'
        first_request = NEW/case['id']/'request_1.json'
        if first_request.exists():
            payload = json.loads(first_request.read_text(encoding='utf-8'))
            fields = payload['tools'][0]['function']['parameters']['properties']
            checks['no_model_dates'] = not bool(set(fields) & {'t0', 'start', 'end', 'window', 'path'})
        rows.append({'id': case['id'], 'question': case['question'], 't0': case['t0'],
                     'checks': checks, 'passed': all(checks.values()), 'answer': d['answer'],
                     'seconds': d['elapsed_seconds'], 'tool_errors': len(d['tool_errors'])})
    assert len(rows) == 27
    baseline = json.loads((BASE/'review_summary.json').read_text(encoding='utf-8'))
    summary = {'cases': len(rows), 'passed_structured_checks': sum(r['passed'] for r in rows),
               'failed_checks': [r for r in rows if not r['passed']],
               'new_tool_errors': sum(r['tool_errors'] for r in rows),
               'baseline_tool_errors': baseline['tool_errors'],
               'new_mean_seconds': round(mean(r['seconds'] for r in rows), 3),
               'baseline_mean_seconds': baseline['mean_seconds'],
               'baseline_manual_verdicts': baseline['counts'],
               'method': 'Mismas preguntas/datos/reglas/modelo. Nueva interfaz recibe ventana del llamador y renderiza respuestas desde Python; no es una puntuación del texto libre de Qwen ni precisión HAR.'}
    (NEW/'verification.json').write_text(json.dumps({'summary': summary, 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    intro = ('La aplicación proporciona t0 y una ventana temporal. Python resuelve fechas, valida el corte, ejecuta métricas/HAR '
             'y construye cifras y horarios. Qwen selecciona una medida/actividad y una operación con una única herramienta. '
             'El esquema del modelo no admite t0, start, end, rutas ni fechas absolutas. Ayer se resuelve como día natural en Python. '
             'El fichero completo permanece en Python: al modelo llega el contexto temporal y después solo los hechos de la consulta.')
    method = ('La nueva respuesta visible es canónica, calculada por el servicio; el texto final libre de Qwen se guarda en auditoría '
              'pero no sustituye sus números. La comparación incluye la mejora de interfaz y ese renderizado. Las ventanas de prueba '
              'se pasan desde el llamador, como contexto explícito; no se pasan al modelo las respuestas ni el objetivo esperado. '
              'Se conservan el fichero, las reglas, los umbrales y el modelo local. No se ha ejecutado el agente de conversación general.')
    result_line = (f"{summary['passed_structured_checks']}/27 casos pasan las comprobaciones de objetivo, operación, fechas y hechos. "
                   f"Errores de herramienta: {summary['baseline_tool_errors']} antes, {summary['new_tool_errors']} ahora. "
                   f"Tiempo medio: {summary['baseline_mean_seconds']} s antes, {summary['new_mean_seconds']} s ahora.")
    parts = []
    text = ['Agente de métricas/HAR: t0 y ventana fijados por Python', '', intro, '', method, '', result_line, '']
    for row in rows:
        old = json.loads((BASE/row['id']/'result.json').read_text(encoding='utf-8'))
        text.extend([row['id']+'. '+row['question'], 't0: '+row['t0'],
                     'Antes: '+old.get('answer', old.get('error', '')), 'Ahora: '+row['answer'],
                     'Comprobaciones: '+json.dumps(row['checks'], ensure_ascii=False), ''])
        parts.append(f'<article><h3>{row["id"]}. {escape(row["question"])}</h3><p class="meta">{escape(row["t0"])} · '
                     f'{"Verificado" if row["passed"] else "Revisar"}</p><details><summary>Respuesta anterior</summary>'
                     f'<pre>{escape(old.get("answer", old.get("error", "")))}</pre></details>'
                     f'<h4>Respuesta nueva</h4><pre>{escape(row["answer"])}</pre>'
                     f'<details><summary>Comprobaciones</summary><pre>{escape(json.dumps(row["checks"], ensure_ascii=False, indent=2))}</pre></details></article>')
    html = '<!doctype html><html lang="es"><meta charset="utf-8"><title>CAREBOT · Ventana temporal fijada</title>'+'''
<style>body{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 22px;color:#23303d;background:#f6f8fa}article{background:white;border:1px solid #d8e0e6;border-radius:12px;padding:20px;margin:20px 0}h1,h3{line-height:1.25}.meta{font-size:13px;color:#556576}pre{font:14px/1.6 system-ui;white-space:pre-wrap;overflow-wrap:anywhere}aside{background:#dceee6;border-radius:10px;padding:20px}details{margin:10px 0}</style>
'''+f'<h1>t0 y ventana resueltos por Python</h1><p>{escape(intro)}</p><aside>{escape(result_line)}</aside>'
    html += '<p>'+escape(method)+'</p><p>Son comprobaciones de coherencia del sistema con datos y reglas; no validan las actividades reales sin anotaciones.</p>'
    html += ''.join(parts)+'</html>'
    (NEW/'comparacion_ventana_fijada.html').write_text(html, encoding='utf-8')
    (NEW/'comparacion_ventana_fijada.txt').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
