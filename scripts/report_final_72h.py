"""Render reviewed A-agent answers next to their deterministic 72h references."""
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from statistics import mean

from smart_home_agent.historical_har import stamp

OUT = Path('temp_data/eval_72h_clean_2026-11-06')

# Case-specific review of the saved native Ollama responses, not an LLM accuracy score.
REVIEWS = {
    'Q01': ('Incorrecta', 'Pidió últimas 24 h en vez de las 72 h explícitas y devolvió una media como respuesta al total. Referencia: 80.947 pasos.'),
    'Q02': ('Incorrecta', 'La herramienta devuelve 32.661 pasos, pero responde con la media 22,681 pasos/min.'),
    'Q03': ('Parcial', 'Ventana y cobertura correctas. La suma es 29.121, pero el texto dice 29.122: error de un paso al redactar.'),
    'Q04': ('No resuelta', 'La consulta de hoy debería devolver 12.365 pasos; construye fechas fuera de t0 y no logra una consulta válida.'),
    'Q05': ('Correcta', '12.878 pasos para el intervalo explícito 12:06–23:59 del 31/10; coincide con la referencia.'),
    'Q06': ('No resuelta', 'Debería informar de 458 minutos de Bedtime. Repite end=23:59:59, incompatible con minutos completos, y termina sin resultado.'),
    'Q07': ('Incorrecta', 'Dice que no hay datos cuando hay 469 minutos de Bedtime. Confunde un error de fecha con ausencia de datos y usa el prefijo Agente B.'),
    'Q08': ('Incorrecta', 'Afirma ausencia de Bedtime. Hay 492 minutos activos y 726 minutos observados; sus consultas terminaban en un futuro respecto a t0.'),
    'Q09': ('No resuelta', 'No obtiene los episodios. El final del tramo Bedtime es 09:57; repite la fecha inválida 23:59:59.'),
    'Q10': ('Incorrecta', 'Inventa 06:00 sin consultar episodios. La referencia Bedtime termina a las 08:30. También confunde 492 minutos activos con 726 disponibles.'),
    'Q11': ('Incorrecta', 'Usa humedad para responder sobre presencia en baño y aplica el 80% al día entero. La regla tiene 42 minutos observados y 15 tramos, no cero episodios.'),
    'Q12': ('Incorrecta', 'Niega presencia en baño a partir de consultas fallidas y humedad. La localización respalda 92 minutos, 31 en tramos >=80% y 61 en candidatos.'),
    'Q13': ('Correcta', 'Horario 12:29–12:30 y humedad 75,88%→86,23% coinciden con la regla. Ese minuto es soporte de la regla, no una duración de ducha validada externamente.'),
    'Q14': ('No resuelta', 'No llega a ejecutar una consulta válida y entrega XML de herramienta como texto. La referencia es un candidato 11:21–11:24, con 2/3 minutos de soporte.'),
    'Q15': ('Incorrecta', 'Inventa 12:30–13:15 y una regla humedad>0. La referencia tiene 13:28–13:29 con soporte y un candidato 12:27–12:30; no consulta la regla HAR.'),
    'Q16': ('Incorrecta', 'Inventa ejercicio 18:45–19:15 y resultados numéricos sin herramientas válidas. Hay cinco tramos >=80%, con 38 minutos de evidencia en total.'),
    'Q17': ('Incorrecta', 'Convierte una ventana de tendencia de pasos en 60 minutos de ejercicio, sin comprobar HR ni la regla. Solo hay candidatos 09:43–10:31 (79%) y 10:37–10:43 (67%).'),
    'Q18': ('Parcial', 'Recupera los seis horarios y separa tramos fuertes/candidatos. Los soportes por tramo son correctos en la herramienta, pero el texto inventa un denominador global de 10 min y confunde huecos con soporte.'),
    'Q19': ('Incorrecta', 'Dice que no hay cocina tras fallar todas sus consultas de fecha. La regla aporta 13 minutos: 4 en tramos >=80% y 9 en candidatos.'),
    'Q20': ('Incorrecta', 'Confunde descanso con Bedtime e inventa 22:30–06:30 y 300 minutos sin una herramienta válida. La regla respalda 375 minutos, 324 en un tramo >=80%.'),
    'Q21': ('Correcta', 'Reproduce los 17 tramos de entrada del 2/11, con 18 minutos de activación. Son eventos del sensor; no verifica dirección de desplazamiento.'),
    'Q22': ('Parcial', 'La conclusión sobre duración fuera indeterminada y los 18 eventos son coherentes. Añade un prefijo Agente B injustificado para una única actividad y texto repetitivo.'),
    'Q23': ('Parcial', 'Pendiente +0,94 bpm/min, medias 105,8/98,3 y diferencia -7,5 son correctas. Mezcla la subida dentro de la ventana actual con el descenso entre medias de ventanas y acaba describiéndolo como un único descenso.'),
    'Q24': ('Parcial', 'Reconoce 1/10 observaciones por ventana y tendencia indeterminada. El inicio “No está subiendo” es injustificado: indeterminado no equivale a falso.'),
    'Q25': ('Incorrecta', 'Devuelve minutos de humedad en lugar de intervalos de localización. La referencia tiene 13 tramos y 41 minutos observados de baño, no un tramo 00:00–11:30.'),
    'Q26': ('No resuelta', 'No responde sobre la actividad; repite instrucciones y fechas posteriores al t0 13:28. Ya existe un candidato 12:27–12:30, con 67% de soporte, antes de ese corte.'),
    'Q28': ('Correcta', 'La protección temporal rechaza consultar 15:00–18:00 cuando t0=12:06. No inventa pasos futuros; el mensaje final es técnico y mejorable.'),
}


def concise_reference(d):
    r = d['reference']
    if 'metrics' in r:
        return {'interval': r['interval'], 'coverage': r['coverage'], 'metrics': r['metrics']}
    if 'episodes' in r:
        return {'interval': r['interval'], 'totals': r['totals'], 'episodes': [
            {k: e[k] for k in ['start', 'end', 'duration_minutes', 'evidence_minutes',
                               'unknown_minutes', 'interruption_minutes', 'support_fraction', 'meets_support_threshold']}
            for e in r['episodes']]}
    if 'current' in r:
        return {k: r[k] for k in ['interval', 'units', 'previous', 'current', 'change_of_window_means', 'status']}
    return r


def main():
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if sha256(Path(manifest['data']).read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('La fuente ya no coincide con la evaluada')
    reviewed = []
    leakage = False
    future_results = []
    for case_id, (verdict, note) in REVIEWS.items():
        d = json.loads((OUT/case_id/'result.json').read_text(encoding='utf-8'))
        d['review'] = {'verdict': verdict, 'analysis': note}
        events = [json.loads(s) for s in (OUT/case_id/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        for e in events:
            if e['event'] == 'result' and 'interval' in e['result']:
                if stamp(e['result']['interval']['end']) > stamp(d['case']['t0']):
                    future_results.append(case_id)
        for req in (OUT/case_id).glob('request_*.json'):
            contents = req.read_text(encoding='utf-8')
            leakage |= any(x in contents for x in ('escenario_', 'actividad_confirmada_usuario', 'confirmacion_origen'))
        reviewed.append(d)
    counts = dict(Counter(d['review']['verdict'] for d in reviewed))
    prompts = [r['prompt_eval_count'] for d in reviewed for r in d['llm_requests'] if r.get('prompt_eval_count')]
    summary = dict(cases=len(reviewed), counts=counts, excluded={'Q27': 'composición; fuera del alcance solicitado'},
                   tool_calls=sum(len(d['tool_calls']) for d in reviewed),
                   tool_errors=sum(len(d['tool_errors']) for d in reviewed),
                   cases_with_tool_errors=sum(bool(d['tool_errors']) for d in reviewed),
                   mean_seconds=round(mean(d['elapsed_seconds'] for d in reviewed), 2),
                   max_prompt_tokens=max(prompts), context_tokens=16384,
                   future_tool_results=future_results, annotation_or_scenario_leak=leakage,
                   review_method='Revisión del asistente de respuestas y trazas frente a resultados deterministas; no precisión HAR sobre actividades reales')
    (OUT/'review_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'reviewed_results.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding='utf-8')
    intro = (
        'Se evaluó exclusivamente el agente A de métricas/HAR, con el fichero limpio de 4.320 minutos '
        '(31/10/2026 12:06 a 03/11/2026 12:06, Europe/Madrid), 17 columnas y el modelo local '
        +manifest['model']+'. El informe incluye 26 consultas del módulo y una protección temporal. '
        'La pregunta de composición Q27 se excluye; no se ha ejecutado el agente de conversación general. '
        'Las respuestas son de Qwen sin editar y las referencias proceden de Python. No se cambiaron '
        'reglas ni prompt durante esta tanda. La ejecución previa sobre el otro fichero queda archivada aparte.'
    )
    diagnosis = [
        'El agente todavía no es fiable como interfaz de respuestas numéricas. En esta tanda, la selección del intervalo, '
        'la herramienta y el campo de respuesta fallan aunque el cálculo determinista esté disponible.',
        'Fechas: genera 23:59:59, días que terminan después de t0 y reintentos idénticos. A veces convierte un error de '
        'consulta en “no hay datos”, incluso con Bedtime completo.',
        'Métricas: confunde observed_sum con mean; cobertura con activación; porcentaje de soporte del episodio con '
        'fracción del día. También altera números al redactar: 29.121 se convierte en 29.122.',
        'HAR: sustituye baño por humedad, ejercicio por pendiente de pasos y descanso por Bedtime. No siempre ejecuta '
        'inferir_actividades y puede inventar horarios sin evidencia de una herramienta válida.',
        'Hay un defecto concreto de control: el prefijo “Agente B:” evita el rechazo final de respuestas sin resumen válido. '
        'Ese prefijo no llama a ningún segundo agente. La pregunta sigue siendo de A.',
        'Las protecciones Python sí rechazan intervalos posteriores a t0 y datos inválidos. No se observó ninguna respuesta '
        'de herramienta con datos posteriores al corte ni acceso a anotaciones/escenarios. Los fallos restantes están en la '
        'orquestación y redacción; los tests unitarios no garantizan que el LLM siga el contrato.',
        'El máximo de contexto usado fue '+str(summary['max_prompt_tokens'])+' tokens de 16.384; esta tanda no apunta a '
        'agotamiento de contexto como causa principal.',
    ]
    priorities = [
        'Cerrar la salida de respuestas con hechos cuando no existe una herramienta válida; una etiqueta Agente B no debe saltarse esa validación.',
        'Resolver hoy/ayer/últimas 24 h en Python desde t0 y evitar que el modelo tenga que construir manualmente el final del día.',
        'Obligar a usar la herramienta HAR correspondiente antes de responder sobre una actividad; buscar_episodios y tendencias solo la complementan.',
        'Renderizar cifras, unidades, cobertura y horarios desde campos tipados de la herramienta. Dejar al LLM la formulación, sin recalcular ni inventar valores.',
        'Detectar reintentos idénticos y distinguir error de consulta, ausencia de muestras, candidato y falta de soporte. Mantener A limitado a una medida/actividad.',
        'Repetir exactamente estas preguntas después de esos cambios, sin tocar simultáneamente los umbrales HAR. La coherencia con reglas y la validez de las actividades reales son evaluaciones diferentes.',
    ]
    text_lines = ['Evaluación del agente de métricas/HAR — fichero limpio de 72 horas', '', intro, '',
                  'RESULTADO: '+json.dumps(counts, ensure_ascii=False), '', 'DIAGNÓSTICO', *diagnosis, '',
                  'PRIORIDADES', *[f'{i+1}. {t}' for i, t in enumerate(priorities)], '', 'CASOS']
    sections = []
    for d in reviewed:
        c, review = d['case'], d['review']
        text_lines.extend(['', f"{c['id']}. {c['question']}", 't0: '+c['t0'],
                           'Valoración: '+review['verdict'], 'Qwen: '+d.get('answer', d.get('error', '')),
                           'Análisis: '+review['analysis'],
                           'Referencia: '+json.dumps(concise_reference(d), ensure_ascii=False)])
        sections.append(f'<article id="{c["id"]}"><h3>{c["id"]} · {escape(c["question"])}</h3>'
                        f'<p class="meta">t0: {escape(c["t0"])} · {escape(review["verdict"])} · {d["elapsed_seconds"]:.1f} s · '
                        f'{len(d["tool_calls"])} llamadas · {len(d["tool_errors"])} errores</p>'
                        f'<h4>Respuesta de Qwen</h4><pre>{escape(d.get("answer", d.get("error", "")))}</pre>'
                        f'<h4>Análisis</h4><p>{escape(review["analysis"])}</p>'
                        f'<details><summary>Referencia Python y llamadas ejecutadas</summary><pre>{escape(json.dumps(concise_reference(d), ensure_ascii=False, indent=2))}</pre>'
                        f'<pre>{escape(json.dumps(d["tool_calls"], ensure_ascii=False, indent=2))}</pre></details></article>')
    (OUT/'evaluacion_agente_metricas_72h.txt').write_text('\n'.join(text_lines), encoding='utf-8')
    html = '<!doctype html><html lang="es"><meta charset="utf-8"><title>CAREBOT · Evaluación de métricas/HAR</title>'+'''
<style>body{font:16px/1.6 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#23303d;background:#f6f8fa}article{background:white;padding:18px 24px;margin:22px 0;border:1px solid #d8e0e6;border-radius:12px}h1,h2,h3{line-height:1.25}.meta{font-size:13px;color:#556576}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 system-ui}details pre{font:12px/1.5 monospace}aside{background:#e9eff6;padding:18px;border-radius:10px}li{margin:10px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #cbd6df}@media print{body{background:white}article{break-inside:avoid}details{display:none}}</style>
'''+f'<h1>Evaluación del agente A de métricas/HAR</h1><p>{escape(intro)}</p>'
    html += '<aside><strong>Balance de esta tanda:</strong> '+escape('; '.join(f'{n} {k.lower()}' for k, n in counts.items()))+'. '
    html += 'Revisión frente a datos y reglas, no precisión HAR medida sobre anotaciones humanas.</aside>'
    html += '<h2>Diagnóstico</h2><ul>'+''.join('<li>'+escape(t)+'</li>' for t in diagnosis)+'</ul>'
    html += '<h2>Qué corregir primero</h2><ol>'+''.join('<li>'+escape(t)+'</li>' for t in priorities)+'</ol>'
    html += '<h2>Las 27 consultas</h2>'+''.join(sections)
    html += '<h2>Reproducibilidad</h2><pre>'+escape(json.dumps(manifest, ensure_ascii=False, indent=2))+'</pre></html>'
    (OUT/'evaluacion_agente_metricas_72h.html').write_text(html, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
