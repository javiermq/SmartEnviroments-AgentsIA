"""Reproduce conversaciones con tres roles de Qwen sobre el fichero local de 72 horas."""
import argparse
from datetime import datetime
from itertools import pairwise
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from .config import Settings
from .historical_har import History, ZONE, iso, median_columns, number, stamp


TV_POWER = 'salon_consumo14_w'
BATH_HUMIDITY = 'bano_humedad7_pct'
EXERCISE_TERMS = re.compile(r'\b(?:ejercici|ejercit|deport|entren|correr|moverme|me mov|actividad f[ií]sica)', re.I)
WATCH_MODE = re.compile(r'\b(?:bedtime|modo\s+(?:de\s+)?(?:descanso|sueño|dormir))\b', re.I)


def supports_exercise_query(question, conversation):
    if EXERCISE_TERMS.search(question):
        return True
    if re.search(r'\b(?:ducha|duch[eé]|humedad|baño|tele|reloj|descanso)\b', question, re.I):
        return False
    previous_question = next((turn['usuario'] for turn in reversed(conversation) if 'usuario' in turn), '')
    return bool(EXERCISE_TERMS.search(previous_question))


def conversation_events(events, rows, t0):
    """Keep the latest declaration and the latest observed power change."""
    cutoff = stamp(t0)
    latest = {event['source']: event for event in events
              if stamp(event['time']) < cutoff
              or (event['source'] == 'usuario' and stamp(event['time']) == cutoff)}
    delta_key = median_columns(TV_POWER)['median_minus_current']
    readings = [(t, number(row, TV_POWER), number(row, delta_key))
                for t, row in rows if row.get(TV_POWER)]
    for (previous_at, previous_w, previous_delta), (at, watts, delta) in pairwise(readings):
        if (at - previous_at <= 5*60 and previous_delta is not None and delta is not None
                and previous_delta > -20 >= delta and watts - previous_w >= 20):
            latest[TV_POWER] = {'event': 'subida_consumo_tele', 'source': TV_POWER, 'time': iso(at),
                                'vatios_antes': previous_w, 'vatios_observados': watts}
    return [
        {**event, 'reciente': cutoff - stamp(event['time']) <= (15 if source == 'usuario' else 10)*60}
        for source, event in latest.items()
    ]


def reply_focus(question, t0, metrics, selected, conversation, events):
    """Give the reply model only the evidence needed for a clear request."""
    text = question.casefold()
    if '?' not in question and any(
        event['source'] == 'usuario' and event['time'] == t0 and event['event'] in question
        for event in events
    ):
        return (
            'Reacciona con naturalidad a lo que acaba de contarte la persona. Habla de tú y respeta '
            'quién ha hecho cada cosa. Mantén una conversación sencilla con lo que ha contado.',
            {},
        )
    if 'ejercicio' in metrics:
        return (
            'Responde sobre ejercicio solo con el resultado calculado. Si tramo es null, di que '
            'no aparece un tramo claro de ejercicio en el intervalo pedido. Si hay un tramo, '
            'describe los indicios o cifras solicitados. Usa la segunda persona. Los datos '
            'no identifican el deporte concreto ni explican cómo se siente la persona.',
            {'ejercicio': metrics['ejercicio']},
        )
    signals = metrics.get('senales', {})
    if TV_POWER in signals and re.search(r'\b(?:tele|televisi[oó]n|vatios)\b', text):
        reading = signals[TV_POWER]
        return (
            'Da la potencia medida en W, y si preguntan por un cambio usa el evento con las dos '
            'potencias si existe. Si preguntan si está encendida, empieza por la lectura y expresa '
            'la posibilidad con «podría» o «parece». Una lectura baja es compatible con espera. '
            'La potencia por sí sola deja incierto el estado exacto y quién mira.',
            {'potencia_tele': reading.get('current_value'),
             'eventos_tele': [event for event in events if event['source'] == TV_POWER]},
        )
    previous_questions = ' '.join(turn.get('usuario', '') for turn in conversation).casefold()
    if BATH_HUMIDITY in signals and ('humed' in text or 'húmed' in text or 'humedad' in previous_questions):
        reading = signals[BATH_HUMIDITY]
        if re.search(r'\b(?:normal|habitual|mediana)\b', text):
            return (
                'Compara la última humedad del baño con su mediana de referencia. Usa porcentaje '
                'para el valor y puntos porcentuales para la diferencia. Describe la comparación '
                'con la mediana, sin interpretarla como un cambio entre dos instantes.',
                {'humedad_pct': reading.get('current_value'),
                 'mediana_pct': reading.get('mediana_referencia'),
                 'diferencia_puntos': reading.get('valor_menos_mediana')},
            )
        if 'sub' in text and re.search(r'\b(?:hora|cu[aá]ndo|empez[oó])\b', text):
            observed = reading.get('current_at')
            return (
                'Distingue cuándo consta la lectura de humedad del baño de cuándo empezó la subida. '
                'Da la hora de la lectura si existe y di que el comienzo exacto es desconocido.',
                {'hora_lectura': observed[11:16] if observed else None,
                 'humedad_pct': reading.get('current_value')},
            )
    activity = (selected or {}).get('actividad')
    if activity == 'ducha' and re.search(r'\b(?:dur[oó]|duraci[oó]n)\b', text):
        return (
            'La pregunta pide duración. La duración real es desconocida. Contesta solo sobre la '
            'duración. Una señal puntual compatible con una ducha no mide cuánto tiempo duró.',
            {},
        )
    if activity == 'ducha' and re.search(r'\b(?:duch\w*|baño|humedad)\b', text):
        return (
            'Habla de una posible ducha en segunda persona. Si pide la hora, da la aproximada. '
            'Si pregunta por los indicios, explica que la ubicación estimada situó a la persona '
            'en el baño y que el sensor de ese baño registró una subida de humedad. '
            'Estos indicios admiten otras causas.',
            {'hora_aproximada': selected['inicio'], 'evidencia': selected.get('evidencia', {})},
        )
    if activity == 'descansar' and re.search(r'\b(?:mover\w*|movimiento|quiet\w*|reposo)\b', text):
        return (
            'Describe solo los indicios de reposo. Hay minutos con cero pasos y pulso bajo. '
            'Habla de compatibilidad con el reposo, sin afirmar inmovilidad continua.',
            {'minutos_con_indicios': selected['min_con_evidencia'], 'tramo': [selected['inicio'], selected['fin']]},
        )
    if selected is None and not any(event['source'] == 'usuario' and event['reciente'] for event in events):
        if re.search(r'\bqu[eé]\b.*\b(?:estoy|ando)\s+haciendo\b', text):
            return ('Explica brevemente que no tienes indicios fiables para saber qué hace ahora la persona.', {})
    return None


# Las escenas fijan la hora y el mensaje. Sus nombres quedan fuera de los prompts.
SCENES = (
    ('Cena', '2026-10-31T20:52:00+01:00', 'Hola'),
    ('Ducha reciente', '2026-10-31T12:40:00+01:00', '¿Sobre qué hora parece que me duché?'),
    ('Tele y ubicación desconocida', '2026-10-31T13:09:00+01:00', 'Hola'),
    ('Ejercicio', '2026-11-01T14:08:00+01:00', 'Hola'),
    ('Fin de Bedtime', '2026-11-01T09:59:00+01:00', 'Buenos días, ¿a qué hora terminó el modo descanso del reloj?'),
    ('Reposo', '2026-10-31T15:47:00+01:00', 'Hola'),
    ('Cocina por la tarde', '2026-11-02T17:09:00+01:00', 'Hola'),
    ('Humedad y actividad', '2026-11-02T13:39:00+01:00', '¿Qué parece que estaba haciendo cuando subió la humedad del baño?'),
    ('Sin contexto fiable', '2026-11-02T18:40:00+01:00', 'Hola'),
    ('Pasos de los últimos tres días', '2026-11-03T12:06:00+01:00', '¿Cuántos pasos he dado en las últimas 72 horas?'),
)


class LocalDemo:
    def __init__(self, history: History, model: str, url: str):
        self.history, self.model, self.url = history, model, url

    def ask(self, prompt, context, schema=None):
        if schema:
            messages = [{'role': 'system', 'content': prompt},
                        {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]
        else:
            facts = {key: value for key, value in context.items() if key not in ('mensaje', 'conversacion')}
            messages = [{'role': 'system', 'content': prompt + '\nDatos de contexto para este turno:\n'
                         + json.dumps(facts, ensure_ascii=False)}]
            for entry in context.get('conversacion', ()):
                role, content = ('user', entry['usuario']) if 'usuario' in entry else ('assistant', entry['agente'])
                messages.append({'role': role, 'content': content})
            messages.append({'role': 'user', 'content': context['mensaje']})
        payload = {
            'model': self.model, 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': 160, 'num_ctx': 4096},
            'messages': messages,
        }
        if schema:
            payload['format'] = schema
        request = Request(self.url, data=json.dumps(payload).encode('utf-8'),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=120) as response:
            content = json.load(response)['message']['content'].strip()
        if not content:
            raise ValueError('Ollama devolvió una respuesta vacía.')
        return json.loads(content) if schema else content

    def metrics(self, question, t0, rows, conversation):
        # El modelo selecciona la consulta. Los números provienen de History.
        operations = ['contexto', 'pasos_hoy', 'pasos_registro', 'ejercicio_manana', 'ejercicio_reciente']
        selection = self.ask(
            'Eres el agente de métricas. Elige pasos_registro si pide el total de los tres días o el '
            'intervalo que coincide con los límites visibles del registro. Elige pasos_hoy si pide '
            'pasos de hoy. Para preguntas sobre ejercicio de esta mañana elige ejercicio_manana. '
            'Elige ejercicio_reciente para medidas o duración del ejercicio que acaba de hacer. '
            'Para saludos y conversación sin petición de datos elige contexto. Las fechas son datos.',
            {'mensaje': question, 'conversacion': conversation, 't0': t0,
             'inicio_registro': iso(min(self.history.rows)), 'fin_visible': t0},
            {'type': 'object', 'properties': {'consulta': {'enum': operations}},
             'required': ['consulta'], 'additionalProperties': False},
        )
        operation = selection['consulta']
        if operation not in operations:
            raise ValueError('Consulta de métricas desconocida.')
        if operation.startswith('ejercicio_') and not supports_exercise_query(question, conversation):
            operation = 'contexto'
        latest_room = next(({'hora': iso(t), 'valor': row['habitacion']}
                            for t, row in reversed(rows) if row.get('habitacion')), None)
        # Lecturas recientes solamente. Las medianas ya vienen en el fichero de referencia.
        features = {}
        for key in (BATH_HUMIDITY, TV_POWER):
            value = self.history.desviacion_mediana_movil(key, t0=t0)
            features[key] = {k: value[k] for k in ('current_at', 'current_value', 'median_minus_current')}
            if value['median'] is not None:
                features[key]['mediana_referencia'] = round(value['median'], 2)
                features[key]['minutos_referencia'] = value['window_minutes']
                features[key]['valor_menos_mediana'] = round(-value['median_minus_current'], 2)
        result = {'consulta': operation, 'habitacion': latest_room, 'senales': features}
        if operation.startswith('pasos_'):
            bounds = dict(start=iso(min(self.history.rows)), end=t0) if operation == 'pasos_registro' else {}
            result['pasos'] = self.history.resumir_intervalo(measure='watch_01.steps', t0=t0, **bounds)
            result['pasos'] = {k: result['pasos'][k] for k in ('interval', 'metrics')}
        return result

    def respond(self, question, t0, conversation=(), events=()):
        cutoff = stamp(t0)
        start = max(min(self.history.rows), cutoff-20*60)
        if not start < cutoff <= self.history.latest:
            raise ValueError('La hora debe caer dentro del registro reproducible.')
        rows = [(t, self.history.rows[t]) for t in range(start, cutoff, 60) if t in self.history.rows]
        metrics = self.metrics(question, t0, rows, conversation)
        events = conversation_events(events, rows, t0)
        operation = metrics['consulta']
        if operation.startswith('ejercicio_'):
            end = cutoff
            if operation == 'ejercicio_manana':
                day = datetime.fromtimestamp(cutoff, ZONE)
                start = max(min(self.history.rows), int(day.replace(hour=0, minute=0).timestamp()))
                end = min(cutoff, int(day.replace(hour=12, minute=0).timestamp()))
            episodes = self.history.inferir_actividades(
                activity='ejercicio', start=iso(start), end=iso(end), t0=t0,
            )['episodes'] if start < end else []
            exercise = {'intervalo_consultado': [iso(start), iso(end)] if start < end else None, 'tramo': None}
            if episodes:
                episode = episodes[-1]
                summary = self.history.resumir_intervalo(t0=t0, start=episode['start'], end=episode['end'])['metrics']
                exercise['tramo'] = {
                    'inicio': episode['start'], 'fin': episode['end'], 'minutos': episode['duration_minutes'],
                    'cortado_por_limite_consulta': episode['boundary_censored'],
                    'pasos_observados': summary['watch_01.steps'].get('observed_sum'),
                    'pulso_medio_bpm': round(summary['watch_01.hr_bpm']['mean'], 1),
                }
            metrics['ejercicio'] = exercise
        else:
            episodes = self.history.inferir_actividades(start=iso(start), end=t0, t0=t0, limit=100)['episodes']
            if WATCH_MODE.search(question):
                episodes = [e for e in episodes if e['activity'] == 'bedtime']
        candidates = []
        for index, e in enumerate(episodes):
            candidates.append({
                'id': index, 'actividad': e['activity'], 'inicio': e['start'][11:16],
                'fin': e['end'][11:16], 'hace_min': (cutoff-stamp(e['end']))//60,
                'min_con_evidencia': e['evidence_minutes'], 'soporte': round(e['support_fraction'], 2),
            })
        choice = self.ask(
            'Eres el HAR. Selecciona un id de la lista o null. Para preguntas explícitas elige la '
            'actividad que responda al mensaje, aunque haya otro episodio más reciente. '
            'Para una explicación sobre cocinar elige cocinar; para ducha elige ducha. El modo del '
            'reloj corresponde a bedtime. Si el tema es un aparato, una cifra o un comentario sin '
            'pregunta sobre actividad, puedes devolver null. Para un saludo elige la actividad más reciente '
            'con soporte de al menos 0.8 y fin hace como máximo 2 minutos. Si no hay una así, null. '
            'El consumo de la tele y la puerta son señales del entorno. No atribuyas esas acciones a la persona. '
            'En declaracion copia un fragmento literal del mensaje actual si el usuario afirma qué está '
            'haciendo o corrige qué hace o ha dejado de hacer. Conserva el detalle que aporta. '
            'Si pregunta, especula o habla de otra persona, devuelve null en declaracion. '
            'Mantén la selección de sensores separada de la declaración.',
            {'mensaje': question, 'conversacion': conversation, 't0': t0,
             'metricas': metrics, 'candidatos': candidates, 'eventos': events},
            {'type': 'object', 'properties': {'id': {'enum': [None, *range(len(candidates))]},
                                             'declaracion': {'type': ['string', 'null']}},
             'required': ['id', 'declaracion'], 'additionalProperties': False},
        )
        greeting = question.lower().strip(' ¡!¿?.') in {'hola', 'buenas', 'buenos días', 'buenas tardes', 'buenas noches'}
        declaration = (choice['declaracion'] or '').strip()
        clauses = re.split(r'(?<=[.!?])\s+|(?=¿)', question)
        if not greeting and declaration and any(declaration in clause and '?' not in clause for clause in clauses):
            events = [event for event in events if event['source'] != 'usuario']
            events.append({'event': declaration, 'source': 'usuario', 'time': t0, 'reciente': True})
        selected = next((c for c in candidates if c['id'] == choice['id']), None)
        if WATCH_MODE.search(question) and selected is None and len(candidates) == 1:
            selected = candidates[0]
        if greeting and selected and (
            selected['actividad'] == 'salir' or selected['hace_min'] > 2 or selected['soporte'] < .8
        ):
            selected = None
        if selected:
            evidence = episodes[selected['id']]['evidence_sample']
            if evidence:
                selected = {**selected, 'evidencia': {k: v for k, v in evidence[-1].items() if k != 'source'}}
        answer, greeting_context = self.general(question, t0, metrics, selected, conversation, events)
        return {'t0': t0, 'mensaje': question, 'metricas': metrics, 'candidatos_har': candidates,
                'seleccion_har': choice, 'contexto_actividad': selected, 'contexto_saludo': greeting_context,
                'eventos': events, 'respuesta': answer}

    def general(self, question, t0, metrics, selected, conversation=(), events=()):
        tone = 'Responde en español, de tú, en una o dos frases naturales. '
        greeting = question.lower().strip(' ¡!¿?.') in {'hola', 'buenas', 'buenos días', 'buenas tardes', 'buenas noches'}
        greeting_context = None
        if greeting and not conversation and not any(e['source'] == 'usuario' for e in events):
            if selected and selected['actividad'] in ('cocinar', 'ejercicio'):
                greeting_context = selected
            prompt = tone + 'Devuelve el saludo con una sola pregunta breve.'
            if greeting_context:
                prompt += (' Haz una pregunta concreta sobre el tema. Usa un verbo cotidiano que '
                           'describa lo que está haciendo la persona. La hora orienta el vocabulario '
                           'de forma discreta. Omite la hora y los detalles del sistema.')
                context = {'mensaje': question, 'hora_local': t0[11:16], 'tema': greeting_context['actividad']}
            else:
                prompt += ' La conversación acaba de empezar. Pregunta simplemente cómo está.'
                context = {'mensaje': question}
            answer = self.ask(prompt, context)
        else:
            focus = reply_focus(question, t0, metrics, selected, conversation, events)
            if focus:
                instruction, facts = focus
                answer = self.ask(tone + instruction,
                                  {'mensaje': question, 'conversacion': conversation, **facts})
                return answer, greeting_context
            response_context = selected
            if selected and selected.get('actividad') == 'ducha':
                response_context = {
                    'actividad_estimada': 'ducha',
                    'hora_aproximada': selected['inicio'],
                    'duracion_real_desconocida': True,
                    'fuentes': 'ubicación estimada por UWB y sensor de humedad',
                    'evidencia': selected.get('evidencia', {}),
                }
            answer = self.ask(
                tone + '\nContesta solo lo que pide. Un comentario personal admite una reacción breve y natural. '
                'Para recordar lo que contó, atribuye la declaración al usuario en segunda persona. Si es '
                'antigua, solo sabes que lo contó entonces. Respeta sus correcciones. Una sugerencia anterior '
                'o un ingrediente disponible no cuenta como algo que hizo, usó o comió.\n'
                'Para cifras y horas usa el resultado del turno y su intervalo. Distingue dato medido, '
                'estimación y declaración. Si falta el dato, dilo. La habitación reciente no sitúa al usuario '
                'en el pasado. Un episodio de cocina indica uso de la cocina; una declaración posterior '
                'puede aclarar lo que hacía, pero no causó la estimación anterior.\n'
                'Tele: empieza por la potencia observada o por «parece»; la potencia es un indicio, no confirma '
                'el estado ni quién mira. Usa ambas lecturas si pregunta por el salto. Humedad: la hora del '
                'sensor indica cuándo se midió, no cuándo empezó a subir. La referencia es la mediana, '
                'y su diferencia se expresa en puntos porcentuales. Ducha: estimación aproximada basada '
                'en baño y humedad, sin saber si usó agua caliente ni cuánto duró.\n'
                'Reloj: el fin de bedtime es el fin del modo, sin determinar cuándo se levantó ni cómo '
                'se desactivó. Reposo: usa pasos y pulso, nunca la potencia ni la habitación. '
                'Ejercicio: si tramo es null, no consta un tramo claro en el intervalo pedido. '
                'Si pide ayuda con un objetivo explícito de actividad física o alimentación, ofrece una '
                'idea sencilla y opcional con lo que contó.',
                {'mensaje': question, 'conversacion': conversation, 't0': t0,
                 'metricas': metrics, 'contexto_actividad': response_context, 'eventos': events},
            )
        return answer, greeting_context


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset_home_real'))
    parser.add_argument('--model', default=settings.ollama_model)
    parser.add_argument('--url', default=settings.ollama_url)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scene', type=int, choices=range(1, len(SCENES)+1))
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    if args.interactive and args.scene is None:
        parser.error('--interactive requiere --scene')
    demo = LocalDemo(History(args.data), args.model, args.url)
    if args.interactive:
        title, t0, _ = SCENES[args.scene-1]
        print(f'{title} · registro a las {t0}', flush=True)
        conversation = []
        events = []
        while True:
            try:
                message = input('Tú > ').strip()
            except (EOFError, KeyboardInterrupt):
                return
            if message == '/salir':
                return
            if message:
                result = demo.respond(message, t0, conversation[-4:], events)
                answer, events = result['respuesta'], result['eventos']
                print('Agente > '+answer, flush=True)
                conversation.extend([{'usuario': message}, {'agente': answer}])
    else:
        results = []
        if args.output is None:
            name = f'demo_tres_agentes_escena_{args.scene:02d}' if args.scene else 'demo_tres_agentes'
            args.output = Path('temp_data') / (name+'.json')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for i, (title, t0, question) in enumerate(SCENES, 1):
            if args.scene is not None and args.scene != i:
                continue
            result = demo.respond(question, t0)
            results.append({'escena': i, 'titulo': title, **result})
            args.output.write_text(json.dumps({'modelo': args.model, 'fuentes': demo.history.sources,
                                               'escenas': results}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{i:02d}. {title} | {question}\n    {result["respuesta"]}', flush=True)


if __name__ == '__main__':
    main()
