"""Reproduce conversaciones con tres roles de Qwen sobre el fichero local de 72 horas."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from .config import Settings
from .historical_har import History, iso, median_columns, number, stamp


TV_POWER = 'salon_consumo14_w'


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
        operations = ['contexto', 'pasos_hoy', 'pasos_registro']
        selection = self.ask(
            'Eres el agente de métricas. Elige pasos_registro si pide el total de los tres días o el '
            'intervalo que coincide con los límites visibles del registro. Elige pasos_hoy si pide '
            'pasos de hoy. Para saludos y preguntas de actividad elige contexto. Las fechas son datos.',
            {'mensaje': question, 'conversacion': conversation, 't0': t0,
             'inicio_registro': iso(min(self.history.rows)), 'fin_visible': t0},
            {'type': 'object', 'properties': {'consulta': {'enum': operations}},
             'required': ['consulta'], 'additionalProperties': False},
        )
        operation = selection['consulta']
        if operation not in operations:
            raise ValueError('Consulta de métricas desconocida.')
        latest_room = next(({'hora': iso(t), 'valor': row['habitacion']}
                            for t, row in reversed(rows) if row.get('habitacion')), None)
        # Lecturas recientes solamente. Las medianas ya vienen en el fichero de referencia.
        features = {}
        for key in ('bano_humedad7_pct', TV_POWER):
            value = self.history.desviacion_mediana_movil(key, t0=t0)
            features[key] = {k: value[k] for k in ('current_at', 'current_value', 'median_minus_current')}
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
        episodes = self.history.inferir_actividades(start=iso(start), end=t0, t0=t0, limit=100)['episodes']
        candidates = []
        for index, e in enumerate(episodes):
            candidates.append({
                'id': index, 'actividad': e['activity'], 'inicio': e['start'][11:16],
                'fin': e['end'][11:16], 'hace_min': (cutoff-stamp(e['end']))//60,
                'min_con_evidencia': e['evidence_minutes'], 'soporte': round(e['support_fraction'], 2),
            })
        choice = self.ask(
            'Eres el HAR. Selecciona un id de la lista o null. Para preguntas explícitas elige la '
            'actividad que responda al mensaje. Para un saludo elige la actividad más reciente '
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
        if greeting and selected and (
            selected['actividad'] == 'salir' or selected['hace_min'] > 2 or selected['soporte'] < .8
        ):
            selected = None
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
            response_context = selected
            if selected and selected['actividad'] == 'ducha':
                response_context = {'actividad_estimada': 'ducha', 'hora_aproximada': selected['inicio']}
            answer = self.ask(
                tone + 'Si pregunta, contesta directamente y termina. Si comparte algo, responde al detalle '
                'que contó y puedes seguir con una pregunta breve. Respeta sus palabras en segunda persona. '
                'Si pregunta qué te contó, empieza por «Me dijiste que». Lo que declara sobre sí mismo '
                'tiene prioridad para hablar de su actividad. Evita añadir '
                'detalles de lugar, compañía o ambiente que el usuario no haya mencionado al contarte algo. '
                'Si una declaración está marcada como reciente=false, trátala como recuerdo y di que '
                'desconoces si sigue haciéndolo. Su hora indica cuándo lo contó. '
                'Menciona un evento ambiental solo si viene al caso. Una subida de potencia de la tele '
                'es indicio de encendido. Un consumo bajo solo indica la potencia observada. La potencia '
                'describe el aparato y no quién lo está viendo. '
                'Si pide pasos usa la suma observada en métricas. Si hay una estimación de ducha, '
                'da solo la hora aproximada y mantén la incertidumbre. Para Bedtime usa el fin del '
                'tramo como fin del modo del reloj. La última habitación es la observada al hacer '
                'la consulta y puede ser distinta de la ubicación durante una actividad pasada. '
                'Si falta contexto para responder, dilo brevemente. Usa únicamente los hechos facilitados.',
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
