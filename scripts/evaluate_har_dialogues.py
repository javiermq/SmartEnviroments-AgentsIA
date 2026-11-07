"""Run five connected turns per replay scene and show the three Qwen calls per turn."""
import argparse
from html import escape
import json
from pathlib import Path

from smart_home_agent.config import Settings
from smart_home_agent.historical_har import History
from smart_home_agent.local_har_demo import LocalDemo, SCENES


FOLLOW_UPS = (
    ('¿Qué te hace pensar que estoy cocinando?', 'Sí, solo estoy preparando un bocadillo.',
     '¿Cuántos pasos llevo hoy?', '¿Y desde que empecé a llevar la cuenta?'),
    ('¿Y qué te hace pensar que fue una ducha?', '¿Cuánto duró?',
     'Puede que solo me lavara las manos.', '¿Entonces a qué hora fue lo del baño?'),
    ('¿Está encendida la tele?', '¿Eso quiere decir que la estoy viendo?',
     'Estoy en otra habitación.', '¿Cuántos vatios marca ahora?'),
    ('Sí, acabo de moverme un poco. ¿Desde cuándo?',
     '¿Cómo me fue con el ejercicio esta mañana? Estoy cansado.',
     '¿Cuántos pasos llevo hoy?', '¿Y solo durante el ejercicio de esta mañana?'),
    ('¿Eso significa que me levanté a esa hora?', 'Me quedé en la cama un rato más.',
     '¿Cuántos pasos llevo hoy?', '¿Y cuándo acabó lo del reloj?'),
    ('Estoy descansando un poco.', '¿Se nota que llevo un rato sin moverme?',
     '¿Desde qué hora, más o menos?', 'Llevo toda la tarde aquí. ¿He dado algún paseo hoy?'),
    ('Estoy en la cocina, pero solo buscando un vaso.', '¿Por qué pensaste que estaba cocinando?',
     '¿Hace mucho que entré?', 'Ya encontré el vaso. ¿Cuánto rato llevo aquí?'),
    ('¿A qué hora empezó a subir?', '¿Está mucho más húmedo de lo normal?',
     'A lo mejor fue solo el vapor. ¿Tú qué crees?', '¿Sigue alta ahora?'),
    ('¿Sabes qué estoy haciendo ahora?', 'Te lo cuento yo, estoy leyendo.',
     '¿Qué te acabo de decir que estaba haciendo?', '¿Cuántos pasos llevo hoy?'),
    ('¿Y cuántos llevo hoy?', '¿Ese número era solo de hoy?',
     '¿Me recuerdas cuántos llevo en estos tres días?', '¿Y cuántos minutos estuvo contando el reloj?'),
)


EVENT_SCENES = (
    ('Lectura y memoria', (
        ('2026-11-02T18:39:00+01:00', 'Hola'),
        ('2026-11-02T18:40:00+01:00', 'Te lo cuento yo, estoy leyendo un libro de misterio.'),
        ('2026-11-02T18:41:00+01:00', '¿Cuántos pasos llevo hoy?'),
        ('2026-11-02T18:42:00+01:00', '¿Está la tele encendida?'),
        ('2026-11-02T18:43:00+01:00', '¿De qué te dije que era el libro?'),
        ('2026-11-02T19:06:00+01:00', '¿Sabes si sigo con el libro?'),
    )),
    ('Corrección en la cocina', (
        ('2026-11-02T17:09:00+01:00', 'Hola'),
        ('2026-11-02T17:09:00+01:00', 'Estoy en la cocina, pero solo buscando un vaso.'),
        ('2026-11-02T17:10:00+01:00', '¿Qué te hizo pensar que estaba cocinando?'),
        ('2026-11-02T17:11:00+01:00', 'Ya encontré el vaso y ahora estoy poniendo la mesa.'),
        ('2026-11-02T17:12:00+01:00', '¿Qué estoy haciendo ahora?'),
    )),
    ('La tele cambia mientras hablamos', (
        ('2026-10-31T12:40:00+01:00', 'Hola'),
        ('2026-10-31T12:41:00+01:00', 'Estoy leyendo una novela en otra habitación.'),
        ('2026-10-31T12:42:00+01:00', 'Me tiene enganchado el libro.'),
        ('2026-10-31T12:43:00+01:00', 'Por cierto, ¿se ha encendido la tele?'),
        ('2026-10-31T12:44:00+01:00', '¿Eso quiere decir que la estoy viendo?'),
    )),
)


class TracedDemo(LocalDemo):
    def ask(self, prompt, context, schema=None):
        role = ('Métricas', 'HAR', 'General')[len(self.calls)]
        call = {'agente': role, 'instrucciones': prompt, 'entrada': context}
        self.calls.append(call)
        try:
            call['salida'] = super().ask(prompt, context, schema)
        except Exception as exc:
            call['error'] = str(exc)
            raise
        return call['salida']


def pretty(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2))


def write_report(output, report):
    output.with_suffix('.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    parts = ['<!doctype html><html lang="es"><meta charset="utf-8">',
             '<title>CAREBOT · Conversaciones locales</title>',
             '<style>body{font:16px/1.55 system-ui;max-width:1000px;margin:32px auto;padding:0 18px;'
             'background:#f7f8fa;color:#1f2937}article{background:white;border:1px solid #d9dee6;'
             'padding:15px;margin:12px 0;border-radius:10px}pre{white-space:pre-wrap;'
             'overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer}nav a{margin-right:12px}'
             '</style><h1>Conversaciones con métricas, HAR y agente general</h1>',
             '<p>Cada turno muestra la hora de consulta del registro. Se conservan hasta dos intercambios '
             'previos y los eventos de la conversación. Los mensajes del usuario son simulados. Las llamadas '
             'proceden de la ejecución local de Qwen.</p><nav>']
    for scene in report['escenas']:
        parts.append(f'<a href="#s{scene["id"]}">{scene["id"]}. {escape(scene["titulo"])}</a>')
    parts.append('</nav>')
    for scene in report['escenas']:
        parts.append(f'<h2 id="s{scene["id"]}">{scene["id"]}. {escape(scene["titulo"])}</h2>'
                     f'<p>{escape(scene["t0"])}</p>')
        for turn in scene['turnos']:
            result = turn.get('resultado', {})
            metrics = result.get('metricas', {}).get('consulta', '—')
            activity = (result.get('contexto_actividad') or {}).get('actividad', 'sin selección')
            answer = result.get('respuesta', turn.get('error', ''))
            parts.append(f'<article id="s{scene["id"]}t{turn["turno"]}">'
                         f'<p><b>{turn["turno"]}. Tú</b> · {escape(turn["mensaje"])} '
                         f'<small>({escape(turn["t0"][11:16])})</small></p>'
                         f'<p><b>Agente</b> · {escape(answer)}</p>'
                         f'<p>Métricas · {escape(metrics)} &nbsp; HAR · {escape(activity)}</p>')
            for call in turn['llamadas']:
                parts.append(f'<details><summary>{escape(call["agente"])} · ver consulta y respuesta</summary>'
                             f'<p><b>Consulta recibida</b></p><pre>{pretty(call["entrada"])}</pre>'
                             f'<p><b>Respuesta</b></p><pre>{pretty(call.get("salida", call.get("error")))}</pre>'
                             f'<details><summary>Instrucciones del agente</summary>'
                             f'<pre>{escape(call["instrucciones"])}</pre></details></details>')
            parts.append(f'<details><summary>Eventos compartidos</summary>'
                         f'<pre>{pretty(result.get("eventos", []))}</pre></details>'
                         f'<details><summary>Métricas calculadas y candidatos del motor</summary>'
                         f'<pre>{pretty(result)}</pre></details></article>')
    output.with_suffix('.html').write_text(''.join(parts)+'</html>', encoding='utf-8')


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', default=settings.ollama_model)
    parser.add_argument('--events', action='store_true', help='Replay the conversational event scenarios.')
    parser.add_argument('--resume', action='store_true', help='Reuse unchanged turns from an earlier run.')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('temp_data/dialogos_har_eventos' if args.events else 'temp_data/dialogos_har_50')
    demo = TracedDemo(History('dataset_home_real'), args.model, settings.ollama_url)
    report = {'modelo': demo.model, 'fuentes': demo.history.sources,
              'evaluacion': 'eventos' if args.events else 'conversacion', 'escenas': []}
    previous = {}
    if args.resume:
        saved = json.loads(args.output.with_suffix('.json').read_text(encoding='utf-8'))
        if any(saved.get(key) != report[key] for key in ('modelo', 'fuentes', 'evaluacion')):
            parser.error('La ejecución anterior usó otra evaluación, modelo o fichero de datos.')
        previous = {(s['id'], s['t0']): s['turnos'] for s in saved['escenas']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenes = EVENT_SCENES if args.events else tuple(
        (title, tuple((t0, message) for message in (opening, *follow_ups)))
        for (title, t0, opening), follow_ups in zip(SCENES, FOLLOW_UPS)
    )
    for index, (title, messages) in enumerate(scenes, 1):
        scene = {'id': index, 'titulo': title, 't0': messages[0][0], 'turnos': []}
        report['escenas'].append(scene)
        conversation = []
        events = []
        reusable = iter(previous.get((index, scene['t0']), ()))
        for number, (t0, message) in enumerate(messages, 1):
            saved_turn = next(reusable, {})
            if saved_turn.get('mensaje') == message and saved_turn.get('t0') == t0 and 'resultado' in saved_turn:
                scene['turnos'].append(saved_turn)
                conversation.extend([{'usuario': message}, {'agente': saved_turn['resultado']['respuesta']}])
                events = saved_turn['resultado']['eventos']
                continue
            reusable = iter(())
            demo.calls = []
            turn = {'turno': number, 'mensaje': message, 't0': t0, 'llamadas': demo.calls}
            scene['turnos'].append(turn)
            try:
                result = demo.respond(message, t0, conversation[-4:], events)
                turn['resultado'] = result
                events = result['eventos']
                conversation.extend([{'usuario': message}, {'agente': result['respuesta']}])
                print(f'{index:02}.{number} {result["respuesta"]}', flush=True)
            except (OSError, ValueError, KeyError) as exc:
                turn['error'] = str(exc)
                print(f'{index:02}.{number} ERROR {exc}', flush=True)
            write_report(args.output, report)
    write_report(args.output, report)


if __name__ == '__main__':
    main()
