"""Paired event-payload probe for the ACCESS27 dialogue prototype."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from smart_home_agent.config import Settings
from smart_home_agent.historical_har import History, stamp
from smart_home_agent.local_har_demo import LocalDemo, TV_POWER, conversation_events


DECLARATIONS = (
    ('Estoy leyendo una novela policíaca de Islandia.',
     '¿De qué país era la novela que te dije que estaba leyendo?', 'Islandia'),
    ('Estoy escuchando un pódcast sobre astronomía.',
     '¿Sobre qué tema era el pódcast que puse?', 'astronomía'),
    ('Estoy preparando un bocadillo de tortilla.',
     '¿De qué era el bocadillo que dije que estaba preparando?', 'tortilla'),
    ('Estoy buscando un cargador verde.',
     '¿De qué color era el cargador que estaba buscando?', 'verde'),
    ('Estoy ordenando los cubiertos en el segundo cajón.',
     '¿En qué cajón te dije que estaba ordenando los cubiertos?', 'segundo'),
    ('Estoy regando la albahaca.',
     '¿Qué planta te dije que estaba regando?', 'albahaca'),
    ('Estoy practicando portugués.',
     '¿Qué idioma dije que estaba practicando?', 'portugués'),
    ('Estoy siguiendo una receta de gazpacho.',
     '¿Qué receta te dije que estaba siguiendo?', 'gazpacho'),
)

POWER_TURNS = (
    ('2026-10-31T12:43:00+01:00', (1.0, 70.6)),
    ('2026-10-31T19:40:00+01:00', (1.0, 64.8)),
    ('2026-11-01T10:25:00+01:00', (0.9, 65.5)),
    ('2026-11-02T09:49:00+01:00', (0.6, 68.6)),
)


def fold(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value.casefold()) if not unicodedata.combining(c))


def gives_values(reply, expected):
    """Check the target word or the ordered before/after values."""
    if isinstance(expected, str):
        return bool(re.search(r'\b'+re.escape(fold(expected))+r'\b', fold(reply)))
    values = (float(n.replace(',', '.')) for n in re.findall(r'\d+(?:[.,]\d+)?', reply))
    return all(target in values for target in expected)


def fixtures(history, dialogue):
    replay = json.loads(dialogue.read_text(encoding='utf-8'))
    scene = next(s for s in replay['escenas'] if s['titulo'] == 'Lectura y memoria')
    context = scene['turnos'][4]['llamadas'][2]['entrada']
    for statement, question, expected in DECLARATIONS:
        event = {'event': statement, 'source': 'usuario', 'time': '2026-11-02T18:40:00+01:00', 'reciente': True}
        yield 'recall', question, expected, context, [event]
    for t0, expected in POWER_TURNS:
        at = stamp(t0)
        rows = [(t, history.rows[t]) for t in range(at-20*60, at, 60) if t in history.rows]
        events = conversation_events([], rows, t0)
        if len(events) != 1 or (events[0]['vatios_antes'], events[0]['vatios_observados']) != expected:
            raise ValueError(f'The snapshot lacks the expected power event for {t0}.')
        power = history.desviacion_mediana_movil(TV_POWER, t0=t0)
        current = {k: power[k] for k in ('current_at', 'current_value', 'median_minus_current')}
        context = {'t0': t0, 'conversacion': [], 'contexto_actividad': None,
                   'metricas': {'consulta': 'contexto', 'senales': {TV_POWER: current}}}
        yield 'power', '¿Cuántos vatios marcaba la tele justo antes y justo después del último salto de consumo?', expected, context, events


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dialogue', type=Path, default=Path('temp_data/access27_dialogues_final.json'))
    parser.add_argument('--output', type=Path, default=Path('temp_data/access27_event_probe.json'))
    args = parser.parse_args()
    history = History('dataset_home_real')
    demo = LocalDemo(history, settings.ollama_model, settings.ollama_url)
    cases = list(fixtures(history, args.dialogue))
    report = {'model': demo.model, 'sources': history.sources, 'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, (group, question, expected, context, events) in enumerate(cases, 1):
        row = {'id': i, 'group': group, 'question': question, 'expected': expected,
               'fixed_context': {k: context[k] for k in ('t0', 'metricas', 'contexto_actividad', 'conversacion')},
               'events': events, 'results': {}}
        conditions = ('with_events', 'without_events')
        for condition in conditions if i % 2 else reversed(conditions):
            reply, _ = demo.general(question, context['t0'], context['metricas'], context['contexto_actividad'],
                                    context['conversacion'], events if condition == 'with_events' else [])
            row['results'][condition] = {'answer': reply, 'target_recovered': gives_values(reply, expected)}
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{i:02} {group}: "+', '.join(f"{k}={v['target_recovered']}" for k, v in row['results'].items()),
              flush=True)


if __name__ == '__main__':
    main()
