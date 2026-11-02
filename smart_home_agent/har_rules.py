"""Explicit operational HAR hypotheses over immutable minute observations.

This layer never repairs the input or equates missing values with inactivity.
Defaults are experimental parameters, not individually calibrated thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import math
from typing import TYPE_CHECKING, Any

from .historical_har import WATCH, iso, stamp, number, segment_states

if TYPE_CHECKING:
    from .historical_har import History

ACTIVITIES = ('cocinar', 'bano', 'ducha', 'bedtime', 'descansar', 'ejercicio', 'salir')


@dataclass(frozen=True)
class HarRules:
    merge_gap: int = 5
    max_total_gap: int = 1440
    min_support_fraction: float = .8
    fridge_lux: float = 10
    humidity_rise_pct_points: float = 8
    humidity_reference_minutes: int = 60
    humidity_delay_minutes: int = 10
    rest_hr_max_bpm: float = 65
    rest_min_minutes: int = 10
    exercise_steps_min: float = 60
    exercise_hr_min_bpm: float = 100
    exercise_min_minutes: int = 5

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('Umbral HAR inválido: '+key)
        if not 0 < self.min_support_fraction <= 1:
            raise ValueError('min_support_fraction debe estar en (0,1]')
        for key in ('merge_gap', 'max_total_gap', 'humidity_reference_minutes',
                    'humidity_delay_minutes', 'rest_min_minutes', 'exercise_min_minutes'):
            if not isinstance(getattr(self, key), int):
                raise ValueError('El parámetro debe ser entero: '+key)
        if self.rest_min_minutes < 1 or self.exercise_min_minutes < 1 or self.merge_gap > 60 or self.humidity_rise_pct_points <= 0:
            raise ValueError('Duración o subida de humedad inválida')


RULE_DESCRIPTIONS = {
    'cocinar': 'Habitación Cocina y, en el mismo minuto, frigo >= umbral lux o vasos/platos=1.',
    'bano': 'Habitación estimada WC/Baño/Bano.',
    'ducha': 'Baño observado y subida de humedad respecto a lectura conocida anterior al tramo de baño; admite retraso limitado. Se localiza la hipótesis en el último minuto de baño observado.',
    'bedtime': 'Modo Bedtime=1.',
    'descansar': 'Pasos observados=0 y HR observada <= umbral, con soporte temporal flexible y duración mínima.',
    'ejercicio': 'Pasos por minuto y HR observados >= sus umbrales, con duración mínima y 80% de soporte temporal configurable. No exige pendiente positiva.',
    'salir': 'Sensor entrada=1: evento compatible con salida o entrada, dirección sin verificar. No determina tiempo fuera.',
}


def infer_activities(history: History, *, activity: str = 'all', rules: HarRules | None = None,
                     t0: str | None = None, window: str = 'hoy',
                     start: str | None = None, end: str | None = None,
                     offset: int = 0, limit: int = 50) -> dict[str, Any]:
    if activity not in (*ACTIVITIES, 'all'):
        raise ValueError('Actividad HAR no permitida')
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (offset, limit)) or offset < 0 or not 1 <= limit <= 100:
        raise ValueError('Paginación HAR inválida')
    cfg = rules or HarRules()
    a, b, ref = history.interval(t0, window, start, end)
    names = ACTIVITIES if activity == 'all' else (activity,)
    states = {name: {t: 'unknown' for t in range(a, b, 60)} for name in names}
    evidence: dict[str, dict[int, dict]] = {name: {} for name in names}
    checks = {'bathroom_observed_minutes': 0, 'humidity_observed_minutes': 0,
              'known_reference_checks': 0, 'rise_checks_passed': 0,
              'kitchen_observed_minutes': 0, 'rest_candidate_minutes': 0, 'exercise_candidate_minutes': 0}

    def mark(name, t, kind, facts=None):
        if name in states and a <= t < b:
            states[name][t] = kind
            if kind == 'match':
                evidence[name][t] = facts or {}

    # Only look back as needed for a known humidity reference / sustained rest.
    lookback = max(cfg.humidity_reference_minutes+cfg.humidity_delay_minutes,
                   cfg.rest_min_minutes, cfg.exercise_min_minutes)*60
    humidity = []
    last_bath = bath_start = None
    for t in range(a-lookback, b, 60):
        row = history.rows.get(t, {})
        room = row.get('habitacion', '')
        bath = room.casefold() in ('wc', 'baño', 'bano')
        if bath:
            if a <= t < b:
                checks['bathroom_observed_minutes'] += 1
            if last_bath != t-60:
                bath_start = t
            last_bath = t
            mark('bano', t, 'match', {'habitacion': room})
        elif room:
            mark('bano', t, 'contradiction')
        if room.casefold() == 'cocina':
            if a <= t < b:
                checks['kitchen_observed_minutes'] += 1
            used = {k: number(row, k) for k in ('cocina_vasos13', 'cocina_platos4') if number(row, k) == 1}
            light = number(row, 'cocina_frigo_luz24_lx')
            if light is not None and light >= cfg.fridge_lux:
                used['cocina_frigo_luz24_lx'] = light
            if used:
                mark('cocinar', t, 'match', {'habitacion': room, **used})
            elif all(number(row, k) is not None for k in ('cocina_vasos13', 'cocina_platos4', 'cocina_frigo_luz24_lx')):
                mark('cocinar', t, 'nonmatch')
        elif room:
            mark('cocinar', t, 'nonmatch', {'habitacion': room})

        bedtime = number(row, WATCH+'bedtime')
        if bedtime is not None:
            mark('bedtime', t, 'match' if bedtime == 1 else 'contradiction', {WATCH+'bedtime': bedtime})
        steps, hr = number(row, WATCH+'steps'), number(row, WATCH+'hr_bpm')
        if steps == 0 and hr is not None and hr <= cfg.rest_hr_max_bpm:
            if a <= t < b:
                checks['rest_candidate_minutes'] += 1
            mark('descansar', t, 'match', {WATCH+'steps': steps, WATCH+'hr_bpm': hr})
        elif (steps is not None and steps > 0) or (hr is not None and hr > cfg.rest_hr_max_bpm):
            mark('descansar', t, 'nonmatch')
        if steps is not None and hr is not None and steps >= cfg.exercise_steps_min and hr >= cfg.exercise_hr_min_bpm:
            if a <= t < b:
                checks['exercise_candidate_minutes'] += 1
            mark('ejercicio', t, 'match', {WATCH+'steps': steps, WATCH+'hr_bpm': hr})
        elif (steps is not None and steps < cfg.exercise_steps_min) or (hr is not None and hr < cfg.exercise_hr_min_bpm):
            mark('ejercicio', t, 'nonmatch')
        h = number(row, 'bano_humedad7_pct')
        if h is not None:
            if a <= t < b:
                checks['humidity_observed_minutes'] += 1
            if last_bath is not None and t-last_bath <= cfg.humidity_delay_minutes*60:
                prior = [(u, v) for u, v in humidity if bath_start-cfg.humidity_reference_minutes*60 <= u < bath_start]
                if prior and a <= t < b:
                    checks['known_reference_checks'] += 1
                if prior and h-prior[-1][1] >= cfg.humidity_rise_pct_points:
                    if a <= t < b:
                        checks['rise_checks_passed'] += 1
                    mark('ducha', last_bath, 'match', {
                        'bath_observed_at': iso(last_bath), 'humidity_observed_at': iso(t),
                        'humidity_pct': h, 'reference_at': iso(prior[-1][0]),
                        'reference_pct': prior[-1][1], 'rise_pct_points': h-prior[-1][1]})
            humidity.append((t, h))
        if number(row, 'entrada_binario7') == 1:
            mark('salir', t, 'match', {'entrada_binario7': 1})
        elif number(row, 'entrada_binario7') == 0:
            mark('salir', t, 'contradiction')

    episodes = []
    totals = {}
    for name in names:
        # Other observed states interrupt support, but may be bridged as explicit gaps.
        flexible_states = {t: ('nonmatch' if value == 'contradiction' else value)
                           for t, value in states[name].items()}
        runs = segment_states(flexible_states, 0 if name == 'salir' else cfg.merge_gap, cfg.max_total_gap)
        minimum = cfg.rest_min_minutes if name == 'descansar' else cfg.exercise_min_minutes if name == 'ejercicio' else 1
        # A short core can meet 80% over a full minimum-length window (e.g. 4/5).
        # Include that explicitly observed/unknown margin without counting it as support.
        eligible = []
        for e in runs:
            if e['duration_minutes'] < minimum:
                first, last = stamp(e['start']), stamp(e['end'])
                windows = []
                for left in range(max(a, last-minimum*60), min(first, b-minimum*60)+1, 60):
                    values = [flexible_states[t] for t in range(left, left+minimum*60, 60)]
                    if values.count('match')/minimum >= cfg.min_support_fraction:
                        windows.append((left, values))
                if not windows:
                    continue
                left, values = min(windows, key=lambda w: abs(w[0]-first))
                e.update(start=iso(left), end=iso(left+minimum*60), duration_minutes=minimum,
                         matching_minutes=values.count('match'), unknown_minutes=values.count('unknown'),
                         nonmatching_minutes=values.count('nonmatch'), gaps=[],
                         boundary_censored=left == a or left+minimum*60 == b)
                for index, value in enumerate(values):
                    if value != 'match':
                        t = left+index*60
                        e['gaps'].append({'start': iso(t), 'end': iso(t+60), 'kind': value})
            eligible.append(e)
        runs = eligible
        for e in runs:
            e['activity'] = name
            e['support_fraction'] = e['matching_minutes']/e['duration_minutes']
            e['required_support_fraction'] = cfg.min_support_fraction if name != 'salir' else None
            e['meets_support_threshold'] = e['support_fraction'] >= cfg.min_support_fraction if name != 'salir' else True
            e['classification'] = 'tramo con soporte' if e['meets_support_threshold'] else 'candidato para interpretación'
            e['support_threshold_applies'] = name != 'salir'
            e['rule'] = RULE_DESCRIPTIONS[name]
            e['evidence_minutes'] = e.pop('matching_minutes')
            e['interruption_minutes'] = e.pop('nonmatching_minutes')
            support = [(t, facts) for t, facts in evidence[name].items()
                       if stamp(e['start']) <= t < stamp(e['end'])]
            sample = support if len(support) <= 4 else support[:2]+support[-2:]
            e['evidence_sample'] = [{'at': iso(t), **facts, 'source': history.rows[t]['_source']}
                                    for t, facts in sample]
            e['evidence_sample_count'] = len(sample)
            e['evidence_sample_is_complete'] = len(sample) == len(support)
            for gap in e['gaps']:
                if gap['kind'] == 'nonmatch':
                    gap['rooms_observed_minutes'] = dict(Counter(
                        history.rows.get(t, {}).get('habitacion') for t in range(stamp(gap['start']), stamp(gap['end']), 60)
                        if history.rows.get(t, {}).get('habitacion')))
            e['warnings'] = ['Umbrales experimentales sin validación HAR ni calibración individual.']
            if e['duration_minutes']-e['evidence_minutes'] > e['duration_minutes']*.4:
                e['warnings'].append('Más del 40% del tramo son huecos/interrupciones; no es actividad sostenida.')
            if name == 'cocinar':
                e['warnings'].append('Etiqueta operativa: uso de cocina.')
            if name == 'bedtime':
                e['warnings'].append('Fuente: modo Bedtime del SO.')
            if name == 'ejercicio':
                e['warnings'].append('Compatible con actividad física ambulatoria; no confirma deporte, intensidad clínica ni detecta necesariamente ejercicio sin pasos. Umbrales provisionales.')
            if name == 'salir':
                e['warnings'].append('Duración del evento de entrada; no es duración fuera de casa ni número real de aperturas.')
            if name == 'ducha':
                e['warnings'].append('Hipótesis situada en baño observado; evidencia de humedad puede ser posterior, siempre anterior al fin consultado. No mide duración real de ducha.')
            episodes.append(e)
        totals[name] = {
            'rule_support_minutes': sum(e['evidence_minutes'] for e in runs),
            'predicate_match_minutes': sum(s == 'match' for s in states[name].values()),
            'supported_episode_evidence_minutes': sum(e['evidence_minutes'] for e in runs if e['meets_support_threshold']),
            'candidate_episode_evidence_minutes': sum(e['evidence_minutes'] for e in runs if not e['meets_support_threshold']),
            'unknown_minutes': sum(s == 'unknown' for s in states[name].values()),
            'observed_other_minutes': sum(s in ('nonmatch', 'contradiction') for s in states[name].values()),
        }
    episodes.sort(key=lambda e: (e['start'], e['activity']))
    result = history.base(a, b, ref)
    result.update(rules=asdict(cfg), rule_descriptions={n: RULE_DESCRIPTIONS[n] for n in names},
                  episode_count=len(episodes), episodes=episodes[offset:offset+limit], totals=totals,
                  prerequisite_checks=checks,
                  next_offset=offset+limit if offset+limit < len(episodes) else None,
                  outside_duration_minutes=None,
                  supported_episode_count=sum(e['meets_support_threshold'] for e in episodes),
                  candidate_episode_count=sum(not e['meets_support_threshold'] for e in episodes),
                  status=('tramos con soporte' if any(e['meets_support_threshold'] for e in episodes)
                          else 'candidatos para interpretación' if episodes else 'evidencia insuficiente'))
    result['warnings'].extend([
        'Reglas HAR operativas: no equivalen a hechos medidos ni precisión validada.',
        'No inferir ausencia o desconexión por huecos, reposo o red. Salir solo usa el sensor de entrada.',
        'Ducha y Baño pueden solaparse. Para totales exclusivos usar unión de minutos o una prioridad explícita; nunca sumar duraciones globales.',
    ])
    return result
