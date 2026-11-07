"""Deterministic, read-only minute tools for the HERA reduced export.

All arithmetic uses UTC seconds; local calendar boundaries use Europe/Madrid.
No carry-forward, interpolation, device access, or dynamic code execution.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import Counter
from statistics import median
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from .har_rules import HarRules

ZONE = ZoneInfo('Europe/Madrid')
WATCH = 'watch_01.'
SIGNALS = {
    WATCH+'steps': 'steps', WATCH+'hr_bpm': 'bpm',
    'cocina_frigo_luz24_lx': 'lx', 'bano_humedad7_pct': '%',
    'salon_consumo14_w': 'W', 'cocina_vasos13': 'binary',
    'cocina_platos4': 'binary', 'entrada_binario7': 'binary',
    WATCH+'bedtime': 'binary', WATCH+'network_evidence': 'code',
}
MEDIAN_WINDOWS = {
    'bano_humedad7_pct': (120, 'bano_humedad_2h', 'bano_humedad_delta_2h_pp'),
    'salon_consumo14_w': (1440, 'tv_power_24h', 'tv_power_delta_24h_w'),
}
REFERENCE_EXPORT = 'all_data_72hours_reference.txt'


def median_columns(column: str) -> dict[str, str]:
    _, prefix, delta = MEDIAN_WINDOWS[column]
    return {
        'median': prefix+'_median',
        'observed_values': prefix+'_observations',
        'current_at': prefix+'_current_at',
        'first_observed_at': prefix+'_first_at',
        'median_minus_current': delta,
    }


SOURCE_COLUMNS = {
    'timestamp', 'timestamp_unix', 'fecha', 'minuto', 'watch_id', 'habitacion',
    'ancla_minima', 'distancia_minima_m', *SIGNALS,
    *(WATCH+k for k in ('battery_max_percent', 'charging_observed', 'ble_gateway',
                       'ble_rssi_dbm', 'store', 'source_minute', 'revision')),
}
SOURCE_COLUMNS.update(name for signal in MEDIAN_WINDOWS for name in median_columns(signal).values())
WARNINGS = [
    'Vacío es desconocido; no se interpolan ni arrastran valores.',
    'Habitación UWB estimada; no es HAR validado ni identifica a la persona.',
    'Ambientales: máximos observados de cambios; indicadores de retenidos omitidos en el export reducido.',
    'HR: última lectura de cada minuto; Bedtime: modo del SO.',
    'Red: 1 contacto, 2 reposo anunciado, 3 desconexión explícita, 4 Wi-Fi sin verificar.',
    't0 limita tiempo de observación, no reconstruye las revisiones conocidas históricamente en t0.',
]


def iso(stamp: int) -> str:
    return datetime.fromtimestamp(stamp, ZONE).isoformat()


def stamp(value: str) -> int:
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.second or dt.microsecond:
        raise ValueError('Usa ISO-8601 con desfase explícito y minutos completos; aclara fechas ambiguas.')
    local = dt.astimezone(ZONE)
    if dt.utcoffset() != local.utcoffset():
        raise ValueError('El desfase debe corresponder a Europe/Madrid en esa fecha.')
    return int(dt.timestamp())


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ''):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Valor no finito en {key}')
    return result


class History:
    """One immutable optimistic snapshot per question, including source hashes."""

    def __init__(self, directory: Path | str, har_rules: HarRules | None = None):
        self.directory = Path(directory)
        self.har_rules = har_rules
        self.rows: dict[int, dict] = {}
        self.sources: list[dict] = []
        def input_paths():
            if self.directory.is_file():
                return [self.directory]
            daily = sorted(self.directory.glob('????-??-??.tsv'))
            consolidated = self.directory/REFERENCE_EXPORT
            if not consolidated.is_file():
                consolidated = self.directory/'all_data_72hours.txt'
            if daily and consolidated.is_file():
                raise ValueError('Hay export diario y consolidado: indica el archivo o directorio exacto')
            return daily or ([consolidated] if consolidated.is_file() else [])
        for _ in range(3):
            paths = input_paths()
            if not paths:
                raise FileNotFoundError('No hay export tabulado en la fuente configurada')
            try:
                before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
                blobs = [p.read_bytes() for p in paths]
                after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
                if before == after and paths == input_paths():
                    break
            except OSError:
                continue
        else:
            raise RuntimeError('Exportación cambiante: vuelve a consultar; no se ha bloqueado ni modificado')
        for path, blob in zip(paths, blobs):
            if not blob.endswith(b'\n'):
                raise ValueError('TSV posiblemente incompleto: '+path.name)
            self.sources.append({'file': path.name, 'sha256': hashlib.sha256(blob).hexdigest()})
            reader = csv.DictReader(io.StringIO(blob.decode('utf-8-sig')), delimiter='\t')
            # The clean 72h export has 17 columns and omits optional network/store metadata.
            required = {'timestamp', 'timestamp_unix', 'watch_id', 'habitacion'} | (set(SIGNALS)-{WATCH+'network_evidence'})
            if not required <= set(reader.fieldnames or []):
                raise ValueError('Esquema incompleto: '+path.name)
            for line, row in enumerate(reader, 2):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Fila incompleta: '+path.name)
                # Extra scenario/annotation fields are not observations or agent inputs.
                row = {k: v for k, v in row.items() if k in SOURCE_COLUMNS}
                t = stamp(row['timestamp'])
                if t != int(row['timestamp_unix']) or t % 60 or row['watch_id'] != 'watch_01':
                    raise ValueError('Timestamp o reloj incompatible')
                for key in SIGNALS:
                    value = number(row, key)
                    if value is not None and SIGNALS[key] == 'binary' and value not in (0, 1):
                        raise ValueError('Binario distinto de 0/1: resolver en preprocesamiento externo')
                    if value is not None and SIGNALS[key] == 'code' and value not in (1, 2, 3, 4):
                        raise ValueError('Código de red fuera de esquema: resolver en preprocesamiento externo')
                revision = int(row[WATCH+'revision']) if row.get(WATCH+'revision') else None
                if row[WATCH+'steps'] and number(row, WATCH+'steps') < 0:
                    raise ValueError('Pasos negativos')
                previous = self.rows.get(t)
                provenance = {'file': path.name, 'line': line, 'revision': revision,
                              'store': row.get(WATCH+'store') or None}
                if previous:
                    raise ValueError(f'Minuto duplicado/conflictivo {iso(t)}: resolver en preprocesamiento externo')
                self.rows[t] = dict(row, _source=provenance)
        if not self.rows:
            raise ValueError('Exportación vacía')
        # Last completed minute endpoint: all included records end at or before t0.
        self.latest = max(self.rows)+60

    def interval(self, t0: str | None = None, window: str = 'hoy',
                 start: str | None = None, end: str | None = None) -> tuple[int, int, int]:
        ref = stamp(t0) if t0 else self.latest
        local = datetime.fromtimestamp(ref, ZONE)
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        if (start is None) != (end is None):
            raise ValueError('Indica inicio y fin juntos')
        if start is not None:
            a, b = stamp(start), stamp(end)
        elif window == 'ultimas_24h':
            a, b = ref-86400, ref
        elif window == 'hoy':
            a, b = int(midnight.timestamp()), ref
        elif window == 'ayer':
            a, b = int((midnight-timedelta(days=1)).timestamp()), int(midnight.timestamp())
        else:
            raise ValueError('Ventana: hoy, ayer o ultimas_24h; para otras fechas indica start/end')
        if a >= b or b > ref:
            raise ValueError('Intervalo vacío, invertido o posterior a t0')
        return a, b, ref

    def base(self, a: int, b: int, ref: int) -> dict[str, Any]:
        count = sum(a <= t < b for t in self.rows)
        return {'interval': {'start': iso(a), 'end': iso(b), 't0': iso(ref), 'closure': '[start,end)'},
                'coverage': {'expected_minutes': (b-a)//60, 'rows': count},
                'units': {'duration': 'min'}, 'warnings': WARNINGS.copy(),
                'snapshot': self.sources}

    def resumir_intervalo(self, t0: str | None = None, window: str = 'hoy',
                         start: str | None = None, end: str | None = None,
                         measure: str | None = None) -> dict[str, Any]:
        if measure is not None and measure not in (*SIGNALS, 'habitacion'):
            raise ValueError('Medida no permitida')
        a, b, ref = self.interval(t0, window, start, end)
        result = self.base(a, b, ref)
        rows = [r for t, r in sorted(self.rows.items()) if a <= t < b]
        metrics = {}
        for key, unit in SIGNALS.items():
            if measure is not None and key != measure:
                continue
            values = [v for r in rows if (v := number(r, key)) is not None]
            stat = {'unit': unit, 'observed_minutes': len(values),
                    'unknown_minutes': (b-a)//60-len(values)}
            if values:
                if unit == 'binary':
                    stat['active_minutes'] = sum(v == 1 for v in values)
                elif unit == 'code':
                    stat['counts'] = dict(Counter(str(int(v)) for v in values))
                else:
                    stat.update(mean=sum(values)/len(values), min=min(values), max=max(values))
                    if key == WATCH+'steps':
                        stat['observed_sum'] = sum(values)
            metrics[key] = stat
        result['metrics'] = metrics
        result['rooms_observed_minutes'] = dict(Counter(r['habitacion'] for r in rows if r['habitacion']))
        result['room_transitions'] = [
            {'at': iso(t), 'from': self.rows[t-60]['habitacion'], 'to': r['habitacion']}
            for t, r in sorted(self.rows.items()) if a < t < b and t-60 in self.rows
            and r['habitacion'] and self.rows[t-60]['habitacion']
            and r['habitacion'] != self.rows[t-60]['habitacion']]
        if measure is not None and measure != 'habitacion':
            result.pop('rooms_observed_minutes')
            result.pop('room_transitions')
        return result

    def consultar_minutos(self, columns: list[str], t0: str | None = None, window: str = 'hoy',
                          start: str | None = None, end: str | None = None,
                          offset: int = 0, limit: int = 60) -> dict[str, Any]:
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 120:
            raise ValueError('Paginación inválida; limit entre 1 y 120')
        if not columns or set(columns)-set(next(iter(self.rows.values()))) or any(k.startswith('_') for k in columns):
            raise ValueError('Columnas no permitidas')
        a, b, ref = self.interval(t0, window, start, end)
        selected = [(t, r) for t, r in sorted(self.rows.items()) if a <= t < b]
        result = self.base(a, b, ref)
        result['records'] = [dict(timestamp=iso(t), values={k: r[k] or None for k in columns},
                                  source=r['_source']) for t, r in selected[offset:offset+limit]]
        result['total_records'] = len(selected)
        result['next_offset'] = offset+limit if offset+limit < len(selected) else None
        result['units'].update({k: SIGNALS.get(k, 'source value') for k in columns})
        return result

    def buscar_episodios(self, conditions: list[dict[str, Any]], mode: str = 'all',
                        t0: str | None = None, window: str = 'hoy',
                        start: str | None = None, end: str | None = None,
                        merge_gap: int = 0, max_total_gap: int = 9,
                        offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Runs matching explicit predicates, without assigning human activities.

        Missing values yield unknown (three-valued logic). Bridging is opt-in;
        nonmatching and unknown minutes remain separately accounted for.
        """
        if mode not in ('all', 'any') or not conditions or len(conditions) > 10:
            raise ValueError('Usa entre 1 y 10 condiciones y mode all/any')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in (merge_gap, max_total_gap, offset, limit)) or not 1 <= limit <= 100 or merge_gap > 60:
            raise ValueError('Paginación o huecos inválidos; limit 1..100, merge_gap 0..60')
        allowed = set(SIGNALS) | {'habitacion'}
        for c in conditions:
            if not isinstance(c, dict) or set(c) != {'column', 'op', 'value'} or c['column'] not in allowed:
                raise ValueError('Condición: column, op, value; columna no permitida')
            if c['op'] not in ('eq', 'ne', 'gt', 'ge', 'lt', 'le'):
                raise ValueError('Operador no permitido')
            if c['column'] == 'habitacion':
                if c['op'] not in ('eq', 'ne') or not isinstance(c['value'], str) or not c['value']:
                    raise ValueError('Habitación admite eq/ne con texto exacto')
            elif isinstance(c['value'], bool) or not isinstance(c['value'], (int, float)) or not math.isfinite(c['value']):
                raise ValueError('La condición numérica exige número JSON sin comillas: value: 1, no value: "1"')
        a, b, ref = self.interval(t0, window, start, end)
        from operator import eq, ne, gt, ge, lt, le
        operators = dict(eq=eq, ne=ne, gt=gt, ge=ge, lt=lt, le=le)
        states: dict[int, str] = {}
        for t in range(a, b, 60):
            row = self.rows.get(t, {})
            values = []
            for c in conditions:
                raw = row.get(c['column'])
                value = raw if c['column'] == 'habitacion' else number(row, c['column'])
                values.append(None if value in (None, '') else operators[c['op']](value, c['value']))
            if mode == 'all':
                match = False if False in values else (None if None in values else True)
            else:
                match = True if True in values else (None if None in values else False)
            states[t] = 'unknown' if match is None else ('match' if match else 'nonmatch')
        episodes = segment_states(states, merge_gap, max_total_gap)
        result = self.base(a, b, ref)
        result.update(conditions=conditions, mode=mode, merge_gap=merge_gap, max_total_gap=max_total_gap,
                      matching_minutes=sum(s == 'match' for s in states.values()),
                      unknown_minutes=sum(s == 'unknown' for s in states.values()),
                      nonmatching_minutes=sum(s == 'nonmatch' for s in states.values()),
                      episode_count=len(episodes), episodes=episodes[offset:offset+limit],
                      next_offset=offset+limit if offset+limit < len(episodes) else None)
        result['warnings'].append('Tramos de condiciones explícitas, sin etiquetas HAR. Duración global incluye huecos; solo matching_minutes cumple la condición. No sumar tramos solapados para totales exclusivos.')
        return result

    def inferir_actividades(self, activity: str = 'all', t0: str | None = None,
                           window: str = 'hoy', start: str | None = None,
                           end: str | None = None, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Separate, explicitly labelled rule layer; raw metric tools stay objective."""
        from .har_rules import infer_activities
        return infer_activities(self, activity=activity, rules=self.har_rules, t0=t0,
                                window=window, start=start, end=end, offset=offset, limit=limit)

    def desviacion_mediana_movil(self, column: str, t0: str | None = None,
                                 window_minutes: int | None = None) -> dict[str, Any]:
        """Trailing observed median minus latest observed value, including current.

        Humidity defaults to 2h and TV power to 24h. Available readings are used
        even with a partial history; no density gate, carry-forward or interpolation.
        """
        if column not in MEDIAN_WINDOWS:
            raise ValueError('Referencia móvil disponible para humedad y potencia de la televisión.')
        default_width = MEDIAN_WINDOWS[column][0]
        width = default_width if window_minutes is None else window_minutes
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 10080:
            raise ValueError('window_minutes debe ser entero entre 1 y 10080.')
        ref = stamp(t0) if t0 else self.latest
        a = ref-width*60
        stored = self.rows.get(ref-60, {})
        fields = median_columns(column)
        precalculated = width == default_width and set(fields.values()) <= stored.keys()
        if precalculated:
            baseline = number(stored, fields['median'])
            count = int(stored[fields['observed_values']])
            first_at = stored[fields['first_observed_at']] or None
            current_at = stored[fields['current_at']] or None
            difference = number(stored, fields['median_minus_current'])
            latest = None
            if current_at is not None:
                current = stamp(current_at)
                value = number(self.rows.get(current, {}), column)
                if not a <= current < ref or value is None:
                    raise ValueError('Lectura de referencia precalculada inválida.')
                latest = current, value
        else:
            points = [(t, value) for t, row in sorted(self.rows.items()) if a <= t < ref
                      and (value := number(row, column)) is not None]
            count = len(points)
            baseline = median(value for _, value in points) if points else None
            latest = points[-1] if points else None
            first_at = iso(points[0][0]) if points else None
            difference = baseline-latest[1] if latest else None
        result = self.base(a, ref, ref)
        result.update(column=column, window_minutes=width,
                      formula='median_observed_window - current_value', includes_current_in_median=True,
                      observed_values=count,
                      available_export_minutes=result['coverage']['rows'],
                      median=baseline, current_value=latest[1] if latest else None,
                      current_at=iso(latest[0]) if latest else None,
                      current_age_minutes=(ref-latest[0])/60 if latest else None,
                      first_observed_at=first_at,
                      median_minus_current=difference,
                      status='available' if count else 'no_observations')
        result['units'].update(level=SIGNALS[column],
                               difference='percentage_points' if column == 'bano_humedad7_pct' else 'W')
        return result

    def analizar_cambios(self, column: str, t0: str | None = None, window: str = 'hoy',
                         start: str | None = None, end: str | None = None,
                         reference_minutes: int = 60) -> dict[str, Any]:
        """Differences between real observed readings, without requiring dense sampling."""
        if column not in SIGNALS or SIGNALS[column] in ('binary', 'code'):
            raise ValueError('Los cambios requieren una señal numérica, no códigos/binarios.')
        if isinstance(reference_minutes, bool) or not isinstance(reference_minutes, int) or not 1 <= reference_minutes <= 1440:
            raise ValueError('reference_minutes debe estar entre 1 y 1440.')
        a, b, ref = self.interval(t0, window, start, end)
        points = [(t, v) for t, row in sorted(self.rows.items()) if a <= t < b
                  and (v := number(row, column)) is not None]
        previous = [(t, v) for t, row in sorted(self.rows.items()) if a-reference_minutes*60 <= t < a
                    and (v := number(row, column)) is not None]
        reference = previous[-1:] if points else []
        chain = reference+points
        def pair(p, q):
            return {'from_at': iso(p[0]), 'to_at': iso(q[0]), 'previous': p[1], 'current': q[1],
                    'difference': q[1]-p[1], 'elapsed_minutes': (q[0]-p[0])/60,
                    'reference_before_window': p[0] < a}
        changes = [pair(p, q) for p, q in zip(chain, chain[1:]) if 0 < q[0]-p[0] <= reference_minutes*60]
        latest = changes[-1] if changes and changes[-1]['to_at'] == iso(points[-1][0]) else None
        increases = [c for c in changes if c['difference'] > 0]
        decreases = [c for c in changes if c['difference'] < 0]
        result = self.base(a, b, ref)
        unit = 'steps/min' if column == WATCH+'steps' else SIGNALS[column]
        result.update(column=column, observed_minutes=len(points), reference_minutes=reference_minutes,
                      latest_change=latest,
                      largest_rise=max(increases, key=lambda c: c['difference']) if increases else None,
                      largest_drop=min(decreases, key=lambda c: c['difference']) if decreases else None,
                      first_last_difference=points[-1][1]-points[0][1] if len(points) >= 2 else None,
                      status='observed_change' if latest is not None else 'insufficient_reference')
        result['units'].update(level=unit, difference='percentage_points' if unit == '%' else unit)
        result['warnings'].append('Diferencias de lecturas observadas; no implican estado continuo entre ellas ni se rellenan huecos.')
        return result

    def analizar_tendencia(self, column: str, window_minutes: int = 10,
                          t0: str | None = None, min_coverage: float = .8) -> dict[str, Any]:
        """Observed linear slope and change between two adjacent trailing windows.

        Uses actual elapsed sample times, not row positions or interpolated data.
        No result is called an instantaneous physiological derivative.
        """
        if column not in SIGNALS or SIGNALS[column] in ('binary', 'code'):
            raise ValueError('La tendencia requiere una señal numérica continua o pasos')
        if isinstance(window_minutes, bool) or not isinstance(window_minutes, int) or not 2 <= window_minutes <= 1440:
            raise ValueError('window_minutes debe ser entero entre 2 y 1440')
        if isinstance(min_coverage, bool) or not isinstance(min_coverage, (float, int)) or not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError('min_coverage debe estar en (0,1]')
        ref = stamp(t0) if t0 else self.latest
        width = window_minutes*60
        unit = 'steps/min' if column == WATCH+'steps' else SIGNALS[column]
        delta_unit = 'percentage_points' if unit == '%' else unit

        def analyze(a, b):
            points = [(t, v) for t in range(a, b, 60)
                      if (v := number(self.rows.get(t, {}), column)) is not None]
            count = len(points)
            coverage = count/window_minutes
            sufficient = count >= 2 and coverage >= min_coverage
            output = {'start': iso(a), 'end': iso(b), 'observed_minutes': count,
                      'unknown_minutes': window_minutes-count, 'coverage_fraction': coverage,
                      'status': 'ok' if sufficient else 'insufficient_coverage',
                      'mean': sum(v for _, v in points)/count if count else None,
                      'first': {'at': iso(points[0][0]), 'value': points[0][1]} if count else None,
                      'last': {'at': iso(points[-1][0]), 'value': points[-1][1]} if count else None,
                      'change_first_last': None, 'observed_span_minutes': None,
                      'linear_slope_per_minute': None}
            if sufficient:
                xs = [(t-a)/60 for t, _ in points]
                xmean = sum(xs)/count
                ymean = output['mean']
                output['linear_slope_per_minute'] = sum((x-xmean)*(y-ymean) for x, (_, y) in zip(xs, points))/sum((x-xmean)**2 for x in xs)
                output['change_first_last'] = points[-1][1]-points[0][1]
                output['observed_span_minutes'] = (points[-1][0]-points[0][0])/60
            return output

        previous, current = analyze(ref-2*width, ref-width), analyze(ref-width, ref)
        comparable = previous['status'] == current['status'] == 'ok'
        result = self.base(ref-2*width, ref, ref)
        result.update(column=column, window_minutes=window_minutes, min_coverage=min_coverage,
                      method='ordinary_least_squares_on_observed_timestamps',
                      previous=previous, current=current,
                      change_of_window_means=current['mean']-previous['mean'] if comparable else None,
                      status='ok' if comparable else 'insufficient_comparison_coverage')
        result['units'].update(level=unit, change=delta_unit, slope=delta_unit+'/min')
        result['warnings'].extend([
            'Pendiente estimada sobre observaciones, no derivada instantánea ni muestras interpoladas.',
            'Los pasos ya son incrementos por minuto: su pendiente mide cambio de cadencia, no pasos acumulados.',
            'La cobertura mínima se aplica por ventana; una tendencia no identifica por sí sola ejercicio.',
        ])
        return result



def segment_states(states: dict[int, str], merge_gap: int, max_total_gap: int) -> list[dict]:
    """Group matched minutes, preserving unknown/observed gaps and hard contradictions."""
    groups: list[list[int]] = []
    for t, state in states.items():
        if state != 'match':
            continue
        if groups:
            gap = (t-groups[-1][-1])//60-1
            total_gap = (t-groups[-1][0])//60+1-len(groups[-1])-1
            if gap <= merge_gap and total_gap <= max_total_gap and not any(
                        states[u] == 'contradiction' for u in range(groups[-1][-1]+60, t, 60)):
                groups[-1].append(t)
                continue
        groups.append([t])
    episodes = []
    for group in groups:
        first, last = group[0], group[-1]+60
        counts = Counter(states[t] for t in range(first, last, 60))
        gaps = []
        for t in range(first, last, 60):
            state = states[t]
            if state == 'match':
                continue
            if gaps and gaps[-1]['kind'] == state and gaps[-1]['end'] == iso(t):
                gaps[-1]['end'] = iso(t+60)
            else:
                gaps.append({'start': iso(t), 'end': iso(t+60), 'kind': state})
        episodes.append({'start': iso(first), 'end': iso(last),
                         'duration_minutes': (last-first)//60,
                         'matching_minutes': counts['match'], 'unknown_minutes': counts['unknown'],
                         'nonmatching_minutes': counts['nonmatch'], 'gaps': gaps,
                         'boundary_censored': first == min(states) or last == max(states)+60})
    return episodes
