"""Trusted temporal context for the metrics agent; no model calendar arithmetic."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import unicodedata

from .historical_har import ZONE, iso, stamp


def plain(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')


def reference_time(t0: str | None, question_time: str | None, offline_fallback: int) -> tuple[int, str]:
    if t0 is not None:
        return stamp(t0), 'explicit_t0'
    if question_time is not None:
        recorded = datetime.fromisoformat(question_time.replace('Z', '+00:00'))
        if recorded.tzinfo is None:
            raise ValueError('El timestamp del registro de pregunta debe incluir zona horaria.')
        local = recorded.astimezone(ZONE).replace(second=0, microsecond=0)
        return stamp(local.isoformat()), 'question_record'
    return offline_fallback, 'latest_data_offline'


@dataclass(frozen=True)
class QueryWindow:
    cutoff: int
    start: int
    end: int
    label: str

    def context(self):
        return {'t0': iso(self.cutoff), 'window': self.label, 'start': iso(self.start),
                'end': iso(self.end), 'minutes': (self.end-self.start)//60, 'timezone': 'Europe/Madrid'}


def resolve_window(question: str, cutoff: int, *, window: str | None = None,
                   window_minutes: int | None = None, start: str | None = None,
                   end: str | None = None) -> QueryWindow:
    """Explicit caller context wins; otherwise resolve a small relative vocabulary.

    Unrecognized explicit dates require caller context instead of silently using today.
    'ayer' uses calendar midnights; elapsed durations use UTC seconds across DST.
    """
    text = plain(question)
    if re.search(r'\b(dare|hare|manana|pasado manana)\b', text) and 'esta manana' not in text:
        raise ValueError('La consulta se refiere al futuro; solo hay observaciones hasta t0.')
    explicit = int(window is not None)+int(window_minutes is not None)+int(start is not None or end is not None)
    if explicit > 1:
        raise ValueError('Indica una sola ventana: window, window_minutes o start/end.')
    if (start is None) != (end is None):
        raise ValueError('El contexto externo debe incluir inicio y fin juntos.')
    local = datetime.fromtimestamp(cutoff, ZONE)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if start is not None:
        a, b, label = stamp(start), stamp(end), 'intervalo fijado externamente'
    else:
        if explicit == 0:
            if re.search(r'\banteayer\b', text):
                raise ValueError('Indica la ventana para anteayer en el contexto externo.')
            if re.search(r'\bayer\b', text):
                window = 'ayer'
            else:
                words = {'dos': 2, 'tres': 3, 'cinco': 5, 'diez': 10, 'quince': 15,
                         'veinte': 20, 'veinticuatro': 24, 'treinta': 30, 'sesenta': 60}
                match = re.search(r'ultim[oa]s?\s+(\d+|'+ '|'.join(words)+r')\s+(minutos?|horas?)', text)
                if match:
                    n = int(match[1]) if match[1].isdigit() else words[match[1]]
                    window_minutes = n*(60 if match[2].startswith('hora') else 1)
                elif re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}|\b(octubre|noviembre|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|diciembre)\b', text):
                    raise ValueError('Para fechas explícitas proporciona t0 y la ventana desde el contexto externo.')
                else:
                    clock = re.search(r'desde (?:las? )?(\d{1,2}):(\d{2})', text)
                    if clock:
                        h, m = int(clock[1]), int(clock[2])
                        if not 0 <= h < 24 or not 0 <= m < 60:
                            raise ValueError('Hora local inválida.')
                        candidate = midnight.replace(hour=h, minute=m)
                        if candidate.replace(fold=0).utcoffset() != candidate.replace(fold=1).utcoffset():
                            raise ValueError('Hora local ambigua o inexistente; fija el intervalo en el contexto externo.')
                        a, b = int(candidate.timestamp()), cutoff
                        if a >= b:
                            raise ValueError('El horario solicitado está fuera del pasado disponible hasta t0.')
                        return QueryWindow(cutoff, a, b, 'desde hora local hasta t0')
                    if 'esta manana' in text or 'desde medianoche hasta las doce' in text:
                        a, b = int(midnight.timestamp()), min(cutoff, int(midnight.replace(hour=12).timestamp()))
                        if a >= b:
                            raise ValueError('La ventana de esta mañana está vacía hasta t0.')
                        return QueryWindow(cutoff, a, b, 'esta mañana')
                    if re.search(r'\b(hace|anoche|tarde|noche|entre las|antes de|despues de)\b|\d{1,2}:\d{2}', text):
                        raise ValueError('Esa referencia temporal requiere una ventana fijada por el llamador.')
                    window = 'hoy'
        if window_minutes is not None:
            if isinstance(window_minutes, bool) or not isinstance(window_minutes, int) or not 1 <= window_minutes <= 525600:
                raise ValueError('window_minutes debe ser un entero positivo, hasta un año.')
            a, b, label = cutoff-window_minutes*60, cutoff, f'últimos {window_minutes} minutos'
        elif window == 'hoy':
            a, b, label = int(midnight.timestamp()), cutoff, 'hoy'
        elif window == 'ayer':
            a, b, label = int((midnight-timedelta(days=1)).timestamp()), int(midnight.timestamp()), 'ayer'
        elif window == 'ultimas_24h':
            a, b, label = cutoff-86400, cutoff, 'últimas 24 horas'
        else:
            raise ValueError('Ventana no admitida: hoy, ayer, ultimas_24h o window_minutes.')
    if a >= b or b > cutoff:
        raise ValueError('El intervalo debe ser pasado, no vacío y terminar como máximo en t0.')
    return QueryWindow(cutoff, a, b, label)


def select_relative_window(base: QueryWindow, period: str = 'referencia', *,
                           minutes: int | None = None, day_offset: int = 0,
                           anchor_offset_minutes: int = 0) -> QueryWindow:
    """Resolve a relative request chosen by Qwen, preserving the question's cutoff."""
    for name, value in [('day_offset', day_offset), ('anchor_offset_minutes', anchor_offset_minutes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value > 0:
            raise ValueError(name+' debe ser un entero no positivo; no se consulta el futuro.')
    if day_offset < -365 or anchor_offset_minutes < -525600:
        raise ValueError('Desplazamiento fuera del rango de un año.')
    if period != 'dia_relativo' and day_offset:
        raise ValueError('day_offset solo corresponde a period=dia_relativo.')
    if period != 'ultimos_minutos' and minutes is not None:
        raise ValueError('minutes solo corresponde a period=ultimos_minutos.')
    anchor = base.cutoff+anchor_offset_minutes*60
    if period == 'referencia':
        return QueryWindow(base.cutoff, base.start+anchor_offset_minutes*60,
                           base.end+anchor_offset_minutes*60, base.label)
    if period in ('hoy', 'ayer'):
        selected = resolve_window('', anchor, window=period)
    elif period == 'ultimos_minutos':
        if minutes is None:
            raise ValueError('Indica minutes para ultimos_minutos.')
        selected = resolve_window('', anchor, window_minutes=minutes)
    elif period == 'dia_relativo':
        local = datetime.fromtimestamp(anchor, ZONE)
        begin = local.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=day_offset)
        finish = min(int((begin+timedelta(days=1)).timestamp()), anchor)
        selected = QueryWindow(anchor, int(begin.timestamp()), finish, f'día relativo {day_offset}')
        if selected.start >= selected.end:
            raise ValueError('Día relativo vacío hasta la referencia.')
    else:
        raise ValueError('Periodo relativo no admitido.')
    return QueryWindow(base.cutoff, selected.start, selected.end, selected.label)
