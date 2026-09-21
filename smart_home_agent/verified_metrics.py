"""Respuestas de medidas personales basadas en consultas, nunca en texto del LLM."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta

try:
    from .sensor_queries import query_aggregation
except ImportError:
    from sensor_queries import query_aggregation


def normalized(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


def metric_types(text: str) -> list[str]:
    text = normalized(text)
    kinds = []
    if re.search(r'\bpasos?\b', text):
        kinds.append('watch.steps')
    if re.search(r'\b(dorm\w*|sueno)\b', text):
        kinds.append('watch.sleep')
    return kinds


def requested_interval(text: str, t0: str) -> tuple[datetime, datetime]:
    """Solo interpreta intervalos explícitos soportados; no inventa noches o semanas."""
    moment = datetime.fromisoformat(t0.replace('Z', '+00:00'))
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    text = normalized(text)
    iso = re.findall(r'\d{4}-\d{2}-\d{2}t\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:\d{2})', text)
    if len(iso) == 2:
        start, end = (datetime.fromisoformat(s.replace('z', '+00:00')) for s in iso)
    else:
        times = re.findall(r'\b(\d{1,2}):(\d{2})\b', text)
        if re.search(r'\b(anoche|noche|seman\w*|mes\w*|anteayer|manana|ultim\w*|antes|despues|hace|dias|lunes|martes|miercoles|jueves|viernes|sabado|domingo|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|media|promedio|maxim\w*|minim\w*)\b|\d{4}-\d{2}-\d{2}', text):
            raise ValueError('Indica el inicio y el fin con dos timestamps ISO-8601 con zona horaria.')
        start, end = (day - timedelta(days=1), day) if 'ayer' in text else (day, moment)
        if times:
            if len(times) != 2:
                raise ValueError('Indica las dos horas del intervalo, por ejemplo entre 09:00 y 09:40.')
            base = start
            start, end = (base.replace(hour=int(h), minute=int(m)) for h, m in times)
        elif re.search(r'\b(entre|desde|hasta|de \d|a las|por la|esta tarde|esta madrugada)\b|\d', text):
            raise ValueError('Indica el intervalo con dos horas HH:MM o dos timestamps ISO-8601.')
    if start.tzinfo is None or end.tzinfo is None or end <= start or end > moment:
        raise ValueError('El intervalo debe tener zona horaria, inicio anterior al fin y no superar t0.')
    return start, end


def verified_answer(user_text: str, t0: str, messages: list[dict], trace=None) -> str | None:
    kinds = metric_types(user_text)
    # Consejos generales de sueño no son consultas de duración personal.
    text = normalized(user_text)
    if kinds == ['watch.sleep'] and not re.search(r'cuant|dormido|dormi\b|horas?|minutos?|sueno.*(hoy|ayer)', text):
        return None
    if not kinds:
        return None
    try:
        start, end = requested_interval(user_text, t0)
    except ValueError as exc:
        if trace:
            trace('validation', {'message': str(exc)})
        return str(exc)
    answers = []
    for kind in kinds:
        arguments = dict(sensor_type=kind, time_init=start.isoformat(), time_end=end.isoformat(), aggregation='total')
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'function': {'name': 'query_aggregation', 'arguments': arguments}}]})
        if trace:
            trace('tool_call', {'name': 'query_aggregation', 'arguments': arguments})
        try:
            result = query_aggregation(**arguments)
            # No presentar cobertura incompleta como un total del intervalo.
            expected = int((end - start).total_seconds() // 60)
            if start.second or end.second or start.microsecond or end.microsecond or result['samples'] != expected:
                raise ValueError('No hay cobertura completa de minutos para ese intervalo.')
        except (ValueError, OSError, TypeError) as exc:
            result = {'error': str(exc)}
            if trace:
                trace('tool_error', {'message': str(exc)})
            answers.append('No puedo verificar el total de ' + ('pasos' if kind == 'watch.steps' else 'sueño') + ' en ese intervalo: ' + str(exc))
        else:
            if trace:
                trace('tool_result', result)
            value = result['result']
            if kind == 'watch.steps':
                answers.append(f'Pasos registrados: {value:g}.')
            else:
                hours, minutes = divmod(value, 60)
                answers.append(f'Sueño registrado: {hours:g} h y {minutes:g} min.')
        messages.append({'role': 'tool', 'tool_name': 'query_aggregation', 'content': json.dumps(result, ensure_ascii=False)})
    return ' '.join(answers) + f' Intervalo consultado: [{start.isoformat()}, {end.isoformat()}).'


def guard_unverified_answer(content: str) -> str:
    """Impide afirmaciones espontáneas sobre medidas fuera de la ruta verificada.

    Es deliberadamente conservador: no intenta validar lenguaje numérico libre.
    """
    number_words = r'\b(cero|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci\w*|veinte|veinti\w*|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien\w*|\w+cientos|mil\w*|millon\w*|media|medio|cuarto|ningun\w*)\b'
    if metric_types(content) and re.search(r'\d|' + number_words, normalized(content)):
        return 'Para darte una cifra verificada de pasos o sueño, pregúntame por esa medida e indica el intervalo.'
    return content
