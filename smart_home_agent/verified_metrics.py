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
    # «Pasos para preparar una receta» no son medidas del reloj.
    procedural = re.search(r'\bpasos (?:para|a seguir|de (?:la |una )?(?:receta|preparacion))\b', text)
    if re.search(r'\bpasos\b|\b(?:un|1) paso\b', text) and not procedural:
        kinds.append('watch.steps')
    if re.search(r'\b(dorm\w*|sueno)\b', text):
        kinds.append('watch.sleep')
    return kinds


def requested_interval(text: str, t0: str) -> tuple[datetime, datetime]:
    """Resuelve horarios cotidianos; la ventana nocturna se declara en la respuesta."""
    moment = datetime.fromisoformat(t0.replace('Z', '+00:00'))
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    text = normalized(text).replace('media noche', 'medianoche')
    iso = re.findall(r'\d{4}-\d{2}-\d{2}t\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:\d{2})', text)
    if len(iso) == 2:
        start, end = (datetime.fromisoformat(s.replace('z', '+00:00')) for s in iso)
    else:
        if re.search(r'\b(seman\w*|mes\w*|anteayer|ultim\w*|antes|despues|hace|dias|lunes|martes|miercoles|jueves|viernes|sabado|domingo|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|promedio|maxim\w*|minim\w*)\b|de media|\d{4}-\d{2}-\d{2}', text):
            raise ValueError('¿Entre qué días y horas quieres que lo consulte? Puedes decir, por ejemplo, de ayer a las diez de la noche a hoy a las siete de la mañana, o escribir dos fechas con sus horas.')
        if re.search(r'\bmanana\b', text) and not re.search(r'(?:esta|la|media) manana', text):
            raise ValueError('Solo puedo consultar datos anteriores a la hora de referencia. ¿Qué horario quieres consultar?')
        night = bool(re.search(r'\banoche\b|\b(?:esta|la|ultima) noche\b', text))
        explicit = text.replace('medianoche', '00:00').replace('media manana', '10:00')
        times = re.findall(r'\b(\d{1,2}):(\d{2})\b', explicit)
        if times:
            if len(times) != 2:
                raise ValueError('¿Entre qué dos horas? Por ejemplo, de nueve a diez de la mañana, o escribe 09:00 y 10:00.')
            base = day - timedelta(days=1) if 'ayer' in text or night else day
            start, end = (base.replace(hour=int(h), minute=int(m)) for h, m in times)
            if night and end <= start:
                end += timedelta(days=1)
        elif night:
            start = day - timedelta(days=1) + timedelta(hours=20)
            end = min(day + timedelta(hours=12), moment)
        elif re.search(r'\b(entre|desde|hasta|de \d|a las|por la|esta tarde|esta madrugada|media)\b|\d', text):
            raise ValueError('¿Entre qué dos horas quieres que lo consulte? Puedes indicar, por ejemplo, 09:00 y 10:00.')
        else:
            start, end = (day - timedelta(days=1), day) if 'ayer' in text else (day, moment)
    if start.tzinfo is None or end.tzinfo is None or end <= start or end > moment:
        raise ValueError('El inicio debe ser anterior al final y ambas horas deben ser anteriores a la hora de referencia. ¿Qué horario quieres consultar?')
    return start, end


def verified_answer(user_text: str, t0: str, messages: list[dict], trace=None, data_dir=None) -> str | None:
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
            result = query_aggregation(**arguments, **({"data_file": data_dir} if data_dir is not None else {}))
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
                answers.append(f'Sueño registrado: {hours:g} horas y {minutes:g} minutos.')
        messages.append({'role': 'tool', 'tool_name': 'query_aggregation', 'content': json.dumps(result, ensure_ascii=False)})
    return ' '.join(answers) + f' Intervalo consultado: de {spoken_time(start)}{day_label(start, t0)} a {spoken_time(end)}{day_label(end, t0)}.'


def guard_unverified_answer(content: str, user_text: str = '') -> str:
    """Impide afirmaciones espontáneas sobre medidas fuera de la ruta verificada.

    Es deliberadamente conservador: no intenta validar lenguaje numérico libre.
    """
    number_words = r'\b(cero|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci\w*|veinte|veinti\w*|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien\w*|\w+cientos|mil\w*|millon\w*|media|medio|cuarto|ningun\w*)\b'
    requested = set(metric_types(user_text))
    sentences = re.split(r'(?<=[.!?])\s+|\n+', content)
    kept = []
    removed = False
    for sentence in sentences:
        kinds = set(metric_types(sentence))
        unsupported = kinds and re.search(r'\d|' + number_words, normalized(sentence))
        off_topic = kinds and not kinds.intersection(requested)
        if unsupported or off_topic:
            removed = True
            continue
        # No conservar una pregunta vacía ligada a una oferta que acabamos de quitar.
        if removed and re.fullmatch(r'[¿¡\s]*(?:te interesa|quieres|te gustaria)[?!.\s]*', normalized(sentence)):
            continue
        kept.append(sentence)
    if not removed:
        return content
    return ' '.join(kept).strip() or 'No tengo una medida verificada para afirmar eso.'


def spoken_time(moment):
    from .speech_text import spoken_clock
    return spoken_clock(moment.hour, moment.minute)


def day_label(moment, t0):
    reference = datetime.fromisoformat(t0.replace('Z', '+00:00')).date()
    days = (reference - moment.date()).days
    if days == 0:
        return ''
    if days == 1:
        return ' del día anterior'
    return f' de hace {days} días'
