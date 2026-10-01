"""Horas legibles para Piper, conservando minutos y referencias de día."""
import re

NUMBERS = ('cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce '
           'trece catorce quince dieciséis diecisiete dieciocho diecinueve veinte '
           'veintiuno veintidós veintitrés veinticuatro veinticinco veintiséis '
           'veintisiete veintiocho veintinueve').split()


def number(value):
    if value < 30:
        return NUMBERS[value]
    tens, units = divmod(value, 10)
    return {3: 'treinta', 4: 'cuarenta', 5: 'cincuenta'}[tens] + (' y ' + NUMBERS[units] if units else '')


def spoken_clock(hour, minute):
    return f'{number(hour)} horas y {number(minute)} minutos'


def for_speech(text):
    text = re.sub(r'\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})',
                  lambda m: spoken_clock(int(m[1]), int(m[2])), text)
    return re.sub(r'\b([01]?\d|2[0-3]):([0-5]\d)\b',
                  lambda m: spoken_clock(int(m[1]), int(m[2])), text)
