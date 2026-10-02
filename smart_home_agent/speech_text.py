"""Texto sin iconos y horas legibles para Piper."""
import re

ICONS = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF'
    '\u2190-\u21FF\u25A0-\u25FF\u2B00-\u2BFF'
    '\u00A9\u00AE\u203C\u2049\u2122\u2139\u3030\u303D\u3297\u3299'
    '\u200D\uFE0E\uFE0F\U000E0020-\U000E007F]'
)


def strip_icons(text):
    """Elimina emojis compuestos, banderas, pictogramas y viñetas decorativas."""
    text = re.sub(r'[#*0-9]\uFE0F?\u20E3', '', text)
    text = ICONS.sub('', text)
    text = re.sub(r'[\u2022\u2023\u2043\u20E3]', '', text)
    text = re.sub(r'(?m)^[ \t]*[-*][ \t]+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +([,.;:!?])', r'\1', text)
    return text.strip()

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
    text = strip_icons(text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})',
                  lambda m: spoken_clock(int(m[1]), int(m[2])), text)
    return re.sub(r'\b([01]?\d|2[0-3]):([0-5]\d)\b',
                  lambda m: spoken_clock(int(m[1]), int(m[2])), text)
