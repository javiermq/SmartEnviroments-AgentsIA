"""Emite sensores completos y HAR de 36 h al iniciar y cada minuto simulado."""
import argparse
import json
import logging
import math
import time
from datetime import timedelta
from pathlib import Path
from threading import Event
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit

from .context import DATA_DIR, build_context, parse_timestamp, read_rows
from .user_sessions import user_id


def endpoint_url(endpoint, path):
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, path, '', ''))


def post(endpoint, payload):
    request = Request(endpoint, data=json.dumps(payload, allow_nan=False).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def environment_payload(user, t0, data_root=DATA_DIR, session_id='reachy',
                        model='qwen3:4b-instruct-2507-q4_K_M'):
    user = user_id(user)
    context = build_context(t0, Path(data_root) / user, history_hours=36)
    row = context['environment_row']
    return dict(user=user, t0=t0, session_id=session_id, model=model, row=row, context=context)


def run(endpoint, users, t0, data_root=DATA_DIR, session_id='reachy',
        model='qwen3:4b-instruct-2507-q4_K_M', period=60.0, stop=None):
    if not math.isfinite(period) or period <= 0:
        raise ValueError('period debe ser positivo')
    moment = parse_timestamp(t0)
    stop = stop or Event()
    last = {}
    for user in users:
        timestamps = [parse_timestamp(row['timestamp']) for row in read_rows(
            Path(data_root) / user_id(user), 'simulated_sensor_data_*.tsv')]
        if not timestamps:
            raise ValueError(f'No hay sensores para {user}')
        last[user] = max(timestamps) + timedelta(minutes=1)
    active = list(users)
    next_tick = time.monotonic() + period
    while active and not stop.is_set():
        for user in list(active):
            if moment >= last[user]:
                print(f'ENVIRONMENT {user}: fin de datos en {moment.isoformat()}', flush=True)
                active.remove(user)
                continue
            payload = environment_payload(user, moment.isoformat(), data_root, session_id, model)
            try:
                post(endpoint_url(endpoint, '/environment'), payload)
                print(f'ENVIRONMENT enviado: {user}, {moment.isoformat()}, '
                      f'HAR={len(payload["context"]["activities_last_36h"])}', flush=True)
            except (OSError, ValueError):
                logging.exception('No se pudo enviar el entorno de %s; se reintentará', user)
        # El reloj avanza desde el t0 elegido, nunca usa fechas futuras del host.
        if not active or stop.wait(max(0.0, next_tick - time.monotonic())):
            break
        moment += timedelta(minutes=1)
        next_tick += period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversation-url', default='http://127.0.0.1:11437/conversation')
    parser.add_argument('--t0', required=True)
    parser.add_argument('--users', nargs='+', choices=['javi', 'mariola'], default=['javi', 'mariola'])
    parser.add_argument('--session-id', default='reachy')
    parser.add_argument('--model', default='qwen3:4b-instruct-2507-q4_K_M')
    parser.add_argument('--data-root', type=Path, default=DATA_DIR)
    parser.add_argument('--period', type=float, default=60, help='Segundos reales por minuto simulado')
    args = parser.parse_args()
    try:
        run(args.conversation_url, args.users, args.t0, args.data_root,
            args.session_id, args.model, args.period)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
