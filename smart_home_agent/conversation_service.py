"""Conversor HTTP: sesiones por usuario, voz y eventos de entorno."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .context import DATA_DIR
from .user_sessions import UserSessions


def make_handler(sessions):
    def trace(event, data):
        labels = {'tool_call': 'llamada', 'tool_result': 'resultado', 'tool_error': 'error'}
        if event in labels:
            print(f'PYTHON {labels[event]}: {json.dumps(data, ensure_ascii=False)}', flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404,
                       {'status': 'ok'} if self.path == '/health' else {'error': 'Ruta desconocida'})

        def do_POST(self):
            routes = {'/conversation': lambda p: sessions.respond(p, trace),
                      '/environment': sessions.environment, '/activate': sessions.activate,
                      '/deactivate': sessions.deactivate, '/events': sessions.events,
                      '/spoken': sessions.spoken}
            if self.path not in routes:
                self.reply(404, {'error': 'Ruta desconocida'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Content-Length inválido')
                payload = json.loads(self.rfile.read(size))
                result = routes[self.path](payload)
                self.reply(200, result)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self.reply(400, {'error': str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11437)
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434/api/chat')
    parser.add_argument('--data-root', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(
        UserSessions(args.ollama_url, args.data_root)))
    print(f'Conversor disponible en http://{args.host}:{args.port}/conversation', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
