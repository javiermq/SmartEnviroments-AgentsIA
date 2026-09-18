"""Puente humano compatible con POST /api/chat de Ollama."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def assistant_message(answer: str) -> dict[str, Any]:
    return {"model": "human-console", "created_at": datetime.now(timezone.utc).isoformat(), "message": {"role": "assistant", "content": answer}, "done": True}


def tool_message(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"model": "human-console", "created_at": datetime.now(timezone.utc).isoformat(), "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "query_aggregation", "arguments": arguments}}]}, "done": True}


class HumanConsole:
    """Una sola consola para que peticiones simultáneas no mezclen prompts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def reply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            messages = payload.get("messages", [])
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            tool_results = [m for m in messages if m.get("role") == "tool"]
            print("\n" + "=" * 72)
            if last_user:
                print(f"REACHY / usuario: {last_user.get('content', '')}")
            if tool_results:
                print("RESULTADO DE HERRAMIENTA:")
                print(tool_results[-1].get("content", ""))
            print("Escribe la respuesta que Reachy debe decir.")
            print("O usa /tool {JSON} para consultar el reloj antes de responder.")
            print("Ejemplo: /tool {\"sensor_type\":\"watch.sleep\",\"time_init\":\"00:00\",\"time_end\":\"18:56\",\"aggregation\":\"total\"}")
            while True:
                answer = input("HUMANO> ").strip()
                if answer.startswith("/tool "):
                    try:
                        arguments = json.loads(answer.removeprefix("/tool "))
                        if not isinstance(arguments, dict):
                            raise ValueError("el JSON debe ser un objeto")
                        return tool_message(arguments)
                    except (json.JSONDecodeError, ValueError) as exc:
                        print(f"Comando /tool inválido: {exc}")
                        continue
                if answer:
                    return assistant_message(answer)
                print("Escribe una respuesta no vacía.")


def make_handler(console: HumanConsole) -> type[BaseHTTPRequestHandler]:
    class BridgeHandler(BaseHTTPRequestHandler):
        server_version = "HumanConsoleBridge/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/api/tags"):
                body = json.dumps({"status": "ok", "models": [{"name": "human-console"}]}).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Content-Length inválido")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("el cuerpo debe ser JSON objeto")
                response = console.reply(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return BridgeHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM humano compatible con Ollama para pruebas de Reachy")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (solo portátil) o 0.0.0.0 (Reachy en LAN)")
    parser.add_argument("--port", default=11435, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(HumanConsole()))
    print(f"Human Console Bridge escuchando en http://{args.host}:{args.port}/api/chat")
    print("Ctrl+C para detenerlo. No lo expongas a Internet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPuente detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
