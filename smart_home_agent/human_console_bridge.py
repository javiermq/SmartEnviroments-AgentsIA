"""Puente humano compatible con POST /api/chat de Ollama."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
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


class LocalSTT:
    """STT local del portátil, cargado solo al recibir la primera voz."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        with self._lock:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Falta faster-whisper en el portátil. Ejecuta: pip install faster-whisper") from exc
            if self._model is None:
                # auto prioriza CUDA, pero mantiene la prueba utilizable si el portátil no
                # tiene las bibliotecas CUDA/CTranslate2 disponibles.
                if self.device == "auto":
                    try:
                        self._model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
                        print(f"STT listo: {self.model_name} en CUDA")
                    except Exception as exc:
                        print(f"CUDA no disponible para STT ({exc}); usando CPU int8.")
                        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                else:
                    compute_type = "float16" if self.device == "cuda" else "int8"
                    self._model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
                    print(f"STT listo: {self.model_name} en {self.device}")
            with NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
                handle.write(wav_bytes)
            try:
                started = time.monotonic()
                segments, _ = self._model.transcribe(
                    str(wav_path), language="es", beam_size=1,
                    condition_on_previous_text=False, vad_filter=True,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                print(f"STT ({time.monotonic() - started:.2f} s): {text or '[sin texto]'}")
                return text
            finally:
                wav_path.unlink(missing_ok=True)


def make_handler(console: HumanConsole, stt: LocalSTT) -> type[BaseHTTPRequestHandler]:
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
            if self.path not in ("/api/chat", "/stt"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000:
                    raise ValueError("Content-Length inválido")
                body_in = self.rfile.read(length)
                if self.path == "/stt":
                    if not body_in.startswith(b"RIFF"):
                        raise ValueError("/stt espera audio WAV")
                    response: dict[str, Any] = {"text": stt.transcribe_wav(body_in)}
                else:
                    payload = json.loads(body_in.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("el cuerpo debe ser JSON objeto")
                    response = console.reply(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
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
    parser.add_argument("--stt-model", default="base", help="Modelo Faster-Whisper local del portátil")
    parser.add_argument("--stt-device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(HumanConsole(), LocalSTT(args.stt_model, args.stt_device)))
    print(f"Human Console Bridge escuchando en http://{args.host}:{args.port}/api/chat")
    print(f"STT local disponible en http://{args.host}:{args.port}/stt ({args.stt_model}, {args.stt_device})")
    print("Ctrl+C para detenerlo. No lo expongas a Internet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPuente detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
