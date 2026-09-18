"""Puente humano compatible con POST /api/chat de Ollama."""

from __future__ import annotations

import argparse
import json
import os
import random
import site
import threading
import time
import wave
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CUDA_DLL_DIRECTORIES: list[object] = []
AUTOMATIC_RESPONSES = (
    "Recibido. Estoy listo para continuar.",
    "Entendido, la comunicación funciona correctamente.",
    "Perfecto, he recibido tu mensaje.",
    "De acuerdo, sigo atento a tus indicaciones.",
    "La prueba de respuesta automática ha sido completada.",
    "Hola, te escucho con claridad.",
    "Gracias, la conexión está activa.",
    "Confirmado, puedo responder de inmediato.",
    "Todo está funcionando como se esperaba.",
    "Mensaje recibido sin necesidad de consultar el modelo.",
    "Estoy preparado para el siguiente turno.",
    "La transmisión de audio se ha procesado correctamente.",
    "Respuesta automática enviada desde el servidor.",
    "Sí, sigo conectado y disponible.",
    "La latencia de esta prueba no incluye Ollama.",
    "Confirmo que el canal de voz está operativo.",
    "He entendido tu petición.",
    "El sistema de prueba responde correctamente.",
    "Puedes continuar cuando quieras.",
    "Prueba completada: respuesta local generada.",
)
SAVE_SIZE_SAMPLES = 100
SAMPLE_GESTURES = (
    {"name": "alerta", "antennas_deg": [28, 28]},
    {"name": "curiosa_derecha", "antennas_deg": [34, 8]},
    {"name": "curiosa_izquierda", "antennas_deg": [8, 34]},
    {"name": "relajada", "antennas_deg": [-18, -18]},
)


class ReceivedWavRing:
    """Conserva los últimos WAV recibidos en un anillo de tamaño fijo."""

    def __init__(self, directory: Path, size: int = SAVE_SIZE_SAMPLES) -> None:
        if size < 1:
            raise ValueError("El tamaño del anillo debe ser al menos 1")
        self.directory = directory
        self.size = size
        self._next_slot = 0
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, wav_bytes: bytes) -> Path:
        """Guarda de forma atómica y reutiliza reachy_000.wav al completar el anillo."""
        with self._lock:
            path = self.directory / f"reachy_{self._next_slot:03d}.wav"
            temporary = path.with_suffix(".wav.part")
            temporary.write_bytes(wav_bytes)
            temporary.replace(path)
            self._next_slot = (self._next_slot + 1) % self.size
            return path


def configure_windows_cuda_dlls() -> None:
    """Hace visibles las DLL instaladas por pip sin modificar PATH global."""
    if os.name != "nt" or CUDA_DLL_DIRECTORIES:
        return
    paths: list[str] = []
    for package_dir in site.getsitepackages():
        root = Path(package_dir) / "nvidia"
        for relative in ("cublas/bin", "cudnn/bin", "cuda_runtime/bin", "cuda_nvrtc/bin"):
            dll_dir = root / relative
            if dll_dir.is_dir():
                path = str(dll_dir)
                paths.append(path)
                CUDA_DLL_DIRECTORIES.append(os.add_dll_directory(path))
    # CTranslate2 carga CUDA dinámicamente; además de add_dll_directory necesita
    # PATH para algunos builds de Windows. Solo afecta a este proceso Python.
    if paths:
        os.environ["PATH"] = os.pathsep.join(paths + [os.environ.get("PATH", "")])


def assistant_message(answer: str, gesture: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    if gesture is not None:
        message["gesture"] = gesture
    return {"model": "human-console", "created_at": datetime.now(timezone.utc).isoformat(), "message": message, "done": True}


def sample_gesture() -> dict[str, Any]:
    """Devuelve una copia para que cada respuesta tenga su gesto independiente."""
    gesture = random.choice(SAMPLE_GESTURES)
    return {"name": gesture["name"], "antennas_deg": list(gesture["antennas_deg"])}


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
                configure_windows_cuda_dlls()
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


class LocalTTS:
    """Piper persistente en el servidor/portátil; evita cargar la voz en Reachy."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._voice = None
        self._lock = threading.Lock()

    def _load_voice(self) -> None:
        if not self.model_path:
            raise RuntimeError("Falta --tts-model con una voz Piper .onnx")
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError("Falta piper-tts en este equipo") from exc
        if self._voice is None:
            self._voice = PiperVoice.load(self.model_path)
            print(f"TTS listo: {self.model_path}")

    def warm(self) -> None:
        with self._lock:
            self._load_voice()

    def synthesize_wav(self, text: str) -> bytes:
        with self._lock:
            self._load_voice()
            with NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
            try:
                started = time.monotonic()
                with wave.open(str(wav_path), "wb") as wav_file:
                    self._voice.synthesize_wav(text, wav_file)
                wav_bytes = wav_path.read_bytes()
                print(f"TTS ({time.monotonic() - started:.2f} s): {text}")
                return wav_bytes
            finally:
                wav_path.unlink(missing_ok=True)


def _trace_chat_request(body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
        messages = payload.get("messages", [])
        latest = messages[-1] if messages else {}
        print("\n" + "=" * 72)
        if latest.get("role") == "tool":
            print(f"HERRAMIENTA → chat: {latest.get('content', '')}")
        else:
            user = next((message for message in reversed(messages) if message.get("role") == "user"), {})
            print(f"REACHY → chat: {user.get('content', '')}")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        print("REACHY → chat: solicitud no interpretable")


def _trace_ollama_response(body: bytes, elapsed: float) -> None:
    try:
        message = json.loads(body.decode("utf-8")).get("message", {})
        calls = message.get("tool_calls", [])
        for call in calls:
            function = call.get("function", {})
            print(f"OLLAMA tool_call: {function.get('name')}({json.dumps(function.get('arguments', {}), ensure_ascii=False)})")
        content = message.get("content", "").strip()
        if content:
            print(f"OLLAMA → Reachy ({elapsed:.2f} s): {content}")
        elif calls:
            print(f"OLLAMA → Reachy ({elapsed:.2f} s): [esperando resultado de herramienta]")
        else:
            print(f"OLLAMA → Reachy ({elapsed:.2f} s): [respuesta vacía]")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        print(f"OLLAMA → Reachy ({elapsed:.2f} s): respuesta no interpretable")


def forward_to_ollama(ollama_url: str, body: bytes, trace: bool) -> tuple[int, bytes]:
    """Proxy local: Reachy nunca necesita acceder al puerto de Ollama."""
    if trace:
        _trace_chat_request(body)
    request = Request(ollama_url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        started = time.monotonic()
        with urlopen(request, timeout=180) as response:
            response_body = response.read()
            if trace:
                _trace_ollama_response(response_body, time.monotonic() - started)
            return response.status, response_body
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError as exc:
        raise RuntimeError(f"No se puede conectar con Ollama local ({ollama_url}): {exc}") from exc


def make_handler(console: HumanConsole, stt: LocalSTT, tts: LocalTTS, received_wavs: ReceivedWavRing, ollama_url: str | None, automatic: bool, echo: bool, trace: bool) -> type[BaseHTTPRequestHandler]:
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
            if self.path not in ("/api/chat", "/stt", "/tts"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000:
                    raise ValueError("Content-Length inválido")
                body_in = self.rfile.read(length)
                content_type = "application/json; charset=utf-8"
                if self.path == "/stt":
                    if not body_in.startswith(b"RIFF"):
                        raise ValueError("/stt espera audio WAV")
                    saved_path = received_wavs.save(body_in)
                    print(f"WAV de Reachy guardado: {saved_path}")
                    response: dict[str, Any] = {"text": stt.transcribe_wav(body_in)}
                    status = HTTPStatus.OK
                    body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                elif self.path == "/tts":
                    payload = json.loads(body_in.decode("utf-8"))
                    text = payload.get("text") if isinstance(payload, dict) else None
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("/tts espera JSON con texto no vacío")
                    status = HTTPStatus.OK
                    body = tts.synthesize_wav(text.strip())
                    content_type = "audio/wav"
                elif ollama_url:
                    status, body = forward_to_ollama(ollama_url, body_in, trace)
                elif automatic:
                    answer = random.choice(AUTOMATIC_RESPONSES)
                    gesture = sample_gesture()
                    if trace:
                        _trace_chat_request(body_in)
                        print(f"AUTOMÁTICO → Reachy: {answer} | gesto={gesture}")
                    status = HTTPStatus.OK
                    body = json.dumps(assistant_message(answer, gesture), ensure_ascii=False).encode("utf-8")
                elif echo:
                    payload = json.loads(body_in.decode("utf-8"))
                    messages = payload.get("messages", []) if isinstance(payload, dict) else []
                    user = next((message for message in reversed(messages) if message.get("role") == "user"), {})
                    answer = str(user.get("content", "")).strip()
                    if not answer:
                        raise ValueError("/api/chat no recibió texto de usuario para eco")
                    gesture = sample_gesture()
                    if trace:
                        print(f"ECO → Reachy: {answer} | gesto={gesture}")
                    status = HTTPStatus.OK
                    body = json.dumps(assistant_message(answer, gesture), ensure_ascii=False).encode("utf-8")
                else:
                    payload = json.loads(body_in.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("el cuerpo debe ser JSON objeto")
                    response = console.reply(payload)
                    status = HTTPStatus.OK
                    body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
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
    parser.add_argument("--tts-model", default="", help="Ruta local de voz Piper .onnx para /tts")
    parser.add_argument("--chat-mode", choices=["human", "ollama", "automatic", "echo"], default="human")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--save-dir", default="temp_data", help="Directorio del anillo de 100 WAV de entrada")
    parser.add_argument("--trace", action="store_true", help="Muestra solicitudes, tools y respuestas de Ollama")
    args = parser.parse_args()
    ollama_url = args.ollama_url if args.chat_mode == "ollama" else None
    tts = LocalTTS(args.tts_model)
    if args.tts_model:
        tts.warm()
    received_wavs = ReceivedWavRing(Path(args.save_dir), SAVE_SIZE_SAMPLES)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(HumanConsole(), LocalSTT(args.stt_model, args.stt_device), tts, received_wavs, ollama_url, args.chat_mode == "automatic", args.chat_mode == "echo", args.trace))
    print(f"Human Console Bridge escuchando en http://{args.host}:{args.port}/api/chat")
    print(f"STT local disponible en http://{args.host}:{args.port}/stt ({args.stt_model}, {args.stt_device})")
    print(f"WAV recibidos: {received_wavs.directory.resolve()} (anillo de {SAVE_SIZE_SAMPLES})")
    print(f"TTS local disponible en http://{args.host}:{args.port}/tts ({args.tts_model or 'sin voz configurada'})")
    chat_status = "Ollama local en " + ollama_url if ollama_url else ({"automatic": "respuesta automática aleatoria", "echo": "eco del texto transcrito"}.get(args.chat_mode, "respuesta humana por consola"))
    print(f"Chat: {chat_status}")
    print("Ctrl+C para detenerlo. No lo expongas a Internet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPuente detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
