"""Arranque único para interfaz de consola o Reachy Mini."""


from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import os
from smart_home_agent.context import DATA_DIR
from smart_home_agent.user_sessions import RemoteConversation

from smart_home_agent.config import Settings
from smart_home_agent.conversation_backend import ConversationBackend
from smart_home_agent.interfaces import ConsoleInterface


def trace_to_console(event: str, payload: dict) -> None:
    if event == "tool_call":
        print(f"TRAZA    > {payload['name']}({json.dumps(payload['arguments'], ensure_ascii=False)})")
    elif event == "tool_result":
        print(f"RESULTADO> {json.dumps(payload, ensure_ascii=False)}")
    elif event == "tool_error":
        print(f"ERROR    > {payload['message']}")


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    interface_name = args.interface or settings.interface
    if interface_name == "console":
        interface = ConsoleInterface()
    elif interface_name == "reachy":
        from smart_home_agent.interfaces.reachy import ReachyInterface
        interface = ReachyInterface(
            settings,
            speech_rms_threshold=args.speech_rms_threshold,
            speech_pre_roll_seconds=args.speech_pre_roll_seconds,
            tts_leading_silence_seconds=args.tts_leading_silence_seconds,
            expressive_motion=args.expressive_motion,
        )
    else:
        raise ValueError("--interface debe ser console o reachy")

    backend = (RemoteConversation(args.conversation_url, args.t0, args.user, settings.ollama_model, args.session_id)
               if args.conversation_url else ConversationBackend(
                   args.t0, model=settings.ollama_model, ollama_url=settings.ollama_url,
                   data_dir=DATA_DIR / args.user))
    await interface.start()
    print(f"[{interface_name} listo] t0={args.t0}")
    try:
        if args.face_recognition:
            if interface_name != "reachy" or not args.conversation_url:
                raise ValueError("--face-recognition requiere Reachy y --conversation-url")
            from smart_home_agent.recognized_conversation import run_recognized
            await run_recognized(interface, settings, args, trace_to_console if args.trace else None)
            return
        if args.conversation_url:
            await asyncio.to_thread(backend.activate)
        while True:
            try:
                if args.conversation_url and interface_name == 'reachy':
                    from smart_home_agent.proactive import listen_with_environment
                    user_text = await listen_with_environment(interface, backend)
                else:
                    user_text = await interface.listen()
            except KeyboardInterrupt:
                break
            except (RuntimeError, TimeoutError) as exc:
                logging.warning("Turno de audio descartado: %s", exc)
                continue
            if interface_name == "console" and user_text.lower() in {"salir", "exit", "quit"}:
                break
            if not user_text:
                continue
            print(f"USUARIO  > {user_text}")
            await interface.on_thinking()
            response = await asyncio.to_thread(backend.respond, user_text, trace_to_console if args.trace else None)
            if interface_name == "reachy":
                interface.set_response_gesture(backend.last_gesture)
            await interface.speak(response)
    finally:
        if args.conversation_url and not args.face_recognition:
            from contextlib import suppress
            with suppress(OSError, RuntimeError, ValueError):
                await asyncio.to_thread(backend.deactivate)
        await interface.close()
        print("[Sesión cerrada]")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Agente de sensores con interfaces intercambiables.")
    parser.add_argument("--interface", choices=["console", "reachy"])
    parser.add_argument("--t0", required=True, help="Timestamp ISO-8601 de referencia.")
    parser.add_argument(
        "--speech-rms-threshold", type=float, default=0.02,
        help="Umbral RMS usado si el sensor DoA de Reachy falla (por defecto: 0.02).",
    )
    parser.add_argument(
        "--speech-pre-roll-seconds", type=float, default=0.35,
        help="Audio previo a la detección de voz que se conserva (por defecto: 0.35 s).",
    )
    parser.add_argument(
        "--tts-leading-silence-seconds", type=float, default=0.20,
        help="Silencio antes de la voz para no perder la primera sílaba (por defecto: 0.20 s).",
    )
    parser.add_argument(
        "--expressive-motion", action="store_true",
        help="Mueve las antenas al hablar y las devuelve a reposo al terminar.",
    )
    parser.add_argument("--user", choices=["javi", "mariola"], default="mariola")
    parser.add_argument("--session-id", default="reachy")
    parser.add_argument("--conversation-url", default=os.getenv("REMOTE_CONVERSATION_URL", ""))
    parser.add_argument("--face-recognition", action="store_true")
    parser.add_argument("--vision-url", default=os.getenv("REMOTE_VISION_URL", "http://192.168.0.28:11436/vision/check"))
    parser.add_argument("--detector-model", default="models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--trace", action="store_true", help="Muestra llamadas a herramientas y resultados.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
