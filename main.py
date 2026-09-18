"""Arranque único para interfaz de consola o Reachy Mini."""


from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

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
        interface = ReachyInterface(settings, speech_rms_threshold=args.speech_rms_threshold)
    else:
        raise ValueError("--interface debe ser console o reachy")

    backend = ConversationBackend(args.t0, model=settings.ollama_model, ollama_url=settings.ollama_url)
    await interface.start()
    print(f"[{interface_name} listo] t0={args.t0}")
    try:
        while True:
            try:
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
            await interface.speak(response)
    finally:
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
    parser.add_argument("--trace", action="store_true", help="Muestra llamadas a herramientas y resultados.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
