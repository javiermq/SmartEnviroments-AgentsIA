"""Consola conversacional local: Qwen/Ollama + consultas del reloj."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from .sensor_queries import query_aggregation
    from .context import system_prompt
except ImportError:  # Ejecución directa: python smart_home_agent/conversation_demo.py
    from sensor_queries import query_aggregation
    from context import system_prompt

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:4b"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_aggregation",
        "description": "Agrega una señal del reloj en un intervalo [inicio, fin).",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sensor_type", "time_init", "time_end", "aggregation"],
            "properties": {
                "sensor_type": {
                    "type": "string",
                    "enum": ["watch.steps", "watch.distance_m", "watch.sleep"],
                },
                "time_init": {"type": "string", "description": "ISO-8601 o HH:MM"},
                "time_end": {"type": "string", "description": "ISO-8601 o HH:MM"},
                "aggregation": {
                    "type": "string",
                    "enum": ["total", "mean", "max", "min"],
                },
            },
        },
    },
}


def _request_ollama(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": [TOOL_SCHEMA],
        "stream": False,
        "think": True,
        "options": {"num_ctx": 8192},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _system_prompt(t0: str) -> str:
    return system_prompt(t0)


def _visible_answer(content: str) -> str:
    """Elimina razonamiento etiquetado que algunos modelos locales emiten."""
    # Algunas versiones de Qwen devuelven el inicio de <think> en un campo
    # separado y dejan solo el cierre dentro de content.
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1].strip()
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
        # Si la etiqueta queda sin cierre, no se muestra ese contenido.
        if "<think>" in content:
            return ""
    return content.strip()


def _run_turn(messages: list[dict], user_text: str, t0: str) -> None:
    messages[0]["content"] = _system_prompt(t0)
    messages.append({"role": "user", "content": user_text})
    print(f"\nUSUARIO  > {user_text}")

    for round_number in range(1, 5):
        try:
            response = _request_ollama(messages)
        except URLError as exc:
            raise RuntimeError("No se puede conectar con Ollama en el puerto 11434.") from exc

        assistant = response["message"]
        assistant = {**assistant, "content": _visible_answer(assistant.get("content", ""))}
        messages.append(assistant)
        visible_text = assistant.get("content", "")
        calls = assistant.get("tool_calls", [])

        if calls:
            print(f"TRAZA    > ronda {round_number}: {len(calls)} llamada(s) a herramienta")
            for call in calls:
                function = call["function"]
                name = function["name"]
                arguments = function.get("arguments", {})
                print(f"TRAZA    > {name}({json.dumps(arguments, ensure_ascii=False)})")
                try:
                    result = query_aggregation(**arguments, reference_day=t0[:10])
                    tool_content = json.dumps(result, ensure_ascii=False)
                    print(f"RESULTADO> {tool_content}")
                except (ValueError, FileNotFoundError, TypeError) as exc:
                    tool_content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    print(f"ERROR    > {tool_content}")
                messages.append({"role": "tool", "tool_name": name, "content": tool_content})
            continue

        if visible_text:
            print(f"ASISTENTE> {visible_text}")
        else:
            print("ASISTENTE> No se obtuvo texto de respuesta.")
        return

    print("ASISTENTE> Se alcanzó el límite de llamadas de herramienta para este turno.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demostración conversacional con Qwen y sensores.")
    parser.add_argument("--t0", required=True, help="Hora de referencia ISO-8601.")
    parser.add_argument("--question", help="Pregunta única; omitir para conversación interactiva.")
    args = parser.parse_args()
    try:
        datetime.fromisoformat(args.t0)
    except ValueError:
        parser.error("--t0 debe ser un timestamp ISO-8601 válido.")

    messages = [{"role": "system", "content": _system_prompt(args.t0)}]
    print(f"Sesión iniciada en t0={args.t0}. Modelo: {MODEL}")
    print("Razonamiento de Qwen activado. La traza muestra herramientas y resultados; la consola muestra la respuesta final.")

    if args.question:
        _run_turn(messages, args.question, args.t0)
        return

    print("Escribe 'salir' para terminar.")
    while True:
        try:
            question = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesión finalizada.")
            return
        if question.lower() in {"salir", "exit", "quit"}:
            print("Sesión finalizada.")
            return
        if question:
            _run_turn(messages, question, args.t0)


if __name__ == "__main__":
    main()
