"""Backend conversacional independiente de la interfaz de entrada/salida."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .sensor_queries import query_aggregation

TraceCallback = Callable[[str, dict[str, Any]], None]

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
                "sensor_type": {"type": "string", "enum": ["watch.steps", "watch.distance_m", "watch.sleep"]},
                "time_init": {"type": "string", "description": "ISO-8601 o HH:MM"},
                "time_end": {"type": "string", "description": "ISO-8601 o HH:MM"},
                "aggregation": {"type": "string", "enum": ["total", "mean", "max", "min"]},
            },
        },
    },
}


class ConversationBackend:
    """Qwen/Ollama y sus herramientas, sin detalles de consola o robot."""

    def __init__(self, t0: str, model: str = "qwen3:4b", ollama_url: str = "http://127.0.0.1:11434/api/chat"):
        self.model = model
        self.ollama_url = ollama_url
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt(t0)}]
        self.last_gesture: dict[str, Any] | None = None

    @staticmethod
    def _system_prompt(t0: str) -> str:
        return f"""Eres un asistente doméstico conversacional. La hora de referencia es {t0}.
Responde en español y sé breve. Cuando el usuario pregunte por pasos, distancia,
sueño, descanso o actividad, DEBES llamar a query_aggregation antes de contestar.
Solo existen watch.steps, watch.distance_m y watch.sleep. Si no se indica un
intervalo, usa 00:00 como inicio y la hora de referencia como fin. Para la
calidad del sueño, aclara que solo hay una estimación de duración y no una
medición clínica. No inventes datos ni expliques razonamiento interno."""

    def respond(self, user_text: str, trace: TraceCallback | None = None) -> str:
        self.last_gesture = None
        self.messages.append({"role": "user", "content": user_text})
        for round_number in range(1, 5):
            assistant = self._request_ollama()
            assistant = {**assistant, "content": self._visible_answer(assistant.get("content", ""))}
            gesture = assistant.get("gesture")
            if isinstance(gesture, dict):
                self.last_gesture = gesture
            self.messages.append(assistant)
            calls = assistant.get("tool_calls", [])
            if not calls:
                return assistant.get("content", "") or "No se obtuvo texto de respuesta."

            if trace:
                trace("tool_round", {"round": round_number, "calls": len(calls)})
            for call in calls:
                function = call["function"]
                name = function["name"]
                arguments = function.get("arguments", {})
                if trace:
                    trace("tool_call", {"name": name, "arguments": arguments})
                try:
                    result = query_aggregation(**arguments)
                    content = json.dumps(result, ensure_ascii=False)
                    if trace:
                        trace("tool_result", result)
                except (ValueError, FileNotFoundError, TypeError) as exc:
                    content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    if trace:
                        trace("tool_error", {"message": str(exc)})
                self.messages.append({"role": "tool", "tool_name": name, "content": content})
        return "Se alcanzó el límite de llamadas de herramienta para este turno."

    def _request_ollama(self) -> dict[str, Any]:
        payload = {"model": self.model, "messages": self.messages, "tools": [TOOL_SCHEMA], "stream": False, "think": False, "options": {"num_ctx": 8192}}
        request = Request(self.ollama_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))["message"]
        except URLError as exc:
            raise RuntimeError("No se puede conectar con Ollama en el puerto 11434.") from exc

    @staticmethod
    def _visible_answer(content: str) -> str:
        if "</think>" in content:
            return content.rsplit("</think>", 1)[-1].strip()
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
            if "<think>" in content:
                return ""
        return content.strip()
