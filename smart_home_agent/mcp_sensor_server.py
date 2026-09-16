"""Servidor MCP por stdio que expone query_aggregation a OpenClaw.

No requiere dependencias de terceros. Implementa las operaciones MCP necesarias
para una única herramienta: initialize, tools/list y tools/call.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from sensor_queries import query_aggregation

TOOL = {
    "name": "query_aggregation",
    "description": (
        "Agrega datos del reloj en un intervalo [time_init, time_end). "
        "Usa watch.sleep para verificar descanso, watch.steps para pasos y "
        "watch.distance_m para distancia."
    ),
    "inputSchema": {
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
}


def _reply(message_id: Any, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    print(json.dumps(response, ensure_ascii=False), flush=True)


def handle(message: dict[str, Any]) -> None:
    """Procesa una petición JSON-RPC; las notificaciones no reciben respuesta."""
    message_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if method == "notifications/initialized":
        return
    if method == "initialize":
        _reply(message_id, {
            "protocolVersion": params.get("protocolVersion", "2025-03-26"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "smart-home-sensors", "version": "0.1.0"},
        })
        return
    if method == "tools/list":
        _reply(message_id, {"tools": [TOOL]})
        return
    if method == "tools/call":
        if params.get("name") != "query_aggregation":
            _reply(message_id, error={"code": -32601, "message": "Herramienta no encontrada."})
            return
        try:
            result = query_aggregation(**params.get("arguments", {}))
            _reply(message_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "structuredContent": result, "isError": False})
        except (ValueError, FileNotFoundError, TypeError) as exc:
            _reply(message_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
        return
    _reply(message_id, error={"code": -32601, "message": f"Método no soportado: {method}"})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(json.loads(line))
        except json.JSONDecodeError:
            # No hay id fiable para una línea que no es JSON.
            print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "JSON inválido."}}), flush=True)


if __name__ == "__main__":
    main()
