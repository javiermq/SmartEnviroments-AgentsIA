# Smart Environments: consultas con OpenClaw

Este prototipo permite que un agente conversacional consulte dinámicamente el
TSV simulado, empezando por las señales del reloj.

## API de agregación

La función Python `query_aggregation(sensor_type, time_init, time_end, aggregation)`
está en `smart_home_agent/sensor_queries.py`.

Tipos de sensor disponibles:

| Tipo | Columna | Unidad de `total` |
| --- | --- | --- |
| `watch.steps` | `steps` | pasos |
| `watch.distance_m` | `distance_m` | m |
| `watch.sleep` | `sleep` | minutos dormidos |

Las agregaciones permitidas son `total`, `mean`, `max` y `min`. El intervalo
es semiabierto: `[time_init, time_end)`. Se aceptan horas `HH:MM` del día
simulado o timestamps ISO-8601 completos.

Ejemplo de uso:

```powershell
python smart_home_agent/sensor_queries.py watch.sleep 22:15 24:00 total
```

## OpenClaw + Qwen local

Se recomienda `qwen3:4b` en Ollama. Con la GPU de 8 GB mostrada y 32 GB de RAM,
es un punto de partida prudente para conversación y llamadas de herramientas.
Mantén el contexto en 8192 tokens al principio. No es aconsejable escoger un
modelo de 8B o mayor como primera configuración si se desea que quede margen de
VRAM para el contexto y el sistema.

1. Instala Python 3.11+ y Ollama para Windows.
2. Ejecuta `ollama pull qwen3:4b`.
3. Instala y configura OpenClaw siguiendo su asistente de instalación.
4. Copia `openclaw/openclaw.json5.example` en `~/.openclaw/openclaw.json`.
   El ejemplo ya contiene las rutas de este equipo; actualízalas si mueves el
   proyecto o cambias la instalación de Python.
5. Copia `openclaw/skills/smart-home-sensors` a
   `~/.openclaw/skills/smart-home-sensors` para que el agente sepa cuándo usar
   la herramienta.
6. Comprueba la conexión con `openclaw mcp doctor --probe` y reinicia o recarga
   el Gateway si procede.

## Ejemplo conversacional

**Usuario:** «¿He dormido bien?»

**Llamada de herramienta:**

```json
{
  "sensor_type": "watch.sleep",
  "time_init": "2026-09-16T00:00:00+02:00",
  "time_end": "2026-09-17T00:00:00+02:00",
  "aggregation": "total"
}
```

**Resultado simulado:** `525` minutos, es decir, 8 horas y 45 minutos de sueño.

**Respuesta esperada del agente:** «En los datos simulados aparecen 8 h 45 min
de sueño. Es una duración suficiente para la mayoría de adultos, aunque este
registro no mide la calidad clínica ni los despertares.»

## Prueba conversacional con traza

Con Ollama ejecutándose, inicia una sesión en una hora de referencia concreta:

```powershell
& "C:/Users/Javier/AppData/Local/Programs/Python/Python311/python.exe" `
  smart_home_agent/conversation_demo.py `
  --t0 "2026-09-16T18:56:00+02:00"
```

O realiza un único turno de prueba:

```powershell
& "C:/Users/Javier/AppData/Local/Programs/Python/Python311/python.exe" `
  smart_home_agent/conversation_demo.py `
  --t0 "2026-09-16T18:56:00+02:00" `
  --question "¿He dormido bien?"
```

La consola muestra el nombre de la función, argumentos y resultado de cada
llamada. Por seguridad y diseño, la traza no expone razonamiento interno
oculto del modelo.

## Interfaces intercambiables

El entrypoint general es `main.py`. Mantiene Qwen y las tools en
`ConversationBackend` y permite cambiar el frontend:

```powershell
python main.py --interface console --t0 "2026-09-16T18:56:00+02:00" --trace
```

La interfaz Reachy Mini se documenta en
[`docs/REACHY_DEPLOYMENT.md`](docs/REACHY_DEPLOYMENT.md). No se importa su SDK
cuando se usa el modo consola.
