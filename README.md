# Smart Environments: consultas con OpenClaw

Este prototipo permite que un agente conversacional consulte dinámicamente el
TSV simulado, empezando por las señales del reloj.

Los datos viven en `sensor_context_data/`, con sensores y actividades de los días
15 y 16 de septiembre de 2026. El perfil editable está en
[`user_context.json`](sensor_context_data/user_context.json). Cada turno incorpora
el perfil, los sensores en `--t0`, la actividad actual y las últimas 12 horas de
actividades. Véase [formato y semántica temporal](sensor_context_data/README.md).

La personalidad cercana, con humor suave y acompañamiento opcional, se edita en
[`agent_style.txt`](sensor_context_data/agent_style.txt). Se recarga en cada turno.
Para conversar con Ollama local y Qwen3 4B, desde la raíz del proyecto:

```powershell
python -u -m smart_home_agent.conversation_demo --t0 "2026-09-16T18:56:00+02:00"
```

Escribe `salir` para terminar. Esta demo usa Ollama en `127.0.0.1:11434` y conserva
el historial de diálogo durante la sesión. La primera respuesta puede tardar por
la carga del modelo y el procesamiento del contexto.

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

El `qwen3:4b` descargado ya usa cuantización `Q4_K_M`; bajar a Q2/Q3 externo
empeora notablemente las llamadas de herramienta. Para priorizar velocidad, usa
el perfil `qwen3:1.7b-fast` que se prepara con `scripts/setup_fast_qwen.ps1`.
Mantén el 4B para las pruebas de herramientas más exigentes.

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

## Puente humano para probar Reachy sin LLM

### Lanzadores por defecto: STT Turbo + eco + TTS remoto

Este es el modo de validación recomendado antes de añadir el LLM: Reachy
captura el audio, el portátil transcribe con Faster-Whisper `turbo`, devuelve
el mismo texto y lo sintetiza con Piper. No utiliza Ollama ni el servidor.

En el portátil Windows, con la voz Piper `es_ES-davefx-medium` descargada en
`C:\Users\Javier\.piper-voices`, inicia el bridge:

```powershell
& "C:\Users\Javier\AppData\Local\Programs\Python\Python311\python.exe" -u `
  smart_home_agent\human_console_bridge.py `
  --host 0.0.0.0 --port 11435 `
  --stt-model turbo --stt-device auto `
  --tts-model "C:\Users\Javier\.piper-voices\es_ES-davefx-medium.onnx" `
  --chat-mode echo --trace
```

En Reachy, con `REMOTE_STT_URL` y `REMOTE_TTS_URL` apuntando al portátil,
arranca el agente:

```dotenv
REMOTE_STT_URL=http://192.168.0.28:11435/stt
REMOTE_TTS_URL=http://192.168.0.28:11435/tts
OLLAMA_URL=http://192.168.0.28:11435/api/chat
OLLAMA_MODEL=human-console
```

```bash
python main.py \
  --interface reachy \
  --t0 "2026-09-16T18:56:00+02:00" \
  --trace \
  --speech-rms-threshold 0.03 \
  --expressive-motion
  
  
/venvs/apps_venv/bin/python -u main.py   --interface reachy   --t0 "2026-09-16T18:56:00+02:00"   --trace   --speech-rms-threshold 0.03   --expressive-motion  
```




El reproductor de Reachy añade por defecto 0,20 segundos de silencio antes de
la respuesta para no perder la primera sílaba. Si fuese necesario, puede
ajustarse con `--tts-leading-silence-seconds 0.30`.

Los últimos 100 WAV que recibe el STT quedan en `temp_data/` del portátil; al
llegar al número 101 se vuelve a escribir el primer archivo.

En modo `echo` y `automatic`, el bridge adjunta un gesto de antenas de una
lista limitada a cada respuesta. Reachy valida que los dos ángulos estén en el
rango seguro de -35° a 35° antes de ejecutarlo. Qwen podrá usar este mismo
campo `gesture` cuando se active el modo LLM.

### Modo humano manual

También puede iniciarse un servidor que habla el protocolo de Ollama pero deja
que una persona escriba cada respuesta:

```powershell
python smart_home_agent/human_console_bridge.py --host 0.0.0.0 --port 11435
```

En el `.env` de Reachy establece `OLLAMA_URL=http://IP_DEL_PORTATIL:11435/api/chat`
y `OLLAMA_MODEL=human-console`. Reachy sigue usando el mismo backend: envía el
texto al puente y pronuncia literalmente lo que se escriba en `HUMANO>`. Desde
esa consola, `/tool {JSON}` simula una llamada a `query_aggregation`; el
resultado aparecerá en la siguiente petición y podrás redactar la respuesta
final. Para probar solo en el portátil, deja el host por defecto `127.0.0.1`.
