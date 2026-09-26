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
Para conversar con Ollama local y Qwen3 4B Instruct 2507 (sin razonamiento), desde la raíz del proyecto:

```powershell
python -u -m smart_home_agent.conversation_demo --t0 "2026-09-16T18:56:00+02:00"
```

Escribe `salir` para terminar. Esta demo usa Ollama en `127.0.0.1:11434` y conserva
el historial de diálogo durante la sesión. La primera respuesta puede tardar por
la carga del modelo y el procesamiento del contexto.

Esta demo requiere `ollama pull qwen3:4b-instruct-2507-q4_K_M` y envía `think: false` con un
contexto de 8192 tokens. En cada turno muestra por stderr el prompt completo,
los sensores, las actividades/HAR, la petición y la respuesta de Ollama.
Las trazas de herramientas aparecen solo cuando el modelo las solicita.
Los sensores y las actividades se leen de archivos locales, no de servicios HTTP.

Las consultas reconocidas de pasos y duración del sueño tienen una ruta verificada
común a la demo y al backend: ejecutan `query_aggregation` y redactan los números
directamente desde su resultado, sin pedir al modelo que los invente o reformule.
En estos turnos verás trazas de herramienta, pero no una llamada a Ollama.
Se admiten hoy (también por defecto), ayer, dos horas HH:MM en el mismo día,
o dos timestamps ISO-8601 con zona horaria. Para intervalos ambiguos como «anoche»
o «esta semana» se pide concretar inicio y fin; no se sustituye por hoy.
No se ofrece un total si faltan muestras o el intervalo supera `t0`.
La muestra del reloj se presenta separadamente como `watch_minute_sample_NOT_TOTALS`.
Fuera de esta ruta, un filtro conservador elimina frases del modelo que combinan
pasos o sueño con cifras (también números escritos en palabras), y ofertas de esas
medidas ajenas al tema del usuario. Conserva las demás frases sin pedir un intervalo.
Reconoce «pasos para preparar…» como instrucciones, no como medidas del reloj;
puede producir falsos positivos en otras formulaciones.
Esta protección no valida otras magnitudes ni todas las formas de expresar una medida
en lenguaje natural; no sustituye una evaluación completa del agente.

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
llamada. La traza detallada incluye el contexto y la respuesta completa de
Ollama; el razonamiento se solicita desactivado.

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

## Configuración actual: Reachy Mini con Ollama y audio remoto

El portátil ejecuta Ollama y el puente de audio. Reachy ejecuta el agente y se
conecta al portátil por la red local. Sustituye `IP_OLLAMA` e `IP_REACHY` por
las IP actuales de cada sesión. No copies direcciones que aparezcan como enlaces
Markdown: deben ser texto plano, por ejemplo `http://10.137.164.220:11435/stt`.

### Listar dispositivos de la red desde PowerShell

El siguiente comando prueba la subred de la interfaz Wi-Fi. Una IP que no
responda a ping no queda confirmada como libre, porque algunos dispositivos
bloquean ICMP.

```powershell
$red = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi' |
  Where-Object { $_.IPAddress -notlike '169.254.*' } |
  Select-Object -First 1).IPAddress -replace '\.\d+$', ''

$tareas = 1..254 | ForEach-Object {
  $ip = "$red.$_"
  $ping = [System.Net.NetworkInformation.Ping]::new()
  [pscustomobject]@{ IP = $ip; Ping = $ping; Task = $ping.SendPingAsync($ip, 800) }
}

$tareas | ForEach-Object {
  try {
    if ($_.Task.GetAwaiter().GetResult().Status -eq 'Success') { $_.IP }
  } finally {
    $_.Ping.Dispose()
  }
}
```

### Portátil Windows

Abre dos ventanas de PowerShell. En la primera, inicia Ollama en la IP del
portátil. Cierra antes Ollama desde el icono de la bandeja si ya estaba abierto
con otra configuración.

```powershell
$IP_OLLAMA = '10.137.164.220' # IP del portátil
$IP_REACHY = '10.137.164.185' # IP actual de Reachy

$env:OLLAMA_HOST = "${IP_OLLAMA}:11434"
ollama serve
```

En la segunda ventana, inicia el puente de audio que reenvía el chat a Ollama.
Descarga antes el modelo una vez con `ollama pull qwen3:4b-instruct-2507-q4_K_M`.

```powershell
cd C:\Users\Javier\Documents\GitHub\SmartEnviroments-AgentsIA

$IP_OLLAMA = '10.137.164.220'
& "C:\Users\Javier\AppData\Local\Programs\Python\Python311\python.exe" -u `
  smart_home_agent\human_console_bridge.py `
  --host $IP_OLLAMA --port 11435 `
  --stt-model turbo --stt-device auto `
  --tts-model "C:\Users\Javier\.piper-voices\es_ES-davefx-medium.onnx" `
  --chat-mode ollama `
  --ollama-url "http://127.0.0.1:11434/api/chat" `
  --trace
```

Reachy accede al puente en el puerto 11435. Ollama queda accesible solamente
desde el propio portátil, porque el puente reenvía a 127.0.0.1:11434.
Para permitir solamente a Reachy acceder al puente, abre PowerShell como
administrador y ejecuta una vez:

```powershell
$IP_REACHY = '10.137.164.185'
New-NetFirewallRule -DisplayName 'Reachy Mini - Audio' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11435 -RemoteAddress $IP_REACHY
```

### Reachy

Desde la raíz del repositorio en Reachy, exporta las direcciones y ejecuta el
Python que contiene el SDK `reachy_mini`:

```bash
export IP_OLLAMA="10.137.164.220"
export IP_REACHY="10.137.164.185"
export REMOTE_STT_URL="http://${IP_OLLAMA}:11435/stt"
export REMOTE_TTS_URL="http://${IP_OLLAMA}:11435/tts"
export OLLAMA_URL="http://${IP_OLLAMA}:11435/api/chat"
export OLLAMA_MODEL="qwen3:4b-instruct-2507-q4_K_M"
export REACHY_HOST="localhost"
export REACHY_CONNECTION_MODE="localhost_only"

curl --connect-timeout 5 "http://${IP_OLLAMA}:11435/health"
/venvs/apps_venv/bin/python -c "import reachy_mini; print('SDK Reachy OK')"
/venvs/apps_venv/bin/python -u main.py --interface reachy --t0 "2026-09-16T18:56:00+02:00" --trace --speech-rms-threshold 0.03 --expressive-motion
```

`IP_REACHY` sirve para restringir las reglas de firewall del portátil. Puesto
que el agente corre dentro de Reachy, conserva `REACHY_HOST=localhost` y
`REACHY_CONNECTION_MODE=localhost_only`.
