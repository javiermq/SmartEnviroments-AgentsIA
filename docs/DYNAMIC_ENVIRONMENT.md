# Sesiones vivas y entorno dinámico

La arquitectura usa procesos separados, con un único conversor por servidor:

| Componente | Puerto por defecto | Función |
| --- | --- | --- |
| Voz | 11435 | STT y TTS, sin decisiones conversacionales |
| Conversor | 11437 | Sesiones, contexto dinámico, herramientas y decisiones de Ollama |
| Entorno | Sin puerto | Envía sensores y HAR al conversor al iniciar y cada minuto |
| Visión | 11436 | Reconocimiento facial existente |
| Reachy | Cliente | Cámara, escucha y reproducción; selecciona el usuario reconocido |

El emisor de entorno lee los TSV actuales de `sensor_context_data/javi` y
`sensor_context_data/mariola`, que representan las filas de la hoja de cálculo.
No necesita abrir Excel. Cada envío contiene la fila completa y el HAR de una
ventana móvil de 36 horas. Solo se incluye el pasado disponible en los archivos:
no se inventa historia anterior ni se revela el final futuro de una actividad.

Las sesiones se identifican por `(session_id, user)`. `t0` se actualiza y no crea
otra sesión. El modelo se fija al crearla. Un minuto real avanza un minuto desde
el `t0` elegido; `--period` permite acelerar una demo. El silencio no cierra la
sesión. El contador `user_turns_last_minute` es una ventana de 60 segundos reales,
calculada a partir de textos no vacíos recibidos de voz, sin contar eventos del
entorno ni STT vacío. No se calcula un tiempo explícito de silencio.

El entorno reemplaza el estado actual. El prompt conserva el perfil, la
conversación reciente, los cambios respecto al estado anterior y una cronología
de intervenciones con su hora simulada. Se mantienen los últimos 24 turnos en el
historial de diálogo y 48 intervenciones en la cronología. Las sesiones viven en
memoria durante la ejecución; reiniciar el conversor elimina su historial.

Cada actualización del usuario activo se ofrece a Ollama como evento de sistema
identificado como `ENVIRONMENT`, nunca como una frase del usuario ni como un rol
personalizado. Ollama decide `speak` o `silent`, sin filtros previos de interés,
umbrales de turnos ni tiempos mínimos entre comentarios. Las sesiones inactivas
siguen recibiendo datos, pero no solicitan intervenciones para el altavoz.

Los eventos actualizan el estado aunque haya una generación en curso. Una sola
generación por sesión evita mezclar su historial. Si llegan varios eventos
mientras Ollama procesa, se conserva el último estado pendiente. Se descartan
decisiones espontáneas cuyo entorno o conversación cambió mientras se generaban.
La solicitud de decisión tiene un timeout de 30 segundos; un fallo deja silencio.
Una petición de voz puede esperar a que termine una generación ya iniciada.

Reachy consulta las decisiones mientras escucha. Una intervención ya detectada
por el micrófono sigue su curso; el comentario de entorno no corta esa frase.
Las decisiones se verifican de nuevo antes del TTS y se integran en el mismo
historial cuando se aceptan para reproducción. Un único flujo usa el altavoz.
Despedirse desactiva la entrega para ese usuario y devuelve Reachy a reconocimiento,
sin borrar su sesión ni detener las actualizaciones del entorno.

## Arranque del servidor en una consola

Ollama y el servidor de visión deben estar disponibles como antes. El lanzador
arranca voz, conversor y entorno en procesos independientes, compartiendo la
salida de consola para ver `WAV`, `STT`, `PYTHON`, `ENVIRONMENT` y `TTS` juntos.
Detén el puente anterior antes de reutilizar su puerto 11435.

```powershell
python -u -m smart_home_agent.server_stack `
  --host 192.168.0.28 `
  --t0 '2026-09-16T18:56:00+02:00' `
  --session-id reachy-mini `
  --stt-model turbo --stt-device auto `
  --tts-model 'C:\Users\Javier\.piper-voices\es_ES-davefx-medium.onnx'
```

Ctrl+C detiene los tres procesos. El entorno termina al agotarse los sensores;
el conversor y voz permanecen disponibles con el último estado. Los huecos
intermedios se representan como datos no disponibles.

También pueden arrancarse por separado:

```powershell
python -u -m smart_home_agent.conversation_service --host 192.168.0.28 --port 11437
python -u -m smart_home_agent.human_console_bridge --service voice --host 192.168.0.28 --port 11435 --stt-model turbo --tts-model 'C:\Users\Javier\.piper-voices\es_ES-davefx-medium.onnx'
python -u -m smart_home_agent.environment_service --conversation-url http://192.168.0.28:11437/conversation --t0 '2026-09-16T18:56:00+02:00' --session-id reachy-mini
```

Permite conexiones del robot al puerto nuevo 11437 con las mismas restricciones
de red privada que los puertos de voz y visión.

## Reachy

La URL de voz conserva su puerto; la conversación pasa al conversor:

```bash
export REMOTE_STT_URL="http://192.168.0.28:11435/stt"
export REMOTE_TTS_URL="http://192.168.0.28:11435/tts"
export REMOTE_CONVERSATION_URL="http://192.168.0.28:11437/conversation"
export REMOTE_VISION_URL="http://192.168.0.28:11436/vision/check"

/venvs/apps_venv/bin/python -u main.py \
  --interface reachy --face-recognition --session-id reachy-mini \
  --t0 "2026-09-16T18:56:00+02:00" \
  --speech-rms-threshold 0.03 --expressive-motion --trace
```

El `session_id` y el modelo deben coincidir entre el emisor de entorno y Reachy.
Si la sesión ya existe, el `t0` antiguo del cliente no retrocede el reloj del servidor.
Sin cámara, `--user javi` o `--user mariola` activa el mismo flujo de voz y entorno.

## Contrato HTTP del conversor

- `POST /environment`: `user`, `session_id`, `t0`, `model`, `row`, `context`.
  Acepta la actualización sin esperar la decisión del LLM.
- `POST /conversation`: `user`, `session_id`, `t0`, `model`, `text`.
  Registra una intervención real y devuelve respuesta, hora actual y trazas Python.
- `POST /activate` y `/deactivate`: seleccionan la persona presente sin borrar memoria.
- `POST /events`: devuelve la decisión pendiente o `silent`.
- `POST /spoken`: acepta una decisión por `revision`; una decisión antigua se rechaza.
- `GET /health`: comprueba que el conversor está disponible.

En la consola se muestra la aceptación del entorno, las decisiones y las llamadas
Python cuando se ejecutan. Las consultas verificadas siguen utilizando el archivo
del usuario y el `t0` actualizado. No se envían instrucciones procedentes de los
sensores como instrucciones de sistema; el perfil y las reglas son del conversor.
