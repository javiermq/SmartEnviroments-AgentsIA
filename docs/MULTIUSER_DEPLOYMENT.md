# Javi y Mariola: conversación y reconocimiento en paralelo

El servidor Windows mantiene STT, TTS y las sesiones de conversación en el
puerto 11435; Ollama escucha localmente en 11434. El servidor DeepFace es otro
proceso en 11436. En Reachy se ejecuta **solo main.py con --face-recognition**:
la cámara trabaja en un hilo, comparte la conexión al SDK con audio y devuelve
eventos al coordinador, que es el único propietario de los movimientos.
No ejecutes también reachy_vision en este modo. Detén otras apps que usen el
robot desde Reachy Mini Control, manteniendo activo el daemon.

## Datos y sesiones

- `sensor_context_data/javi/`: perfil, estilo, sensores y actividades de Javi.
- `sensor_context_data/mariola/`: perfil, estilo, sensores y actividades de Mariola.

Mariola conserva el perfil y el escenario anteriores. Javi tiene un escenario
simulado diferente para los dos días, con horarios, pasos, sueño y señales de
vivienda coherentes con sus propias actividades. Su perfil indica que es el
marido de Mariola y profesor universitario de informática y robótica; no se
le atribuyen hábitos reales desconocidos.
**Todos estos datos son simulados, no mediciones reales.** Sustituye los archivos
por los de cada persona cuando dispongas de datos reales.
Los archivos de la raíz se conservan para las demos y CLI anteriores.

`POST /conversation` recibe `user`, `session_id`, `t0`, `model` y `text`.
El servidor usa la carpeta del usuario, tanto para el prompt como para las
consultas verificadas y las herramientas del modelo. El historial se guarda
por `(session_id, user, t0, model)`, con bloqueo por sesión. Cambiar de usuario
no borra el historial del anterior. Identificadores desconocidos se rechazan.
Las sesiones viven en memoria y se borran al reiniciar el puente. Usa un
`--session-id` diferente para cada robot o cliente independiente.

## Windows: tres ventanas de PowerShell

Ollama (si ya responde en 11434, no inicies otra instancia):

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11434'
ollama serve
```

Audio y conversación (el modelo y la voz deben estar ya descargados):

```powershell
cd C:\Users\Javier\Documents\GitHub\SmartEnviroments-AgentsIA
& 'C:\Users\Javier\AppData\Local\Programs\Python\Python311\python.exe' -u -m smart_home_agent.human_console_bridge `
  --host 192.168.0.28 --port 11435 `
  --stt-model turbo --stt-device auto `
  --tts-model 'C:\Users\Javier\.piper-voices\es_ES-davefx-medium.onnx' `
  --chat-mode ollama --ollama-url 'http://127.0.0.1:11434/api/chat' --trace
```

Visión (entorno instalado según VISION_DEPLOYMENT.md y fotos de referencia en
`vision_data/users/javi/` y `vision_data/users/mariola/`):

```powershell
cd C:\Users\Javier\Documents\GitHub\SmartEnviroments-AgentsIA
& '.\.venv-vision\Scripts\python.exe' -u -m smart_home_agent.vision_service `
  --host 192.168.0.28 --port 11436
```

Si falta la regla de firewall, PowerShell como administrador, una vez:

```powershell
$robotIP = Read-Host 'IP de Reachy Mini'
New-NetFirewallRule -DisplayName 'Reachy Mini - Audio y Vision' `
  -Direction Inbound -Action Allow -Protocol TCP `
  -LocalPort 11435,11436 -RemoteAddress $robotIP
```

## Reachy: terminal Bash desde la raíz del repositorio actualizado

```bash
export REMOTE_STT_URL="http://192.168.0.28:11435/stt"
export REMOTE_TTS_URL="http://192.168.0.28:11435/tts"
export REMOTE_CONVERSATION_URL="http://192.168.0.28:11435/conversation"
export REMOTE_VISION_URL="http://192.168.0.28:11436/vision/check"
export OLLAMA_MODEL="qwen3:4b-instruct-2507-q4_K_M"
export REACHY_HOST="localhost"
export REACHY_CONNECTION_MODE="localhost_only"

curl --fail http://192.168.0.28:11435/health
curl --fail http://192.168.0.28:11436/health

/venvs/apps_venv/bin/python -u main.py \
  --interface reachy --face-recognition --session-id reachy-mini \
  --t0 "2026-09-16T18:56:00+02:00" \
  --speech-rms-threshold 0.03 --expressive-motion --trace
```

Se mantienen los requisitos de OpenCV, YuNet y cámara de VISION_DEPLOYMENT.md.
Sin reconocimiento, puedes usar `--user javi` o `--user mariola` y el mismo
`--conversation-url` para elegir el perfil manualmente.

## Reconocimiento, voz y movimiento

El umbral inicial es el de distancia coseno de DeepFace para Facenet512;
se exige separación de 0.05 respecto al otro candidato y acuerdo de al menos
dos de tres capturas. No se presenta la distancia como porcentaje de confianza.
Debe calibrarse con imágenes ajenas a las referencias antes de uso real.

Mientras recoge y verifica el lote, Reachy describe un círculo suave con
pitch y roll de hasta seis grados, entrada gradual de 0.8 segundos y una
vuelta cada tres segundos, usando la API del SDK. Finaliza en neutro
al reconocer, rechazar, perder la cara, fallar la petición o cerrar el agente.
Una ventana de captura que se atasca se abandona a los ocho segundos; la
petición al servidor tiene un timeout de sesenta segundos.

La voz se activa tras reconocer a Javi o Mariola: dice «Hola Javi» o
«Hola Mariola», fija su sesión y pausa el reconocimiento facial durante toda
la conversación. Perder la cara, guardar silencio o agotar el timeout de
escucha no cambia de usuario. Al escuchar «adiós» (también «adiós Reachy»,
«bueno, adiós» o «gracias, adiós»), se despide sin consultar al LLM, vuelve a
neutro y reactiva el reconocimiento para el siguiente usuario. El historial
por persona se conserva en el servidor para futuras conversaciones.
Se dejan diez segundos entre verificaciones fallidas mientras busca una cara.

Los intervalos se expresan con horas y minutos en palabras, sin corchetes,
timestamps ni offsets de zona horaria. Si el intervalo empieza el día anterior,
se indica verbalmente. Los argumentos y trazas de herramientas mantienen ISO.
