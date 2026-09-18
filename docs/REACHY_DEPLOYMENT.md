# Despliegue en Reachy Mini

Este proyecto mantiene el agente Qwen y sus tools en el backend. Reachy solo
proporciona audio y gestos a través de `ReachyInterface`.

## Antes de instalar

No detengas ni modifiques `reachy daemon`, `motor_controller`, servicios de
audio, launcher ni `conversation_app`. Una app de Reachy toma el control del
robot mientras está activa, por lo que detén manualmente cualquier app que lo
esté usando desde Reachy Mini Control antes de iniciar este agente. Comprueba el
estado sin cambiar nada:

```bash
whoami
pwd
find /venvs -maxdepth 2 -type f -name python 2>/dev/null
python -m pip show reachy-mini reachy_mini conversation-app conversation_app
python -c "import reachy_mini, inspect; from reachy_mini import ReachyMini; print(reachy_mini.__file__); print(inspect.signature(ReachyMini))"
systemctl --no-pager --type=service | grep -Ei 'reachy|motor|audio|conversation'
ss -ltnp | grep ':8000' || true
```

El SDK de Reachy debe conectarse al daemon ya existente. No uses
`spawn_daemon=True` en este proyecto.

## Copiar el proyecto

Desde el ordenador de desarrollo, crea y publica una rama de integración:

```bash
cd C:/Users/Javier/Documents/GitHub/SmartEnviroments-AgentsIA
git switch -c feature/reachy-interface
git add .
git commit -m "add Reachy Mini conversation interface"
git push -u origin feature/reachy-interface
```

En el Reachy, como el usuario normal que ejecutará el agente, usa una ruta en
su directorio personal. Sustituye `REACHY_USER` por el resultado de `whoami`:

```bash
export REACHY_USER="$(whoami)"
export PROJECT_DIR="$HOME/smart-environments-agent"
git clone --branch feature/reachy-interface https://github.com/javiermq/SmartEnviroments-AgentsIA.git "$PROJECT_DIR"
cd "$PROJECT_DIR"
```

Si la red no permite Git, desde el ordenador de desarrollo copia el repositorio
sin `.git` mediante `scp -r` a la misma ruta. No copies secretos ni `.env`.

## Entorno Python y dependencias

Primero inspecciona `/venvs/apps_venv`. Si contiene `reachy_mini` y coincide
con la versión del daemon, se puede usar para las pruebas de SDK, pero no se
deben instalar dependencias de este proyecto en ese entorno compartido. Crea un
virtualenv independiente con el mismo Python mayor que usa el robot:

```bash
cd "$PROJECT_DIR"
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reachy.txt
```

Si el SDK preinstalado no está disponible desde el virtualenv, instala en este
virtualenv una versión compatible con el daemon, obtenida con el comando de
inspección anterior. No cambies el paquete del sistema ni el daemon.

Piper necesita un modelo de voz local `.onnx`. Descarga una voz española
compatible siguiendo la documentación de Piper y guarda su ruta fuera del
repositorio, por ejemplo `~/.local/share/piper/es_ES-voice.onnx`.

## Configuración

```bash
cd "$PROJECT_DIR"
cp .env.example .env
```

Edita como mínimo. STT y TTS se ejecutan localmente: Hugging Face solo se usa
una vez para descargar pesos, nunca durante una conversación.

```dotenv
INTERFACE=reachy
REACHY_HOST=localhost
REACHY_CONNECTION_MODE=localhost_only
STT_MODEL=base
STT_LANGUAGE=es
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
STT_MODEL_PATH=
TTS_MODEL_PATH=/home/REACHY_USER/.local/share/piper/es_ES-voice.onnx
CONVERSATION_TIMEOUT=20
WAKE_WORD_ENABLED=false
```

Para Reachy Mini Wireless que ejecute el agente en otro equipo, usa
`REACHY_CONNECTION_MODE=network` y `REACHY_HOST=<IP-o-host-del-robot>`. El
cliente remoto requiere el backend multimedia apropiado del SDK; la primera
prueba recomendada se realiza directamente en el Reachy.

El backend Qwen/Ollama puede vivir en la misma máquina o en otra accesible por
red. Ajusta `OLLAMA_URL` a `http://<host-ollama>:11434/api/chat` cuando sea
remoto.

### Descargar modelos una vez, ejecutar siempre localmente

`faster-whisper` puede descargar el peso `base` durante el despliegue. Hazlo
una vez, no en el arranque normal del agente:

```bash
source .venv/bin/activate
python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("base", device="cpu", compute_type="int8")
print("Modelo STT local preparado")
PY
```

Para TTS, Piper carga una voz local `.onnx`. Descarga una voz española una vez:

```bash
source .venv/bin/activate
python -m piper.download_voices --data-dir "$HOME/.local/share/piper" es_ES-davefx-medium
```

Después fija `STT_MODEL_PATH` si almacenas el modelo de Whisper en una ruta
propia y establece `TTS_MODEL_PATH` con el `.onnx` descargado. La conversación
posterior no utiliza ningún endpoint cloud para STT ni TTS.

## Pruebas escalonadas

Activa el entorno antes de cada prueba:

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate
python -m tests.test_reachy_connection --host localhost --connection-mode localhost_only
python -m tests.test_reachy_audio --host localhost --connection-mode localhost_only
```

Prueba STT con un WAV de voz de 16 kHz:

```bash
python - <<'PY'
import asyncio
import wave
import numpy as np
from smart_home_agent.interfaces.reachy import AudioClip, FasterWhisperSTT
with wave.open("prueba_es.wav", "rb") as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
print(asyncio.run(FasterWhisperSTT("small", "es").transcribe(AudioClip(audio, 16000))))
PY
```

Prueba TTS y altavoz con el programa final, después de confirmar micrófono y
STT. Piper se inicializa en la primera respuesta hablada.

## Ejecución

Consola, para verificar backend sin Reachy:

```bash
python main.py --interface console --t0 "2026-09-16T18:56:00+02:00" --trace
```

Reachy:

```bash
python main.py --interface reachy --t0 "2026-09-16T18:56:00+02:00" --trace
```

En modo Reachy, habla tras el mensaje de escucha. La detección de voz usa el
indicador de habla del SDK para terminar el turno tras un segundo de silencio.

## Servicio systemd opcional

No lo habilites hasta haber completado las pruebas manuales. Crea una copia del
template y sustituye los tres marcadores con rutas reales y el usuario normal:

```bash
cd "$PROJECT_DIR"
sed \
  -e "s|REACHY_USER|$(whoami)|g" \
  -e "s|PROJECT_DIR|$PROJECT_DIR|g" \
  -e "s|VENV_DIR|$PROJECT_DIR/.venv|g" \
  deploy/reachy-agent.service > /tmp/reachy-agent.service
sudo cp /tmp/reachy-agent.service /etc/systemd/system/reachy-agent.service
sudo systemctl daemon-reload
sudo systemctl enable reachy-agent.service
sudo systemctl start reachy-agent.service
```

El unit usa `Restart=on-failure` y no se ejecuta como root. Logs:

```bash
systemctl status reachy-agent.service --no-pager
journalctl -u reachy-agent.service -f
```

Para detenerlo sin tocar ningún servicio de Reachy:

```bash
sudo systemctl stop reachy-agent.service
sudo systemctl disable reachy-agent.service
```
