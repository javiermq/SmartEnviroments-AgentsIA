# Visión independiente: Reachy Mini → DeepFace

El agente de cámara detecta caras con YuNet, recorta cada bounding box con un
25 % de margen por lado y envía JPEG al ordenador. No inicia grabación de audio,
STT, TTS ni conversación. En esta fase imprime las respuestas JSON. La futura
integración de audio deberá activarse después de confirmar la identidad.

El servidor compara embeddings Facenet512 de DeepFace con las fotos de la casa
y analiza solo `emotion`. Los modelos se cargan al arrancar y las referencias
se calculan una vez; reinicia el servicio si cambias las fotos.

## Preparar el ordenador (Windows, Python 3.11)

Ejecuta desde la raíz del repositorio:

```powershell
python -m venv .venv-vision
.venv-vision\Scripts\python.exe -m pip install -r requirements-vision-server.txt
New-Item -ItemType Directory -Force vision_data/users/javi, vision_data/users/mariola
```

Coloca varias fotos JPG/PNG de cada persona en su carpeta, con exactamente una
cara visible por imagen. Usa fotos nítidas y algunas variaciones de iluminación
y orientación. No pongas imágenes de otras personas en esas carpetas.
Como punto de partida, usa 3–5 fotos por persona de unos 640 × 640 píxeles,
con la cara de al menos 150–200 píxeles de ancho y margen alrededor. Son
recomendaciones para las pruebas, no límites del modelo: no es obligatorio que
sean cuadradas ni del mismo tamaño. El modelo adapta la entrada internamente.

```text
vision_data/users/
├── javi/
│   ├── frontal.jpg
│   └── lateral_suave.jpg
└── mariola/
    ├── frontal.jpg
    └── otra_luz.jpg
```

Las carpetas se llaman `javi` y `mariola`; los valores de respuesta son exactamente
`Javi`, `mariola` y `unknow` (se conserva la grafía solicitada). Las fotos sin
una cara detectada o con varias caras se omiten con un aviso que indica el archivo.
El arranque falla si no queda ninguna referencia válida. Una persona sin
referencias válidas produce un aviso y no podrá reconocerse. No se desactiva
`enforce_detection`: una foto rechazada nunca se utiliza como plantilla facial.

```powershell
.venv-vision\Scripts\python.exe -u -m smart_home_agent.vision_service --host 0.0.0.0 --port 11436
```

El primer arranque necesita Internet para descargar los pesos de Facenet512 y
Emotion en la caché de DeepFace (`~/.deepface/weights`). Después la inferencia
es local. La configuración inicial funciona en CPU; no requiere que TensorFlow
utilice la GPU de Windows.

```powershell
Invoke-RestMethod http://127.0.0.1:11436/health
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@vision_data/users/javi/frontal.jpg" http://127.0.0.1:11436/vision/check
```

Para el ensayo HTTP usa un JPEG de entre 32 y 2048 píxeles por lado y menos de
2 MB. Las fotos de referencia pueden tener otros tamaños y también ser PNG.

El servicio usa HTTP en la red local, igual que el puente de audio; no incluye
autenticación ni TLS. Si el firewall bloquea a Reachy, permite el puerto 11436
solo desde su IP (PowerShell como administrador, sustituyendo la IP):

```powershell
$IP_REACHY = '192.168.0.50'
New-NetFirewallRule -DisplayName 'Reachy Mini - Vision' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11436 -RemoteAddress $IP_REACHY
```

## Preparar Reachy Mini

Usa un entorno dedicado y el SDK compatible con el daemon instalado, como en
[el despliegue de audio](REACHY_DEPLOYMENT.md). No instales paquetes en el
entorno compartido `/venvs/apps_venv`. El agente se conecta al daemon existente.
Detén desde Reachy Mini Control cualquier aplicación que esté usando el robot
antes de esta prueba independiente; no detengas el daemon.

```bash
python3 -m venv .venv-vision
.venv-vision/bin/python -m pip install --upgrade pip
.venv-vision/bin/python -m pip install -r requirements-vision-reachy.txt
mkdir -p models
curl --fail --location \
  https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  --output models/face_detection_yunet_2023mar.onnx
export REMOTE_VISION_URL="http://IP_DEL_ORDENADOR:11436/vision/check"
.venv-vision/bin/python -u -m smart_home_agent.reachy_vision
```

La instalación en Reachy permite NumPy 2 y exige paquetes precompilados
(`--only-binary=:all:` en requirements). No uses aquí las restricciones de
NumPy del servidor TensorFlow. Si pip muestra un `.tar.gz` o `Preparing metadata`
para NumPy, interrumpe con Ctrl+C y actualiza este archivo de requisitos.
Si no hay un paquete compatible, pip terminará con un error en vez de compilar;
comprueba `.venv-vision/bin/python --version` y `uname -m` antes de cambiar de
intérprete o arquitectura. No modifiques el entorno compartido del robot.

El intervalo por defecto es **1 segundo**, según la configuración acordada para
este proyecto; no hace falta pasar `--interval`. Se espera ese segundo tras
terminar el lote anterior, además del tiempo de procesamiento y envío.

Opciones: `--interval 1` establece explícitamente ese mismo intervalo;
`--min-face 80` ignora caras menores de 80 píxeles en la imagen original;
`--host localhost --connection-mode localhost_only` conecta dentro del robot.
También se admite `--url` y `--detector-model`.

La detección usa imágenes reducidas hasta 640 píxeles de ancho, pero el recorte
se obtiene de la imagen original. Solo se envían recortes, hasta 640 píxeles por
lado, y nunca imágenes sin detecciones. Se procesa una petición cada vez. Un
fallo de red se registra y se vuelve a intentar en un ciclo posterior.
Si hay varias caras, se envían por separado; no se identifica al hablante ni
se selecciona automáticamente un interlocutor. Un recorte con dos caras que
el servidor detecte se rechaza como `multiple_faces`.

## Respuesta y validación

`POST /vision/check`, cuerpo JPEG y `Content-Type: image/jpeg`. Ejemplo
abreviado (valores ilustrativos):

```json
{
  "user": "Javi",
  "status": "matched",
  "emotion": "happy",
  "emotion_scores": {"happy": 80.0, "neutral": 20.0},
  "model": "Facenet512",
  "capture_id": "face_000",
  "request_id": "identificador-unico",
  "received_at": "2026-09-26T15:00:00+00:00"
}
```

`candidates` contiene la mejor distancia coseno de cada usuario y el umbral.
Cuanto menor la distancia, mayor similitud; no es una probabilidad de identidad.
Se acepta el mejor candidato si pasa el umbral y aventaja al siguiente al menos
`--margin` (0.05 por defecto). `--threshold` permite sustituir el umbral de
DeepFace. Ajusta ambos usando fotos de validación distintas de las referencias,
incluyendo personas ajenas a la galería.

Estados: `matched`, `no_match`, `ambiguous`, `no_face`, `multiple_faces` e
`invalid_scores`. Todos salvo `matched` devuelven `unknow`. Un fallo del modelo
devuelve HTTP 500 y `inference_error`; un fallo solo de expresión conserva la
identidad y devuelve `emotion: null` con `emotion_error`. HTTP 503 significa
que hay otra inferencia activa; no se acumula una cola de imágenes pendientes.

`emotion` usa las etiquetas de DeepFace: `angry`, `disgust`, `fear`, `happy`,
`sad`, `surprise`, `neutral`. `emotion_scores` son puntuaciones de 0 a 100 del
clasificador. Son estimaciones de expresión facial, no una medida fiable del
estado de ánimo. Esta primera versión tampoco incorpora prueba de presencia
real frente a una fotografía.

En `temp_data/vision/` se guardan **100 registros**: `face_000.jpg` a
`face_099.jpg`, cada uno con su JSON. Se guardan antes de inferir, también cuando
la inferencia falla. Al completar el anillo se sobrescribe el más antiguo; el
cursor persiste al reiniciar. `capture_id` se reutiliza, `request_id` distingue
las capturas. Los cuerpos inválidos y peticiones rechazadas por servicio ocupado
no entran en el anillo. Usa un solo proceso de servidor por directorio de anillo.

Las referencias, capturas, modelos y entorno virtual están excluidos de Git.

## Pruebas

```powershell
.venv-vision\Scripts\python.exe -m unittest tests.test_vision -v
```

Las pruebas HTTP usan JPEG reales e inferencia simulada: verifican el contrato,
persistencia, errores y servicio ocupado. También comprueban la rotación de
100 registros tras reiniciar, rechazo de identidades ambiguas, recorte y
limitación de frecuencia. No miden exactitud facial ni rendimiento en Reachy.

Referencias: [DeepFace](https://github.com/serengil/deepface/tree/v0.0.93),
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
[cámara del SDK Reachy](https://huggingface.co/docs/reachy_mini/examples/take_picture).
