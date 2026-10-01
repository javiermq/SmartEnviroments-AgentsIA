# Visión por lotes: Reachy Mini → DeepFace

El servidor de visión es independiente del audio y de la conversación.
Para activar sesiones con un saludo y movimiento de reconocimiento, consulta
[MULTIUSER_DEPLOYMENT.md](MULTIUSER_DEPLOYMENT.md). El cliente independiente
`reachy_vision` descrito aquí no activa STT ni TTS. El agente imprime un resultado por lote y el servidor muestra trazas
con cada paso, distancias de identidad y puntuaciones de expresión.

## Flujo y valores por defecto

1. Reachy detecta con YuNet una única cara completa de al menos 120 píxeles
   por lado. Si hay varias, desaparece, queda cortada o se desplaza demasiado,
   se reinicia el lote. El solapamiento IoU con la primera caja debe ser >= 0.65.
2. Reúne 5 capturas distintas durante al menos 1 segundo, separadas al menos
   0.25 segundos. La imagen congelada no cuenta como una nueva captura. Esto
   es seguimiento geométrico, no una garantía de identidad ni prueba de vida.
3. Puntúa calidad sobre la región facial normalizada a 128 × 128: nitidez
   (varianza del Laplaciano), exposición media y fracción de píxeles saturados.
   Elige las 3 mejores. La puntuación es relativa: un lote entero de mala
   calidad todavía puede ser rechazado por el servidor.
4. Envía los tres recortes JPEG en una petición JSON a `/vision/batch`.
   El recorte tiene margen del 25 % por lado y un máximo de 640 píxeles por lado.
5. El servidor detecta y alinea con YuNet tanto referencias como capturas,
   obtiene embeddings Facenet512 y calcula para cada imagen la menor distancia
   coseno a las referencias de cada usuario.
6. Agrega por usuario la mediana de las distancias disponibles (mínimo 2).
   Exige pasar el umbral de DeepFace, separación >= 0.05 respecto al siguiente
   candidato y al menos 2 de 3 decisiones individuales coincidentes. En caso
   contrario devuelve `unknow`. Una captura ausente o fallida no aporta voto.
7. Combina por mediana las puntuaciones de expresión y renormaliza a 100.
   Si hay identidad reconocida usa solo las imágenes que votaron por ella.
   Exige al menos dos resultados de expresión y una diferencia >= 10 puntos
   entre las dos categorías superiores; en otro caso devuelve `uncertain`.

El intervalo por defecto entre envíos sigue siendo **1 segundo**. Es una pausa
tras finalizar la petición anterior; después se recoge una nueva ventana de
capturas. La captura y la inferencia añaden tiempo: no se promete un resultado
por segundo. No se acumulan lotes pendientes durante una petición.

## Ordenador Windows

Desde la raíz del repositorio, con Python 3.11:

```powershell
python -m venv .venv-vision
& ".\.venv-vision\Scripts\python.exe" -m pip install -r requirements-vision-server.txt
New-Item -ItemType Directory -Force vision_data/users/javi, vision_data/users/mariola
```

Pon referencias JPG/PNG en `vision_data/users/javi/` y
`vision_data/users/mariola/`. Como punto de partida: 3–5 fotos por persona,
unos 640 × 640 píxeles, cara nítida de al menos 150–200 píxeles de ancho y
margen alrededor. No necesitan tamaño exacto ni ser cuadradas. Incluye
variaciones moderadas de iluminación y orientación y una única cara por foto.

Las fotos sin cara detectable o con varias se omiten con aviso. Si no queda
ninguna referencia válida, el servicio no arranca. Una persona sin referencias
no podrá reconocerse. Reinicia el servidor cuando cambies las referencias;
se calculan al arrancar, siempre con el mismo detector que las capturas.

```powershell
& ".\.venv-vision\Scripts\python.exe" -u -m smart_home_agent.vision_service --host 0.0.0.0 --port 11436
```

El primer arranque descarga los pesos de Facenet512, Emotion y YuNet en
`~/.deepface/weights`. Después se ejecutan localmente, también en CPU.
Las trazas están activadas por defecto, con fecha y hora. Incluyen:

- Recepción, número de imágenes, bytes, identificador de petición y slot guardado.
- Detección y embedding de cada imagen, resultado y distancia de cada usuario.
- Puntuaciones de expresión por imagen.
- Medianas, votos, identidad y expresión finales, y tiempo total en milisegundos.

La distancia es menor cuanto mayor es la similitud; **no es una probabilidad**.
`--threshold` cambia el umbral de identidad y `--margin` la separación entre
usuarios. No aumentes el umbral solo para aceptar las capturas fallidas: calibra
con imágenes distintas de las referencias y con personas fuera de la galería.
`--detector opencv` permite comparar con el detector anterior; por defecto es
`yunet`. La expresión facial no mide con fiabilidad el estado de ánimo real.

## Reachy Mini: entorno ya instalado

En el robot utiliza el entorno que ya funcionó para el audio. No instales ni
actualices dependencias en `/venvs/apps_venv`, ni detengas el daemon. Detén desde
Reachy Mini Control cualquier otra aplicación que use el robot durante la prueba.

```bash
/venvs/apps_venv/bin/python -c "import reachy_mini, gi, cv2, numpy; print('SDK y cámara OK'); print(hasattr(cv2, 'FaceDetectorYN'))"
mkdir -p models
curl --fail --location https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx --output models/face_detection_yunet_2023mar.onnx
export REMOTE_VISION_URL="http://192.168.0.28:11436/vision/check"
curl --fail --connect-timeout 5 http://192.168.0.28:11436/health
/venvs/apps_venv/bin/python -u -m smart_home_agent.reachy_vision
```

Sustituye `192.168.0.28` si cambia la IP del ordenador. Es texto plano, sin
corchetes Markdown. El agente convierte `/vision/check` en `/vision/batch`;
también acepta la URL acabada directamente en `/vision/batch`.

Opciones: `--interval 1`, `--stable-frames 5`, `--stable-seconds 1`,
`--min-face 120`, `--detector-model`, `--host localhost` y
`--connection-mode localhost_only`. Siempre se seleccionan 3 imágenes; el
número de capturas de la ventana debe ser al menos 3.

El archivo `requirements-vision-reachy.txt` admite NumPy 2 y exige wheels.
La instalación completa en un entorno vacío puede fallar porque PyGObject
necesita las dependencias nativas del robot; no es el procedimiento recomendado
para este Reachy. Utiliza el entorno ya verificado arriba.

El servicio es HTTP local sin autenticación ni TLS. Si el firewall bloquea,
permite el puerto 11436 solo desde la IP de Reachy. Por ejemplo, en PowerShell
administrador, sustituyendo su IP:

```powershell
$IP_REACHY = '192.168.0.50'
New-NetFirewallRule -DisplayName 'Reachy Mini - Vision' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11436 -RemoteAddress $IP_REACHY
```

## Contrato y anillo de validación

`POST /vision/batch`, `Content-Type: application/json`:

```json
{
  "version": 1,
  "frames": [
    {"jpeg_base64": "JPEG_1_EN_BASE64", "bbox": [10, 20, 150, 180], "quality": {"score": 3.2}, "captured_at": "2026-09-26T17:00:00+00:00"},
    {"jpeg_base64": "JPEG_2_EN_BASE64"},
    {"jpeg_base64": "JPEG_3_EN_BASE64"}
  ]
}
```

Se exigen tres JPEG diferentes de hasta 2 MB cada uno, entre 32 y 2048 píxeles
por lado. Límite total del JSON: 8.1 MB. Se rechazan lotes duplicados o inválidos
antes de guardar o inferir. HTTP 503 indica otra inferencia en curso.

La respuesta incluye `user` (`Javi`, `mariola` o `unknow`), `status`,
`candidates` (distancia mediana, umbral, votos y número de capturas válidas),
`emotion`, `emotion_scores`, `frames` (los tres resultados individuales),
`capture_metadata`, `capture_id`, `request_id`, `received_at` y `processing_ms`.
Los fallos individuales quedan en `frames`; no aportan voto. Si no hay dos
comparaciones válidas, el resultado es `insufficient_valid_frames`.
El endpoint JPEG antiguo `/vision/check` sigue disponible para diagnóstico
individual, sin consenso de lote.

En `temp_data/vision/` se conservan **100 peticiones**, hasta 300 JPEG y 100
JSON. Cada slot utiliza `face_000.jpg`, `face_000_1.jpg`, `face_000_2.jpg` y
`face_000.json`, hasta el slot 099. Los índices coinciden con el orden de
`frames` y `capture_metadata`. El cursor persiste al reiniciar y se sobrescribe
el lote más antiguo completo. Una petición JPEG antigua reutiliza el mismo
anillo y elimina las imágenes adicionales del slot sobrescrito. No se borran
las capturas existentes durante la actualización; se sustituyen al rotar.

Se guardan antes de inferir y el JSON final conserva los errores. Un corte
abrupto puede dejar un registro `processing`; no garantiza una transacción de
varios archivos frente a un apagado. Usa un solo servidor por directorio.
`capture_id` se reutiliza; `request_id` distingue peticiones. Referencias,
capturas, pesos y entornos están excluidos de Git.

## Pruebas

```powershell
& ".\.venv-vision\Scripts\python.exe" -m unittest tests.test_vision tests.test_vision_batch -v
```

Se prueban estabilidad, selección, transporte, persistencia, votos, medianas,
incertidumbre de expresión y errores. Las pruebas automáticas simulan la
inferencia; no demuestran exactitud de reconocimiento ni rendimiento físico.
