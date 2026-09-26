"""Agente de cámara independiente: YuNet, recorte JPEG y servicio de visión."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

LOG = logging.getLogger(__name__)


def crop_bounds(box, width: int, height: int, margin: float = 0.25):
    x, y, w, h = [float(value) for value in box[:4]]
    return (max(0, int(x - w * margin)), max(0, int(y - h * margin)),
            min(width, int(x + w * (1 + margin))), min(height, int(y + h * (1 + margin))))


def send_face(endpoint: str, jpeg: bytes, timeout: float = 30) -> dict:
    request = Request(endpoint, data=jpeg, headers={"Content-Type": "image/jpeg"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if result.get("user") not in {"Javi", "mariola", "unknow"}:
        raise ValueError("Respuesta de identidad inválida")
    return result


def run_camera(robot, detector, endpoint: str, interval: float, min_face: int):
    import cv2

    last_sent = float("-inf")
    while True:
        frame = robot.media.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)))
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        if faces is not None and time.monotonic() - last_sent >= interval:
            # Envía cada cara por separado; el resultado no identifica al hablante.
            for face in faces:
                box = face[:4] / scale
                if min(box[2:4]) < min_face:
                    continue
                left, top, right, bottom = crop_bounds(box, width, height)
                crop = frame[top:bottom, left:right]
                if not crop.size:
                    continue
                if max(crop.shape[:2]) > 640:
                    factor = 640 / max(crop.shape[:2])
                    crop = cv2.resize(crop, (round(crop.shape[1] * factor), round(crop.shape[0] * factor)))
                ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                try:
                    result = send_face(endpoint, jpeg.tobytes())
                    print(json.dumps({"bbox": [int(value) for value in box], **result}, ensure_ascii=False), flush=True)
                except (URLError, TimeoutError, ValueError) as exc:
                    LOG.warning("Servicio de visión no disponible: %s", exc)
                    break
            last_sent = time.monotonic()
        time.sleep(0.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("REMOTE_VISION_URL", "http://127.0.0.1:11436/vision/check"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--host", default=os.getenv("REACHY_HOST", "localhost"))
    parser.add_argument("--connection-mode", default=os.getenv("REACHY_CONNECTION_MODE", "localhost_only"))
    parser.add_argument("--interval", type=float, default=2.0, help="Segundos mínimos entre lotes enviados")
    parser.add_argument("--min-face", type=int, default=80, help="Tamaño mínimo de cara en píxeles originales")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0 or args.min_face < 32:
        parser.error("interval debe ser > 0 y min-face >= 32")
    if not args.detector_model.is_file():
        parser.error(f"Falta el modelo YuNet: {args.detector_model}; consulta docs/VISION_DEPLOYMENT.md")
    import cv2
    from reachy_mini import ReachyMini

    logging.basicConfig(level=logging.INFO)
    detector = cv2.FaceDetectorYN.create(str(args.detector_model), "", (640, 480), 0.9)
    try:
        with ReachyMini(host=args.host, connection_mode=args.connection_mode, media_backend="default") as robot:
            run_camera(robot, detector, args.url, args.interval, args.min_face)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
