"""Agente de cámara independiente: YuNet, recorte JPEG y servicio de visión."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
import json
import logging
import math
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

LOG = logging.getLogger(__name__)


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0


class StableFace:
    """Exige una única cara completa y estable respecto al inicio de la ventana."""

    def __init__(self, seconds=1.0, frames=5, min_iou=0.65):
        self.seconds, self.frames, self.min_iou = seconds, frames, min_iou
        self.reset()

    def reset(self):
        self.anchor = None
        self.count = 0
        self.started = self.last_seen = None

    def update(self, box, now):
        if box is None:
            self.reset()
            return False
        if (self.anchor is None or now - self.last_seen > 1.0
                or box_iou(self.anchor, box) < self.min_iou):
            self.anchor = tuple(float(value) for value in box)
            self.count, self.started = 0, now
        self.last_seen = now
        self.count += 1
        return self.count >= self.frames and now - self.started >= self.seconds


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


def image_quality(face):
    import cv2
    gray = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float(((gray < 10) | (gray > 245)).mean())
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    return {"score": math.log1p(sharpness) * exposure * (1 - clipped),
            "sharpness": sharpness, "brightness": brightness, "clipped_fraction": clipped}


def select_best(samples):
    return sorted(samples, key=lambda sample: sample["quality"]["score"], reverse=True)[:3]


def batch_endpoint(endpoint):
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.hostname or "[" in endpoint:
        raise ValueError("URL inválida: usa http://IP:11436/vision/check sin formato Markdown")
    if parts.path not in {"/vision/check", "/vision/batch"}:
        raise ValueError("La URL debe terminar en /vision/check o /vision/batch")
    return urlunsplit((parts.scheme, parts.netloc, "/vision/batch", "", ""))


def send_batch(endpoint, samples, timeout=60):
    if len(samples) != 3:
        raise ValueError("El lote debe contener tres capturas")
    frames = [{**{key: value for key, value in sample.items() if key != "jpeg"},
               "jpeg_base64": base64.b64encode(sample["jpeg"]).decode("ascii")} for sample in samples]
    request = Request(batch_endpoint(endpoint), data=json.dumps({"version": 1, "frames": frames}, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get("user") not in {"Javi", "mariola", "unknow"}:
        raise ValueError("Respuesta de identidad inválida")
    return result


def run_camera(robot, detector, endpoint: str, interval: float, min_face: int,
               stable_seconds: float = 1.0, stable_frames: int = 5, event=None, stop=None, permit=None):
    import cv2
    import numpy as np

    last_sent = float("-inf")
    stability = StableFace(stable_seconds, stable_frames)
    samples = []
    previous_frame = None
    last_sample = float("-inf")
    sample_period = stable_seconds / (stable_frames - 1)
    recognizing = False
    recognition_started = 0.0
    while stop is None or not stop.is_set():
        if permit is not None and not permit.is_set():
            time.sleep(0.05)
            continue
        frame = robot.media.get_frame()
        now = time.monotonic()
        if frame is None:
            if recognizing and event:
                event("finished", None)
            recognizing = False
            stability.reset()
            samples.clear()
            time.sleep(0.05)
            continue
        if now - last_sent < interval:
            time.sleep(0.05)
            continue
        if recognizing and now - recognition_started > 8.0:
            recognizing = False
            stability.reset()
            samples.clear()
            if event:
                event("finished", None)
        # Una cámara congelada no debe producir votos duplicados.
        if previous_frame is not None and np.array_equal(frame, previous_frame):
            time.sleep(0.05)
            continue
        previous_frame = frame.copy()
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)))
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        box = None
        if faces is not None and len(faces) == 1:
            candidate = faces[0][:4] / scale
            x, y, w, h = candidate
            if min(w, h) >= min_face and x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                box = candidate
        if box is not None and not recognizing:
            recognizing = True
            recognition_started = now
            if event:
                event("recognizing", None)
        elif box is None and recognizing:
            recognizing = False
            if event:
                event("finished", None)
        ready = stability.update(box, now)
        if box is None or stability.count == 1:
            if samples:
                LOG.info("Lote reiniciado: cara ausente, múltiple, cortada o desplazada")
            samples.clear()
            last_sample = float("-inf")
        if box is not None and now - last_sample >= sample_period:
            x, y, w, h = [int(value) for value in box]
            quality = image_quality(frame[y:y + h, x:x + w])
            left, top, right, bottom = crop_bounds(box, width, height)
            crop = frame[top:bottom, left:right]
            if max(crop.shape[:2]) > 640:
                factor = 640 / max(crop.shape[:2])
                crop = cv2.resize(crop, (round(crop.shape[1] * factor), round(crop.shape[0] * factor)))
            ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok and all(sample["jpeg"] != jpeg.tobytes() for sample in samples):
                samples.append({"jpeg": jpeg.tobytes(), "bbox": [x, y, w, h], "quality": quality,
                                "captured_at": datetime.now(timezone.utc).isoformat()})
                samples = samples[-stable_frames:]
                last_sample = now
        if ready and len(samples) == stable_frames:
            selected = select_best(samples)
            LOG.info("Enviando 3 de %s capturas estables; calidad=%s", stable_frames,
                     [round(sample["quality"]["score"], 3) for sample in selected])
            try:
                result = send_batch(endpoint, selected)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if event:
                    event("identity", result)
            except (URLError, TimeoutError, ValueError) as exc:
                LOG.warning("Servicio de visión no disponible: %s", exc)
            finally:
                recognizing = False
                if event:
                    event("finished", None)
            last_sent = time.monotonic()
            stability.reset()
            samples.clear()
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("REMOTE_VISION_URL", "http://127.0.0.1:11436/vision/check"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--host", default=os.getenv("REACHY_HOST", "localhost"))
    parser.add_argument("--connection-mode", default=os.getenv("REACHY_CONNECTION_MODE", "localhost_only"))
    parser.add_argument("--interval", type=float, default=1.0, help="Segundos mínimos entre lotes enviados (por defecto: 1)")
    parser.add_argument("--min-face", type=int, default=120, help="Tamaño mínimo de cara en píxeles originales (120)")
    parser.add_argument("--stable-seconds", type=float, default=1.0, help="Duración mínima de estabilidad (1 s)")
    parser.add_argument("--stable-frames", type=int, default=5, help="Detecciones consecutivas mínimas (5)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0 or args.min_face < 32:
        parser.error("interval debe ser > 0 y min-face >= 32")
    if not math.isfinite(args.stable_seconds) or args.stable_seconds <= 0 or args.stable_frames < 3:
        parser.error("stable-seconds debe ser > 0 y stable-frames >= 3")
    try:
        batch_endpoint(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.detector_model.is_file():
        parser.error(f"Falta el modelo YuNet: {args.detector_model}; consulta docs/VISION_DEPLOYMENT.md")
    import cv2
    from reachy_mini import ReachyMini

    logging.basicConfig(level=logging.INFO)
    detector = cv2.FaceDetectorYN.create(str(args.detector_model), "", (640, 480), 0.9)
    try:
        with ReachyMini(host=args.host, connection_mode=args.connection_mode, media_backend="default") as robot:
            run_camera(robot, detector, args.url, args.interval, args.min_face, args.stable_seconds, args.stable_frames)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
