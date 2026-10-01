"""Servicio independiente de identidad y expresión facial (sin audio)."""
from __future__ import annotations

import argparse
import base64
import binascii
import time
import json
import io
import logging
import math
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .vision_batch import aggregate_batch, choose_identity

LOG = logging.getLogger(__name__)
USERS = {"javi": "Javi", "mariola": "mariola"}


class CaptureRing:
    """100 registros de hasta tres JPEG y un JSON; el cursor persiste al reiniciar."""

    def __init__(self, directory: Path, size: int = 100):
        if not 1 <= size <= 100:
            raise ValueError("El anillo debe contener entre 1 y 100 registros")
        self.directory, self.size = directory, size
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cursor_path = directory / "cursor.json"
        try:
            self.next_slot = int(json.loads(self.cursor_path.read_text())["next_slot"]) % size
        except (OSError, ValueError, KeyError, TypeError):
            paths = [directory / f"face_{i:03d}.jpg" for i in range(size)]
            self.next_slot = next((i for i, p in enumerate(paths) if not p.exists()),
                                  min(range(size), key=lambda i: paths[i].stat().st_mtime_ns if paths[i].exists() else 0))

    def save(self, data: bytes, result: dict) -> str:
        return self.save_batch([data], result)

    def save_batch(self, images: list[bytes], result: dict) -> str:
        if not 1 <= len(images) <= 3:
            raise ValueError("Se admiten de una a tres imágenes")
        with self.lock:
            path = self.directory / f"face_{self.next_slot:03d}.jpg"
            image_paths = [path] + [path.with_name(f"{path.stem}_{i}.jpg") for i in range(1, len(images))]
            for i in range(len(images), 3):
                path.with_name(f"{path.stem}_{i}.jpg").unlink(missing_ok=True)
            entries = list(zip(image_paths, images)) + [(path.with_suffix(".json"), json.dumps(result, ensure_ascii=False).encode())]
            for target, content in entries:
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(content)
                temporary.replace(target)
            self.next_slot = (self.next_slot + 1) % self.size
            temporary = self.cursor_path.with_suffix(".json.part")
            temporary.write_text(json.dumps({"next_slot": self.next_slot}), encoding="utf-8")
            temporary.replace(self.cursor_path)
            return path.stem




class FaceEngine:
    def __init__(self, directory: Path, model: str = "Facenet512", threshold: float | None = 0.5, margin: float = 0.05, detector: str = "yunet"):
        from deepface import DeepFace

        self.deepface, self.model = DeepFace, model
        self.detector = detector
        self.threshold, self.margin = threshold, margin
        self.gallery: dict[str, list] = {}
        for folder, name in USERS.items():
            location = directory / folder
            location.mkdir(parents=True, exist_ok=True)
            embeddings = []
            for path in sorted(location.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    faces = self.represent(str(path))
                except ValueError as exc:
                    if "Face could not be detected" in str(exc):
                        LOG.warning("Referencia omitida (no se detectó una cara): %s", path)
                        continue
                    raise ValueError(f"Referencia inválida: {path}: {exc}") from exc
                if len(faces) != 1:
                    LOG.warning("Referencia omitida (se requiere exactamente una cara; detectadas %s): %s", len(faces), path)
                    continue
                embeddings.append(faces[0]["embedding"])
            self.gallery[name] = embeddings
            if not embeddings:
                LOG.warning("Sin referencias para %s: no podrá ser reconocido", name)
        if not any(self.gallery.values()):
            raise ValueError(f"Añade fotos a {directory}/javi y {directory}/mariola antes de arrancar")
        DeepFace.build_model("Emotion", task="facial_attribute")
        LOG.info("Galería cargada: %s", {user: len(rows) for user, rows in self.gallery.items()})

    def represent(self, image):
        return self.deepface.represent(img_path=image, model_name=self.model, detector_backend=self.detector, enforce_detection=True, align=True)

    def analyze(self, image) -> dict:
        base = {"user": "unknow", "emotion": None, "emotion_scores": {}, "model": self.model}
        try:
            faces = self.represent(image)
        except ValueError as exc:
            if "Face could not be detected" in str(exc):
                return {**base, "status": "no_face"}
            raise
        if len(faces) != 1:
            return {**base, "status": "multiple_faces"}
        candidates = []
        for user, references in self.gallery.items():
            comparisons = [self.deepface.verify(
                faces[0]["embedding"], reference, model_name=self.model,
                distance_metric="cosine", threshold=self.threshold, silent=True,
            ) for reference in references]
            if comparisons:
                best = min(comparisons, key=lambda item: item["distance"])
                candidates.append({"user": user, "distance": float(best["distance"]), "threshold": float(best["threshold"])})
        user, status = choose_identity(candidates, self.margin)
        result = {**base, "user": user, "status": status, "candidates": candidates}
        # Reutiliza la región validada para analizar la misma cara, sin inferir otros atributos.
        area = faces[0]["facial_area"]
        x, y, w, h = (int(area[key]) for key in ("x", "y", "w", "h"))
        crop = image[max(0, y):min(image.shape[0], y + h), max(0, x):min(image.shape[1], x + w)]
        try:
            analysis = self.deepface.analyze(crop, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)[0]
            result.update(emotion=analysis["dominant_emotion"], emotion_scores={key: float(value) for key, value in analysis["emotion"].items()})
        except Exception:
            LOG.exception("Fallo de análisis de expresión")
            result["emotion_error"] = "analysis_failed"
        return result

    def analyze_batch(self, images):
        results = []
        for index, image in enumerate(images):
            started = time.monotonic()
            LOG.info("Captura %s/3: detección %s y embedding %s", index + 1, self.detector, self.model)
            try:
                result = self.analyze(image)
            except Exception:
                LOG.exception("Captura %s: fallo de inferencia", index + 1)
                result = {"user": "unknow", "status": "inference_error", "emotion": None, "emotion_scores": {}}
            result["processing_ms"] = round((time.monotonic() - started) * 1000)
            result["index"] = index
            trace_result(f"Captura {index + 1}", result)
            results.append(result)
        result = aggregate_batch(results, self.margin)
        result.update(model=self.model, detector=self.detector)
        return result


def trace_result(label, result):
    LOG.info("%s: usuario=%s estado=%s", label, result.get("user"), result.get("status"))
    for candidate in result.get("candidates", []):
        LOG.info("%s: %s distancia=%.4f umbral=%.4f votos=%s", label, candidate["user"],
                 candidate["distance"], candidate["threshold"], candidate.get("votes", "individual"))
    LOG.info("%s: emoción=%s puntuaciones=%s", label, result.get("emotion"),
             {key: round(value, 2) for key, value in result.get("emotion_scores", {}).items()})


def decode_jpeg(data):
    from PIL import Image
    import cv2
    import numpy as np
    if not 0 < len(data) <= 2_000_000:
        raise ValueError("invalid_image_length")
    try:
        with Image.open(io.BytesIO(data)) as encoded:
            if encoded.format != "JPEG" or min(encoded.size) < 32 or max(encoded.size) > 2048:
                raise ValueError("invalid_image")
            encoded.verify()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid_image") from exc
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid_image")
    return image


def parse_batch(data):
    try:
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("invalid_batch_version")
        frames = payload.get("frames")
        if not isinstance(frames, list) or len(frames) != 3:
            raise ValueError("expected_three_frames")
        raw, images, metadata = [], [], []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("jpeg_base64"), str):
                raise ValueError("invalid_frame")
            jpeg = base64.b64decode(frame["jpeg_base64"], validate=True)
            raw.append(jpeg)
            images.append(decode_jpeg(jpeg))
            # Solo metadatos acotados; nunca aceptar rutas de archivos del cliente.
            info = {key: frame[key] for key in ("bbox", "quality", "captured_at") if key in frame}
            if len(json.dumps(info, allow_nan=False)) > 2048:
                raise ValueError("metadata_too_large")
            metadata.append(info)
        if len(set(raw)) != 3:
            raise ValueError("duplicate_frames")
        return raw, images, metadata
    except (TypeError, UnicodeError, binascii.Error) as exc:
        raise ValueError("invalid_batch") from exc


def make_handler(engine, ring: CaptureRing):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "users": {key: len(value) for key, value in engine.gallery.items()}})
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            is_batch = self.path == "/vision/batch"
            if self.path not in {"/vision/check", "/vision/batch"}:
                self.reply(404, {"error": "not_found"})
                return
            expected_type = "application/json" if is_batch else "image/jpeg"
            if self.headers.get("Content-Type", "").split(";")[0] != expected_type:
                self.reply(415, {"error": "expected_" + expected_type})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (8_100_000 if is_batch else 2_000_000):
                    raise ValueError()
            except ValueError:
                self.reply(413, {"error": "invalid_image_length"})
                return
            try:
                data = self.rfile.read(length)
            except TimeoutError:
                self.reply(408, {"error": "upload_timeout"})
                return
            try:
                if len(data) != length:
                    raise ValueError("incomplete_upload")
                if is_batch:
                    raw, images, capture_metadata = parse_batch(data)
                else:
                    raw, images, capture_metadata = [data], [decode_jpeg(data)], [{}]
            except ValueError as exc:
                self.reply(400, {"error": str(exc)})
                return
            if not inference_lock.acquire(blocking=False):
                self.reply(503, {"error": "busy"})
                return
            try:
                metadata = {"request_id": str(uuid.uuid4()), "received_at": datetime.now(timezone.utc).isoformat()}
                # Guarda antes de inferir para conservar también las capturas que fallen.
                started = time.monotonic()
                LOG.info("[%s] Recibido: %s capturas, %s bytes", metadata["request_id"], len(raw), length)
                slot = ring.save_batch(raw, {**metadata, "status": "processing", "capture_metadata": capture_metadata})
                LOG.info("[%s] Guardado en %s", metadata["request_id"], slot)
                code = 200
                try:
                    analysis = engine.analyze_batch(images) if is_batch else engine.analyze(images[0])
                    result = {**metadata, **analysis, "capture_id": slot}
                except Exception:
                    LOG.exception("Fallo de inferencia")
                    code = 500
                    result = {**metadata, "user": "unknow", "emotion": None, "status": "inference_error", "capture_id": slot}
                result["capture_metadata"] = capture_metadata
                result["processing_ms"] = round((time.monotonic() - started) * 1000)
                trace_result(f"[{metadata['request_id']}] Resultado {slot}", result)
                LOG.info("[%s] Duración total: %s ms", metadata["request_id"], result["processing_ms"])
                # Convierte aquí para que resultados no finitos sean un error explícito.
                try:
                    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
                except (ValueError, TypeError):
                    code = 500
                    result = {**metadata, "user": "unknow", "emotion": None, "status": "invalid_result", "capture_id": slot}
                    serialized = json.dumps(result)
                target = ring.directory / f"{slot}.json"
                temporary = target.with_suffix(".json.part")
                temporary.write_text(serialized, encoding="utf-8")
                temporary.replace(target)
                self.reply(code, result)
            except OSError:
                LOG.exception("Error guardando o devolviendo la captura")
                self.reply(500, {"error": "storage_or_connection_error"})
            finally:
                inference_lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--users-dir", type=Path, default=Path("vision_data/users"))
    parser.add_argument("--save-dir", type=Path, default=Path("temp_data/vision"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de distancia coseno (por defecto: 0.5)")
    parser.add_argument("--margin", type=float, default=0.05, help="Separación mínima entre candidatos")
    parser.add_argument("--detector", choices=["yunet", "opencv"], default="yunet", help="Detector común para referencias y capturas (yunet)")
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0 or (args.threshold is not None and not 0 < args.threshold <= 2):
        parser.error("margin debe ser >= 0 y threshold estar en (0, 2]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = FaceEngine(args.users_dir, threshold=args.threshold, margin=args.margin, detector=args.detector)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, CaptureRing(args.save_dir)))
    LOG.info("Visión lista en http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
