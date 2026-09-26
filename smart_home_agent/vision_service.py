"""Servicio independiente de identidad y expresión facial (sin audio)."""
from __future__ import annotations

import argparse
import json
import io
import logging
import math
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = logging.getLogger(__name__)
USERS = {"javi": "Javi", "mariola": "mariola"}


class CaptureRing:
    """100 pares JPEG/JSON; continúa por el registro más antiguo al reiniciar."""

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
        with self.lock:
            path = self.directory / f"face_{self.next_slot:03d}.jpg"
            for target, content in ((path, data), (path.with_suffix(".json"), json.dumps(result, ensure_ascii=False).encode())):
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(content)
                temporary.replace(target)
            self.next_slot = (self.next_slot + 1) % self.size
            temporary = self.cursor_path.with_suffix(".json.part")
            temporary.write_text(json.dumps({"next_slot": self.next_slot}), encoding="utf-8")
            temporary.replace(self.cursor_path)
            return path.stem


def choose_identity(candidates: list[dict], margin: float) -> tuple[str, str]:
    if any(not math.isfinite(item[key]) for item in candidates for key in ("distance", "threshold")):
        return "unknow", "invalid_scores"
    ordered = sorted(candidates, key=lambda item: item["distance"])
    if not ordered:
        return "unknow", "empty_gallery"
    best = ordered[0]
    if best["distance"] > best["threshold"]:
        return "unknow", "no_match"
    if len(ordered) > 1 and ordered[1]["distance"] - best["distance"] < margin:
        return "unknow", "ambiguous"
    return best["user"], "matched"


class FaceEngine:
    def __init__(self, directory: Path, model: str = "Facenet512", threshold: float | None = None, margin: float = 0.05):
        from deepface import DeepFace

        self.deepface, self.model = DeepFace, model
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
        return self.deepface.represent(img_path=image, model_name=self.model, detector_backend="opencv", enforce_detection=True, align=True)

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
            if self.path != "/vision/check":
                self.reply(404, {"error": "not_found"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "image/jpeg":
                self.reply(415, {"error": "expected_image_jpeg"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError()
            except ValueError:
                self.reply(413, {"error": "invalid_image_length"})
                return
            try:
                data = self.rfile.read(length)
            except TimeoutError:
                self.reply(408, {"error": "upload_timeout"})
                return
            # Comprueba el formato y tamaño antes de descomprimir con OpenCV.
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(io.BytesIO(data)) as encoded:
                    if encoded.format != "JPEG" or min(encoded.size) < 32 or max(encoded.size) > 2048:
                        raise ValueError("invalid_image")
                    encoded.verify()
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                self.reply(400, {"error": "invalid_image"})
                return
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if len(data) != length or image is None or min(image.shape[:2]) < 32 or max(image.shape[:2]) > 2048:
                self.reply(400, {"error": "invalid_image"})
                return
            if not inference_lock.acquire(blocking=False):
                self.reply(503, {"error": "busy"})
                return
            try:
                metadata = {"request_id": str(uuid.uuid4()), "received_at": datetime.now(timezone.utc).isoformat()}
                # Guarda antes de inferir para conservar también las capturas que fallen.
                slot = ring.save(data, {**metadata, "status": "processing"})
                code = 200
                try:
                    result = {**metadata, **engine.analyze(image), "capture_id": slot}
                except Exception:
                    LOG.exception("Fallo de inferencia")
                    code = 500
                    result = {**metadata, "user": "unknow", "emotion": None, "status": "inference_error", "capture_id": slot}
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
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de distancia coseno; por defecto el de DeepFace")
    parser.add_argument("--margin", type=float, default=0.05, help="Separación mínima entre candidatos")
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0 or (args.threshold is not None and not 0 < args.threshold <= 2):
        parser.error("margin debe ser >= 0 y threshold estar en (0, 2]")
    logging.basicConfig(level=logging.INFO)
    engine = FaceEngine(args.users_dir, threshold=args.threshold, margin=args.margin)
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
