import json
import threading
import sys
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from smart_home_agent.reachy_vision import crop_bounds, run_camera, send_face
from smart_home_agent.vision_service import CaptureRing, FaceEngine, choose_identity, make_handler

try:
    import cv2
    import numpy as np
    from PIL import Image
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


class IdentityTests(unittest.TestCase):
    def test_gallery_requires_faces_and_caches_each_reference(self):
        deepface = Mock()
        deepface.represent.return_value = [{"embedding": [1.0]}]
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"deepface": SimpleNamespace(DeepFace=deepface)}):
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Añade fotos"):
                FaceEngine(root)
            (root / "javi" / "one.jpg").write_bytes(b"test fixture")
            (root / "mariola" / "two.png").write_bytes(b"test fixture")
            engine = FaceEngine(root)
            self.assertEqual(engine.gallery, {"Javi": [[1.0]], "mariola": [[1.0]]})
            self.assertEqual(deepface.represent.call_count, 2)
            deepface.represent.return_value = [{"embedding": [1.0]}, {"embedding": [2.0]}]
            with self.assertRaisesRegex(ValueError, "Añade fotos"):
                FaceEngine(root)

    def test_missing_face_skipped_but_model_error_not_hidden(self):
        deepface = Mock()
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"deepface": SimpleNamespace(DeepFace=deepface)}):
            root = Path(tmp)
            (root / "javi").mkdir()
            for name in ("j1.png", "j6.png"):
                (root / "javi" / name).write_bytes(b"fixture")
            deepface.represent.side_effect = [
                [{"embedding": [1.0]}], ValueError("Face could not be detected in j6.png")]
            with self.assertLogs("smart_home_agent.vision_service", level="WARNING") as logs:
                engine = FaceEngine(root)
            self.assertEqual(engine.gallery["Javi"], [[1.0]])
            self.assertTrue(any("j6.png" in line for line in logs.output))
            deepface.represent.side_effect = ValueError("Face could not be detected")
            with self.assertRaisesRegex(ValueError, "Añade fotos"):
                FaceEngine(root)
            deepface.represent.side_effect = ValueError("broken model")
            with self.assertRaisesRegex(ValueError, "broken model"):
                FaceEngine(root)

    def test_threshold_ambiguity_and_empty_gallery(self):
        javi = {"user": "Javi", "distance": 0.2, "threshold": 0.3}
        mariola = {"user": "mariola", "distance": 0.6, "threshold": 0.3}
        self.assertEqual(choose_identity([mariola, javi], 0.05), ("Javi", "matched"))
        self.assertEqual(choose_identity([mariola], 0.05), ("unknow", "no_match"))
        self.assertEqual(choose_identity([javi, {**mariola, "distance": 0.22}], 0.05), ("unknow", "ambiguous"))
        self.assertEqual(choose_identity([], 0.05), ("unknow", "empty_gallery"))
        self.assertEqual(choose_identity([{**javi, "distance": float("nan")}], 0.05), ("unknow", "invalid_scores"))

    def test_ring_preserves_latest_100_across_restart(self):
        with TemporaryDirectory() as tmp:
            ring = CaptureRing(Path(tmp))
            for i in range(105):
                ring.save(str(i).encode(), {"sequence": i})
            ring = CaptureRing(Path(tmp))
            self.assertEqual(ring.save(b"105", {"sequence": 105}), "face_005")
            records = [json.loads(p.read_text()) for p in Path(tmp).glob("face_*.json")]
            self.assertEqual(sorted(row["sequence"] for row in records), list(range(6, 106)))
            self.assertEqual(len(list(Path(tmp).glob("face_*.jpg"))), 100)
            for path in Path(tmp).glob("face_*.jpg"):
                self.assertEqual(int(path.read_bytes()), json.loads(path.with_suffix(".json").read_text())["sequence"])

    def test_crop_margin_clipped_at_frame_edges(self):
        self.assertEqual(crop_bounds([0, 0, 100, 100], 120, 110), (0, 0, 120, 110))
        self.assertEqual(crop_bounds([100, 100, 80, 80], 640, 480), (80, 80, 200, 200))


@unittest.skipUnless(HAS_VISION, "Instala requirements-vision-server.txt")
class VisionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.engine = Mock()
        self.engine.gallery = {"Javi": [[0.1]], "mariola": [[0.2]]}
        self.engine.analyze.return_value = {"user": "Javi", "emotion": "happy", "status": "matched"}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine, CaptureRing(Path(self.directory.name))))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.jpeg = cv2.imencode(".jpg", np.full((100, 120, 3), 100, dtype=np.uint8))[1].tobytes()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def test_client_server_and_persisted_result(self):
        result = send_face(self.base + "/vision/check", self.jpeg)
        self.assertEqual(result["user"], "Javi")
        root = Path(self.directory.name)
        self.assertEqual((root / (result["capture_id"] + ".jpg")).read_bytes(), self.jpeg)
        self.assertEqual(json.loads((root / (result["capture_id"] + ".json")).read_text()), result)
        with urlopen(self.base + "/health") as response:
            self.assertEqual(json.load(response)["users"]["mariola"], 1)

    def test_invalid_uploads_do_not_reach_engine_or_ring(self):
        for content_type, data, status in [("text/plain", b"bad", 415), ("image/jpeg", b"bad", 400), ("image/jpeg", b"", 413)]:
            with self.subTest(status=status), self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/vision/check", data=data, headers={"Content-Type": content_type}))
            self.assertEqual(caught.exception.code, status)
        self.engine.analyze.assert_not_called()
        self.assertEqual(list(Path(self.directory.name).glob("*.jpg")), [])

    def test_wrong_format_and_oversized_dimensions_rejected(self):
        png = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))[1].tobytes()
        wide_jpeg = cv2.imencode(".jpg", np.zeros((32, 2049, 3), dtype=np.uint8))[1].tobytes()
        for data in [png, wide_jpeg]:
            with self.assertRaises(HTTPError) as caught:
                send_face(self.base + "/vision/check", data)
            self.assertEqual(caught.exception.code, 400)
        self.engine.analyze.assert_not_called()

    def test_inference_failure_keeps_capture_and_server_recovers(self):
        self.engine.analyze.side_effect = RuntimeError("test failure")
        with self.assertRaises(HTTPError) as caught:
            send_face(self.base + "/vision/check", self.jpeg)
        self.assertEqual(caught.exception.code, 500)
        record = json.loads((Path(self.directory.name) / "face_000.json").read_text())
        self.assertEqual(record["status"], "inference_error")
        self.engine.analyze.side_effect = None
        self.assertEqual(send_face(self.base + "/vision/check", self.jpeg)["user"], "Javi")

    def test_busy_returns_503_without_extra_capture(self):
        entered, release = threading.Event(), threading.Event()
        def slow(image):
            entered.set()
            release.wait(5)
            return {"user": "unknow", "emotion": "neutral"}
        self.engine.analyze.side_effect = slow
        worker = threading.Thread(target=send_face, args=(self.base + "/vision/check", self.jpeg))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaises(HTTPError) as caught:
                send_face(self.base + "/vision/check", self.jpeg)
            self.assertEqual(caught.exception.code, 503)
        finally:
            release.set()
            worker.join()
        self.assertEqual(len(list(Path(self.directory.name).glob("*.jpg"))), 1)


@unittest.skipUnless(HAS_VISION, "Instala requirements-vision-server.txt")
class CameraAndEngineTests(unittest.TestCase):
    def test_camera_sends_only_detected_crop_and_throttles(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot, detector = Mock(), Mock()
        robot.media.get_frame.side_effect = [frame, frame, frame, KeyboardInterrupt()]
        face = np.array([[100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .99]], dtype=np.float32)
        detector.detect.side_effect = [(None, None), (None, face), (None, face)]
        with patch("smart_home_agent.reachy_vision.send_face", return_value={"user": "Javi"}) as send, patch("smart_home_agent.reachy_vision.time.sleep"), patch("smart_home_agent.reachy_vision.time.monotonic", side_effect=[10, 10, 10.1]), patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                run_camera(robot, detector, "http://test", 2, 80)
        send.assert_called_once()
        decoded = cv2.imdecode(np.frombuffer(send.call_args.args[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (150, 150, 3))

    def test_engine_compares_cached_references_and_only_emotion(self):
        engine = FaceEngine.__new__(FaceEngine)
        engine.model, engine.threshold, engine.margin = "Facenet512", None, .05
        engine.gallery = {"Javi": [[1.0], [2.0]], "mariola": [[3.0]]}
        engine.deepface = Mock()
        engine.deepface.represent.return_value = [{"embedding": [4.0], "facial_area": {"x": 10, "y": 20, "w": 60, "h": 70}}]
        engine.deepface.verify.side_effect = [{"distance": d, "threshold": .3} for d in [.25, .15, .7]]
        engine.deepface.analyze.return_value = [{"dominant_emotion": "happy", "emotion": {"happy": np.float32(90)}}]
        result = engine.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result["user"], "Javi")
        self.assertEqual(result["emotion"], "happy")
        self.assertEqual(engine.deepface.analyze.call_args.kwargs["actions"], ["emotion"])
        self.assertEqual(engine.deepface.analyze.call_args.args[0].shape, (70, 60, 3))
        engine.deepface.represent.side_effect = ValueError("Face could not be detected in numpy array.")
        self.assertEqual(engine.analyze(None)["status"], "no_face")
        engine.deepface.represent.side_effect = ValueError("broken model")
        with self.assertRaises(ValueError):
            engine.analyze(None)

    def test_multiple_faces_and_failed_emotion_do_not_mix_identities(self):
        engine = FaceEngine.__new__(FaceEngine)
        engine.model, engine.threshold, engine.margin = "Facenet512", None, .05
        engine.gallery = {"Javi": [[1.0]]}
        engine.deepface = Mock()
        face = {"embedding": [2.0], "facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}}
        engine.deepface.represent.return_value = [face, face]
        self.assertEqual(engine.analyze(None)["status"], "multiple_faces")
        engine.deepface.verify.assert_not_called()
        engine.deepface.represent.return_value = [face]
        engine.deepface.verify.return_value = {"distance": .2, "threshold": .3}
        engine.deepface.analyze.side_effect = RuntimeError("emotion failure")
        with self.assertLogs("smart_home_agent.vision_service", level="ERROR"):
            result = engine.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result["user"], "Javi")
        self.assertIsNone(result["emotion"])
        self.assertEqual(result["emotion_error"], "analysis_failed")


if __name__ == "__main__":
    unittest.main()
