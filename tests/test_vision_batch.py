import base64
import json
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from smart_home_agent.reachy_vision import StableFace, batch_endpoint, select_best, send_batch
from smart_home_agent.vision_batch import aggregate_batch
from smart_home_agent.vision_service import CaptureRing, FaceEngine
import tests.test_vision as legacy
HAS_VISION = legacy.HAS_VISION


def result(javi=.2, mariola=.7, user="Javi", emotion=None):
    return {"user": user, "status": "matched" if user != "unknow" else "no_match",
            "candidates": [{"user": "Javi", "distance": javi, "threshold": .3},
                           {"user": "mariola", "distance": mariola, "threshold": .3}],
            "emotion_scores": emotion or {"happy": 80., "neutral": 20.}}


class BatchTests(unittest.TestCase):
    def test_outlier_does_not_override_two_matching_votes(self):
        combined = aggregate_batch([result(), result(.25), result(.8, .7, "unknow")])
        self.assertEqual(combined["user"], "Javi")
        self.assertEqual(combined["candidates"][0]["distance"], .25)
        self.assertEqual(combined["emotion"], "happy")

    def test_single_good_capture_is_not_enough(self):
        self.assertEqual(aggregate_batch([result(), result(.7, .8, "unknow"), result(.8, .7, "unknow")])["user"], "unknow")
        self.assertEqual(aggregate_batch([result(), {"status": "no_face"}, {"status": "no_face"}])["status"], "insufficient_valid_frames")
        combined = aggregate_batch([result(), result(.21, .6, "unknow"), result(.22, .6, "unknow")])
        self.assertEqual(combined["status"], "insufficient_agreement")

    def test_emotion_tie_or_single_valid_prediction_is_uncertain(self):
        near_tie = {"angry": 49., "neutral": 51.}
        self.assertEqual(aggregate_batch([result(emotion=near_tie)] * 3)["emotion"], "uncertain")
        self.assertEqual(aggregate_batch([result(), {"status": "no_face"}, {"status": "no_face"}])["emotion"], "uncertain")

    def test_stability_resets_on_absence_drift_and_time_gap(self):
        stable = StableFace(1., 5)
        box = [100, 100, 120, 140]
        for t in [0, .25, .5, .75]:
            self.assertFalse(stable.update(box, t))
        self.assertTrue(stable.update(box, 1.))
        self.assertFalse(stable.update(None, 1.1))
        self.assertFalse(stable.update(box, 1.2))
        self.assertFalse(stable.update([300, 100, 120, 140], 1.3))
        self.assertEqual(stable.count, 1)
        self.assertFalse(stable.update(box, 4.))
        self.assertEqual(stable.count, 1)

    def test_quality_selection_and_url(self):
        samples = [{"quality": {"score": score}} for score in [1, 5, 2, 4, 3]]
        self.assertEqual([s["quality"]["score"] for s in select_best(samples)], [5, 4, 3])
        self.assertEqual(batch_endpoint("http://localhost:11436/vision/check"), "http://localhost:11436/vision/batch")
        with self.assertRaises(ValueError):
            batch_endpoint("[http://localhost](http://localhost)")

    def test_ring_replaces_whole_batch_and_removes_stale_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ring = CaptureRing(root, size=2)
            ring.save_batch([b"a", b"b", b"c"], {"sequence": 0})
            ring.save_batch([b"d", b"e", b"f"], {"sequence": 1})
            ring = CaptureRing(root, size=2)
            ring.save(b"new", {"sequence": 2})
            self.assertEqual((root / "face_000.jpg").read_bytes(), b"new")
            self.assertFalse((root / "face_000_1.jpg").exists())
            self.assertFalse((root / "face_000_2.jpg").exists())
            self.assertEqual((root / "face_001_2.jpg").read_bytes(), b"f")

    def test_ring_bounds_three_image_batches_to_100_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ring = CaptureRing(root)
            for i in range(103):
                ring.save_batch([str(i).encode()] * 3, {"sequence": i})
            self.assertEqual(len(list(root.glob("*.jpg"))), 300)
            records = [json.loads(path.read_text()) for path in root.glob("face_*.json")]
            self.assertEqual(sorted(row["sequence"] for row in records), list(range(3, 103)))

    def test_engine_batch_keeps_failed_frame_and_uses_other_two(self):
        engine = FaceEngine.__new__(FaceEngine)
        engine.margin, engine.detector, engine.model = .05, "yunet", "Facenet512"
        engine.analyze = Mock(side_effect=[result(), RuntimeError("bad frame"), result(.25)])
        with self.assertLogs("smart_home_agent.vision_service", level="INFO") as logs:
            combined = engine.analyze_batch([None] * 3)
        self.assertEqual(combined["user"], "Javi")
        self.assertEqual(combined["frames"][1]["status"], "inference_error")
        self.assertTrue(any("Captura 3" in line for line in logs.output))

    @unittest.skipUnless(HAS_VISION, "Instala requirements-vision-server.txt")
    def test_frozen_camera_cannot_make_a_batch(self):
        import numpy as np
        from smart_home_agent.reachy_vision import run_camera
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot, detector = Mock(), Mock()
        robot.media.get_frame.side_effect = [frame] * 10 + [KeyboardInterrupt()]
        detector.detect.return_value = (None, np.array([[100, 100, 150, 150] + [0] * 11], dtype=float))
        with patch("smart_home_agent.reachy_vision.time.sleep"), patch("smart_home_agent.reachy_vision.send_batch") as send:
            with self.assertRaises(KeyboardInterrupt):
                run_camera(robot, detector, "http://test/vision/check", 1, 120)
        send.assert_not_called()


@unittest.skipUnless(HAS_VISION, "Instala requirements-vision-server.txt")
class BatchHTTPTests(unittest.TestCase):
    setUp = legacy.VisionHTTPTests.setUp
    tearDown = legacy.VisionHTTPTests.tearDown
    def samples(self):
        import cv2
        import numpy as np
        return [{"jpeg": cv2.imencode(".jpg", np.full((100, 120, 3), value, dtype=np.uint8))[1].tobytes(),
                 "quality": {"score": float(value)}} for value in [50, 100, 150]]

    def test_batch_round_trip_and_traces(self):
        self.engine.analyze_batch.return_value = aggregate_batch([result()] * 3)
        with self.assertLogs("smart_home_agent.vision_service", level="INFO") as logs:
            response = send_batch(self.base + "/vision/check", self.samples())
        self.assertEqual(response["user"], "Javi")
        self.assertEqual(len(response["frames"]), 3)
        self.assertEqual(len(list(Path(self.directory.name).glob("*.jpg"))), 3)
        self.assertTrue(any("distancia=" in line for line in logs.output))
        self.assertTrue(any("puntuaciones=" in line for line in logs.output))

    def test_bad_batch_rejected_before_inference(self):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        raw = base64.b64encode(self.jpeg).decode()
        for payload in [{"version": 1, "frames": []}, {"version": 1, "frames": [{"jpeg_base64": raw}] * 3},
                        {"version": 1, "frames": [{"jpeg_base64": "bad!"}] * 3}]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/vision/batch", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}))
            self.assertEqual(caught.exception.code, 400)
        self.engine.analyze_batch.assert_not_called()
        self.assertEqual(list(Path(self.directory.name).glob("*.jpg")), [])


if __name__ == "__main__":
    unittest.main()
