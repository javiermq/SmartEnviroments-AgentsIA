import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smart_home_agent.human_console_bridge import ReceivedWavRing, assistant_message, tool_message


class HumanConsoleBridgeTests(unittest.TestCase):
    def test_plain_answer_is_native_ollama_message(self):
        response = assistant_message("Has dormido 8 horas.")
        self.assertTrue(response["done"])
        self.assertEqual(response["message"], {"role": "assistant", "content": "Has dormido 8 horas."})

    def test_tool_answer_has_expected_shape(self):
        response = tool_message({"sensor_type": "watch.sleep"})
        call = response["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "query_aggregation")
        self.assertEqual(call["arguments"]["sensor_type"], "watch.sleep")

    def test_received_wav_ring_overwrites_first_slot_after_capacity(self):
        with TemporaryDirectory() as temporary:
            ring = ReceivedWavRing(Path(temporary), size=2)
            first = ring.save(b"RIFF-first")
            second = ring.save(b"RIFF-second")
            overwritten = ring.save(b"RIFF-third")
            self.assertEqual(first, overwritten)
            self.assertEqual(first.read_bytes(), b"RIFF-third")
            self.assertEqual(second.read_bytes(), b"RIFF-second")
