import unittest

from smart_home_agent.conversation_backend import ConversationBackend


class ConversationBackendTests(unittest.TestCase):
    def test_filters_model_thinking_markup(self):
        self.assertEqual(ConversationBackend._visible_answer("hidden</think>Respuesta"), "Respuesta")

    def test_executes_sensor_tool_and_returns_answer(self):
        backend = ConversationBackend("2026-09-16T18:56:00+02:00")
        responses = iter([
            {"content": "", "tool_calls": [{"function": {"name": "query_aggregation", "arguments": {"sensor_type": "watch.steps", "time_init": "09:00", "time_end": "09:40", "aggregation": "total"}}}]},
            {"content": "Has hecho 4855 pasos.", "tool_calls": []},
        ])
        backend._request_ollama = lambda: next(responses)  # type: ignore[method-assign]
        answer = backend.respond("Consulta mi actividad")
        self.assertNotIn("4855", answer)
        self.assertTrue(any(m.get('role') == 'tool' for m in backend.messages))

    def test_keeps_gesture_from_bridge_response(self):
        backend = ConversationBackend("2026-09-16T18:56:00+02:00")
        backend._request_ollama = lambda: {  # type: ignore[method-assign]
            "content": "Hola.", "tool_calls": [],
            "gesture": {"name": "curiosa", "antennas_deg": [20, 8]},
        }
        self.assertEqual(backend.respond("Hola"), "Hola.")
        self.assertEqual(backend.last_gesture, {"name": "curiosa", "antennas_deg": [20, 8]})


if __name__ == "__main__":
    unittest.main()
