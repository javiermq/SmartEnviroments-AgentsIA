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
        self.assertEqual(backend.respond("¿Cuántos pasos hice?"), "Has hecho 4855 pasos.")


if __name__ == "__main__":
    unittest.main()
