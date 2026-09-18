import unittest

from smart_home_agent.human_console_bridge import assistant_message, tool_message


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
