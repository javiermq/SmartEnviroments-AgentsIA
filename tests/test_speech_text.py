import unittest

from smart_home_agent.conversation_backend import ConversationBackend
from smart_home_agent.speech_text import for_speech, strip_icons


class SpeechTextTests(unittest.TestCase):
    def test_removes_composite_emojis_and_icons(self):
        self.assertEqual(strip_icons('Hola 👋🏽, familia 👨‍👩‍👧‍👦 🇪🇸 ❤️ ✅ ⭐!'),
                         'Hola, familia!')
        self.assertEqual(strip_icons('1️⃣ Descansa ☕.\n• Respira.\n- Sonríe.'),
                         'Descansa.\n Respira.\nSonríe.')

    def test_preserves_spanish_numbers_and_punctuation(self):
        text = '¡Ánimo, Javi! ¿Qué tal? 25 pasos, 3,5 km y -2 °C.'
        self.assertEqual(strip_icons(text), text)
        self.assertEqual(for_speech('🕒 A las 18:56. 😊'),
                         'A las dieciocho horas y cincuenta y seis minutos.')

    def test_filters_model_output_with_or_without_thinking(self):
        for text in ('Hola 😊.', '<think>interno</think>Hola 😊.'):
            self.assertEqual(ConversationBackend._visible_answer(text), 'Hola.')


if __name__ == '__main__':
    unittest.main()
