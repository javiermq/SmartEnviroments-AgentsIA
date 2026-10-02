import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from smart_home_agent.config import Settings
from smart_home_agent.interfaces.reachy import ListenTimeout, ReachyInterface


class RecentAudioTests(unittest.IsolatedAsyncioTestCase):
    def interface(self):
        interface = ReachyInterface(Settings(), stt=Mock(), tts=Mock())
        interface.robot = SimpleNamespace(media=Mock())
        interface.robot.media.get_input_audio_samplerate.return_value = 16000
        return interface

    async def test_discards_thirty_seconds_before_transcribing_new_voice(self):
        interface = self.interface()
        old = np.full((16000, 2), 0.9, dtype=np.float32)
        fresh = np.full((1600, 2), 0.1, dtype=np.float32)
        silence = np.zeros_like(fresh)
        interface.robot.media.get_audio_sample.side_effect = [old] * 30 + [None, fresh, silence]
        interface.robot.media.get_DoA.side_effect = [(0, True), (0, False)]
        interface.stt.transcribe = AsyncMock(return_value='Hola de ahora')
        with patch.dict('os.environ', {'SPEECH_SILENCE_SECONDS': '0'}):
            self.assertEqual(await interface.listen(), 'Hola de ahora')
        audio = interface.stt.transcribe.call_args.args[0]
        np.testing.assert_array_equal(audio.samples, np.concatenate([fresh, silence]))

    def test_empty_queue_does_not_discard_next_sample(self):
        interface = self.interface()
        interface.robot.media.get_audio_sample.return_value = None
        interface._discard_pending_audio()
        interface.robot.media.get_audio_sample.assert_called_once()

    def test_live_blocking_backend_finishes_drain(self):
        interface = self.interface()
        interface.robot.media.get_audio_sample.return_value = np.zeros((1600, 2))
        with patch('smart_home_agent.interfaces.reachy.time.monotonic',
                   side_effect=[0, 0, 0, 0.1]):
            interface._discard_pending_audio()
        interface.robot.media.get_audio_sample.assert_called_once()

    def test_drain_timeout_does_not_send_old_audio_to_stt(self):
        interface = self.interface()
        interface.robot.media.get_audio_sample.return_value = np.zeros((16000, 2))
        with patch('smart_home_agent.interfaces.reachy.time.monotonic',
                   side_effect=[0, 0, 0, 0, 3]):
            with self.assertRaises(ListenTimeout):
                interface._discard_pending_audio()
        interface.stt.transcribe.assert_not_called()
