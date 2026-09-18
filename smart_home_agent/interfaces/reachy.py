"""Frontend Reachy Mini: audio/gestos, sin lógica de negocio del agente.

Las importaciones del SDK son diferidas para que el modo consola no dependa de
Reachy. El uso de media se limita a métodos documentados del SDK oficial.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings
from .base import ConversationInterface, InterfaceState

LOGGER = logging.getLogger(__name__)


class AudioError(RuntimeError):
    pass


class ListenTimeout(TimeoutError):
    pass


@dataclass
class AudioClip:
    samples: object  # numpy.ndarray, importado solo al utilizar Reachy.
    sample_rate: int


class SpeechToText(Protocol):
    async def transcribe(self, audio: AudioClip) -> str: ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str) -> AudioClip: ...


class FasterWhisperSTT:
    """Adaptador local de faster-whisper, cargado bajo demanda."""

    def __init__(self, model_name: str, language: str, device: str = "cpu", compute_type: str = "int8"):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    async def transcribe(self, audio: AudioClip) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: AudioClip) -> str:
        try:
            import numpy as np
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AudioError("Falta faster-whisper. Instálalo en el entorno de Reachy.") from exc
        if self._model is None:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(audio.sample_rate)
                wav.writeframes(pcm.tobytes())
            segments, _ = self._model.transcribe(str(wav_path), language=self.language, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            wav_path.unlink(missing_ok=True)


class PiperTTS:
    """Adaptador Piper CLI: genera WAV; Reachy reproduce por su SDK oficial."""

    def __init__(self, model_path: str, executable: str = "piper"):
        self.model_path = model_path
        self.executable = executable

    async def synthesize(self, text: str) -> AudioClip:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> AudioClip:
        if not self.model_path:
            raise AudioError("Define TTS_MODEL_PATH con una voz Piper .onnx.")
        try:
            import numpy as np
        except ImportError as exc:
            raise AudioError("Falta numpy en el entorno de Reachy.") from exc
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            process = subprocess.run(
                [self.executable, "--model", self.model_path, "--output_file", str(wav_path)],
                input=text, text=True, capture_output=True, timeout=90, check=False,
            )
            if process.returncode != 0:
                raise AudioError(f"Piper falló: {process.stderr.strip()}")
            with wave.open(str(wav_path), "rb") as wav:
                rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                if width != 2:
                    raise AudioError("La voz Piper debe generar WAV PCM de 16 bits.")
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            return AudioClip(samples=samples[:, None], sample_rate=rate)
        except FileNotFoundError as exc:
            raise AudioError("No se encontró el ejecutable 'piper'.") from exc
        finally:
            wav_path.unlink(missing_ok=True)


class ReachyInterface(ConversationInterface):
    """Captura/audio/gestos Reachy. Nunca conversa con Ollama directamente."""

    def __init__(self, settings: Settings, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        self.settings = settings
        stt_model = settings.stt_model_path or settings.stt_model
        self.stt = stt or FasterWhisperSTT(stt_model, settings.stt_language, settings.stt_device, settings.stt_compute_type)
        self.tts = tts or PiperTTS(settings.tts_model_path)
        self.robot = None

    async def start(self) -> None:
        try:
            from reachy_mini import ReachyMini
        except ImportError as exc:
            raise RuntimeError("No está instalado reachy_mini en este entorno.") from exc
        try:
            self.robot = await asyncio.to_thread(
                ReachyMini,
                host=self.settings.reachy_host,
                connection_mode=self.settings.reachy_connection_mode,
                media_backend="default",
            )
            await asyncio.to_thread(self.robot.__enter__)
            await asyncio.to_thread(self.robot.media.start_recording)
            await asyncio.to_thread(self.robot.media.start_playing)
        except Exception as exc:
            await self.close()
            raise RuntimeError(f"No se pudo conectar a Reachy Mini: {exc}") from exc
        self.state = InterfaceState.ACTIVE
        LOGGER.info("Reachy conectado y audio inicializado")
        await self.on_idle()

    async def listen(self) -> str:
        if self.robot is None:
            raise RuntimeError("Reachy no está conectado.")
        await self.on_listening()
        try:
            import numpy as np
        except ImportError as exc:
            raise AudioError("Falta numpy en el entorno de Reachy.") from exc
        chunks, speech_started = [], False
        deadline = time.monotonic() + self.settings.conversation_timeout
        silence_deadline = deadline
        while time.monotonic() < deadline:
            sample = await asyncio.to_thread(self.robot.media.get_audio_sample)
            _, speech = await asyncio.to_thread(self.robot.media.get_DoA)
            if speech:
                speech_started = True
                silence_deadline = time.monotonic() + 1.0
            if speech_started and sample is not None:
                chunks.append(sample)
                if time.monotonic() >= silence_deadline:
                    break
            await asyncio.sleep(0.01)
        if not chunks:
            raise ListenTimeout("No se detectó voz antes del timeout.")
        audio = AudioClip(np.concatenate(chunks, axis=0), self.robot.media.get_input_audio_samplerate())
        text = (await self.stt.transcribe(audio)).strip()
        if not text:
            raise AudioError("STT no devolvió texto.")
        self.state = InterfaceState.ACTIVE
        return text

    async def speak(self, text: str) -> None:
        if self.robot is None:
            raise RuntimeError("Reachy no está conectado.")
        self.state = InterfaceState.SPEAKING
        await self.on_speaking()
        try:
            clip = await self.tts.synthesize(text)
            samples, rate = self._resample_for_reachy(clip)
            await asyncio.to_thread(self.robot.media.push_audio_sample, samples)
            await asyncio.sleep(len(samples) / rate)
        finally:
            self.state = InterfaceState.ACTIVE
            await self.on_idle()

    def _resample_for_reachy(self, clip: AudioClip) -> tuple[object, int]:
        import numpy as np
        target_rate = self.robot.media.get_output_audio_samplerate()
        samples = np.asarray(clip.samples, dtype=np.float32)
        if clip.sample_rate == target_rate:
            return samples, target_rate
        target_count = round(len(samples) * target_rate / clip.sample_rate)
        source_x = np.linspace(0, 1, len(samples), endpoint=False)
        target_x = np.linspace(0, 1, target_count, endpoint=False)
        mono = samples.mean(axis=1) if samples.ndim == 2 else samples
        return np.interp(target_x, source_x, mono).astype(np.float32)[:, None], target_rate

    async def on_listening(self) -> None:
        LOGGER.info("Reachy escuchando")

    async def on_thinking(self) -> None:
        LOGGER.info("Reachy procesando")

    async def on_speaking(self) -> None:
        # API oficial: activa un movimiento sutil reactivo al audio reproducido.
        try:
            await asyncio.to_thread(self.robot.enable_wobbling)
        except Exception as exc:
            LOGGER.warning("No se pudo activar wobbling: %s", exc)

    async def on_idle(self) -> None:
        LOGGER.debug("Reachy en espera")

    async def close(self) -> None:
        robot, self.robot = self.robot, None
        if robot is not None:
            for operation in (robot.media.stop_recording, robot.media.stop_playing):
                try:
                    await asyncio.to_thread(operation)
                except Exception as exc:
                    LOGGER.warning("Error cerrando Reachy: %s", exc)
            try:
                await asyncio.to_thread(robot.__exit__, None, None, None)
            except Exception as exc:
                LOGGER.warning("Error cerrando contexto Reachy: %s", exc)
        await super().close()
