"""Conversación por voz con Reachy Mini sobre el registro real de Mariola."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import io
import logging
import os
from pathlib import Path
from time import monotonic
import wave

from .config import Settings
from .historical_har import History
from .human_console_bridge import LocalTTS
from .interfaces.reachy import AudioClip, ListenTimeout, ReachyInterface
from .local_har_demo import LocalDemo


T0 = '2026-11-02T17:09:00+01:00'
DATA = Path(__file__).resolve().parent.parent / 'dataset_home_real' / 'all_data_72hours_reference.txt'
OPENING = ('Hola, Mariola. Parece que acabas de pasar del salón a la cocina. '
           '¿Estás preparando algo para tomar?')
IDLE_SECONDS = 15 * 60
LOGGER = logging.getLogger(__name__)


class MariolaDemo(LocalDemo):
    def ask(self, prompt, context, schema=None):
        if schema is None:
            prompt += (
                ' La señal cocina_vasos13 indica el estado del sensor del armario de los vasos. '
                ' Una activación aporta un indicio de uso de la cocina. '
                ' Respeta el minuto de cada lectura y deja como desconocido qué se preparaba '
                ' hasta que la persona lo cuente.'
            )
        return super().ask(prompt, context, schema)


class PiperVoice:
    """Mantiene cargada la voz Piper durante la conversación."""

    def __init__(self, voice: LocalTTS) -> None:
        self.voice = voice

    async def synthesize(self, text: str) -> AudioClip:
        import numpy as np

        audio = await asyncio.to_thread(self.voice.synthesize_wav, text)
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
            return AudioClip(samples.astype(np.float32)[:, None] / 32768, wav.getframerate())


async def run(demo: LocalDemo, interface: ReachyInterface) -> None:
    await interface.start()
    try:
        media = interface.robot.media
        for _ in range(60):
            sample = await asyncio.to_thread(media.get_audio_sample)
            if sample is not None and len(sample):
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError('El micrófono de Reachy no entregó audio.')

        print(f'Escena de Mariola a las {T0}', flush=True)
        await interface.speak(OPENING)
        print('Reachy está escuchando. Di «adiós» para terminar.', flush=True)
        conversation, events = [{'agente': OPENING}], []
        deadline = monotonic() + IDLE_SECONDS
        while monotonic() < deadline:
            try:
                text = await interface.listen()
            except ListenTimeout:
                continue
            if not text:
                continue
            if text.casefold().strip(' .,¡!¿?') in {'adiós', 'adiós reachy', 'adiós, reachy'}:
                await interface.speak('Hasta luego.')
                break

            await interface.on_thinking()
            started = monotonic()
            try:
                result = await asyncio.to_thread(demo.respond, text, T0, conversation, events)
            except (OSError, RuntimeError, ValueError):
                LOGGER.exception('Error en el turno de los tres agentes')
                await interface.speak('Ha habido un problema al consultar el registro. ¿Puedes repetirlo?')
            else:
                answer = result['respuesta']
                events = result['eventos']
                conversation = (conversation + [{'usuario': text}, {'agente': answer}])[-4:]
                LOGGER.info('Respuesta de los tres agentes en %.1f s', monotonic() - started)
                await interface.speak(answer)
            deadline = monotonic() + IDLE_SECONDS
    finally:
        await interface.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='Dirección de Reachy Mini para usar audio por red.')
    parser.add_argument('--tts-model', help='Ruta del modelo de voz Piper en español.')
    parser.add_argument('--check', action='store_true', help='Prueba los tres agentes desde la consola.')
    args = parser.parse_args()

    settings = Settings.from_env()
    demo = MariolaDemo(History(DATA), settings.ollama_model, settings.ollama_url)
    if args.check:
        started = monotonic()
        result = demo.respond('¿Qué indicios hay de que estuve usando la cocina justo antes?', T0)
        print(result['contexto_actividad'])
        print(result['respuesta'])
        print(f'Tiempo de los tres agentes: {monotonic() - started:.1f} s')
        return

    if args.host:
        settings = replace(settings, reachy_host=args.host, reachy_connection_mode='network')
    voice = None
    if not os.getenv('REMOTE_TTS_URL', '').strip():
        model_path = args.tts_model or settings.tts_model_path
        if not model_path or not Path(model_path).expanduser().is_file():
            parser.error('Indica un modelo Piper válido con --tts-model o TTS_MODEL_PATH.')
        local_tts = LocalTTS(str(Path(model_path).expanduser()))
        local_tts.warm()
        voice = PiperVoice(local_tts)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    interface = ReachyInterface(settings, tts=voice, speech_rms_threshold=0.015)
    try:
        asyncio.run(run(demo, interface))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
