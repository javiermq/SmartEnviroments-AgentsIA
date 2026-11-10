"""Conversación por voz con Reachy Mini sobre el registro real de Mariola."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import io
import logging
import os
from pathlib import Path
import sys
from time import monotonic
import wave

from .config import Settings
from .historical_har import History
from .human_console_bridge import LocalTTS, configure_windows_cuda_dlls
from .interfaces.reachy import AudioClip, ListenTimeout, ReachyInterface
from .local_har_demo import EXERCISE_CHECK, LocalDemo


T0 = '2026-11-02T17:09:00+01:00'
DATA = Path(__file__).resolve().parent.parent / 'dataset_home_real' / 'all_data_72hours_reference.txt'
STYLE = DATA.parent / 'mariola' / 'agent_style.txt'
OPENING = 'Hola, Mariola. ¿Qué tal va la tarde?'
IDLE_SECONDS = 15 * 60
LOGGER = logging.getLogger(__name__)


class MariolaDemo(LocalDemo):
    def __init__(self, history: History, model: str, url: str, *, trace: bool = False) -> None:
        super().__init__(history, model, url)
        self.trace = trace
        self.style = STYLE.read_text(encoding='utf-8-sig').strip()

    def ask(self, prompt, context, schema=None):
        if schema is None:
            voice_style = (
                'Eres Reachy, un robot. Mariola es la persona que habla contigo. Sus comidas, '
                'sueño y otras actividades le pertenecen a ella. Tú conversas y das ideas. '
                'No cocinas, comes ni duermes. Si no entiendes la frase, pídele que la repita. '
                'Cuando falte el dato que pide, di qué queda por saber.\n'
            )
            activity = context.get('contexto_actividad') or {}
            if activity.get('actividad') == 'cocinar':
                voice_style += (
                    'La etiqueta «cocinar» es un indicio de uso de la cocina. El valor '
                    'cocina_vasos13=1 indica que se activó el sensor del armario de los vasos. '
                    'Deja sin determinar qué se preparaba hasta que Mariola lo cuente.\n'
                )
            prompt = f'{voice_style}\nESTILO DE CONVERSACIÓN:\n{self.style}\n\n{prompt}'
        if not self.trace:
            return super().ask(prompt, context, schema)

        agent = 'CONVERSACIÓN'
        if schema is not None:
            agent = 'MÉTRICAS' if 'consulta' in schema['properties'] else 'HAR'
        print(f'[{agent}] Procesando…', flush=True)
        started = monotonic()
        answer = super().ask(prompt, context, schema)
        print(f'[{agent}] {monotonic() - started:.1f} s · {answer}', flush=True)
        return answer

    def metrics(self, question, t0, rows, conversation):
        result = super().metrics(question, t0, rows, conversation)
        if self.trace and 'descanso' in result:
            print(f"[MÉTRICAS] Modo descanso del reloj · {result['descanso']}", flush=True)
        return result

    def general(self, question, t0, metrics, selected, conversation=(), events=()):
        answer, context = super().general(question, t0, metrics, selected, conversation, events)
        if self.trace and 'descanso' in metrics:
            print(f'[HAR] Tramo del reloj · {selected}', flush=True)
            print(f'[CONVERSACIÓN] Respuesta con datos verificados · {answer}', flush=True)
        elif self.trace and EXERCISE_CHECK.fullmatch(question):
            print(f"[HAR] Ejercicio en el intervalo · {metrics['ejercicio']}", flush=True)
            print(f'[CONVERSACIÓN] Respuesta con datos verificados · {answer}', flush=True)
        return answer, context


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


async def run(demo: LocalDemo, interface: ReachyInterface, *, trace: bool = False) -> None:
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
                if trace:
                    print('[STT] Audio recibido, sin frase reconocible.', flush=True)
                continue
            if trace:
                print(f'\n[STT] {text}', flush=True)
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
                if trace:
                    print(f'[TTS] Sintetizando para el altavoz de Reachy · {answer}', flush=True)
                await interface.speak(answer)
                if trace:
                    print('[REACHY] Reproducción terminada. Vuelve a escuchar.', flush=True)
            deadline = monotonic() + IDLE_SECONDS
    finally:
        await interface.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='Dirección de Reachy Mini para usar audio por red.')
    parser.add_argument('--tts-model', help='Ruta del modelo de voz Piper en español.')
    parser.add_argument('--check', action='store_true', help='Prueba los tres agentes desde la consola.')
    parser.add_argument('--trace', action='store_true', help='Muestra la transcripción y el paso por cada agente.')
    args = parser.parse_args()

    settings = Settings.from_env()
    demo = MariolaDemo(History(DATA), settings.ollama_model, settings.ollama_url, trace=args.trace)
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    if settings.stt_device == 'cuda':
        configure_windows_cuda_dlls()
    interface = ReachyInterface(settings, tts=voice, speech_rms_threshold=0.015, speech_pre_roll_seconds=0.6)
    try:
        asyncio.run(run(demo, interface, trace=args.trace))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
