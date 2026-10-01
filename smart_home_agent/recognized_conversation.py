"""Coordina cámara y audio con un único propietario de los motores."""
import asyncio
import logging
import math
import threading
import re
import unicodedata
from contextlib import suppress
from pathlib import Path

from .reachy_vision import run_camera
from .user_sessions import RemoteConversation, user_id


def is_goodbye(text):
    text = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')
    text = ' '.join(re.findall(r'\w+', text))
    return bool(re.fullmatch(
        r'(?:(?:bueno|vale|muchas gracias|gracias) )*adios'
        r'(?: reachy(?: mini)?)?(?: gracias)?', text))


def accepted_identity(result):
    if not isinstance(result, dict) or result.get('status') != 'matched':
        return None
    try:
        user = user_id(result.get('user'))
        candidate = next(c for c in result.get('candidates', []) if user_id(c['user']) == user)
        distance, threshold = candidate['distance'], candidate['threshold']
        if (not math.isfinite(distance) or not math.isfinite(threshold)
                or distance < 0 or threshold <= 0 or distance > threshold
                or candidate.get('votes', 0) < 2):
            return None
        return user
    except (ValueError, KeyError, StopIteration, TypeError):
        return None


async def circle(interface):
    from reachy_mini.utils import create_head_pose
    started = asyncio.get_running_loop().time()
    try:
        while True:
            phase = (asyncio.get_running_loop().time() - started) * (2 * math.pi / 3)
            amplitude = 6 * min(1.0, (asyncio.get_running_loop().time() - started) / 0.8)
            # Círculo más visible, con entrada gradual y una vuelta cada tres segundos.
            interface.robot.set_target(head=create_head_pose(
                roll=amplitude * math.sin(phase), pitch=amplitude * math.cos(phase), degrees=True))
            await asyncio.sleep(0.05)
    finally:
        await asyncio.to_thread(interface._idle_gesture)


async def run_recognized(interface, settings, args, trace):
    import cv2
    model = Path(args.detector_model)
    if not model.is_file():
        raise ValueError(f'Falta el modelo YuNet: {model}')
    detector = cv2.FaceDetectorYN.create(str(model), '', (640, 480), 0.9)
    events = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop, permit = threading.Event(), threading.Event()
    permit.set()

    def emit(kind, result):
        # Bloquea el siguiente lote antes de que el coordinador salude.
        if kind == 'identity' and accepted_identity(result):
            permit.clear()
        loop.call_soon_threadsafe(events.put_nowait, (kind, result))

    def capture():
        try:
            run_camera(interface.robot, detector, args.vision_url, 10.0, 120,
                       event=emit, stop=stop, permit=permit)
        except Exception:
            logging.exception('Captura facial detenida')
        finally:
            emit('finished', None)
            emit('camera_stopped', None)

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    motion = None
    current, backend, pending_identity = None, None, None
    listening = None
    event_task = None

    async def stop_motion():
        nonlocal motion
        if motion:
            motion.cancel()
            with suppress(asyncio.CancelledError):
                await motion
            motion = None

    try:
        while True:
            # Audio habilitado únicamente después de una identificación válida.
            if current and motion is None and listening is None:
                listening = asyncio.create_task(interface.listen())
            event_task = asyncio.create_task(events.get())
            waiting = [event_task] + ([listening] if listening else [])
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if event_task in done:
                kind, result = event_task.result()
                if kind == 'camera_stopped':
                    raise RuntimeError('La cámara se ha detenido; reinicia el agente')
                # Una identidad permanece activa hasta la despedida. Ignora
                # eventos de visión que ya estuvieran en tránsito al activarla.
                if current:
                    continue
                if kind == 'recognizing':
                    if listening:
                        listening.cancel()
                        with suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
                            await listening
                        listening = None
                    await stop_motion()
                    # Termina el gesto anterior antes de tomar los motores.
                    gesture = interface._gesture_task
                    if gesture:
                        with suppress(asyncio.CancelledError):
                            await gesture
                    motion = asyncio.create_task(circle(interface))
                elif kind == 'identity':
                    pending_identity = accepted_identity(result)
                elif kind == 'finished':
                    await stop_motion()
                    recognized = pending_identity
                    pending_identity = None
                    if recognized is None:
                        current, backend = None, None
                    elif recognized != current:
                        current = recognized
                        backend = RemoteConversation(args.conversation_url, args.t0, current,
                                                     settings.ollama_model, args.session_id)
                        permit.clear()
                        await interface.speak('Hola ' + ('Javi' if current == 'javi' else 'Mariola'))
                        logging.info('Modo conversación: %s; reconocimiento pausado', current)
                continue
            event_task.cancel()
            with suppress(asyncio.CancelledError):
                await event_task
            task, listening = listening, None
            try:
                text = task.result()
            except (RuntimeError, TimeoutError) as exc:
                logging.warning('Turno descartado: %s', exc)
                continue
            if not text:
                continue
            if is_goodbye(text):
                interface.set_response_gesture(None)
                await interface.speak('Adiós ' + ('Javi' if current == 'javi' else 'Mariola'))
                gesture = interface._gesture_task
                if gesture:
                    with suppress(asyncio.CancelledError):
                        await gesture
                await asyncio.to_thread(interface._idle_gesture)
                current, backend, pending_identity = None, None, None
                # Ningún evento antiguo debe activar al siguiente usuario.
                while not events.empty():
                    kind, _ = events.get_nowait()
                    if kind == 'camera_stopped':
                        raise RuntimeError('La cámara se ha detenido; reinicia el agente')
                permit.set()
                logging.info('Modo reconocimiento: conversación finalizada')
                continue
            await interface.on_thinking()
            answer = await asyncio.to_thread(backend.respond, text, trace)
            interface.set_response_gesture(backend.last_gesture)
            await interface.speak(answer)
    finally:
        if event_task and not event_task.done():
            event_task.cancel()
            with suppress(asyncio.CancelledError):
                await event_task
        stop.set()
        permit.set()
        if listening:
            listening.cancel()
            with suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
                await listening
        await stop_motion()
        # El HTTP de visión tiene timeout: no cerrar media mientras sigue capturando.
        while worker.is_alive():
            await asyncio.sleep(0.1)
        await asyncio.to_thread(interface._idle_gesture)
