"""Entrega de decisiones de entorno al mismo altavoz de la conversación."""
import asyncio
import logging


async def wait_for_environment(backend, interface):
    while True:
        await asyncio.sleep(1)
        if getattr(interface, 'user_speaking', False):
            continue
        try:
            event = await asyncio.to_thread(backend.poll)
            if event.get('action') == 'speak':
                return event
        except (OSError, RuntimeError, ValueError):
            logging.exception('No se pudo consultar la decisión de entorno')


async def listen_with_environment(interface, backend):
    """Recibe voz y decisiones simultáneamente; un solo flujo reproduce TTS."""
    from contextlib import suppress
    while True:
        listening = asyncio.create_task(interface.listen())
        proactive = asyncio.create_task(wait_for_environment(backend, interface))
        try:
            done, _ = await asyncio.wait([listening, proactive], return_when=asyncio.FIRST_COMPLETED)
            if listening in done:
                return listening.result()
            event = proactive.result()
            # La detección puede haber empezado mientras la petición HTTP viajaba.
            if getattr(interface, 'user_speaking', False):
                return await listening
            listening.cancel()
            with suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
                await listening
            accepted = await asyncio.to_thread(backend.accept, event)
            if not accepted:
                continue
            interface.set_response_gesture(None)
            await interface.speak(event['text'])
        finally:
            for task in (listening, proactive):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
                    await task
