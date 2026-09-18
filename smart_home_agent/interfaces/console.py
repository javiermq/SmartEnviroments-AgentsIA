from __future__ import annotations

import asyncio

from .base import ConversationInterface


class ConsoleInterface(ConversationInterface):
    async def listen(self) -> str:
        return (await asyncio.to_thread(input, "\nTú > ")).strip()

    async def speak(self, text: str) -> None:
        print(f"ASISTENTE> {text}")
