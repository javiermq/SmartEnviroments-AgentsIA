"""Contrato común para cada frontend conversacional."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class InterfaceState(str, Enum):
    SLEEP = "sleep"
    ACTIVE = "active"
    SPEAKING = "speaking"


class ConversationInterface(ABC):
    state: InterfaceState = InterfaceState.SLEEP

    async def start(self) -> None:
        self.state = InterfaceState.ACTIVE

    @abstractmethod
    async def listen(self) -> str:
        """Obtiene un turno de usuario como texto."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Entrega una respuesta textual al usuario."""

    async def close(self) -> None:
        self.state = InterfaceState.SLEEP

    async def on_listening(self) -> None:
        pass

    async def on_thinking(self) -> None:
        pass

    async def on_speaking(self) -> None:
        pass

    async def on_idle(self) -> None:
        pass
