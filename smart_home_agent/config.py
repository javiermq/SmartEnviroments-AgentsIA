"""Configuración por entorno; .env no se versiona."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


@dataclass(frozen=True)
class Settings:
    interface: str = "console"
    ollama_model: str = "qwen3:4b"
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    reachy_host: str = "localhost"
    reachy_connection_mode: str = "localhost_only"
    stt_model: str = "base"
    stt_language: str = "es"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_model_path: str = ""
    tts_model_path: str = ""
    conversation_timeout: float = 20.0
    wake_word_enabled: bool = False
    wake_word: str = "hola reachy"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            interface=os.getenv("INTERFACE", "console"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
            reachy_host=os.getenv("REACHY_HOST", "localhost"),
            reachy_connection_mode=os.getenv("REACHY_CONNECTION_MODE", "localhost_only"),
            stt_model=os.getenv("STT_MODEL", "base"),
            stt_language=os.getenv("STT_LANGUAGE", "es"),
            stt_device=os.getenv("STT_DEVICE", "cpu"),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "int8"),
            stt_model_path=os.getenv("STT_MODEL_PATH", ""),
            tts_model_path=os.getenv("TTS_MODEL_PATH", ""),
            conversation_timeout=float(os.getenv("CONVERSATION_TIMEOUT", "20")),
            wake_word_enabled=os.getenv("WAKE_WORD_ENABLED", "false").lower() == "true",
            wake_word=os.getenv("WAKE_WORD", "hola reachy"),
        )
