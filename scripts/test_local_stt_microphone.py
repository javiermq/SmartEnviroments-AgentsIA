"""Minitest interactivo: micrófono local -> Faster-Whisper -> texto."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import wave

from smart_home_agent.human_console_bridge import LocalSTT


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba STT local usando el micrófono predeterminado de Windows.")
    parser.add_argument("--seconds", type=float, default=4.0, help="Duración de la grabación (por defecto: 4).")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Instala dependencias: python -m pip install sounddevice numpy") from exc
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.seconds <= 0:
        raise SystemExit("--seconds debe ser mayor que cero.")
    rate = 16_000
    print(f"Micrófono predeterminado: {sd.query_devices(kind='input')['name']}")
    print(f"Habla ahora durante {args.seconds:.1f} segundos...")
    samples = sd.rec(round(args.seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = handle.name
    try:
        pcm = (np.clip(samples[:, 0], -1, 1) * 32767).astype("<i2")
        with wave.open(wav_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm.tobytes())
        started = time.monotonic()
        text = LocalSTT(args.model, args.device).transcribe_wav(open(wav_path, "rb").read())
        print(f"\nTexto: {text or '[sin texto]'}")
        print(f"STT total: {time.monotonic() - started:.2f} s")
    finally:
        os.unlink(wav_path)


if __name__ == "__main__":
    main()
