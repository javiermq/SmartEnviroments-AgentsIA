"""Comprueba que el SDK puede reservar el micrófono y recibir una muestra."""

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--connection-mode", default="localhost_only")
    parser.add_argument("--timeout", type=float, default=5.0, help="Segundos máximos para esperar la primera muestra")
    args = parser.parse_args()
    from reachy_mini import ReachyMini

    with ReachyMini(host=args.host, connection_mode=args.connection_mode, media_backend="default") as robot:
        robot.media.start_recording()
        try:
            deadline = time.monotonic() + args.timeout
            sample = None
            while time.monotonic() < deadline and sample is None:
                sample = robot.media.get_audio_sample()
                if sample is None:
                    time.sleep(0.02)
            if sample is None:
                raise RuntimeError(
                    f"No llegó ninguna muestra de micrófono en {args.timeout:.1f} s. "
                    "Comprueba que el daemon de Reachy y el backend de audio estén activos."
                )
            print(f"Mic OK: shape={sample.shape}, rate={robot.media.get_input_audio_samplerate()} Hz")
        finally:
            robot.media.stop_recording()


if __name__ == "__main__":
    main()
