"""Comprueba que el SDK puede reservar el micrófono y recibir una muestra."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--connection-mode", default="localhost_only")
    args = parser.parse_args()
    from reachy_mini import ReachyMini

    with ReachyMini(host=args.host, connection_mode=args.connection_mode, media_backend="default") as robot:
        robot.media.start_recording()
        try:
            sample = robot.media.get_audio_sample()
            print(f"Mic OK: shape={sample.shape}, rate={robot.media.get_input_audio_samplerate()} Hz")
        finally:
            robot.media.stop_recording()


if __name__ == "__main__":
    main()
