"""No modifica servicios: comprueba únicamente la conexión SDK con el daemon."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--connection-mode", default="localhost_only")
    args = parser.parse_args()
    from reachy_mini import ReachyMini

    with ReachyMini(host=args.host, connection_mode=args.connection_mode) as robot:
        print(f"Reachy connected: {robot!r}")


if __name__ == "__main__":
    main()
