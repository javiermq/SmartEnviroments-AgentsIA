"""Consola interactiva A/B con trazas verbose."""
import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smart_home_agent.context import DATA_DIR, parse_timestamp
from smart_home_agent.sensor_queries import DEFAULT_T0
from smart_home_agent.two_agent_backend import TwoAgentBackend


def verbose_trace(event, data):
    print(f"\n[TRACE] {event}\n{json.dumps(data, ensure_ascii=False, indent=2)}", file=sys.stderr, flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="A extrae contexto; B responde con estilo.")
    parser.add_argument("--t0", default=DEFAULT_T0)
    parser.add_argument("--question", help="Omitir para escribir en consola.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--goal", default="Acompañar al usuario y ayudarle con lo que pide sin insistir.")
    args = parser.parse_args()
    try:
        parse_timestamp(args.t0)
    except ValueError as exc:
        parser.error(str(exc))
    backend = TwoAgentBackend(args.t0, args.model, args.ollama_url, args.data_dir, args.goal)
    print(f"Sesión A/B en t0={args.t0}. Modelo: {args.model}")
    print("Verbose por stderr; think=false. Escribe 'salir' para terminar.")
    def turn(question):
        try:
            print("ASISTENTE> " + backend.respond(question, verbose_trace if args.verbose else None))
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"ERROR> {exc}", file=sys.stderr)
    if args.question:
        turn(args.question)
        return
    while True:
        try:
            question = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesión finalizada.")
            return
        if question.lower() in {"salir", "exit", "quit"}:
            return
        if question:
            turn(question)


if __name__ == "__main__":
    main()
