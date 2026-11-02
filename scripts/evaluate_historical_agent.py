"""Opt-in live Qwen smoke questions, with objective Python reference outputs.

Run from repo root: python -m scripts.evaluate_historical_agent
This does not score HAR accuracy. Inspect answers and audit for evidence quality.
"""
import json
from pathlib import Path

from smart_home_agent.historical_agent import HistoricalAgent
from smart_home_agent.historical_har import History


def main():
    data = Path('dataset_home_real/mariola')
    output = Path('temp_data/har_evaluation')
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ('¿Qué estaba haciendo a las 08:30 del 31/10/2026? Consulta el minuto [08:30,08:31).', '2026-10-31T08:31:00+01:00'),
        ('¿Cuántos pasos di en las últimas 24 horas hasta t0?', '2026-11-02T12:00:00+01:00'),
        ('¿A qué hora parece que me levanté hoy?', '2026-11-02T12:00:00+01:00'),
        ('¿Cuándo estuve cocinando hoy y cuánto tiempo? Distingue observaciones e hipótesis.', '2026-11-02T12:00:00+01:00'),
        ('¿Hay evidencia de una ducha hoy?', '2026-11-02T12:00:00+01:00'),
        ('¿Cuánto tiempo estuve fuera de casa hoy?', '2026-11-02T12:00:00+01:00'),
    ]
    agent = HistoricalAgent(data, audit=output/'audit.jsonl')
    for i, (question, t0) in enumerate(cases, 1):
        print(f'Caso {i}/{len(cases)}: inicio', flush=True)
        h = History(data)
        reference = h.resumir_intervalo(t0=t0, window='ultimas_24h' if i == 2 else 'hoy')
        try:
            answer = agent.respond(question, t0)
            result = dict(question=question, t0=t0, answer=answer, python_reference=reference)
        except Exception as exc:
            result = dict(question=question, t0=t0, error=str(exc), python_reference=reference)
        (output/f'case_{i}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Caso {i}: '+('error' if 'error' in result else 'completado'), flush=True)


if __name__ == '__main__':
    main()
