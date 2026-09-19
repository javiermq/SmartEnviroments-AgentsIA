"""Consultas agregadas y seguras para el histórico de sensores simulados."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parent.parent / "sensor_context_data"

# API pública: no se permite que el modelo elija nombres de columna o ficheros.
SENSOR_TYPES = {
    "watch.steps": ("steps", "steps"),
    "watch.distance_m": ("distance_m", "m"),
    "watch.sleep": ("sleep", "minutes"),
}
AGGREGATIONS = {"total", "mean", "max", "min"}


def _parse_time(value: str, reference_day: str) -> datetime:
    """Acepta ISO-8601 o HH:MM; la hora sola usa el día del conjunto de datos."""
    try:
        if value == "24:00":
            return datetime.fromisoformat(f"{reference_day}T00:00:00+02:00") + timedelta(days=1)
        if len(value) == 5 and value[2] == ":":
            return datetime.fromisoformat(f"{reference_day}T{value}:00+02:00")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Las horas deben ser ISO-8601 o HH:MM.") from exc


def query_aggregation(
    sensor_type: str,
    time_init: str,
    time_end: str,
    aggregation: str,
    data_file: Path | str = DATA_FILE,
    reference_day: str | None = None,
) -> dict[str, Any]:
    """Agrega una señal del reloj en el intervalo semiabierto [time_init, time_end).

    Args:
        sensor_type: ``watch.steps``, ``watch.distance_m`` o ``watch.sleep``.
        time_init: Inicio ISO-8601 o ``HH:MM``.
        time_end: Fin ISO-8601 o ``HH:MM``.
        aggregation: ``total``, ``mean``, ``max`` o ``min``.
        data_file: TSV o directorio; por defecto carga todos los días simulados.
        reference_day: Fecha para HH:MM; por defecto el último día de datos.

    Para ``watch.sleep``, ``total`` representa minutos dormidos y ``mean`` la
    fracción de minutos dormidos en el intervalo.
    """
    if sensor_type not in SENSOR_TYPES:
        allowed = ", ".join(SENSOR_TYPES)
        raise ValueError(f"sensor_type no válido. Valores admitidos: {allowed}.")
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation debe ser total, mean, max o min.")

    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {path}")

    paths = sorted(path.glob("simulated_sensor_data_*.tsv")) if path.is_dir() else [path]
    rows = []
    for source in paths:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("El archivo de datos está vacío.")

    reference_day = reference_day or max(row["timestamp"][:10] for row in rows)
    start = _parse_time(time_init, reference_day)
    end = _parse_time(time_end, reference_day)
    if end <= start:
        raise ValueError("time_end debe ser posterior a time_init.")

    column, unit = SENSOR_TYPES[sensor_type]
    values = [
        float(row[column])
        for row in rows
        if start <= datetime.fromisoformat(row["timestamp"]) < end
    ]
    if not values:
        raise ValueError("No hay muestras dentro del intervalo solicitado.")

    if aggregation == "total":
        result = sum(values)
    elif aggregation == "mean":
        result = sum(values) / len(values)
    elif aggregation == "max":
        result = max(values)
    else:
        result = min(values)

    return {
        "sensor_type": sensor_type,
        "aggregation": aggregation,
        "time_init": start.isoformat(),
        "time_end": end.isoformat(),
        "interval": "[time_init, time_end)",
        "samples": len(values),
        "result": round(result, 2),
        "unit": unit,
    }


def main() -> None:
    """CLI JSON para pruebas y para usarla desde una herramienta externa."""
    import argparse

    parser = argparse.ArgumentParser(description="Consulta agregada de sensores.")
    parser.add_argument("sensor_type", choices=SENSOR_TYPES)
    parser.add_argument("time_init")
    parser.add_argument("time_end")
    parser.add_argument("aggregation", choices=AGGREGATIONS)
    args = parser.parse_args()
    print(json.dumps(query_aggregation(**vars(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
