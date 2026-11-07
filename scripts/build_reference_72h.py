"""Materialize causal median features in the agent's tab-separated reference file."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

from smart_home_agent.historical_har import History, MEDIAN_WINDOWS, REFERENCE_EXPORT, iso, median_columns


def main():
    folder = Path('dataset_home_real')
    source = History(folder/'all_data_72hours.txt')
    rows = sorted(source.rows.items())
    original_columns = [name for name in rows[0][1] if name != '_source']
    features = {column: median_columns(column) for column in MEDIAN_WINDOWS}
    columns = original_columns + [name for fields in features.values() for name in fields.values()]
    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=folder, delete=False) as output:
        temporary = Path(output.name)
        try:
            writer = csv.DictWriter(output, columns, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            for minute, row in rows:
                record = {name: row[name] for name in original_columns}
                for column, fields in features.items():
                    result = source.desviacion_mediana_movil(column, t0=iso(minute+60))
                    record.update({name: result[key] for key, name in fields.items()})
                writer.writerow(record)
            output.close()
            temporary.replace(folder/REFERENCE_EXPORT)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
