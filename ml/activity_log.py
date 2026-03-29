import csv
from datetime import datetime, timezone
from pathlib import Path


ML_DIR = Path(__file__).parent
CSV_PATH = ML_DIR / "leaf_activity_log.csv"
CSV_HEADERS = [
    "S No",
    "Event Type",
    "Disease Name",
    "Time",
    "Record ID",
    "Source",
    "Confidence",
    "Notes",
]


def ensure_log_file():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADERS)


def next_serial_number():
    ensure_log_file()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        return 1
    return len(rows)


def append_log(event_type, disease_name, record_id="", source="", confidence="", notes=""):
    ensure_log_file()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                next_serial_number(),
                event_type,
                disease_name,
                datetime.now(timezone.utc).isoformat(),
                record_id,
                source,
                confidence,
                notes,
            ]
        )

