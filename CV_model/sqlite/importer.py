"""Import only accepted JSONL inference events into the SQLite database."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:  # Supports direct execution: python sqlite/importer.py
    from .config import DATABASE_FILE, INFERENCE_FILE
    from .database import get_connection, initialize_database
except ImportError:
    from config import DATABASE_FILE, INFERENCE_FILE
    from database import get_connection, initialize_database


@dataclass
class ImportResult:
    records_read: int = 0
    accepted_records: int = 0
    non_accepted_skipped: int = 0
    malformed_json_skipped: int = 0
    invalid_records_skipped: int = 0
    inserted: int = 0
    duplicate_records_skipped: int = 0


def make_donation_id(record: dict[str, Any], denomination: int) -> str:
    """Build a stable ID so rerunning the importer cannot reinsert an event."""
    identity = {
        "pocket_id": record.get("pocket_id"),
        "timestamp": record.get("timestamp"),
        "denomination": denomination,
        "source": record.get("source"),
        "model_version": record.get("model_version"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"missing required field '{field}'")
    return str(value)


def _denomination(record: dict[str, Any]) -> int:
    value = record.get("predicted_denomination")
    if isinstance(value, bool):
        raise ValueError("predicted_denomination must be an integer")
    try:
        denomination = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError("invalid predicted_denomination") from error
    if denomination <= 0:
        raise ValueError("predicted_denomination must be positive")
    return denomination


def _confidence(record: dict[str, Any], line_number: int) -> Optional[float]:
    value = record.get("classifier_confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: line {line_number}: invalid classifier_confidence; storing NULL.")
        return None



def import_single_record(record: dict[str, Any], database_file: Path = DATABASE_FILE) -> ImportResult:
    """Insert a single accepted record directly into the database."""
    result = ImportResult()
    result.records_read = 1
    
    if record.get("decision") != "ACCEPT":
        result.non_accepted_skipped += 1
        return result

    result.accepted_records += 1
    try:
        initialize_database(database_file)
        with get_connection(database_file) as connection:
            denomination = _denomination(record)
            pocket_id = _required_text(record, "pocket_id")
            timestamp = _required_text(record, "timestamp")

            donation_id = make_donation_id(record, denomination)
            values = (
                donation_id, pocket_id, timestamp, record.get("item_status"), denomination,
                _confidence(record, 1), record.get("source"), record.get("target_bin"),
                record.get("model_version"), datetime.now(timezone.utc).isoformat(),
            )
            cursor = connection.execute(
                """
                INSERT INTO donations (
                    donation_id, pocket_id, timestamp, item_status, denomination,
                    classifier_confidence, source, target_bin, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(donation_id) DO NOTHING
                """,
                values,
            )
            if cursor.rowcount == 1:
                result.inserted += 1
            else:
                result.duplicate_records_skipped += 1
            connection.commit()
    except (ValueError, sqlite3.Error, RuntimeError) as error:
        result.invalid_records_skipped += 1
        print(f"Warning: single record import failed ({error}).")
    
    return result

def import_inference_file(
    inference_file: Path = INFERENCE_FILE, database_file: Path = DATABASE_FILE
) -> ImportResult:
    """Read a JSONL audit log and insert only valid ``decision == ACCEPT`` records."""
    result = ImportResult()
    inference_file = Path(inference_file)

    if not inference_file.is_file():
        print(f"Error: inference file not found: {inference_file}")
        return result

    try:
        initialize_database(database_file)
        with get_connection(database_file) as connection, inference_file.open(
            "r", encoding="utf-8"
        ) as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                result.records_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    result.malformed_json_skipped += 1
                    print(f"Warning: line {line_number}: malformed JSON skipped ({error.msg}).")
                    continue
                if not isinstance(record, dict):
                    result.invalid_records_skipped += 1
                    print(f"Warning: line {line_number}: JSON value is not an object; skipped.")
                    continue
                if record.get("decision") != "ACCEPT":
                    result.non_accepted_skipped += 1
                    continue

                result.accepted_records += 1
                try:
                    denomination = _denomination(record)
                    pocket_id = _required_text(record, "pocket_id")
                    timestamp = _required_text(record, "timestamp")
                except ValueError as error:
                    result.invalid_records_skipped += 1
                    print(f"Warning: line {line_number}: accepted record skipped ({error}).")
                    continue

                donation_id = make_donation_id(record, denomination)
                values = (
                    donation_id, pocket_id, timestamp, record.get("item_status"), denomination,
                    _confidence(record, line_number), record.get("source"), record.get("target_bin"),
                    record.get("model_version"), datetime.now(timezone.utc).isoformat(),
                )
                try:
                    cursor = connection.execute(
                        """
                        INSERT INTO donations (
                            donation_id, pocket_id, timestamp, item_status, denomination,
                            classifier_confidence, source, target_bin, model_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(donation_id) DO NOTHING
                        """,
                        values,
                    )
                except sqlite3.Error as error:
                    result.invalid_records_skipped += 1
                    print(f"Warning: line {line_number}: SQLite insert failed ({error}).")
                    continue
                if cursor.rowcount == 1:
                    result.inserted += 1
                else:
                    result.duplicate_records_skipped += 1
    except (OSError, sqlite3.Error, RuntimeError) as error:
        print(f"Error: could not import '{inference_file}': {error}")
    return result


if __name__ == "__main__":
    summary = import_inference_file()
    print(f"Records read: {summary.records_read}")
    print(f"ACCEPT records: {summary.accepted_records}")
    print(f"Non-ACCEPT records skipped: {summary.non_accepted_skipped}")
    print(f"Inserted: {summary.inserted}")
    print(f"Duplicate records skipped: {summary.duplicate_records_skipped}")
