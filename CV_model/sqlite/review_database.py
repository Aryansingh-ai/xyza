"""
CV_model/sqlite/review_database.py

SQLite persistence for REVIEW events (unknown objects, non-banknotes, low-confidence detections).

This is a SEPARATE table from the donations table. Review events are NEVER donations.

Schema:
  review_events  — one row per unique unknown-object event (RANDOM_001, RANDOM_002, ...)
  review_counter — single-row counter ensuring unique sequential IDs across restarts
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import DATABASE_FILE
except ImportError:
    from config import DATABASE_FILE

REVIEW_DB_FILE = DATABASE_FILE.parent / "review_events.db"

REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id               TEXT UNIQUE NOT NULL,
    review_seq              INTEGER NOT NULL,
    pocket_id               TEXT NOT NULL,
    timestamp               TEXT NOT NULL,
    reason                  TEXT NOT NULL,
    candidate_label         TEXT,
    classifier_confidence   REAL,
    detector_stage          TEXT,
    user_message            TEXT,
    created_at              TEXT NOT NULL
);
"""

COUNTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_counter (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    next_seq INTEGER NOT NULL DEFAULT 1
);
"""

COUNTER_INIT = "INSERT OR IGNORE INTO review_counter (id, next_seq) VALUES (1, 1);"


def get_review_connection(db_file: Path = REVIEW_DB_FILE) -> sqlite3.Connection:
    db_file = Path(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def initialize_review_database(db_file: Path = REVIEW_DB_FILE) -> Path:
    """Create review_events and review_counter tables if they don't exist."""
    db_file = Path(db_file)
    with get_review_connection(db_file) as conn:
        conn.execute(REVIEW_SCHEMA)
        conn.execute(COUNTER_SCHEMA)
        conn.execute(COUNTER_INIT)
        conn.commit()
    return db_file


def get_next_review_id(conn: sqlite3.Connection) -> tuple:
    """
    Atomically read and increment the review counter.
    Returns (review_id_str, seq_int), e.g. ("RANDOM_001", 1).
    Must be called within a transaction.
    """
    row = conn.execute(
        "SELECT next_seq FROM review_counter WHERE id = 1;"
    ).fetchone()
    if row is None:
        conn.execute("INSERT INTO review_counter (id, next_seq) VALUES (1, 2);")
        seq = 1
    else:
        seq = int(row["next_seq"])
        conn.execute(
            "UPDATE review_counter SET next_seq = ? WHERE id = 1;",
            (seq + 1,),
        )
    review_id = f"RANDOM_{seq:03d}"
    return review_id, seq


def insert_review_event(
    review_id: str,
    review_seq: int,
    pocket_id: str,
    reason: str,
    candidate_label: str | None = None,
    classifier_confidence: float | None = None,
    detector_stage: str | None = None,
    user_message: str | None = None,
    db_file: Path = REVIEW_DB_FILE,
) -> str:
    """
    Insert a review event. Returns the review_id.
    Does nothing if review_id already exists (idempotent).
    """
    now = datetime.now(timezone.utc).isoformat()
    db_file = Path(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with get_review_connection(db_file) as conn:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO review_events
                  (review_id, review_seq, pocket_id, timestamp, reason,
                   candidate_label, classifier_confidence, detector_stage,
                   user_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    review_id,
                    review_seq,
                    pocket_id,
                    now,
                    reason,
                    candidate_label,
                    classifier_confidence,
                    detector_stage,
                    user_message,
                    now,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"[ReviewDB] Warning: could not insert review event {review_id}: {e}")
    return review_id


def allocate_review_id(db_file: Path = REVIEW_DB_FILE) -> tuple:
    """
    Allocate (and persist) the next sequential review ID.
    Returns (review_id_str, seq_int).
    Safe to call without immediately inserting an event.
    """
    db_file = Path(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with get_review_connection(db_file) as conn:
        # Ensure tables exist
        conn.execute(REVIEW_SCHEMA)
        conn.execute(COUNTER_SCHEMA)
        conn.execute(COUNTER_INIT)
        review_id, seq = get_next_review_id(conn)
        conn.commit()
    return review_id, seq

