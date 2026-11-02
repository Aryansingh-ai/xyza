"""SQLite schema and connection helpers for confirmed donations."""

import sqlite3
from pathlib import Path

try:  # Supports both ``python sqlite/database.py`` and package imports.
    from .config import DATABASE_FILE
except ImportError:
    from config import DATABASE_FILE


SCHEMA = """
CREATE TABLE IF NOT EXISTS donations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    donation_id TEXT UNIQUE NOT NULL,
    pocket_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    item_status TEXT,
    denomination INTEGER NOT NULL,
    classifier_confidence REAL,
    source TEXT,
    target_bin TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL
)
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_donations_timestamp ON donations(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_donations_denomination ON donations(denomination)",
    "CREATE INDEX IF NOT EXISTS idx_donations_source ON donations(source)",
)


def get_connection(database_file: Path = DATABASE_FILE) -> sqlite3.Connection:
    """Return a connection with rows accessible by column name."""
    connection = sqlite3.connect(str(database_file))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_file: Path = DATABASE_FILE) -> Path:
    """Create the database, table, and indexes when they do not already exist."""
    database_file = Path(database_file)
    database_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with get_connection(database_file) as connection:
            connection.execute(SCHEMA)
            for index_sql in INDEXES:
                connection.execute(index_sql)
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not initialize SQLite database '{database_file}': {error}") from error
    return database_file
