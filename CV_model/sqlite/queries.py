"""Read-only reporting queries for the donation SQLite database."""

import sqlite3
from pathlib import Path

try:
    from .config import DATABASE_FILE, SUPPORTED_DENOMINATIONS
    from .database import get_connection, initialize_database
except ImportError:
    from config import DATABASE_FILE, SUPPORTED_DENOMINATIONS
    from database import get_connection, initialize_database


def _rows(database_file: Path, sql: str, parameters: tuple = ()) -> list[dict]:
    initialize_database(database_file)
    try:
        with get_connection(database_file) as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]
    except sqlite3.Error as error:
        print(f"Error: SQLite query failed: {error}")
        return []


def get_all_donations(database_file: Path = DATABASE_FILE) -> list[dict]:
    return _rows(database_file, "SELECT * FROM donations ORDER BY timestamp ASC, id ASC")


def get_total_donation_amount(database_file: Path = DATABASE_FILE) -> int:
    rows = _rows(database_file, "SELECT COALESCE(SUM(denomination), 0) AS total FROM donations")
    return int(rows[0]["total"]) if rows else 0


def get_donation_count(database_file: Path = DATABASE_FILE) -> int:
    rows = _rows(database_file, "SELECT COUNT(*) AS count FROM donations")
    return int(rows[0]["count"]) if rows else 0


def get_denomination_counts(database_file: Path = DATABASE_FILE) -> dict[int, int]:
    counts = {denomination: 0 for denomination in SUPPORTED_DENOMINATIONS}
    for row in _rows(database_file, "SELECT denomination, COUNT(*) AS count FROM donations GROUP BY denomination"):
        counts[int(row["denomination"])] = int(row["count"])
    return counts


def get_daily_total(date: str, database_file: Path = DATABASE_FILE) -> int:
    """Return the total for an ISO date such as ``2026-09-09``."""
    rows = _rows(
        database_file,
        "SELECT COALESCE(SUM(denomination), 0) AS total FROM donations WHERE substr(timestamp, 1, 10) = ?",
        (date,),
    )
    return int(rows[0]["total"]) if rows else 0


def get_recent_donations(limit: int = 10, database_file: Path = DATABASE_FILE) -> list[dict]:
    if limit < 1:
        return []
    return _rows(database_file, "SELECT * FROM donations ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,))
