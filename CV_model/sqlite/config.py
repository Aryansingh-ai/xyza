"""Filesystem locations used by the independent SQLite module."""

from pathlib import Path


SQLITE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SQLITE_DIRECTORY.parent
INFERENCE_FILE = PROJECT_ROOT / "logs" / "inference.jsonl"
DATABASE_FILE = SQLITE_DIRECTORY / "donation_box.db"

# These are pre-populated by get_denomination_counts(), even if no records exist.
SUPPORTED_DENOMINATIONS = (50, 100, 200, 500)
