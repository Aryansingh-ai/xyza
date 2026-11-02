"""Independent SQLite storage for confirmed donation-box inferences."""

from .database import initialize_database

__all__ = ["initialize_database"]
