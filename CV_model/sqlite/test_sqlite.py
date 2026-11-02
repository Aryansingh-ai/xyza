"""Runnable demonstration of the independent donation SQLite importer."""

try:
    from .config import DATABASE_FILE, INFERENCE_FILE, SUPPORTED_DENOMINATIONS
    from .database import initialize_database
    from .importer import import_inference_file
    from .queries import (
        get_denomination_counts,
        get_donation_count,
        get_recent_donations,
        get_total_donation_amount,
    )
except ImportError:
    from config import DATABASE_FILE, INFERENCE_FILE, SUPPORTED_DENOMINATIONS
    from database import initialize_database
    from importer import import_inference_file
    from queries import get_denomination_counts, get_donation_count, get_recent_donations, get_total_donation_amount


def main() -> None:
    initialize_database()
    result = import_inference_file()
    counts = get_denomination_counts()

    print("=" * 40)
    print("DONATION BOX SQLITE IMPORT")
    print("=" * 40)
    print(f"\nInput:\n{INFERENCE_FILE}")
    print(f"\nRecords read: {result.records_read}")
    print(f"ACCEPT records: {result.accepted_records}")
    print(f"REVIEW/non-ACCEPT records skipped: {result.non_accepted_skipped}")
    print(f"Malformed JSON records skipped: {result.malformed_json_skipped}")
    print(f"Invalid ACCEPT records skipped: {result.invalid_records_skipped}")
    print(f"Duplicate records skipped: {result.duplicate_records_skipped}")
    print(f"\nDatabase:\n{DATABASE_FILE}")
    print(f"\nTotal confirmed donations: {get_donation_count()}")
    print(f"Total amount: Rs.{get_total_donation_amount()}")
    print("\nDenominations:")
    for denomination in SUPPORTED_DENOMINATIONS:
        print(f"Rs.{denomination:<3}: {counts[denomination]}")
    print("\nRecent accepted donations:")
    for donation in get_recent_donations():
        print(f"- {donation['timestamp']} | Rs.{donation['denomination']} | {donation['pocket_id']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
