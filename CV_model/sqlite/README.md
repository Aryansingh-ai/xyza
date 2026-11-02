# Donation Box SQLite Module

This independent module imports confirmed donation events from the ML audit log
into a local SQLite database. It does not modify the inference system or
`logs/inference.jsonl`.

`logs/inference.jsonl` is the raw ML inference/audit log. `sqlite/donation_box.db`
is confirmed donation transaction storage.

## Filtering and duplicate prevention

Only JSON objects whose `decision` is exactly `ACCEPT` are eligible for storage.
`REVIEW` (and every other decision) is skipped. Blank lines and malformed JSON are
safely skipped with a warning.

Each accepted record gets a deterministic SHA-256 `donation_id`, generated from
its pocket ID, timestamp, denomination, source, and model version. The database
enforces uniqueness on that ID, so rerunning the importer will skip an already
stored event instead of inserting it twice.

## Database structure

The `donations` table contains `id`, `donation_id`, `pocket_id`, `timestamp`,
`item_status`, `denomination`, `classifier_confidence`, `source`, `target_bin`,
`model_version`, and `created_at`. Indexes are created for timestamp,
denomination, and source.

## Run

From the project root:

```powershell
python sqlite/test_sqlite.py
# or import only
python sqlite/importer.py
```

The first command initializes `sqlite/donation_box.db`, imports the JSONL log,
and prints import counts, total amount, denomination counts, and recent accepted
donations. Paths are resolved from the project root, so the commands work from
any current directory.

Example (the precise counts depend on the audit log):

```text
Records read: 10
ACCEPT records: 4
REVIEW/non-ACCEPT records skipped: 6
Duplicate records skipped: 0
Total amount: Rs.800
```

## Inspect the database

With the SQLite command-line client installed:

```powershell
sqlite3 sqlite/donation_box.db
sqlite> SELECT * FROM donations;
sqlite> SELECT SUM(denomination) FROM donations;
```

You can also use a GUI such as DB Browser for SQLite and open
`sqlite/donation_box.db`.
