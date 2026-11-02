"""
prepare_data.py

Organizes the raw Kaggle "Indian Currency Dataset" folders into
data/train, data/val, data/test for a 3-class MobileNetV3 baseline:
    100, 200, 500

WHY SESSION-AWARE SPLITTING:
Sequential filenames like IMG_1666262689009 / IMG_1666262695803 are very
likely bursts of the SAME physical note photographed against different
backgrounds a few seconds apart. If we split individual images randomly,
the same physical note can leak into both train and test, which inflates
accuracy in a way that will not hold up under questioning or in the
live demo. Instead we group images into "sessions" using timestamp
proximity, and split whole sessions.

USAGE:
    python src/prepare_data.py \
        --raw_dir data/raw \
        --out_dir data \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
        --session_gap_sec 60

EXPECTED raw_dir STRUCTURE (as downloaded from Kaggle, unzipped):
    data/raw/hundred_new/*.jpg
    data/raw/hundred_old/*.jpg
    data/raw/two_hundred/*.jpg
    data/raw/five_hundred/*.jpg
    (other denomination folders are ignored unless --include_all is passed)
"""

import argparse
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict

# Maps raw Kaggle folder names -> our target class label
# Full 6-denomination scope (₹2000 excluded: demonetized/rare, not relevant
# to temple donations, and not in the original spec's training class list).
CLASS_MAP = {
    "ten_new": "10",
    "ten_old": "10",
    "twenty_new": "20",
    "twenty_old": "20",
    "fifty_new": "50",
    "fifty_old": "50",
    "hundred_new": "100",
    "hundred_old": "100",
    "two_hundred": "200",
    "five_hundred": "500",
}

TIMESTAMP_RE = re.compile(r"(\d{10,13})")


def extract_timestamp(filename: str):
    """Pull the numeric timestamp out of filenames like IMG_1666262689009.jpg.
    Returns None if no timestamp-like number is found (file is then treated
    as its own standalone session)."""
    match = TIMESTAMP_RE.search(filename)
    if match:
        return int(match.group(1))
    return None


def group_into_sessions(files, session_gap_ms: int):
    """Sort files by timestamp and group consecutive files into sessions
    where the gap between them is below session_gap_ms. Files without a
    parseable timestamp each become their own session (safe default)."""
    timestamped = []
    untimestamped = []
    for f in files:
        ts = extract_timestamp(f.name)
        if ts is not None:
            timestamped.append((ts, f))
        else:
            untimestamped.append(f)

    timestamped.sort(key=lambda x: x[0])

    sessions = []
    current_session = []
    last_ts = None
    for ts, f in timestamped:
        if last_ts is None or (ts - last_ts) <= session_gap_ms:
            current_session.append(f)
        else:
            sessions.append(current_session)
            current_session = [f]
        last_ts = ts
    if current_session:
        sessions.append(current_session)

    # each untimestamped file is its own session
    for f in untimestamped:
        sessions.append([f])

    return sessions


def split_sessions(sessions, train_ratio, val_ratio, test_ratio, seed):
    """Assign whole sessions to train/val/test using a greedy image-count
    balance instead of a naive session-count split.

    WHY: with only a few dozen sessions per class and uneven session sizes
    (some bursts have far more photos than others), splitting by session
    COUNT (e.g. 70% of 17 sessions) can badly skew the actual IMAGE-count
    ratio -- e.g. one oversized session landing in "test" can make test
    bigger than train. This still keeps every session intact in exactly
    one split (no physical note leaks across splits), but distributes
    sessions to track the target image-count ratio as closely as possible,
    using a largest-first greedy bin-packing approach.
    """
    rng = random.Random(seed)
    sessions = sessions[:]
    rng.shuffle(sessions)
    # largest sessions first: placing big items first gives the greedy
    # balancer the best chance of hitting the target ratios closely
    sessions.sort(key=len, reverse=True)

    total_images = sum(len(s) for s in sessions)
    targets = {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
        "test": total_images * test_ratio,
    }
    assigned = {"train": 0, "val": 0, "test": 0}
    buckets = {"train": [], "val": [], "test": []}

    for session in sessions:
        # assign to whichever split currently has the largest deficit
        # (target - assigned), i.e. is furthest behind its target
        deficits = {k: targets[k] - assigned[k] for k in targets}
        best_split = max(deficits, key=deficits.get)
        buckets[best_split].append(session)
        assigned[best_split] += len(session)

    return buckets["train"], buckets["val"], buckets["test"]


def copy_sessions(sessions, dest_dir: Path, class_label: str):
    # wipe any previous split output for this class first, so re-running
    # prepare_data.py (e.g. after this fix) doesn't leave stale files from
    # an old split mixed in with the new one
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for session in sessions:
        for f in session:
            target = dest_dir / f"{f.stem}_{f.suffix.lstrip('.')}{f.suffix}"
            # simpler: keep original name, dest dir already scopes by class/split
            target = dest_dir / f.name
            shutil.copy2(f, target)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--session_gap_sec", type=float, default=60.0,
                         help="Images within this many seconds of each other "
                              "are treated as the same physical note session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_all", action="store_true",
                         help="Include all denomination folders found, not just "
                              "the CLASS_MAP entries. Off by default to exclude "
                              "two_thousand (out of scope per project spec).")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found. Download the Kaggle dataset, unzip it, and "
            f"place the denomination folders (hundred_new, hundred_old, "
            f"two_hundred, five_hundred, ...) inside {raw_dir}/"
        )

    folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No subfolders found in {raw_dir}")

    # group raw folders by target class
    class_to_folders = defaultdict(list)
    for folder in folders:
        if folder.name in CLASS_MAP:
            class_to_folders[CLASS_MAP[folder.name]].append(folder)
        elif args.include_all:
            class_to_folders[folder.name].append(folder)
        else:
            print(f"Skipping '{folder.name}' (not in current class scope; "
                  f"pass --include_all to include it)")

    if not class_to_folders:
        raise RuntimeError(
            "No matching class folders found. Expected folder names like "
            "hundred_new, hundred_old, two_hundred, five_hundred inside "
            f"{raw_dir}. Found: {[f.name for f in folders]}"
        )

    session_gap_ms = args.session_gap_sec * 1000  # Kaggle timestamps look ms-based

    summary = {}
    for class_label, class_folders in class_to_folders.items():
        all_files = []
        for folder in class_folders:
            all_files.extend(sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png")))

        sessions = group_into_sessions(all_files, session_gap_ms)
        train_s, val_s, test_s = split_sessions(
            sessions, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

        n_train = copy_sessions(train_s, out_dir / "train" / class_label, class_label)
        n_val = copy_sessions(val_s, out_dir / "val" / class_label, class_label)
        n_test = copy_sessions(test_s, out_dir / "test" / class_label, class_label)

        summary[class_label] = {
            "total_images": len(all_files),
            "sessions": len(sessions),
            "train_images": n_train,
            "val_images": n_val,
            "test_images": n_test,
        }

    print("\n=== Data preparation summary ===")
    for cls, stats in summary.items():
        print(f"Class {cls}: {stats}")
    print(f"\nOutput written to: {out_dir}/train, {out_dir}/val, {out_dir}/test")


if __name__ == "__main__":
    main()
