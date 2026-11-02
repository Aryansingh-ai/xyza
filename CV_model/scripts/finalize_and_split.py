"""
scripts/finalize_and_split.py

Step 1: Build audit/physical_note_mapping_final.csv
  - Keep all 2,616 already-resolved mappings from physical_note_mapping.csv
  - Accept all HIGH and MEDIUM proposals from physical_note_review.csv
  - Assign unique LOW_<source_id> IDs to each LOW record (one per source, never merged)

Step 2: Build data_clean/ with leak-free train/val/test split
  - Split is done at physical_note_id level, stratified by denomination
  - Every synthetic image from a physical note stays in the same split as that note
  - Source images are also copied into data_clean/
  - Produces data_clean/split_manifest.csv

DOES NOT modify any existing files or datasets.
"""

import csv, shutil, random, json
from pathlib import Path
from collections import defaultdict

DIVYAKOSH   = Path(__file__).resolve().parent.parent
MAPPING_CSV = DIVYAKOSH / "audit/physical_note_mapping.csv"
REVIEW_CSV  = DIVYAKOSH / "audit/physical_note_review.csv"
SRC_CSV     = DIVYAKOSH / "synthetic_dataset/metadata/source_manifest.csv"
SYN_CSV     = DIVYAKOSH / "synthetic_dataset/metadata/synthetic_manifest.csv"
FINAL_MAP   = DIVYAKOSH / "audit/physical_note_mapping_final.csv"
DATA_CLEAN  = DIVYAKOSH / "data_clean"
BB_DIR      = DIVYAKOSH / "synthetic_dataset/black_box"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build final mapping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=== STEP 1: Building final physical_note_id mapping ===")

# Load base mapping (2,886 rows)
base_map = {}
with open(MAPPING_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        base_map[row["source_id"]] = row["physical_note_id"]

# Load review proposals
review_proposals = {}
with open(REVIEW_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        review_proposals[row["source_id"]] = row

# Build final map
final_map = {}  # source_id -> physical_note_id
handling = {"already_resolved": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

for src_id, base_pn in base_map.items():
    if base_pn != "REVIEW":
        final_map[src_id] = base_pn
        handling["already_resolved"] += 1
    elif src_id in review_proposals:
        row = review_proposals[src_id]
        conf = row["confidence"]
        proposed = row["proposed_physical_note_id"]
        if conf in ("HIGH", "MEDIUM") and proposed != "REVIEW":
            final_map[src_id] = proposed
            handling[conf] += 1
        else:  # LOW or still REVIEW
            final_map[src_id] = f"LOW_{src_id}"
            handling["LOW"] += 1
    else:
        # Safety: shouldn't happen, but give unique id
        final_map[src_id] = f"LOW_{src_id}"
        handling["LOW"] += 1

# Load source details for enrichment
src_details = {}
with open(SRC_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        src_details[row["source_id"]] = row

# Write final mapping
with open(FINAL_MAP, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "physical_note_id", "denomination", "source_type", "path"])
    for src_id, pn in sorted(final_map.items()):
        d = src_details.get(src_id, {})
        writer.writerow([src_id, pn, d.get("denomination",""), d.get("source_type",""), d.get("original_path","")])

# Validate: unique source_ids, no conflicts
assert len(final_map) == len(base_map), "Mismatch in source count!"
conflict_check = defaultdict(set)
for src_id, pn in final_map.items():
    conflict_check[src_id].add(pn)
conflicts = {k: v for k, v in conflict_check.items() if len(v) > 1}

print(f"  Already resolved : {handling['already_resolved']}")
print(f"  HIGH accepted    : {handling['HIGH']}")
print(f"  MEDIUM accepted  : {handling['MEDIUM']}")
print(f"  LOW (unique IDs) : {handling['LOW']}")
print(f"  Total mapped     : {len(final_map)}")
print(f"  Unique PN IDs    : {len(set(final_map.values()))}")
print(f"  Conflicts        : {len(conflicts)}")

# Count by denomination
denom_counts = defaultdict(lambda: defaultdict(int))
for src_id, pn in final_map.items():
    d = src_details.get(src_id, {})
    denom_counts[d.get("denomination","?")][pn] += 1

print("\n  Physical note groups per denomination:")
for denom in sorted(denom_counts):
    n_pn = len(denom_counts[denom])
    n_src = sum(denom_counts[denom].values())
    print(f"    {denom:>4}: {n_pn} physical notes, {n_src} source images")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Split at physical_note_id level
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== STEP 2: Building leak-free split ===")

# Group physical notes by denomination
pn_by_denom = defaultdict(set)
for src_id, pn in final_map.items():
    denom = src_details[src_id]["denomination"]
    pn_by_denom[denom].add(pn)

# Stratified split per denomination
rng = random.Random(SEED)
pn_split = {}  # physical_note_id -> "train"/"val"/"test"

for denom, pns in pn_by_denom.items():
    pns = sorted(pns)
    rng.shuffle(pns)
    n = len(pns)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val   = max(1, round(n * VAL_RATIO))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_val -= 1
        n_test = 1
    for pn in pns[:n_train]:
        pn_split[pn] = "train"
    for pn in pns[n_train:n_train+n_val]:
        pn_split[pn] = "val"
    for pn in pns[n_train+n_val:]:
        pn_split[pn] = "test"

# Load synthetic manifest
syn_rows = []
with open(SYN_CSV, encoding="utf-8") as f:
    syn_rows = list(csv.DictReader(f))

# Build src_id -> list of synthetic rows
src_to_syn = defaultdict(list)
for row in syn_rows:
    src_to_syn[row["source_id"]].append(row)

# Create data_clean directories
for split in ("train", "val", "test"):
    for denom in ("10", "20", "50", "100", "200", "500"):
        (DATA_CLEAN / split / denom).mkdir(parents=True, exist_ok=True)

manifest_rows = []

# Copy synthetic images
print("  Copying synthetic images...")
syn_copied = 0
for src_id, syn_list in src_to_syn.items():
    pn = final_map.get(src_id, f"LOW_{src_id}")
    split = pn_split.get(pn, "train")  # fallback safety
    for syn in syn_list:
        denom = syn["denomination"]
        syn_id = syn["synthetic_id"]
        src_file = BB_DIR / denom / f"{syn_id}.jpg"
        if not src_file.exists():
            continue
        dst_file = DATA_CLEAN / split / denom / f"{syn_id}.jpg"
        shutil.copy2(src_file, dst_file)
        manifest_rows.append({
            "image_path": str(dst_file.relative_to(DIVYAKOSH)),
            "split": split,
            "denomination": denom,
            "source_type": syn["source_type"],
            "source_id": src_id,
            "synthetic_id": syn_id,
            "physical_note_id": pn,
        })
        syn_copied += 1
print(f"    Synthetic images copied: {syn_copied}")

# Copy real_currency source images (real_camera type only, selected_for_synthesis=yes)
print("  Copying real-camera source images...")
real_copied = 0
for src_id, detail in src_details.items():
    if detail["source_type"] != "real_camera":
        continue
    pn = final_map.get(src_id, f"LOW_{src_id}")
    split = pn_split.get(pn, "train")
    denom = detail["denomination"]
    src_path = DIVYAKOSH / detail["original_path"]
    if not src_path.exists():
        continue
    dst_file = DATA_CLEAN / split / denom / f"RC_{src_id}{src_path.suffix}"
    shutil.copy2(src_path, dst_file)
    manifest_rows.append({
        "image_path": str(dst_file.relative_to(DIVYAKOSH)),
        "split": split,
        "denomination": denom,
        "source_type": "real_camera",
        "source_id": src_id,
        "synthetic_id": "",
        "physical_note_id": pn,
    })
    real_copied += 1
print(f"    Real-camera images copied: {real_copied}")

# Write split manifest
manifest_path = DATA_CLEAN / "split_manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path","split","denomination","source_type","source_id","synthetic_id","physical_note_id"])
    writer.writeheader()
    writer.writerows(manifest_rows)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Validation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== STEP 3: Validation ===")

split_totals = defaultdict(int)
split_denom  = defaultdict(lambda: defaultdict(int))
split_pns    = defaultdict(set)
split_real   = defaultdict(int)

for row in manifest_rows:
    s = row["split"]
    d = row["denomination"]
    split_totals[s] += 1
    split_denom[s][d] += 1
    split_pns[s].add(row["physical_note_id"])
    if row["source_type"] == "real_camera":
        split_real[s] += 1

# Check for leakage: any pn in multiple splits
all_pns = defaultdict(set)
for row in manifest_rows:
    all_pns[row["physical_note_id"]].add(row["split"])
leaked = {pn: splits for pn, splits in all_pns.items() if len(splits) > 1}

print(f"\n  Total images     : {len(manifest_rows)}")
print(f"  Splits created   : {sorted(split_totals.keys())}")
for s in ("train","val","test"):
    print(f"\n  [{s.upper()}] total={split_totals[s]}  unique_pns={len(split_pns[s])}  real_cam={split_real[s]}")
    for d in sorted(split_denom[s]):
        print(f"    {d:>4}: {split_denom[s][d]}")

print(f"\n  Physical notes leaking across splits: {len(leaked)}")
if leaked:
    for pn, splits in list(leaked.items())[:5]:
        print(f"    {pn} -> {splits}")

print(f"\n  Split manifest: {manifest_path}")
print("\nDONE. No existing files modified.")

