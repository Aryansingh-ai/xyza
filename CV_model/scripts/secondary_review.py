"""
scripts/secondary_review.py

Secondary review of 270 REVIEW source images.
Produces audit/physical_note_review.csv with:
  source_id, path, denomination, current_status, proposed_physical_note_id, confidence, reason

Confidence levels:
  HIGH   = strong evidence (same WhatsApp session, same IMG burst, CAND group)
  MEDIUM = reasonable evidence (UUID cluster under same denom, small gap)
  LOW    = insufficient evidence; keep REVIEW

Does NOT modify any existing manifest.
"""

import csv, re, datetime
from pathlib import Path
from collections import defaultdict

DIVYAKOSH = Path(__file__).resolve().parent.parent
RC_CSV         = DIVYAKOSH / "data2/data/real_currency/manifest.csv"
SRC_CSV        = DIVYAKOSH / "synthetic_dataset/metadata/source_manifest.csv"
PN_MAPPING_CSV = DIVYAKOSH / "audit/physical_note_mapping.csv"
OUTPUT_CSV     = DIVYAKOSH / "audit/physical_note_review.csv"

WA_RE  = re.compile(r"WhatsApp Image (\d{4}-\d{2}-\d{2}) at (\d{2})\.(\d{2})\.(\d{2})")
IMG_RE = re.compile(r"IMG(\d{8})(\d{6})")   # YYYYMMDDHHMMSS
CAND_RE = re.compile(r"(CAND_\d+_\d+)")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}", re.I)

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def wa_datetime(fname):
    m = WA_RE.search(fname)
    if not m:
        return None
    d = m.group(1)
    return datetime.datetime(int(d[:4]), int(d[5:7]), int(d[8:10]),
                             int(m.group(2)), int(m.group(3)), int(m.group(4)))

def img_datetime(fname):
    m = IMG_RE.search(fname)
    if not m:
        return None
    s = m.group(1) + m.group(2)
    return datetime.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                             int(s[8:10]), int(s[10:12]), int(s[12:14]))

def cluster_by_time(items, gap_seconds=300, prefix=""):
    """items = list of (datetime, key). Returns {key: cluster_label}."""
    items = sorted(items, key=lambda x: x[0])
    result = {}
    cluster = 0
    last_dt = None
    for dt, key in items:
        if last_dt is None or (dt - last_dt).total_seconds() > gap_seconds:
            cluster += 1
        result[key] = f"{prefix}C{cluster:02d}"
        last_dt = dt
    return result

# â”€â”€ load real_currency manifest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

rc_rows = {}
with open(RC_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rc_rows[row["image_path"]] = row

# â”€â”€ load source_manifest, keep REVIEW rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

src_details = {}
with open(SRC_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        src_details[row["source_id"]] = row

review_src_ids = set()
with open(PN_MAPPING_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["physical_note_id"] == "REVIEW":
            review_src_ids.add(row["source_id"])

# Build pathâ†’source_id for REVIEW sources (real_currency only)
path_to_src = {}
for sid in review_src_ids:
    path_to_src[src_details[sid]["original_path"]] = sid

# â”€â”€ Per-denom, per-filetype clustering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# key: (denom, filepath) â†’ (proposed_pn, confidence, reason)
proposals = {}

# --- CAND groups (manually verified) ----------------------------------------
for ipath, rc_row in rc_rows.items():
    m = CAND_RE.search(rc_row.get("review_notes", ""))
    if m:
        full_path = f"data2/data/real_currency/{ipath}"
        if full_path in path_to_src:
            sid = path_to_src[full_path]
            denom = src_details[sid]["denomination"]
            proposals[sid] = (m.group(1), "HIGH", "CAND group from manually verified review_notes")

# --- WhatsApp: cluster by time per denomination ------------------------------
wa_by_denom = defaultdict(list)
for full_path, sid in path_to_src.items():
    if sid in proposals:
        continue
    fname = Path(full_path).name
    dt = wa_datetime(fname)
    if dt is None:
        continue
    denom = src_details[sid]["denomination"]
    wa_by_denom[denom].append((dt, sid))

for denom, items in wa_by_denom.items():
    clusters = cluster_by_time(items, gap_seconds=300, prefix=f"WA_{denom}_")
    for sid, label in clusters.items():
        dt = next(dt for dt, s in items if s == sid)
        # single-day burst â†’ HIGH; multi-day spans within same cluster â†’ MEDIUM
        cluster_items = [d for d, s in items if clusters[s] == label]
        span_days = (max(cluster_items) - min(cluster_items)).days
        if span_days == 0:
            conf, reason = "HIGH", f"WhatsApp burst same day ({dt.date()}), gap <5 min"
        else:
            conf, reason = "MEDIUM", f"WhatsApp cluster {label}, spans {span_days} day(s) â€” verify"
        proposals[sid] = (label, conf, reason)

# --- IMG YYYYMMDDHHMMSS: cluster per denomination ----------------------------
img_by_denom = defaultdict(list)
for full_path, sid in path_to_src.items():
    if sid in proposals:
        continue
    fname = Path(full_path).name
    dt = img_datetime(fname)
    if dt is None:
        continue
    denom = src_details[sid]["denomination"]
    img_by_denom[denom].append((dt, sid))

for denom, items in img_by_denom.items():
    clusters = cluster_by_time(items, gap_seconds=120, prefix=f"IMG_{denom}_")
    for sid, label in clusters.items():
        dt = next(dt for dt, s in items if s == sid)
        cluster_items = [d for d, s in items if clusters[s] == label]
        span = (max(cluster_items) - min(cluster_items)).total_seconds()
        if span < 60:
            conf, reason = "HIGH", f"IMG burst same session ({dt}), span <60 s"
        else:
            conf, reason = "MEDIUM", f"IMG cluster {label}, span {span:.0f}s â€” verify"
        proposals[sid] = (label, conf, reason)

# --- UUID filenames: no timestamp â†’ LOW (cannot cluster safely) --------------
for full_path, sid in path_to_src.items():
    if sid in proposals:
        continue
    fname = Path(full_path).name
    if UUID_RE.match(fname):
        denom = src_details[sid]["denomination"]
        proposals[sid] = ("REVIEW", "LOW",
                          "UUID filename â€” no timestamp, cannot safely cluster without human inspection")

# --- Anything remaining â†’ LOW ------------------------------------------------
for sid in review_src_ids:
    if sid not in proposals:
        proposals[sid] = ("REVIEW", "LOW", "No grouping evidence found")

# â”€â”€ Write output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "path", "denomination", "current_status",
                     "proposed_physical_note_id", "confidence", "reason"])
    for sid in sorted(review_src_ids):
        detail = src_details[sid]
        pn, conf, reason = proposals.get(sid, ("REVIEW", "LOW", "No proposal"))
        counts[conf] += 1
        writer.writerow([sid, detail["original_path"], detail["denomination"],
                         "REVIEW", pn, conf, reason])

print("=== SECONDARY REVIEW RESULTS ===")
print(f"Total REVIEW sources : 270")
print(f"HIGH confidence      : {counts['HIGH']}  â†’ can be auto-resolved")
print(f"MEDIUM confidence    : {counts['MEDIUM']}  â†’ recommend human spot-check before accepting")
print(f"LOW / unresolved     : {counts['LOW']}  â†’ keep REVIEW; need human visual inspection")
print(f"\nOutput: {OUTPUT_CSV}")
print("\nNOTE: No manifests modified.")

