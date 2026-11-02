"""Prepare a validated, stratified, leakage-free YOLO dataset.

The source validation split is intentionally ignored because it is an exact
duplicate of the source training split. Only source images/train is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("OpenCV is required to decode source images.") from exc


CLASS_NAMES = {
    0: "10 INR",
    1: "20 INR",
    2: "50 INR",
    3: "100 INR",
    4: "200 INR",
    5: "500 INR",
}
SPLITS = ("train", "val", "test")
PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_counts(class_counts: dict[int, int], totals: dict[str, int]) -> dict[int, dict[str, int]]:
    """Allocate exact split totals with largest-remainder rounding."""
    allocation = {class_id: {split: math.floor(count * PROPORTIONS[split])
                             for split in SPLITS}
                  for class_id, count in class_counts.items()}
    for split in SPLITS:
        missing = totals[split] - sum(row[split] for row in allocation.values())
        ranked = sorted(
            class_counts,
            key=lambda class_id: (
                -(class_counts[class_id] * PROPORTIONS[split]
                  - math.floor(class_counts[class_id] * PROPORTIONS[split])),
                class_id,
            ),
        )
        for class_id in ranked[:missing]:
            allocation[class_id][split] += 1
    for class_id, row in allocation.items():
        row["test"] = class_counts[class_id] - row["train"] - row["val"]
    return allocation


def parse_label(label_path: Path, image_path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            errors.append("image could not be decoded")
        height, width = (image.shape[:2] if image is not None else (0, 0))
    except Exception as exc:
        errors.append(f"image decode error: {exc}")
        height, width = 0, 0

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return 0, errors + [f"label could not be read: {exc}"]
    if not lines:
        errors.append("empty annotation")
    classes: list[int] = []
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected five fields")
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric class or coordinate")
            continue
        if class_id not in CLASS_NAMES:
            errors.append(f"line {line_number}: class ID {class_id} is outside 0-5")
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"line {line_number}: non-finite coordinate")
        x_center, y_center, box_width, box_height = coordinates
        if box_width <= 0 or box_height <= 0:
            errors.append(f"line {line_number}: width and height must be positive")
        if (x_center - box_width / 2 < 0 or x_center + box_width / 2 > 1
                or y_center - box_height / 2 < 0 or y_center + box_height / 2 > 1):
            errors.append(f"line {line_number}: bounding box lies outside image bounds")
        classes.append(class_id)
    return (classes[0] if classes else 0), errors


def validate_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    images = sorted(path for path in image_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = {path.stem: path for path in label_dir.glob("*.txt") if path.is_file()}
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    missing_labels = [path.name for path in images if path.stem not in labels]
    orphaned_labels = sorted(stem for stem in labels
                             if not any(path.stem == stem for path in images))
    for image_path in images:
        label_path = labels.get(image_path.stem)
        if label_path is None:
            invalid.append({"file": image_path.name, "problems": ["missing annotation"]})
            continue
        class_id, errors = parse_label(label_path, image_path)
        if errors:
            invalid.append({"file": image_path.name, "problems": errors})
        samples.append({
            "image": image_path,
            "label": label_path,
            "class_id": class_id,
            "hash": sha256(image_path),
        })
    validation = {
        "source_image_count": len(images),
        "unique_image_count": len({sample["hash"] for sample in samples}),
        "invalid_samples": invalid,
        "invalid_image_count": sum(1 for item in invalid if "image could not be decoded" in " ".join(item["problems"])),
        "invalid_annotation_count": sum(1 for item in invalid if any(
            "annotation" in problem or "line " in problem or "label could" in problem
            for problem in item["problems"])),
        "missing_image_label_pair_count": len(missing_labels),
        "missing_labels": missing_labels,
        "orphaned_labels": orphaned_labels,
    }
    return samples, validation


def duplicate_counts(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    within: dict[str, int] = {}
    hashes_by_split: dict[str, set[str]] = {}
    for split, samples in groups.items():
        counts = Counter(sample["hash"] for sample in samples)
        within[split] = sum(count - 1 for count in counts.values() if count > 1)
        hashes_by_split[split] = set(counts)
    across = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            across[f"{first}_vs_{second}"] = len(hashes_by_split[first] & hashes_by_split[second])
    return {"within_split_duplicate_images": within, "across_split_duplicate_images": across}


def write_report(output: Path, report: dict[str, Any]) -> None:
    audit = output / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "dataset_creation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    distributions = report["per_class_image_counts"]
    lines = [
        "# Dataset Creation Report", "",
        f"- Created: {report['created_at']}",
        f"- Random seed: `{report['random_seed']}`",
        "- Split proportions: train 70%, val 15%, test 15%",
        f"- Source images: **{report['source_image_count']}**",
        f"- Unique source images: **{report['unique_image_count']}**",
        f"- Final counts: train **{report['split_image_counts']['train']}**, "
        f"val **{report['split_image_counts']['val']}**, "
        f"test **{report['split_image_counts']['test']}**", "",
        "## Per-denomination image counts", "",
        "| Denomination | Train | Val | Test |", "|---|---:|---:|---:|",
    ]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"| {name} | {distributions['train'][name]} | "
                     f"{distributions['val'][name]} | {distributions['test'][name]} |")
    lines += [
        "", "## Validation",
        f"- Total annotations: **{report['total_annotation_count']}**",
        f"- Invalid images: **{report['invalid_image_count']}**",
        f"- Invalid annotations: **{report['invalid_annotation_count']}**",
        f"- Missing image-label pairs: **{report['missing_image_label_pair_count']}**",
        f"- Exact duplicates within splits: `{report['duplicates']['within_split_duplicate_images']}`",
        f"- Exact duplicates across splits: `{report['duplicates']['across_split_duplicate_images']}`",
        "- Duplicate checks use exact SHA-256 image hashes only; near-duplicates and "
        "physical-note leakage were not assessed.",
        "- Original dataset modified: **No**",
        "", "## Paths",
        f"- Dataset: `{report['dataset_path']}`",
        f"- YAML: `{report['yaml_path']}`",
        f"- Reports: `{(output / 'audit').resolve()}`",
    ]
    (audit / "DATASET_CREATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(source: Path, output: Path, seed: int, overwrite: bool, dry_run: bool) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}. Use --overwrite.")
    samples, validation = validate_source(source)
    if validation["invalid_samples"] or validation["missing_image_label_pair_count"] or validation["orphaned_labels"]:
        print("Source validation failed:", file=sys.stderr)
        for item in validation["invalid_samples"]:
            print(f"  {item['file']}: {'; '.join(item['problems'])}", file=sys.stderr)
        if validation["missing_labels"]:
            print(f"  Missing labels: {', '.join(validation['missing_labels'])}", file=sys.stderr)
        if validation["orphaned_labels"]:
            print(f"  Orphaned labels: {', '.join(validation['orphaned_labels'])}", file=sys.stderr)
        raise SystemExit("Source validation failed; see invalid samples and pair errors above.")
    if len(samples) != 1312:
        raise SystemExit(f"Expected 1312 valid source samples, found {len(samples)}.")

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_class[sample["class_id"]].append(sample)
    if set(by_class) != set(CLASS_NAMES):
        raise SystemExit(f"Expected all classes 0-5, found {sorted(by_class)}.")
    totals = {"train": round(len(samples) * 0.70), "val": round(len(samples) * 0.15)}
    totals["test"] = len(samples) - totals["train"] - totals["val"]
    allocation = allocate_counts({key: len(value) for key, value in by_class.items()}, totals)
    rng = random.Random(seed)
    groups = {split: [] for split in SPLITS}
    for class_id in sorted(by_class):
        rng.shuffle(by_class[class_id])
        start = 0
        for split in SPLITS:
            end = start + allocation[class_id][split]
            groups[split].extend(by_class[class_id][start:end])
            start = end
    if not dry_run:
        if output.exists() and overwrite:
            shutil.rmtree(output)
        for split in SPLITS:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
            for sample in groups[split]:
                shutil.copy2(sample["image"], output / "images" / split / sample["image"].name)
                shutil.copy2(sample["label"], output / "labels" / split / sample["label"].name)
        (output / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "names:\n" + "\n".join(f'  {key}: "{value}"' for key, value in CLASS_NAMES.items()) + "\n",
            encoding="utf-8",
        )
        (output / "README.md").write_text(
            "# Processed YOLO Dataset\n\n"
            "Created from the unique images in `../Data-yolo/images/train` using a "
            "deterministic stratified 70/15/15 split (seed 42). The original dataset "
            "was not modified. Exact duplicate checks use SHA-256 image hashes; "
            "near-duplicates and physical-note leakage were not assessed.\n",
            encoding="utf-8",
        )
    distributions = {
        split: {name: sum(1 for sample in groups[split] if sample["class_id"] == class_id)
                for class_id, name in CLASS_NAMES.items()}
        for split in SPLITS
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "split_proportions": PROPORTIONS,
        "source_path": str(source.resolve()),
        "dataset_path": str(output.resolve()),
        "yaml_path": str((output / "data.yaml").resolve()),
        "source_image_count": validation["source_image_count"],
        "unique_image_count": validation["unique_image_count"],
        "split_image_counts": {split: len(groups[split]) for split in SPLITS},
        "per_class_image_counts": distributions,
        "total_annotation_count": sum(len(sample["label"].read_text(encoding="utf-8").splitlines())
                                      for sample in samples),
        "invalid_image_count": validation["invalid_image_count"],
        "invalid_annotation_count": validation["invalid_annotation_count"],
        "missing_image_label_pair_count": validation["missing_image_label_pair_count"],
        "invalid_samples": validation["invalid_samples"],
        "duplicates": duplicate_counts(groups),
        "original_dataset_modified": False,
        "source_validation": validation,
        "dry_run": dry_run,
    }
    if not dry_run:
        write_report(output, report)
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "CV_model" / "Data-yolo")
    parser.add_argument("--output", type=Path, default=root / "CV_model" / "Data-yolo-processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.seed, args.overwrite, args.dry_run)
    print(json.dumps({
        "status": "validated" if args.dry_run else "created",
        "counts": report["split_image_counts"],
        "per_class": report["per_class_image_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
