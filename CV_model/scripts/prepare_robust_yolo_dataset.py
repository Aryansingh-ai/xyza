"""Prepare a provenance-aware, augmented YOLO banknote detection dataset.

Only samples with trustworthy boxes are included:
* Data-yolo-processed: existing validated YOLO labels.
* synthetic_dataset/black_box: conservative black-background segmentation.
* data_clean: additional synthetic ₹500 images, segmented the same way.

Unannotated real-camera/classification images are reported as unresolved and
are never assigned a guessed full-image box.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = {
    0: "10 INR",
    1: "20 INR",
    2: "50 INR",
    3: "100 INR",
    4: "200 INR",
    5: "500 INR",
}
DENOM_TO_ID = {name.split()[0]: class_id for class_id, name in CLASS_NAMES.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
PROPORTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}


@dataclass
class Sample:
    image: Path
    class_id: int
    bbox: tuple[int, int, int, int]
    source: str
    group_id: str
    provenance: str
    image_hash: str = ""
    phash: int = 0


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    median = float(np.median(dct[1:]))
    bits = 0
    for value in dct.flat:
        bits = (bits << 1) | int(value > median)
    return bits


def phash_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def read_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def validate_yolo_label(label_path: Path, image: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None, list[str]]:
    height, width = image.shape[:2]
    errors: list[str] = []
    rows = label_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        errors.append("expected exactly one annotation line")
    if not rows:
        return 0, None, ["empty annotation"]
    fields = rows[0].split()
    if len(fields) != 5:
        return 0, None, ["annotation must have five fields"]
    try:
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError:
        return 0, None, ["annotation contains non-numeric fields"]
    if class_id not in CLASS_NAMES:
        errors.append(f"class ID {class_id} outside 0-5")
    if not all(math.isfinite(value) for value in values):
        errors.append("annotation contains non-finite values")
    x_center, y_center, box_width, box_height = values
    if box_width <= 0 or box_height <= 0:
        errors.append("box width and height must be positive")
    if (x_center - box_width / 2 < 0 or x_center + box_width / 2 > 1
            or y_center - box_height / 2 < 0 or y_center + box_height / 2 > 1):
        errors.append("box lies outside normalized image boundaries")
    if errors:
        return class_id, None, errors
    x1 = max(0, int(round((x_center - box_width / 2) * width)))
    y1 = max(0, int(round((y_center - box_height / 2) * height)))
    x2 = min(width, int(round((x_center + box_width / 2) * width)))
    y2 = min(height, int(round((y_center + box_height / 2) * height)))
    if x2 <= x1 or y2 <= y1:
        return class_id, None, ["pixel bounding box is empty"]
    return class_id, (x1, y1, x2, y2), []


def segment_black_background(image: np.ndarray) -> tuple[tuple[int, int, int, int] | None, str]:
    """Find a conservative foreground component against the synthetic matte box."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([
        gray[:5].ravel(), gray[-5:].ravel(),
        gray[:, :5].ravel(), gray[:, -5:].ravel(),
    ])
    threshold = max(18.0, float(np.median(border)) + 12.0)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no foreground component"
    contour = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(contour)
    area_ratio = cv2.contourArea(contour) / float(width * height)
    if box_width < 20 or box_height < 20:
        return None, "foreground component too small"
    if area_ratio < 0.01:
        return None, "foreground area ratio below 0.01"
    if area_ratio > 0.90:
        return None, "foreground component covers nearly the whole frame"
    return (x, y, x + box_width, y + box_height), ""


def load_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest_path = root / "data_clean" / "split_manifest.csv"
    by_hash: dict[str, dict[str, str]] = {}
    if not manifest_path.exists():
        return by_hash
    for row in csv.DictReader(manifest_path.open(encoding="utf-8")):
        path = root / row["image_path"]
        if path.exists():
            by_hash[image_hash(path)] = row
    return by_hash


def add_processed_samples(root: Path, samples: list[Sample], report: dict[str, Any]) -> None:
    image_root = root / "Data-yolo-processed" / "images"
    label_root = root / "Data-yolo-processed" / "labels"
    invalid = report["invalid_by_source"]["processed_yolo"]
    for split in SPLITS:
        for image_path in sorted(image_root.joinpath(split).iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = read_image(image_path)
            label_path = label_root / split / f"{image_path.stem}.txt"
            if image is None:
                invalid.append({"file": str(image_path), "reason": "image could not be decoded"})
                continue
            if not label_path.exists():
                invalid.append({"file": str(image_path), "reason": "missing YOLO label"})
                continue
            class_id, bbox, errors = validate_yolo_label(label_path, image)
            if errors:
                invalid.append({"file": str(image_path), "reason": "; ".join(errors)})
                continue
            samples.append(Sample(
                image_path, class_id, bbox, "processed_yolo",
                f"processed:{image_hash(image_path)}", str(image_path),
            ))
    report["source_counts"]["processed_yolo"]["images"] = sum(
        1 for split in SPLITS
        for path in (image_root / split).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def add_synthetic_samples(root: Path, samples: list[Sample], manifest: dict[str, dict[str, str]],
                          report: dict[str, Any]) -> None:
    candidates: list[tuple[Path, str]] = []
    synthetic_root = root / "synthetic_dataset" / "black_box"
    for path in sorted(synthetic_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append((path, "synthetic_black_box"))
    clean_root = root / "data_clean"
    for path in sorted(clean_root.rglob("SYN_500_*.jpg")):
        candidates.append((path, "data_clean_500_synthetic"))
    invalid = report["invalid_by_source"]["synthetic"]
    for image_path, source in candidates:
        image = read_image(image_path)
        if image is None:
            invalid.append({"file": str(image_path), "reason": "image could not be decoded"})
            continue
        file_hash = image_hash(image_path)
        metadata = manifest.get(file_hash, {})
        denomination = metadata.get("denomination") or image_path.parent.name
        class_id = DENOM_TO_ID.get(denomination)
        if class_id is None:
            invalid.append({"file": str(image_path), "reason": f"unknown denomination {denomination}"})
            continue
        bbox, reason = segment_black_background(image)
        if bbox is None:
            invalid.append({"file": str(image_path), "reason": reason})
            continue
        physical_id = metadata.get("physical_note_id") or f"unmapped:{file_hash}"
        samples.append(Sample(
            image_path, class_id, bbox, source, f"physical:{physical_id}",
            metadata.get("image_path", str(image_path)),
            image_hash=file_hash, phash=perceptual_hash(image),
        ))
    report["source_counts"]["synthetic_black_box"]["images"] = len(
        list((root / "synthetic_dataset" / "black_box").rglob("*.jpg"))
    )
    report["source_counts"]["data_clean_500_synthetic"]["images"] = len(
        list((root / "data_clean").rglob("SYN_500_*.jpg"))
    )


def audit_clean_unresolved(root: Path, manifest: dict[str, dict[str, str]],
                           report: dict[str, Any]) -> None:
    synthetic_hashes = {image_hash(path) for path in (root / "synthetic_dataset" / "black_box").rglob("*.jpg")}
    all_images = [path for path in (root / "data_clean").rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    report["source_counts"]["data_clean"]["images"] = len(all_images)
    for path in all_images:
        row = manifest.get(image_hash(path), {})
        if image_hash(path) in synthetic_hashes or path.name.startswith("SYN_500_"):
            continue
        report["unresolved_data_clean"].append({
            "file": str(path),
            "denomination": row.get("denomination", path.parent.name),
            "source_type": row.get("source_type", "unknown"),
            "reason": "no reliable bounding-box annotation or retained mask",
        })


def assign_splits(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    groups_by_class: dict[int, list[list[Sample]]] = defaultdict(list)
    for group in groups.values():
        groups_by_class[group[0].class_id].append(group)
    for class_id in sorted(groups_by_class):
        class_groups = groups_by_class[class_id]
        rng.shuffle(class_groups)
        class_groups.sort(key=len, reverse=True)
        targets = {split: sum(s.class_id == class_id for s in samples) * PROPORTIONS[split]
                   for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        # Ensure every available class is represented in all three splits.
        guaranteed = min(len(class_groups), len(SPLITS))
        for index in range(guaranteed):
            split = SPLITS[index]
            group = class_groups[index]
            result[split].extend(group)
            counts[split] += len(group)
        for group in class_groups[guaranteed:]:
            best_split = min(
                SPLITS,
                key=lambda split: (
                    counts[split] / max(targets[split], 1),
                    counts[split],
                ),
            )
            result[best_split].extend(group)
            counts[best_split] += len(group)
    return result


def transform_sample(image: np.ndarray, bbox: tuple[int, int, int, int],
                     rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.90, 1.10)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-0.08, 0.08) * width
    matrix[1, 2] += rng.uniform(-0.08, 0.08) * height
    transformed = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    x1, y1, x2, y2 = bbox
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    new_corners = cv2.transform(corners, matrix).reshape(-1, 2)
    nx1, ny1 = np.min(new_corners, axis=0)
    nx2, ny2 = np.max(new_corners, axis=0)
    if rng.random() < 0.7:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-20, 20)
        transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        gamma = rng.uniform(0.75, 1.30)
        lookup = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        transformed = cv2.LUT(transformed, lookup)
    if rng.random() < 0.25:
        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)
    if rng.random() < 0.25:
        noise = rng.uniform(1.0, 4.0)
        transformed = np.clip(
            transformed.astype(np.float32) + np.random.default_rng(rng.randrange(2**32)).normal(0, noise, transformed.shape),
            0, 255,
        ).astype(np.uint8)
    nx1 = max(0, min(width - 1, int(math.floor(nx1))))
    ny1 = max(0, min(height - 1, int(math.floor(ny1))))
    nx2 = max(nx1 + 1, min(width, int(math.ceil(nx2))))
    ny2 = max(ny1 + 1, min(height, int(math.ceil(ny2))))
    return transformed, (nx1, ny1, nx2, ny2)


def write_yolo_label(path: Path, class_id: int, bbox: tuple[int, int, int, int],
                     image_shape: tuple[int, ...]) -> None:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    values = (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    values = tuple(max(0.0, min(1.0, value)) for value in values)
    center_x, center_y, box_width, box_height = values
    box_width = min(box_width, 2 * min(center_x, 1 - center_x))
    box_height = min(box_height, 2 * min(center_y, 1 - center_y))
    values = (center_x, center_y, box_width, box_height)
    path.write_text(f"{class_id} " + " ".join(f"{value:.8f}" for value in values) + "\n", encoding="utf-8")


def draw_preview(path: Path, sample: Sample) -> None:
    image = read_image(sample.image)
    if image is None:
        return
    x1, y1, x2, y2 = sample.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(image, f"{CLASS_NAMES[sample.class_id]} | {sample.source}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def exact_and_near_duplicates(groups: dict[str, list[Sample]]) -> dict[str, Any]:
    exact = {split: 0 for split in SPLITS}
    hashes = {}
    phashes = {}
    for split, samples in groups.items():
        hash_counts = Counter(sample.image_hash or image_hash(sample.image) for sample in samples)
        exact[split] = sum(count - 1 for count in hash_counts.values() if count > 1)
        hashes[split] = set(hash_counts)
        phashes[split] = [sample.phash or perceptual_hash(read_image(sample.image)) for sample in samples]
    across_exact = {}
    across_near = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            across_exact[f"{first}_vs_{second}"] = len(hashes[first] & hashes[second])
            second_hashes = phashes[second]
            buckets: dict[int, list[int]] = defaultdict(list)
            for candidate in second_hashes:
                buckets[candidate >> 48].append(candidate)
            near = 0
            for value in phashes[first]:
                if any(phash_distance(value, candidate) <= 6
                       for candidate in buckets.get(value >> 48, [])):
                    near += 1
            across_near[f"{first}_vs_{second}"] = near
    return {
        "exact_duplicates_within_split": exact,
        "exact_duplicates_across_splits": across_exact,
        "near_duplicate_threshold_hamming": 6,
        "near_duplicate_candidates_across_splits": across_near,
    }


def write_reports(output: Path, reports: dict[str, Any]) -> None:
    audit = output / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    for filename, value in reports.items():
        if filename == "final":
            continue
        (audit / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final = reports["final"]
    lines = [
        "# Robust YOLO Dataset Preparation Report", "",
        f"- Created: {final['created_at']}",
        f"- Random seed: `{final['seed']}`",
        f"- Dataset: `{final['dataset_path']}`",
        f"- YAML: `{final['yaml_path']}`", "",
        "## Source usage",
        f"- Processed YOLO images used: **{final['source_usage']['processed_yolo']['used']}**",
        f"- Synthetic black-box images used: **{final['source_usage']['synthetic_black_box']['used']}**",
        f"- Additional data_clean ₹500 synthetic images used: **{final['source_usage']['data_clean_500_synthetic']['used']}**",
        f"- data_clean unresolved images: **{final['source_usage']['data_clean']['unresolved']}**", "",
        "## Final split counts",
        "| Split | Images |", "|---|---:|",
    ]
    for split in SPLITS:
        lines.append(f"| {split} | {final['split_counts'][split]} |")
    lines += ["", "| Denomination | Train | Val | Test |", "|---|---:|---:|---:|"]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"| {name} | {final['class_counts']['train'][name]} | "
                     f"{final['class_counts']['val'][name]} | {final['class_counts']['test'][name]} |")
    lines += [
        "", "## Annotation and augmentation",
        f"- Valid original annotations: **{final['valid_original_annotations']}**",
        f"- Invalid/rejected annotations: **{final['invalid_annotations']}**",
        f"- Augmented training images: **{final['augmented_images']}**",
        "- Augmentation is applied only after grouped splitting and only to training samples.",
        "- Geometric augmentation does not create physically realistic folded or crumpled notes.",
        "", "## Limitations",
        "- Exact duplicate checks use SHA-256; near-duplicate candidates use 64-bit DCT perceptual hashes.",
        "- Physical-note grouping is available for synthetic families through the clean manifest; "
        "processed YOLO images have no shared physical-note metadata.",
        "- No retained masks were present in the synthetic output, so boxes were derived from the "
        "black-background foreground and must be visually reviewed.",
        "- Folded/crumpled-note presence was not established automatically; real-camera samples were unresolved.",
        "- Empty/unrelated-object negatives are not included in this positive detection dataset.",
        "- Original datasets and checkpoints were not modified.",
    ]
    (audit / "DATASET_PREPARATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = root / "Data-yolo-robust"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}. Use --overwrite.")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)

    manifest = load_manifest(root)
    report = {
        "source_counts": {
            "processed_yolo": {"images": 0},
            "data_clean": {"images": 0},
            "synthetic_black_box": {"images": 0},
            "data_clean_500_synthetic": {"images": 0},
        },
        "invalid_by_source": {"processed_yolo": [], "synthetic": []},
        "unresolved_data_clean": [],
    }
    samples: list[Sample] = []
    add_processed_samples(root, samples, report)
    add_synthetic_samples(root, samples, manifest, report)
    audit_clean_unresolved(root, manifest, report)
    samples_by_hash: dict[str, Sample] = {}
    duplicate_source_samples = 0
    for sample in samples:
        sample.image_hash = sample.image_hash or image_hash(sample.image)
        if sample.image_hash in samples_by_hash:
            duplicate_source_samples += 1
            continue
        samples_by_hash[sample.image_hash] = sample
    samples = list(samples_by_hash.values())
    if not samples:
        raise SystemExit("No reliable annotated samples were found; refusing to create a misleading dataset.")
    groups = assign_splits(samples, args.seed)
    if any(not groups[split] for split in SPLITS):
        raise SystemExit("Grouped split produced an empty subset.")

    if args.audit_only:
        return {
            "source_report": report,
            "sample_count": len(samples),
            "split_counts": {split: len(groups[split]) for split in SPLITS},
        }

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_seen: set[tuple[str, int]] = set()
    rng = random.Random(args.seed)
    augmented = 0
    for split, split_samples in groups.items():
        for index, sample in enumerate(split_samples):
            image = read_image(sample.image)
            if image is None:
                continue
            base_name = f"{sample.source}_{sample.image_hash[:12]}_{index}"
            image_out = output / "images" / split / f"{base_name}.jpg"
            label_out = output / "labels" / split / f"{base_name}.txt"
            shutil.copy2(sample.image, image_out)
            if sample.source == "processed_yolo":
                original_label = (
                    sample.image.parents[2] / "labels" / sample.image.parent.name
                    / f"{sample.image.stem}.txt"
                )
                label_out.write_text(original_label.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                write_yolo_label(label_out, sample.class_id, sample.bbox, image.shape)
            preview_key = (sample.source, sample.class_id)
            if preview_key not in preview_seen:
                draw_preview(output / "audit" / "previews" / f"{sample.source}_{sample.class_id}.jpg", sample)
                preview_seen.add(preview_key)
            if split == "train" and args.augmentations_per_image:
                for aug_index in range(args.augmentations_per_image):
                    augmented_image, augmented_bbox = transform_sample(image, sample.bbox, rng)
                    aug_name = f"{base_name}_aug{aug_index + 1}"
                    cv2.imwrite(str(output / "images" / split / f"{aug_name}.jpg"), augmented_image,
                                [cv2.IMWRITE_JPEG_QUALITY, 94])
                    write_yolo_label(output / "labels" / split / f"{aug_name}.txt",
                                     sample.class_id, augmented_bbox, augmented_image.shape)
                    augmented += 1

    class_counts = {
        split: {name: sum(sample.class_id == class_id for sample in groups[split])
                for class_id, name in CLASS_NAMES.items()}
        for split in SPLITS
    }
    final = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_path": str(output.resolve()),
        "yaml_path": str((output / "data.yaml").resolve()),
        "source_usage": {
            "processed_yolo": {"available": report["source_counts"]["processed_yolo"]["images"],
                               "used": sum(s.source == "processed_yolo" for s in samples)},
            "synthetic_black_box": {"available": report["source_counts"]["synthetic_black_box"]["images"],
                                    "used": sum(s.source == "synthetic_black_box" for s in samples)},
            "data_clean_500_synthetic": {
                "available": report["source_counts"]["data_clean_500_synthetic"]["images"],
                "used": sum(s.source == "data_clean_500_synthetic" for s in samples),
            },
            "data_clean": {
                "available": report["source_counts"]["data_clean"]["images"],
                "unresolved": len(report["unresolved_data_clean"]),
            },
        },
        "split_counts": {split: len(groups[split]) for split in SPLITS},
        "class_counts": class_counts,
        "valid_original_annotations": len(samples),
        "invalid_annotations": sum(len(values) for values in report["invalid_by_source"].values()),
        "augmented_images": augmented,
        "duplicate_source_samples_removed": duplicate_source_samples,
        "unresolved_data_clean_count": len(report["unresolved_data_clean"]),
        "folded_or_crumpled_examples": "not established; unresolved real-camera images were excluded",
        "negative_testing_images": 0,
    }
    write_reports(output, {
        "source_dataset_report.json": {
            **report,
            "masks_present_in_synthetic_output": (root / "synthetic_dataset" / "masks").exists(),
            "segmentation_method": "largest connected foreground component above matte-black border threshold",
        },
        "augmentation_report.json": {
            "seed": args.seed,
            "augmentations_per_training_image": args.augmentations_per_image,
            "augmented_images": augmented,
            "operations": ["rotation", "scale", "translation", "brightness_contrast",
                           "gamma", "blur", "noise"],
            "training_only": True,
            "physical_fold_fabrication": False,
        },
        "split_report.json": {
            "seed": args.seed,
            "proportions": PROPORTIONS,
            "grouping_key": "physical_note_id from data_clean manifest where available; image hash otherwise",
            "split_counts": final["split_counts"],
            "class_counts": final["class_counts"],
            "duplicates": exact_and_near_duplicates(groups),
        },
        "annotation_validation_report.json": {
            "valid_original_annotations": len(samples),
            "invalid_annotations": report["invalid_by_source"],
            "generated_boxes": "synthetic images use conservative foreground segmentation",
            "preview_directory": str((output / "audit" / "previews").resolve()),
        },
        "final": final,
    })
    (output / "data.yaml").write_text(
        f"path: '{output.resolve()}'\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n" + "\n".join(f'  {class_id}: "{name}"' for class_id, name in CLASS_NAMES.items()) + "\n",
        encoding="utf-8",
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentations-per-image", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
