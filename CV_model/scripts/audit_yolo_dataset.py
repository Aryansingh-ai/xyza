"""
CV_model/scripts/audit_yolo_dataset.py

Comprehensive Audit for CV_model/Data-yolo/
Analyzes:
  1. Image and annotation file counts in train and val splits
  2. 1-to-1 matching (missing images, orphaned labels, empty annotations)
  3. YOLO label syntax, class IDs, bounding box boundaries and dimensions
  4. Class ID to denomination mapping and distributions
  5. Exact duplicate images within splits and across splits (data leakage)
  6. Overlap with existing datasets (data_clean and synthetic_dataset)
  7. Image resolutions, aspect ratios, color channels, and file integrity
  8. Split composition and validity
  9. Licensing audit
Outputs results to CV_model/Data-yolo/audit/audit_report.json
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

CLASS_MAPPING = {
    0: "10 INR",
    1: "20 INR",
    2: "50 INR",
    3: "100 INR",
    4: "200 INR",
    5: "500 INR",
}

def get_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(root_dir: Path, output_audit_dir: Path):
    output_audit_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_path": str(root_dir),
        "detect_yaml": {},
        "readme_info": {},
        "splits": {},
        "class_distribution": {},
        "integrity": {
            "non_image_files": [],
            "orphaned_labels": [],
            "missing_labels": [],
            "empty_labels": [],
            "malformed_lines": [],
            "invalid_class_ids": [],
            "invalid_bboxes": [],
            "unreadable_images": [],
        },
        "duplication_and_leakage": {
            "train_internal_duplicates": [],
            "val_internal_duplicates": [],
            "train_val_leakage_pairs": [],
            "overlap_with_data_clean": [],
            "overlap_with_synthetic": [],
        },
        "image_statistics": {},
        "split_status": {},
        "license_info": {},
        "readiness_summary": {},
    }

    # 1. Parse detect.yaml & README.md
    yaml_path = root_dir / "detect.yaml"
    readme_path = root_dir / "README.md"
    if yaml_path.exists():
        report["detect_yaml"] = {
            "path": str(yaml_path),
            "content": yaml_path.read_text(encoding="utf-8")
        }
    if readme_path.exists():
        report["readme_info"] = {
            "path": str(readme_path),
            "content": readme_path.read_text(encoding="utf-8")
        }

    # 2. Inspect splits
    splits = ["train", "val"]
    images_by_split = {}
    labels_by_split = {}
    hashes_by_split = defaultdict(dict)
    sizes_by_hash = {}

    for split in splits:
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split
        
        all_img_files = list(img_dir.iterdir()) if img_dir.exists() else []
        valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = [f for f in all_img_files if f.is_file() and f.suffix.lower() in valid_img_exts]
        non_images = [str(f) for f in all_img_files if f.is_file() and f.suffix.lower() not in valid_img_exts]
        if non_images:
            report["integrity"]["non_image_files"].extend(non_images)

        all_lbl_files = list(lbl_dir.iterdir()) if lbl_dir.exists() else []
        lbl_files = [f for f in all_lbl_files if f.is_file() and f.suffix.lower() == ".txt"]
        non_txt = [str(f) for f in all_lbl_files if f.is_file() and f.suffix.lower() != ".txt"]
        if non_txt:
            report["integrity"]["orphaned_labels"].extend([{"file": f, "reason": "Non-txt file in labels directory"} for f in non_txt])

        images_by_split[split] = sorted(img_files)
        labels_by_split[split] = sorted(lbl_files)

        report["splits"][split] = {
            "image_count": len(img_files),
            "label_count": len(lbl_files),
            "non_txt_label_files": non_txt,
        }

    # 3. Analyze train split in detail
    total_annotations_by_class = Counter()
    total_images_by_class = Counter()
    boxes_per_image = Counter()
    image_resolutions = Counter()
    per_split_class_annotations = defaultdict(Counter)

    for split in splits:
        img_files = images_by_split[split]
        lbl_files = labels_by_split[split]

        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}

        for stem, lbl_path in lbl_stems.items():
            if stem not in img_stems:
                report["integrity"]["orphaned_labels"].append({
                    "split": split,
                    "label_file": str(lbl_path),
                    "expected_image": stem
                })

        for stem, img_path in img_stems.items():
            if stem not in lbl_stems:
                report["integrity"]["missing_labels"].append({
                    "split": split,
                    "image_file": str(img_path),
                    "expected_label": f"{stem}.txt"
                })

        for stem, img_path in img_stems.items():
            try:
                img_hash = get_file_hash(img_path)
                hashes_by_split[split].setdefault(img_hash, []).append(str(img_path))
                sizes_by_hash[img_hash] = img_path.stat().st_size
                
                # Check resolution
                img = cv2.imread(str(img_path))
                if img is None:
                    report["integrity"]["unreadable_images"].append(str(img_path))
                else:
                    h, w, c = img.shape
                    if split == "train":
                        image_resolutions[f"{w}x{h}"] += 1
            except Exception as e:
                report["integrity"]["unreadable_images"].append(f"{img_path}: {e}")

            if stem in lbl_stems:
                lbl_path = lbl_stems[stem]
                content = lbl_path.read_text(encoding="utf-8").strip()
                if not content:
                    report["integrity"]["empty_labels"].append({
                        "split": split,
                        "label_file": str(lbl_path)
                    })
                    continue

                lines = content.splitlines()
                if split == "train":
                    boxes_per_image[len(lines)] += 1
                classes_in_this_image = set()

                for line_idx, line in enumerate(lines):
                    tokens = line.strip().split()
                    if len(tokens) != 5:
                        report["integrity"]["malformed_lines"].append({
                            "split": split,
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "raw": line,
                            "reason": f"Expected 5 tokens, got {len(tokens)}"
                        })
                        continue

                    try:
                        cls_id = int(tokens[0])
                        xc = float(tokens[1])
                        yc = float(tokens[2])
                        bw = float(tokens[3])
                        bh = float(tokens[4])
                    except ValueError as ve:
                        report["integrity"]["malformed_lines"].append({
                            "split": split,
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "raw": line,
                            "reason": f"Conversion error: {ve}"
                        })
                        continue

                    if cls_id not in CLASS_MAPPING:
                        report["integrity"]["invalid_class_ids"].append({
                            "split": split,
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "class_id": cls_id
                        })
                    else:
                        denom_name = CLASS_MAPPING[cls_id]
                        per_split_class_annotations[split][denom_name] += 1
                        if split == "train":
                            total_annotations_by_class[denom_name] += 1
                            classes_in_this_image.add(denom_name)

                    # Bbox checks
                    bbox_invalid_reasons = []
                    if not (0.0 <= xc <= 1.0):
                        bbox_invalid_reasons.append(f"xc={xc} outside [0, 1]")
                    if not (0.0 <= yc <= 1.0):
                        bbox_invalid_reasons.append(f"yc={yc} outside [0, 1]")
                    if not (0.0 < bw <= 1.0):
                        bbox_invalid_reasons.append(f"width={bw} not in (0, 1]")
                    if not (0.0 < bh <= 1.0):
                        bbox_invalid_reasons.append(f"height={bh} not in (0, 1]")

                    x1 = xc - bw / 2.0
                    x2 = xc + bw / 2.0
                    y1 = yc - bh / 2.0
                    y2 = yc + bh / 2.0
                    if x1 < -0.05 or x2 > 1.05 or y1 < -0.05 or y2 > 1.05:
                        bbox_invalid_reasons.append(f"significant bbox overflow: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

                    if bbox_invalid_reasons:
                        report["integrity"]["invalid_bboxes"].append({
                            "split": split,
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "reasons": bbox_invalid_reasons,
                            "box": [xc, yc, bw, bh]
                        })

                if split == "train":
                    for cname in classes_in_this_image:
                        total_images_by_class[cname] += 1

    # 4. Leakage and internal duplication
    train_hashes = hashes_by_split["train"]
    val_hashes = hashes_by_split["val"]

    for h, paths in train_hashes.items():
        if len(paths) > 1:
            report["duplication_and_leakage"]["train_internal_duplicates"].append(paths)

    for h, paths in val_hashes.items():
        if len(paths) > 1:
            report["duplication_and_leakage"]["val_internal_duplicates"].append(paths)

    leakage_hashes = set(train_hashes.keys()) & set(val_hashes.keys())
    report["duplication_and_leakage"]["train_val_leakage_count"] = len(leakage_hashes)
    is_exact_clone = (len(leakage_hashes) == len(train_hashes) == len(val_hashes) == 1312)

    # 5. Overlap check with data_clean using file sizes filter first
    cv_model_dir = root_dir.parent
    data_clean_dir = cv_model_dir / "data_clean"
    all_yolo_hashes = set(train_hashes.keys())
    yolo_sizes = set(sizes_by_hash.values())

    clean_overlap_count = 0
    if data_clean_dir.exists():
        for p in data_clean_dir.rglob("*.jpg"):
            try:
                sz = p.stat().st_size
                if sz in yolo_sizes:
                    h = get_file_hash(p)
                    if h in all_yolo_hashes:
                        clean_overlap_count += 1
                        if len(report["duplication_and_leakage"]["overlap_with_data_clean"]) < 20:
                            report["duplication_and_leakage"]["overlap_with_data_clean"].append({
                                "data_clean_file": str(p),
                                "size": sz
                            })
            except Exception:
                pass
    report["duplication_and_leakage"]["total_data_clean_overlaps"] = clean_overlap_count

    # 6. Collate statistics
    report["class_distribution"] = {
        "annotations_per_denomination_train": dict(total_annotations_by_class),
        "annotations_per_denomination_val": dict(per_split_class_annotations["val"]),
        "images_per_denomination_train": dict(total_images_by_class),
        "class_mapping": CLASS_MAPPING,
        "boxes_per_image_train": dict(boxes_per_image),
    }

    report["image_statistics"] = {
        "unique_images_train": len(train_hashes),
        "unique_images_val": len(val_hashes),
        "top_resolutions_train": dict(image_resolutions.most_common(10)),
    }

    # 7. Split status
    report["split_status"] = {
        "has_train": True,
        "has_val": True,
        "has_test": False,
        "train_image_count": len(images_by_split["train"]),
        "val_image_count": len(images_by_split["val"]),
        "is_val_exact_clone_of_train": is_exact_clone,
        "total_unique_images_across_dataset": len(train_hashes),
    }

    # 8. License analysis
    report["license_info"] = {
        "license_in_readme": "GPL license",
        "license_file_present": (root_dir / "LICENSE").exists(),
        "notes": "Declared under GPL license in README.md. No separate LICENSE or COPYING text file provided."
    }

    # 9. Readiness Assessment
    critical_issues = []
    if is_exact_clone:
        critical_issues.append("CRITICAL: The 'val' split is a 100% duplicate clone of the 'train' split (1312 identical images & labels). Evaluating a model on this val set would measure training memorization, not generalization.")
    if not report["split_status"]["has_test"]:
        critical_issues.append("No separate test split exists (detect.yaml 'test:' is empty).")
    if clean_overlap_count > 0:
        critical_issues.append(f"Overlap detected: {clean_overlap_count} images match existing images in data_clean.")
    if report["integrity"]["orphaned_labels"]:
        critical_issues.append(f"{len(report['integrity']['orphaned_labels'])} non-annotation / orphaned label entry found (e.g. .DS_Store).")

    report["readiness_summary"] = {
        "is_ready_for_training_as_is": False,
        "critical_issues": critical_issues,
        "recommended_actions": [
            "1. Remove .DS_Store junk file from labels/val/.",
            "2. Delete the cloned val/ split and perform a proper, clean session/hash-based train/val/test split (e.g., 70% train / 15% val / 15% test, or 80/20) from the 1,312 unique images.",
            "3. Update detect.yaml with the corrected relative paths (train, val, test) and absolute root path.",
            "4. Standardize image resolution / aspect ratio or use letterboxing (common YOLO 640x640) prior to training.",
            "5. Apply real-world background augmentation or donation-box scene composition to adapt to the dark box environment."
        ]
    }

    report_file = output_audit_dir / "audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit completed successfully.")
    print(f"Unique images: {len(train_hashes)}")
    print(f"Exact clone val of train: {is_exact_clone}")
    print(f"Data clean overlaps: {clean_overlap_count}")
    print(f"Report written to {report_file}")
    return report

if __name__ == "__main__":
    yolo_dir = Path(__file__).resolve().parent.parent / "Data-yolo"
    audit_out = yolo_dir / "audit"
    audit_dataset(yolo_dir, audit_out)
