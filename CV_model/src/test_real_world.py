"""
test_real_world.py

Runs the trained model against a folder of real photos YOU captured
(phone camera, realistic conditions: folded, crumpled, varied background,
rushed/slightly blurred) -- separate from the Kaggle-derived test set.

This is the test that actually matters for deployment credibility, since
the Kaggle val/test images likely share background/lighting style within
each denomination's photo session, which the model could be exploiting.

USAGE:
1. Photograph 15-20+ real notes with your phone. Save them into a folder,
   e.g. data/real_test/, with filenames STARTING with the true denomination,
   e.g.:
       100_flat.jpg
       100_folded.jpg
       200_crumpled_lowlight.jpg
       500_blurred.jpg
   (the number before the first underscore is read as the ground truth label)

2. Run:
    python src/test_real_world.py --folder data/real_test --checkpoint models/mobilenet_v3_baseline.pt

OUTPUT:
    Per-image prediction + confidence + decision, and a final accuracy /
    confusion summary against the filename-derived ground truth.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image

from infer import load_model, classify_image


def extract_true_label(filename: str, valid_classes):
    prefix = filename.split("_")[0]
    if prefix in valid_classes:
        return prefix
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/mobilenet_v3_baseline.pt")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Provisional confidence threshold; calibrate before deployment.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.checkpoint, device)
    print(f"Loaded model. Classes: {class_names}\n")

    folder = Path(args.folder)
    image_paths = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}")

    correct = 0
    total_with_label = 0
    dangerous_misclass = 0  # confidently wrong (ACCEPT decision but wrong label)
    confusion = defaultdict(lambda: defaultdict(int))

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        result = classify_image(model, class_names, image, device, args.threshold,
                                pocket_id=img_path.name, source="real_world_test")

        true_label = extract_true_label(img_path.name, class_names)
        pred_label = result["predicted_denomination"]
        conf = result["confidence"]
        decision = result["decision"]

        status = ""
        if true_label:
            total_with_label += 1
            is_correct = (pred_label == true_label)
            correct += int(is_correct)
            confusion[true_label][pred_label] += 1
            if not is_correct and decision == "ACCEPT":
                dangerous_misclass += 1
                status = "  <-- DANGEROUS: confidently wrong"
            elif not is_correct:
                status = "  <-- wrong but caught by REVIEW"
        else:
            status = "  (no ground truth label parsed from filename)"

        print(f"{img_path.name:35s} true={true_label or '?':5s} pred={pred_label:5s} "
              f"conf={conf:.3f} decision={decision:7s}{status}")

    print("\n=== Summary ===")
    if total_with_label > 0:
        acc = correct / total_with_label
        print(f"Accuracy (of {total_with_label} labeled images): {acc:.3f}")
        print(f"Dangerous misclassifications (confidently ACCEPTed but wrong): {dangerous_misclass}")
        print("\nConfusion (true -> predicted counts):")
        for true_cls in sorted(confusion.keys()):
            print(f"  {true_cls}: {dict(confusion[true_cls])}")
    else:
        print("No filenames had a parseable ground-truth label "
              "(name files like '100_description.jpg'). Review predictions above manually.")


if __name__ == "__main__":
    main()
