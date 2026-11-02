"""
CV_model/scripts/collect_negatives.py

Tool for capturing real-world negative examples (non-banknotes) inside the
donation box setup for binary detector and classifier validation.

Captures images of:
  - hand / fingers inside box
  - empty black box
  - cardboard and box surfaces
  - random paper, receipts, envelopes
  - other unknown objects (keys, cups, coins, packaging)

Usage:
    python scripts/collect_negatives.py --category hand
    python scripts/collect_negatives.py --category empty_box
    python scripts/collect_negatives.py --category random_paper
    python scripts/collect_negatives.py --category random_object
    python scripts/collect_negatives.py --camera 0

Keys:
    SPACE : capture current frame
    Q     : quit
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_BASE = PROJECT_ROOT / "data_negatives"

VALID_CATEGORIES = ("hand", "empty_box", "random_paper", "random_object", "cardboard")


def main():
    parser = argparse.ArgumentParser(description="Capture negative examples for banknote detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--category",
        type=str,
        default="hand",
        choices=VALID_CATEGORIES,
        help="Negative category to capture",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (default: data_negatives/<category>)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUT_BASE / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    win_name = f"DivyaKosh Negative Collector - [{args.category}]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    count = len(list(out_dir.glob("*.jpg")))
    print("=" * 60)
    print(f"Capturing category : {args.category}")
    print(f"Output directory   : {out_dir}")
    print(f"Existing images    : {count}")
    print("Controls: SPACE = capture, Q = quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"Category: {args.category} | Captured: {count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "SPACE = Capture | Q = Quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                count += 1
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                fname = f"NEG_{args.category}_{ts}.jpg"
                fpath = out_dir / fname
                cv2.imwrite(str(fpath), frame)
                print(f"Saved: {fname} (total: {count})")
                time.sleep(0.2)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished. Total images in {out_dir}: {count}")


if __name__ == "__main__":
    main()

