"""
scripts/capture_note_count_dataset.py

Simple webcam data collection tool for a future YOLO note-count detector.
This script captures ROI images (exactly matching live_camera.py) to build
a dataset containing:
- exactly 1 physical note (SINGLE)
- 2+ overlapping/stacked physical notes (MULTIPLE)
- 0 notes / background (EMPTY)

Usage:
    python scripts/capture_note_count_dataset.py
    python scripts/capture_note_count_dataset.py --camera 1

Controls:
    [S] : Save current ROI as SINGLE (1 note)
    [M] : Save current ROI as MULTIPLE (2+ notes)
    [E] : Save current ROI as EMPTY (0 notes)
    [Q] : Quit
"""

import os
import cv2
import argparse
from pathlib import Path

# Reuse exact ROI calculation from live_camera.py
def compute_roi(frame_h, frame_w):
    # Central region approximately matching banknote proportions.
    roi_w = int(frame_w * 0.55)
    roi_h = int(roi_w * 0.45)
    roi_h = min(roi_h, int(frame_h * 0.65))

    x1 = (frame_w - roi_w) // 2
    y1 = (frame_h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # Create directories
    base_dir = Path("note_count_raw")
    dirs = {
        "single": base_dir / "single",
        "multiple": base_dir / "multiple",
        "empty": base_dir / "empty"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Count existing files to initialize counters
    counters = {
        "single": len(list(dirs["single"].glob("*.jpg"))),
        "multiple": len(list(dirs["multiple"].glob("*.jpg"))),
        "empty": len(list(dirs["empty"].glob("*.jpg"))),
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        return

    window_name = "YOLO Data Capture - Press S/M/E/Q"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Started camera {args.camera}. Focus on the window to use hotkeys.")
    print("Controls:")
    print("  [S] Save SINGLE (1 note)")
    print("  [M] Save MULTIPLE (2+ notes)")
    print("  [E] Save EMPTY (0 notes)")
    print("  [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = compute_roi(h, w)

        # The region we want to save
        roi_crop = frame[y1:y2, x1:x2].copy()

        # Draw ROI overlay on display frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Display counters and instructions
        y_offset = 30
        cv2.putText(display_frame, f"SINGLE (S): {counters['single']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"MULTIPLE (M): {counters['multiple']}", (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display_frame, f"EMPTY (E): {counters['empty']}", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display_frame, "QUIT (Q)", (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            counters["single"] += 1
            path = dirs["single"] / f"single_{counters['single']:05d}.jpg"
            cv2.imwrite(str(path), roi_crop)
            print(f"Saved {path}")
        elif key == ord('m'):
            counters["multiple"] += 1
            path = dirs["multiple"] / f"multiple_{counters['multiple']:05d}.jpg"
            cv2.imwrite(str(path), roi_crop)
            print(f"Saved {path}")
        elif key == ord('e'):
            counters["empty"] += 1
            path = dirs["empty"] / f"empty_{counters['empty']:05d}.jpg"
            cv2.imwrite(str(path), roi_crop)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Capture complete. Goodbye.")

if __name__ == "__main__":
    main()
