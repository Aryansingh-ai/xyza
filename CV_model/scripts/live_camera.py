"""
DaanDristi live webcam inference demo — DivyaKosh Smart Donation Box.

5-Stage Moving-Box Computer Vision Pipeline:
  Stage A: Donation-box presence detection (adaptive centre-ROI, NOT global-mean based)
  Stage B: Movement & sharpness quality gate (on note crop, not full frame)
  Stage C: Banknote candidate detection (geometry, skin filter, edge density)
  Stage D: Dynamic crop + MobileNetV3 inference (immediate live prediction)
  Stage E: Multi-frame consensus + deduplication (donation confirmation only)

UI features:
  - Live denomination + confidence shown IMMEDIATELY on first detection
  - Class probability panel (all 6 denominations)
  - Bounding-box ROI on detected note
  - Box status / note status / donation status
  - FPS + sharpness telemetry
  - Rejection reason display

Run:
    python scripts/live_camera.py
    python scripts/live_camera.py --camera 1
    python scripts/live_camera.py --accept_confidence 0.85 --consensus_frames 3
    python scripts/live_camera.py --debug           # Print per-frame diagnostics
    python scripts/live_camera.py --timeout 10      # Custom inactivity timeout (default 7s)
    python scripts/live_camera.py --continuous      # Don't exit after first acceptance (MobileNet pipeline only)

YOLO pipeline (recommended):
    python scripts/live_camera.py --yolo_pipeline
    - Detects the first valid banknote within --timeout seconds (default 7).
    - Stops scanning immediately after the first successful detection.
    - If no banknote is found within the timeout, exits with a clear message.
    - --continuous is ignored in YOLO pipeline mode (always one-shot).

Press Q to quit. Press R to manually reset the tracker.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# PROJECT ROOT — set before any local imports
# ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────
# ML IMPORTS
# ─────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ─────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────
from src.validator import (
    ValidationConfig,
    BanknoteTracker,
    ValidationOutcome,
    DENOM_LABELS,
    SUPPORTED_DENOMINATIONS,
    EVAL_TRANSFORM,
    crop_and_classify,
)
from src.review_tracker import ReviewTracker
from sqlite.review_database import REVIEW_DB_FILE

# ─────────────────────────────────────────────────────────────────
# BLOCKCHAIN
# ─────────────────────────────────────────────────────────────────
from src.blockchain import register_blockchain_proof

# ─────────────────────────────────────────────────────────────────
# SQLITE
# ─────────────────────────────────────────────────────────────────
try:
    from sqlite.importer import import_single_record
    from sqlite.config import DATABASE_FILE as _SQLITE_DB_FILE
    _SQLITE_AVAILABLE = True
except Exception as _sqlite_err:
    _SQLITE_AVAILABLE = False
    print(
        f"Warning: SQLite module unavailable ({_sqlite_err}). "
        "Donations will only be written to the JSONL log."
    )

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
WINDOW_NAME    = "DaanDristi - Live Demo"
WINDOW_WIDTH   = 1280
WINDOW_HEIGHT  = 720
WINDOW_POS_X   = 50
WINDOW_POS_Y   = 50
MODEL_VERSION  = "mobilenet_v3_clean_v1"
YOLO_MODEL_VERSION = "yolo26n_robust_train_seed42_20_20261101_203031"

# Colours (BGR)
COL_WHITE   = (255, 255, 255)
COL_GRAY    = (170, 170, 170)
COL_DARK    = (80,  80,  80)
COL_GREEN   = (50,  220, 50)
COL_AMBER   = (30,  180, 255)
COL_RED     = (50,  50,  230)
COL_CYAN    = (200, 210, 20)
COL_BLUE    = (220, 120, 30)
COL_NEUTRAL = (150, 150, 150)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Device       : {device}")
    print(f"Model path   : {checkpoint_path}")
    print(f"Ckpt epoch   : {ckpt.get('epoch', '?')}")
    print(f"Ckpt val_acc : {ckpt.get('val_acc', '?')}")
    print(f"Classes      : {class_names}")
    return model, class_names


def load_yolo_model(checkpoint_path: str):
    """Load the optional YOLO robust banknote detector once at application startup."""
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed; YOLO mode is unavailable.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
    model = YOLO(str(checkpoint))
    print(f"YOLO model path: {checkpoint}")
    return model


def clip_yolo_bbox(xyxy, frame_shape):
    """Validate and clip a YOLO xyxy box to the original frame."""
    values = np.asarray(xyxy, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        return None
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = values
    x1 = max(0, min(frame_width, int(np.floor(x1))))
    y1 = max(0, min(frame_height, int(np.floor(y1))))
    x2 = max(0, min(frame_width, int(np.ceil(x2))))
    y2 = max(0, min(frame_height, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_yolo_detection(result, frame_shape, confidence_threshold: float):
    """Return the highest-confidence valid banknote detection in a result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    candidates = []
    for index in range(len(boxes)):
        confidence = float(boxes.conf[index].item())
        class_id = int(boxes.cls[index].item())
        if confidence < confidence_threshold or class_id not in range(6):
            continue
        bbox = clip_yolo_bbox(boxes.xyxy[index].cpu().numpy(), frame_shape)
        if bbox is not None:
            candidates.append((confidence, class_id, bbox))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def append_review_event(review_log_path: Path, camera_index: int, review_id: str,
                        reason: str, candidate_label, confidence: float,
                        stage: str, user_message: str) -> None:
    record = {
        "event_type": "REVIEW",
        "review_id": review_id,
        "pocket_id": f"webcam-{camera_index}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "candidate_label": candidate_label,
        "classifier_confidence": confidence,
        "stage": stage,
        "user_message": user_message,
    }
    review_log_path.parent.mkdir(parents=True, exist_ok=True)
    with review_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"[REVIEW EVENT CREATED] {review_id} | reason: {reason} | {user_message}")


def run_yolo_detection_window(cap, yolo_model, window_seconds: float,
                              confidence_threshold: float, window_name: str,
                              camera_index: int, review_tracker: ReviewTracker,
                              review_log_path: Path):
    """Scan for a single banknote for at most *window_seconds*.

    The timeout is measured **once** from the moment this function is called; it
    is never reset on a per-frame basis.  The function returns as soon as the
    first valid banknote is detected — it does NOT continue scanning afterward.

    Return values
    -------------
    - ``(frame, (x, y, w, h), class_id)`` — banknote found; caller must process
      it exactly once.
    - ``(last_frame, None, None)``          — timeout expired; no banknote found.
    - ``None``                              — user pressed 'q' to quit.
    """
    scan_start = time.monotonic()           # Absolute scan-start time
    deadline   = scan_start + window_seconds
    last_frame = None

    while True:
        now = time.monotonic()
        if now >= deadline:
            # Timeout: emit a single review event then return "no banknote"
            created, review_id = review_tracker.should_create_event("not_a_banknote", None, 0.0)
            if created and review_id:
                append_review_event(
                    review_log_path, camera_index, review_id, "not_a_banknote",
                    None, 0.0, "YOLO_DETECTION", "No banknote detected during the detection window.",
                )
            return (last_frame, None, None)

        ret, frame = cap.read()
        if not ret:
            # Frame read failure is distinct from timeout — propagate so the caller
            # can report a useful error message instead of silently timing out.
            raise RuntimeError("Failed to read frame from camera during YOLO detection.")
        last_frame = frame

        try:
            results = yolo_model.predict(source=frame, conf=confidence_threshold, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        detection = None
        for result in results:
            detection = select_yolo_detection(result, frame.shape, confidence_threshold)
            if detection is not None:
                break

        display = frame.copy()
        remaining = max(0.0, deadline - time.monotonic())
        _put(display, f"Scanning… {remaining:.1f}s remaining", (12, 28), color=COL_WHITE, thickness=2)

        if detection is not None:
            confidence, class_id, (x1, y1, x2, y2) = detection
            cv2.rectangle(display, (x1, y1), (x2, y2), COL_GREEN, 2)
            _put(display, f"Banknote detected {confidence:.2f}", (x1, max(20, y1 - 8)),
                 color=COL_GREEN, thickness=2)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            # Return immediately — stop scanning; caller processes exactly once
            return frame, (x1, y1, x2 - x1, y2 - y1), class_id

        _put(display, "No banknote detected", (12, 58), color=COL_AMBER, thickness=2)
        cv2.imshow(window_name, display)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return None


# ─────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────
def _put(frame, text, pos, scale=0.55, color=COL_GRAY, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _bar(frame, x, y, w, h, fill_frac, col_fill, col_bg=COL_DARK):
    cv2.rectangle(frame, (x, y), (x + w, y + h), col_bg, -1)
    fw = max(0, int(w * min(fill_frac, 1.0)))
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + h), col_fill, -1)


def _corner_brackets(frame, x1, y1, x2, y2, color, length=18, thickness=2):
    pts = [
        (x1, y1, +1, +1), (x2, y1, -1, +1),
        (x1, y2, +1, -1), (x2, y2, -1, -1),
    ]
    for cx, cy, dx, dy in pts:
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness)


def draw_overlay(
    frame: np.ndarray,
    outcome: "ValidationOutcome",
    fps: float,
    class_names: list,
    search_bbox: tuple,   # (x,y,w,h) of box search ROI
    debug: bool = False,
):
    """
    Render all UI elements on the frame.

    Layout:
      Top-left     : Title + FPS/Sharpness
      Centre ROI   : Box search rectangle (thin dark border)
      Note ROI     : Bright bounding box + corner brackets when detected
      Left panel   : Status lines (box / note / prediction / decision)
      Right panel  : Class probability bars
      Bottom-left  : Quit / Reset help
    """
    h, w = frame.shape[:2]

    # ── Search-ROI border ─────────────────────────────────────────────────
    bx, by, bw, bh = search_bbox
    bx2, by2 = bx + bw, by + bh
    box_border_col = (70, 70, 70) if outcome.box_ready else (40, 40, 180)
    cv2.rectangle(frame, (bx, by), (bx2, by2), box_border_col, 1)
    _corner_brackets(frame, bx, by, bx2, by2, box_border_col, length=16, thickness=2)
    _put(frame, "[ DONATION BOX CAVITY ]", (bx + 8, by - 8), scale=0.45, color=box_border_col, thickness=1)

    # ── Note bounding box ────────────────────────────────────────────────
    if outcome.note_bbox is not None:
        nx, ny, nw, nh = outcome.note_bbox
        if outcome.decision == "ACCEPT":
            note_col = COL_GREEN
        elif outcome.decision == "REVIEW":
            note_col = COL_AMBER
        elif outcome.note_detected:
            note_col = COL_CYAN
        else:
            note_col = COL_NEUTRAL
        cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), note_col, 2)
        _corner_brackets(frame, nx, ny, nx + nw, ny + nh, note_col, length=20, thickness=3)

    # ── Status text panel (left-ish, inside or near box cavity) ───────────
    panel_x = bx + 12
    panel_y = by + 28

    # BOX status
    if outcome.box_ready:
        _put(frame, "BOX: DETECTED", (panel_x, panel_y), scale=0.52, color=COL_GREEN, thickness=1)
    else:
        _put(frame, "BOX: NOT FOUND", (panel_x, panel_y), scale=0.52, color=COL_RED, thickness=2)
    panel_y += 24

    # NOTE status
    if outcome.note_detected:
        _put(frame, "NOTE: DETECTED", (panel_x, panel_y), scale=0.52, color=COL_GREEN, thickness=2)
    elif outcome.box_ready:
        _put(frame, "NOTE: NOT DETECTED (Empty Box)", (panel_x, panel_y), scale=0.50, color=COL_GRAY, thickness=1)
    panel_y += 24

    # ── Main status message ───────────────────────────────────────────────
    if outcome.decision == "ACCEPT":
        main_col = COL_GREEN
    elif outcome.decision == "REVIEW":
        main_col = COL_AMBER
    elif "Hand" in outcome.user_message or "unclear" in outcome.user_message:
        main_col = COL_AMBER
    elif not outcome.box_ready:
        main_col = COL_RED
    else:
        main_col = COL_WHITE

    # Wrap long messages
    msg = outcome.user_message
    _put(frame, msg, (panel_x, panel_y), scale=0.60, color=main_col, thickness=2)
    panel_y += 30

    # ── Live denomination + confidence (shown as soon as note is detected) ─
    if outcome.note_detected and outcome.predicted_label is not None:
        denom_str = DENOM_LABELS.get(outcome.predicted_label, f"Rs.{outcome.predicted_label}")
        conf_pct  = f"{outcome.confidence * 100:.1f}%"

        if outcome.decision == "ACCEPT":
            pred_col = COL_GREEN
        elif outcome.decision == "REVIEW":
            pred_col = COL_AMBER
        else:
            pred_col = COL_CYAN

        _put(frame, f"{denom_str}  |  {conf_pct}", (panel_x, panel_y),
             scale=0.90, color=pred_col, thickness=2)
        panel_y += 32

        # Confidence bar under the denomination label
        _bar(frame, panel_x, panel_y, min(260, bw - 24), 10,
             outcome.confidence, pred_col)
        panel_y += 20

    # ── Rejection / rejection-reason sub-line ────────────────────────────
    reason = outcome.details.get("reason", "")
    if reason and reason not in ("box_detected", "quality_acceptable",
                                  "banknote_candidate_verified", "duplicate_same_note_suppressed"):
        _put(frame, f"({reason})", (panel_x, panel_y), scale=0.42, color=COL_DARK, thickness=1)
        panel_y += 16

    # ── Debug diagnostics ─────────────────────────────────────────────────
    if debug:
        diag_lines = [
            f"sharpness:{outcome.sharpness:.0f}",
            f"dark_ratio:{outcome.details.get('dark_ratio', '?'):.2f}" if 'dark_ratio' in outcome.details else "",
            f"mean_lum:{outcome.details.get('mean_luminance', '?'):.1f}" if 'mean_luminance' in outcome.details else "",
            f"dark_std:{outcome.details.get('dark_color_std', '?'):.1f}" if 'dark_color_std' in outcome.details else "",
            f"skin:{outcome.details.get('skin_ratio', '?'):.3f}" if 'skin_ratio' in outcome.details else "",
            f"edge:{outcome.details.get('edge_density', '?'):.4f}" if 'edge_density' in outcome.details else "",
        ]
        dy = panel_y
        for line in diag_lines:
            if line:
                _put(frame, line, (panel_x, dy), scale=0.38, color=(100, 200, 100), thickness=1)
                dy += 14

    # ── Class probability panel (right side) ─────────────────────────────
    if outcome.all_probs and outcome.note_detected:
        prob_x = max(w - 200, bx2 + 8)
        prob_x = min(prob_x, w - 195)
        prob_y = by + 20
        bar_w  = 140
        bar_h  = 10

        _put(frame, "Class probabilities", (prob_x, prob_y), scale=0.42, color=COL_GRAY, thickness=1)
        prob_y += 16

        sorted_classes = sorted(class_names)
        for cn in sorted_classes:
            p = outcome.all_probs.get(cn, 0.0)
            label_str = DENOM_LABELS.get(cn, cn)
            if cn == outcome.predicted_label:
                bar_col = COL_GREEN if outcome.decision == "ACCEPT" else COL_CYAN
                txt_col = COL_WHITE
            else:
                bar_col = COL_NEUTRAL
                txt_col = COL_GRAY
            _put(frame, f"{label_str}", (prob_x, prob_y), scale=0.38, color=txt_col, thickness=1)
            prob_y += 12
            _bar(frame, prob_x, prob_y, bar_w, bar_h, p, bar_col)
            _put(frame, f"{p*100:.1f}%", (prob_x + bar_w + 4, prob_y + 8),
                 scale=0.35, color=txt_col, thickness=1)
            prob_y += bar_h + 6

    # ── FPS + Sharpness telemetry ─────────────────────────────────────────
    _put(frame, f"FPS:{fps:.1f}  Sharp:{outcome.sharpness:.0f}",
         (max(10, w - 220), 22), scale=0.48, color=COL_GRAY, thickness=1)

    # ── Title ─────────────────────────────────────────────────────────────
    _put(frame, "DivyaKosh - Smart Donation Box",
         (12, 28), scale=0.70, color=COL_WHITE, thickness=2)

    # ── Help text ─────────────────────────────────────────────────────────
    _put(frame, "Q=quit  R=reset  S=snapshot  C=save ROI", (12, h - 12), scale=0.44, color=COL_DARK, thickness=1)


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def append_log(log_path: Path, camera_index: int, label: str, confidence: float) -> dict:
    """Write an ACCEPT record to JSONL + SQLite. Called only on confirmed donations."""
    record = {
        "pocket_id":              f"webcam-{camera_index}",
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "item_status":            "NOTE_DETECTED",
        "predicted_denomination": label,
        "classifier_confidence":  confidence,
        "confidence":             confidence,
        "source":                 "webcam",
        "decision":               "ACCEPT",
        "target_bin":             f"BIN_{label}",
        "model_version":          MODEL_VERSION,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    if _SQLITE_AVAILABLE:
        try:
            result = import_single_record(record=record, database_file=_SQLITE_DB_FILE)
            print(f"SQLite: {result.inserted} record(s) inserted → {_SQLITE_DB_FILE}")
        except Exception as _db_err:
            print(f"Warning: SQLite write failed ({_db_err}). JSONL record is still safe.")

    return record


def run_yolo_pipeline(cap, yolo_model, model, class_names, device, args,
                      review_tracker, log_path: Path, review_log_path: Path) -> None:
    """Run the YOLO-first one-shot camera flow.

    Behaviour
    ---------
    - Scans for *at most* ``args.timeout`` seconds from the moment scanning begins.
    - As soon as one valid banknote is detected it stops scanning immediately,
      processes the crop exactly once, then returns.
    - If the timeout expires without a detection it exits with a clear message.
    - ``--continuous`` is intentionally ignored here: the YOLO pipeline is always
      one-shot.  Looping would risk duplicate events and is not required for the
      donation-box use-case.
    """
    config = ValidationConfig(
        accept_confidence=args.accept_confidence,
        review_confidence=args.review_confidence,
    )

    # ── Single detection attempt ─────────────────────────────────────────────
    event = run_yolo_detection_window(
        cap=cap,
        yolo_model=yolo_model,
        window_seconds=args.timeout,
        confidence_threshold=args.yolo_confidence,
        window_name=WINDOW_NAME,
        camera_index=args.camera,
        review_tracker=review_tracker,
        review_log_path=review_log_path,
    )

    if event is None:
        # User pressed 'q'
        return

    frame, note_bbox, _yolo_class_id = event

    if note_bbox is None:
        # Timeout expired — no banknote was detected
        print(f"No banknote detected within {args.timeout:.0f} seconds. Camera stopped.")
        if frame is not None:
            disp = frame.copy()
            _put(disp, f"No banknote detected ({args.timeout:.0f}s timeout)", (12, 32), color=COL_AMBER, thickness=2)
            cv2.imshow(WINDOW_NAME, disp)
            cv2.waitKey(1200)
        return

    # ── Exactly-once classification ──────────────────────────────────────────
    x, y, width, height = note_bbox
    crop = frame[y:y + height, x:x + width]
    if crop.size == 0:
        created, review_id = review_tracker.should_create_event("low_confidence", None, 0.0)
        if created and review_id:
            append_review_event(
                review_log_path, args.camera, review_id, "low_confidence",
                None, 0.0, "YOLO_CROP", "The detected banknote crop was empty.",
            )
        return

    display = frame.copy()
    cv2.rectangle(display, (x, y), (x + width, y + height), COL_GREEN, 2)
    try:
        classification = crop_and_classify(
            model=model,
            class_names=class_names,
            frame=frame,
            note_bbox=note_bbox,
            device=device,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"MobileNet inference failed: {exc}") from exc

    label = DENOM_LABELS.get(classification.label, classification.label)
    _put(
        display,
        f"MobileNet: {label} {classification.confidence * 100:.1f}% "
        f"[{classification.decision}]",
        (12, 32),
        color=COL_GREEN if classification.decision == "ACCEPT" else COL_AMBER,
        thickness=2,
    )
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(2000)

    if classification.decision == "ACCEPT":
        review_tracker.reset()
        record = append_log(
            log_path, args.camera, classification.label, classification.confidence
        )
        bc_result = register_blockchain_proof(record)
        if bc_result:
            print(f"Blockchain proof: {bc_result.get('transaction_hash', 'N/A')}")
        else:
            print("Blockchain registration skipped / offline. JSONL record is safe.")
        print(json.dumps(record, indent=2))
    else:
        reason = "low_confidence"
        created, review_id = review_tracker.should_create_event(
            reason, classification.label, classification.confidence
        )
        if created and review_id:
            append_review_event(
                review_log_path, args.camera, review_id, reason,
                classification.label, classification.confidence,
                "MOBILENET_CLASSIFICATION",
                f"MobileNet prediction requires review: {label}.",
            )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DivyaKosh moving-box banknote recognition and verification"
    )
    parser.add_argument("--checkpoint",       type=str,   default=str(_PROJECT_ROOT / "models" / "mobilenet_v3_clean.pt"))
    parser.add_argument("--camera",           type=int,   default=0)
    parser.add_argument("--log_path",         type=str,   default=str(_PROJECT_ROOT / "logs" / "inference.jsonl"))
    parser.add_argument("--review_log",       type=str,   default=str(_PROJECT_ROOT / "logs" / "review_events.jsonl"))
    parser.add_argument("--accept_confidence",type=float, default=0.80,  help="Confidence for ACCEPT (default 0.80)")
    parser.add_argument("--review_confidence",type=float, default=0.45,  help="Confidence for REVIEW (default 0.45)")
    parser.add_argument("--sharpness_thresh", type=float, default=15.0,  help="Note-crop sharpness gate (default 15.0)")
    parser.add_argument("--consensus_frames", type=int,   default=3,     help="Frames for donation confirmation (default 3)")
    parser.add_argument("--continuous",       action="store_true",        help="Don't exit after first acceptance (MobileNet pipeline only; ignored by --yolo_pipeline)")
    parser.add_argument("--debug",            action="store_true",        help="Print per-frame diagnostics to console and UI")
    parser.add_argument("--box_roi",          type=str,   default="",    help="Calibrated box cavity ROI x,y,w,h (e.g. 150,90,720,420)")
    parser.add_argument("--calibrate",        action="store_true",        help="Save box cavity ROI when pressing 'c'")
    parser.add_argument("--yolo_pipeline",    action="store_true",        help="Use YOLO-first one-shot banknote detection (stops after first note; respects --timeout)")
    parser.add_argument("--yolo_checkpoint",  type=str, default=str(_PROJECT_ROOT / "models" / "yolo26n_robust" / "train_seed42_20_20261101_203031" / "weights" / "best.pt"))
    parser.add_argument("--yolo_window_seconds", type=float, default=None, help="Deprecated alias for --timeout (use --timeout instead)")
    parser.add_argument("--yolo_confidence", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--timeout",          type=float, default=7.0,  help="Inactivity timeout in seconds before camera stops (YOLO pipeline, default 7)")
    args = parser.parse_args()

    # Resolve effective scan timeout:
    # --yolo_window_seconds is a deprecated alias; --timeout takes precedence.
    # If a caller only passed the old flag, honour it.
    if args.yolo_window_seconds is not None and args.timeout == 7.0:
        # Old flag supplied, new flag at default → use old value for back-compat
        args.timeout = args.yolo_window_seconds

    # Device — prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, class_names = load_model(args.checkpoint, device)
    yolo_model = load_yolo_model(args.yolo_checkpoint) if args.yolo_pipeline else None

    # Camera
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        # Explicit camera-open failure — do NOT treat as a timeout
        print(f"ERROR: Could not open camera index {args.camera}. "
              "Check that the camera is connected and not in use by another application. "
              "Try --camera 1 for an external USB camera.")
        return

    log_path = Path(args.log_path)
    review_log_path = Path(args.review_log)

    # Load calibrated ROI if available
    calibrated_roi = None
    if args.box_roi:
        try:
            parts = [int(p.strip()) for p in args.box_roi.split(",")]
            if len(parts) == 4:
                calibrated_roi = tuple(parts)
                print(f"Calibrated box cavity ROI from CLI: {calibrated_roi}")
        except Exception as e:
            print(f"Warning: Failed to parse --box_roi ({e})")
    else:
        roi_file = _PROJECT_ROOT / "box_roi.json"
        if roi_file.exists():
            try:
                with roi_file.open("r") as rf:
                    rdata = json.load(rf)
                    broi = rdata.get("box_roi", [])
                    if len(broi) == 4:
                        calibrated_roi = tuple(broi)
                        print(f"Loaded calibrated box cavity ROI from {roi_file}: {calibrated_roi}")
            except Exception as e:
                print(f"Notice: Failed to read {roi_file}: {e}")

    # Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, WINDOW_POS_X, WINDOW_POS_Y)

    print()
    print("=" * 54)
    print("DivyaKosh Banknote Recognition System Online")
    print(f"  Device                 : {device}")
    print(f"  Confidence threshold   : {args.accept_confidence:.2f}")
    print(f"  Review threshold       : {args.review_confidence:.2f}")
    print(f"  Sharpness threshold    : {args.sharpness_thresh:.1f}")
    print(f"  Consensus frames       : {args.consensus_frames}")
    print(f"  Continuous mode        : {args.continuous}")
    print(f"  Debug mode             : {args.debug}")
    print(f"  Review log             : {review_log_path}")
    if calibrated_roi:
        print(f"  Box Cavity ROI         : {calibrated_roi}")
    if args.yolo_pipeline:
        print(f"  YOLO model version     : {YOLO_MODEL_VERSION}")
        print(f"  YOLO checkpoint        : {args.yolo_checkpoint}")
        print(f"  Scan timeout           : {args.timeout:.1f}s (one-shot, stops on first note)")
        print(f"  YOLO conf threshold    : {args.yolo_confidence:.2f}")
    print("=" * 54)
    print()

    # Pipeline config & tracker
    config = ValidationConfig(
        box_roi=calibrated_roi,
        accept_confidence=args.accept_confidence,
        review_confidence=args.review_confidence,
        min_sharpness=args.sharpness_thresh,
        consensus_frames=args.consensus_frames,
    )
    review_tracker = ReviewTracker(
        cooldown_frames=8,
        db_file=REVIEW_DB_FILE,
        pocket_id=f"webcam-{args.camera}",
    )
    tracker = BanknoteTracker(config=config, review_tracker=review_tracker)

    prev_gray = None
    prev_time = time.perf_counter()
    fps       = 0.0

    try:
        if args.yolo_pipeline:
            print(f"YOLO-first mode: one-shot, {args.timeout:.1f}s inactivity timeout")
            run_yolo_pipeline(
                cap=cap,
                yolo_model=yolo_model,
                model=model,
                class_names=class_names,
                device=device,
                args=args,
                review_tracker=review_tracker,
                log_path=log_path,
                review_log_path=review_log_path,
            )
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Camera disconnected?")
                break

            fh, fw = frame.shape[:2]

            # FPS
            now = time.perf_counter()
            dt  = now - prev_time
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if (fps > 0 and dt > 0) else (1.0 / max(dt, 1e-6))
            prev_time = now

            # Run full 5-stage pipeline
            outcome = tracker.process_frame(
                frame=frame,
                model=model,
                class_names=class_names,
                previous_gray=prev_gray,
                device=device,
            )

            # Update previous grayscale for motion estimation
            # Use center strip of frame to avoid bright edges
            cx1, cx2 = int(fw * 0.15), int(fw * 0.85)
            cy1, cy2 = int(fh * 0.15), int(fh * 0.85)
            prev_gray = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)

            # Determine display ROI (box search area; fall back to centre if unknown)
            search_bbox = outcome.details.get("center_roi", None)
            if search_bbox is None:
                margin_x = int(fw * 0.15)
                margin_y = int(fh * 0.15)
                search_bbox = (margin_x, margin_y, fw - 2 * margin_x, fh - 2 * margin_y)
            bx, by, bw, bh = search_bbox

            # Console debug
            if args.debug:
                det_info = (
                    f"[{outcome.stage:11s}] box={outcome.box_ready} "
                    f"note={outcome.note_detected} "
                    f"pred={outcome.predicted_label} "
                    f"conf={outcome.confidence:.3f} "
                    f"dec={outcome.decision} "
                    f"| {outcome.details.get('reason','')}"
                )
                print(det_info)

            # Draw UI overlay
            draw_overlay(
                frame=frame,
                outcome=outcome,
                fps=fps,
                class_names=class_names,
                search_bbox=(bx, by, bw, bh),
                debug=args.debug,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # ── Review Event Logging (unknown object / non-banknote) ───────
            if outcome.is_review_event and outcome.review_id:
                review_record = {
                    "event_type": "REVIEW",
                    "review_id": outcome.review_id,
                    "pocket_id": f"webcam-{args.camera}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": outcome.review_reason,
                    "candidate_label": outcome.predicted_label,
                    "classifier_confidence": outcome.confidence,
                    "stage": outcome.stage,
                    "user_message": outcome.user_message,
                }
                review_log_path.parent.mkdir(parents=True, exist_ok=True)
                with review_log_path.open("a", encoding="utf-8") as rf:
                    rf.write(json.dumps(review_record, separators=(",", ":")) + "\n")
                print(f"[REVIEW EVENT CREATED] {outcome.review_id} | reason: {outcome.review_reason} | {outcome.user_message}")

            # ── Donation acceptance event ──────────────────────────────────
            if outcome.decision == "ACCEPT" and tracker.accepted_event_ready:
                print()
                print("=" * 54)
                print(f"DONATION CONFIRMED: Rs.{outcome.predicted_label}  ({outcome.confidence * 100:.1f}%)")
                print("=" * 54)

                record = append_log(log_path, args.camera, outcome.predicted_label, outcome.confidence)

                bc_result = register_blockchain_proof(record)
                if bc_result:
                    print("\n=== BLOCKCHAIN PROOF ANCHORED ===")
                    print(f"  Proof hash  : {bc_result.get('proof_hash','N/A')}")
                    print(f"  Transaction : {bc_result.get('transaction_hash','N/A')}")
                    print(f"  Block       : {bc_result.get('block_number','N/A')}")
                else:
                    print("\nBlockchain registration skipped / offline. JSONL record is safe.")

                print("\n=== RECORD SAVED ===")
                print(json.dumps(record, indent=2))
                print()

                cv2.waitKey(1500)  # Pause to show ACCEPTED frame

                if not args.continuous:
                    print("Single-detection mode complete. Exiting.")
                    break

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Manual reset — release deduplication lock
                tracker.is_locked = False
                tracker.last_accepted_signature = None
                tracker.empty_box_streak = 0
                tracker.consensus_streak = 0
                tracker.current_candidate_label = None
                tracker.state = "WAITING_FOR_BOX"
                review_tracker.reset()
                print("[R] Tracker manually reset.")
            elif key == ord("s"):
                # Save diagnostic snapshot
                debug_dir = _PROJECT_ROOT / "logs" / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_path = debug_dir / f"frame_{ts}.jpg"
                cv2.imwrite(str(frame_path), frame)
                snap_info = {
                    "timestamp": ts,
                    "stage": outcome.stage,
                    "box_ready": outcome.box_ready,
                    "note_detected": outcome.note_detected,
                    "note_bbox": outcome.note_bbox,
                    "pred": outcome.predicted_label,
                    "conf": outcome.confidence,
                    "all_probs": outcome.all_probs,
                    "decision": outcome.decision,
                    "details": outcome.details,
                    "search_bbox": search_bbox,
                }
                with (debug_dir / f"debug_{ts}.json").open("w", encoding="utf-8") as jf:
                    json.dump(snap_info, jf, indent=2)
                print(f"[SNAPSHOT SAVED] {frame_path} and debug_{ts}.json")
            elif key == ord("c"):
                # Save cavity coordinates to box_roi.json
                roi_file = _PROJECT_ROOT / "box_roi.json"
                with roi_file.open("w", encoding="utf-8") as rf:
                    json.dump({"box_roi": list(search_bbox)}, rf, indent=2)
                print(f"[CALIBRATION SAVED] {roi_file} -> {search_bbox}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
