"""
CV_model/src/validator.py

Multi-Stage Computer-Vision Validation Pipeline for Banknote Recognition
in Moving Donation Box & Rotating Pocket Mechanisms.

Stages:
  Stage A: Donation-box presence detection (adaptive, does NOT require full-frame darkness)
  Stage B: Movement, sharpness & image quality (Laplacian variance, motion magnitude)
  Stage C: Banknote detection before classification (geometry, aspect ratio, skin filter, texture)
  Stage D: Dynamic banknote cropping & preprocessing
  Stage E: Multi-frame consensus, ACCEPT/REVIEW decisions & deduplication

Key design principles:
  - Stage A works on the CENTER ROI of the frame, not the global image mean.
    The box is always roughly centred; bright room edges/hands must not kill detection.
  - Stage A is tolerant of bright-note-in-dark-box scenes (bimodal luminance bypass).
  - Live predictions are returned immediately once a candidate is found.
    Consensus (Stage E) is only used to CONFIRM a donation, not to SUPPRESS display.
  - All thresholds are configurable and documented.

Preserves:
  - MobileNetV3-Large checkpoint (loaded once, read-only)
  - Class mapping: ['10', '100', '20', '200', '50', '500']
  - Exact JSONL and SQLite record schemas
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# ============================================================
# SUPPORTED ALLOWLIST
# ============================================================
SUPPORTED_DENOMINATIONS = ("10", "20", "50", "100", "200", "500")

DENOM_LABELS = {
    "10": "Rs.10",
    "20": "Rs.20",
    "50": "Rs.50",
    "100": "Rs.100",
    "200": "Rs.200",
    "500": "Rs.500",
}

# ImageNet normalization matching training
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ValidationConfig:
    """Configurable thresholds for all validation stages."""

    # Stage A: Box Readiness
    box_roi: Optional[Tuple[int, int, int, int]] = None  # Explicit (x, y, w, h) ROI of box cavity if calibrated
    box_center_roi_fraction: float = 0.70  # Central fraction of frame to analyse (0.70 = inner 70%)
    box_min_center_lum: float = 2.0        # Reject pitch-black / lens cap (nearly zero)
    box_max_center_lum: float = 220.0      # Reject blinding white (no box at all); very permissive
    box_min_dark_ratio: float = 0.15       # Min fraction of center ROI that is dark (≤80 lum); lowered for lit-note scenes
    box_min_dark_area_px: int = 2000       # Min absolute dark pixel count; lowered for smaller boxes / cropped views
    box_max_color_std_dark: float = 65.0   # Max color std-dev on dark pixels; raised to tolerate box walls + lit note interior

    # Stage B: Movement & Sharpness (applied to detected note crop, not whole frame)
    min_sharpness: float = 15.0            # Laplacian variance on note crop (lenient for real cameras)
    max_motion_displacement: float = 35.0  # Max inter-frame motion mean-diff
    fov_margin: int = 4                    # Min pixels from frame border for non-truncation check

    # Stage C: Banknote Detection
    min_note_area_ratio: float = 0.015     # Min note area relative to search ROI (1.5%)
    min_note_area_px: int = 1000           # Min absolute pixel area for note candidate
    min_note_w: int = 35                   # Min candidate width in px
    min_note_h: int = 22                   # Min candidate height in px
    max_note_area_ratio: float = 0.85      # Max note area (rejects full-frame occlusion)
    min_aspect_ratio: float = 1.15         # Indian banknotes: ~1.8-2.3 flat; ≥1.15 for partial/tilted/angled
    max_aspect_ratio: float = 4.50         # Wider range for tilted/folded notes
    max_skin_ratio: float = 0.60           # Max fraction of skin-tone pixels
    min_edge_density: float = 0.008        # Min Canny edge density (engravings/text vs smooth paper)
    min_foreground_contrast: float = 18.0  # Luminance delta from box floor (foreground lift)
    foreground_thresh_delta: float = 18.0  # How far above median to threshold foreground

    # Stage E: Decision & Consensus
    accept_confidence: float = 0.80        # Threshold for ACCEPT (donation confirmed)
    review_confidence: float = 0.45        # Threshold for REVIEW (below this is REJECT)
    consensus_frames: int = 3              # Consecutive frames agreeing on denomination for acceptance
    cooldown_frames: int = 12              # Cooldown after an ACCEPT before new detection
    empty_reset_frames: int = 4            # Consecutive empty frames to reset deduplication lock


# ============================================================
# DATA TRANSFER OBJECTS
# ============================================================
@dataclass
class BoxReadinessResult:
    is_ready: bool
    box_bbox: Tuple[int, int, int, int]  # (x, y, w, h) — the search ROI
    mean_luminance: float                # of center ROI
    dark_ratio: float
    reason: str
    user_message: str
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityResult:
    is_clear: bool
    sharpness: float
    motion_score: float
    is_truncated: bool
    reason: str
    user_message: str


@dataclass
class BanknoteDetectionResult:
    is_detected: bool
    note_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) in frame coordinates
    contour: Optional[np.ndarray] = None
    aspect_ratio: float = 0.0
    area_ratio: float = 0.0
    skin_ratio: float = 0.0
    edge_density: float = 0.0
    sharpness: float = 0.0
    reason: str = ""
    user_message: str = ""


@dataclass
class ClassificationResult:
    """Result of running the classifier on a crop."""
    label: str
    confidence: float
    decision: str        # ACCEPT / REVIEW / REJECT
    target_bin: str
    all_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationOutcome:
    """Full pipeline validation outcome for a single frame."""
    stage: str           # BOX / QUALITY / DETECTION / CLASSIFICATION / CONSENSUS
    passed: bool
    box_ready: bool
    quality_ok: bool
    note_detected: bool
    note_bbox: Optional[Tuple[int, int, int, int]]
    sharpness: float
    predicted_label: Optional[str]
    confidence: float
    all_probs: Dict[str, float]
    decision: str        # ACCEPT / REVIEW / REJECT / WAITING
    user_message: str
    details: Dict[str, Any] = field(default_factory=dict)
    review_id: Optional[str] = None
    review_reason: str = ""
    is_review_event: bool = False


# ============================================================
# STAGE A: DONATION BOX READINESS
# ============================================================
def validate_box_readiness(
    frame: np.ndarray,
    config: ValidationConfig = ValidationConfig(),
) -> BoxReadinessResult:
    """
    Determine whether a donation-box interior is visible.

    IMPORTANT DESIGN CHANGE from original implementation:
    - Measurements are taken on the CENTER ROI of the frame (configurable fraction),
      NOT the global mean. Bright hands, desk edges, and room light at the frame
      periphery must not cause rejection when the dark box is centred.
    - box_max_luminance is very permissive (200) since we rely on dark_ratio instead.
    - Color variance check uses a raised threshold (45) to tolerate cardboard texture
      and shadow gradients typical of real-world box interiors.

    Rejects only genuinely bad scenes:
      - Pitch-black / lens cap (near-zero luminance everywhere in ROI)
      - Almost no dark pixels in the centre (box simply not there)
    """
    if frame is None or frame.size == 0:
        return BoxReadinessResult(
            is_ready=False,
            box_bbox=(0, 0, 0, 0),
            mean_luminance=0.0,
            dark_ratio=0.0,
            reason="empty_frame",
            user_message="Camera error: Empty frame received.",
        )

    fh, fw = frame.shape[:2]

    # Use explicit box cavity ROI if configured, otherwise compute center ROI
    if config.box_roi is not None:
        bx, by, bw, bh = config.box_roi
        rx1 = max(0, min(bx, fw - 1))
        ry1 = max(0, min(by, fh - 1))
        rx2 = min(fw, rx1 + bw)
        ry2 = min(fh, ry1 + bh)
        rw = max(1, rx2 - rx1)
        rh = max(1, ry2 - ry1)
    else:
        # Compute center ROI (the inner fraction of the frame)
        margin_x = int(fw * (1.0 - config.box_center_roi_fraction) / 2.0)
        margin_y = int(fh * (1.0 - config.box_center_roi_fraction) / 2.0)
        rx1 = max(0, margin_x)
        ry1 = max(0, margin_y)
        rx2 = min(fw, fw - margin_x)
        ry2 = min(fh, fh - margin_y)
        rw = max(1, rx2 - rx1)
        rh = max(1, ry2 - ry1)

    roi_gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
    roi_frame = frame[ry1:ry2, rx1:rx2]
    mean_lum = float(np.mean(roi_gray))

    debug = {
        "center_roi": (rx1, ry1, rw, rh),
        "mean_lum": mean_lum,
    }

    # Reject pitch-black (camera covered or power failure)
    if mean_lum < config.box_min_center_lum:
        return BoxReadinessResult(
            is_ready=False,
            box_bbox=(rx1, ry1, rw, rh),
            mean_luminance=mean_lum,
            dark_ratio=1.0,
            reason="ambient_darkness_no_box",
            user_message="Please position the donation box in front of the camera.",
            debug=debug,
        )

    # Reject blinding-white scene (e.g. camera aimed at the sky or white wall)
    if mean_lum > config.box_max_center_lum:
        return BoxReadinessResult(
            is_ready=False,
            box_bbox=(rx1, ry1, rw, rh),
            mean_luminance=mean_lum,
            dark_ratio=0.0,
            reason="scene_too_bright_no_box",
            user_message="Please position the donation box in front of the camera.",
            debug=debug,
        )

    # Measure dark interior coverage (pixels ≤ 80 luminance)
    dark_mask = (roi_gray <= 80).astype(np.uint8) * 255
    dark_count = int(cv2.countNonZero(dark_mask))
    dark_ratio = dark_count / float(rw * rh + 1e-5)
    debug["dark_ratio"] = dark_ratio
    debug["dark_count"] = dark_count

    # Must have a meaningful dark region (not just a tiny shadow)
    if dark_ratio < config.box_min_dark_ratio or dark_count < config.box_min_dark_area_px:
        return BoxReadinessResult(
            is_ready=False,
            box_bbox=(rx1, ry1, rw, rh),
            mean_luminance=mean_lum,
            dark_ratio=dark_ratio,
            reason="insufficient_dark_interior",
            user_message="Please position the donation box in front of the camera.",
            debug=debug,
        )

    # Color variance on dark pixels (tolerant threshold for real-world cardboard/shadows).
    # IMPORTANT: A lit banknote inside the dark box creates a bimodal luminance distribution:
    #   - very dark pixels (box walls) → low std
    #   - The dark_mask already excludes the bright note (>80 lum), so dark_pixels
    #     should be mostly uniform box-floor pixels. High std here means the "dark"
    #     zone itself is wildly varied (e.g. a cluttered room without a box).
    # We only apply this check when mean_lum is low enough that a bright note
    # is NOT dominating. When mean_lum > 60 AND dark_ratio > 0.25, the scene is
    # very likely "box with note" — skip the color_std rejection.
    dark_pixels = roi_frame[dark_mask == 255]
    color_std = float(np.std(dark_pixels)) if dark_pixels.size > 0 else 0.0
    debug["dark_color_std"] = color_std

    # Only reject on color_std when scene does NOT look like a lit-note-inside-dark-box.
    # Condition for color_std check: mean luminance is low (dim scene, no bright note).
    # When mean_lum > 55 and dark_ratio > 0.20, the bright note is lifting the average —
    # this is the exact valid "box with note" scenario; skip color_std rejection.
    note_in_box_bypass = (mean_lum > 55.0 and dark_ratio > 0.20)
    if not note_in_box_bypass and color_std > config.box_max_color_std_dark:
        return BoxReadinessResult(
            is_ready=False,
            box_bbox=(rx1, ry1, rw, rh),
            mean_luminance=mean_lum,
            dark_ratio=dark_ratio,
            reason="scene_not_box_interior",
            user_message="Please position the donation box in front of the camera.",
            debug=debug,
        )

    # Box detected — return the center ROI as the search area
    return BoxReadinessResult(
        is_ready=True,
        box_bbox=(rx1, ry1, rw, rh),
        mean_luminance=mean_lum,
        dark_ratio=dark_ratio,
        reason="box_detected",
        user_message="Donation box ready.",
        debug=debug,
    )


# ============================================================
# STAGE B: MOVEMENT & IMAGE QUALITY
# ============================================================
def assess_movement_and_quality(
    current_frame: np.ndarray,
    previous_gray: Optional[np.ndarray],
    roi_bbox: Tuple[int, int, int, int],
    config: ValidationConfig = ValidationConfig(),
) -> QualityResult:
    """
    Assess inter-frame motion and blur.
    Sharpness is measured on the ROI (usually the note crop).
    An empty smooth box floor naturally has low sharpness — this is NOT a fault.
    The sharpness gate is only applied when a textured object is present (roi_std >= 8).
    """
    h, w = current_frame.shape[:2]
    x, y, bw, bh = roi_bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    curr_roi = current_frame[y:y + bh, x:x + bw]
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(curr_gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Inter-frame motion
    motion_score = 0.0
    if previous_gray is not None and previous_gray.shape == curr_gray.shape:
        diff = cv2.absdiff(curr_gray, previous_gray)
        motion_score = float(np.mean(diff))

    # Truncation check (only for localized objects, not full-camera ROI)
    is_full_frame = (bw >= int(w * 0.85) and bh >= int(h * 0.85))
    is_truncated = False
    if not is_full_frame:
        is_truncated = (
            x <= config.fov_margin or
            y <= config.fov_margin or
            (x + bw) >= (w - config.fov_margin) or
            (y + bh) >= (h - config.fov_margin)
        )

    if is_truncated:
        return QualityResult(
            is_clear=False, sharpness=sharpness, motion_score=motion_score,
            is_truncated=True, reason="truncated_at_boundary",
            user_message="Please reposition the donation box.",
        )

    if motion_score > config.max_motion_displacement:
        return QualityResult(
            is_clear=False, sharpness=sharpness, motion_score=motion_score,
            is_truncated=is_truncated, reason="excessive_motion",
            user_message="Hold the box steady.",
        )

    # Sharpness check only when texture/object is present (std >= 8)
    roi_std = float(np.std(curr_gray))
    if roi_std >= 8.0 and sharpness < config.min_sharpness:
        return QualityResult(
            is_clear=False, sharpness=sharpness, motion_score=motion_score,
            is_truncated=is_truncated, reason="motion_blurred_or_defocused",
            user_message="Image unclear. Please wait.",
        )

    return QualityResult(
        is_clear=True, sharpness=sharpness, motion_score=motion_score,
        is_truncated=False, reason="quality_acceptable",
        user_message="Image clear.",
    )


# ============================================================
# STAGE C: BANKNOTE DETECTION
# ============================================================
def detect_banknote(
    frame: np.ndarray,
    box_bbox: Tuple[int, int, int, int],
    config: ValidationConfig = ValidationConfig(),
) -> BanknoteDetectionResult:
    """
    Detect whether a banknote candidate is present inside the box search ROI.

    Uses adaptive foreground thresholding relative to the ROI median,
    so it works regardless of absolute scene brightness.

    Filter order:
      1. Area (size gate — removes dust specks and full-frame floods)
      2. Aspect ratio (geometry — rejects round objects)
      3. Skin filter (YCrCb — rejects hands if dominant)
      4. Edge density (Canny — rejects blank paper)
      5. Sharpness (Laplacian — rejects heavily blurred crops; skipped for uniform surfaces)

    Returns the BEST candidate (highest area that passes all filters).
    """
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = box_bbox
    bx = max(0, min(bx, fw - 1))
    by = max(0, min(by, fh - 1))
    bw = max(1, min(bw, fw - bx))
    bh = max(1, min(bh, fh - by))

    box_crop = frame[by:by + bh, bx:bx + bw]
    box_gray = cv2.cvtColor(box_crop, cv2.COLOR_BGR2GRAY)
    box_area = float(bw * bh)

    # Adaptive threshold: foreground = pixels significantly brighter than the dark floor
    floor_median = float(np.median(box_gray))
    thresh_val = min(235, max(28, int(floor_median + config.foreground_thresh_delta)))
    _, fg_mask = cv2.threshold(box_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleaning: close gaps in banknote printing / folds, remove tiny speckle noise
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BanknoteDetectionResult(
            is_detected=False, note_bbox=None,
            reason="empty_box", user_message="No banknote detected. Please insert a note.",
        )

    # Filter out tiny noise contours before clustering
    valid_raw = [c for c in contours if cv2.contourArea(c) >= 100]
    if not valid_raw:
        return BanknoteDetectionResult(
            is_detected=False, note_bbox=None,
            reason="empty_box", user_message="No banknote detected. Please insert a note.",
        )

    # Multi-component clustering: Group nearby disjoint contours (within 35px) of same banknote
    raw_boxes = []
    for c in valid_raw:
        bx_c, by_c, bw_c, bh_c = cv2.boundingRect(c)
        raw_boxes.append([bx_c, by_c, bw_c, bh_c, [c]])

    merged_candidates = []
    while raw_boxes:
        curr = raw_boxes.pop(0)
        cx, cy, cw, ch, c_cnts = curr
        i = 0
        while i < len(raw_boxes):
            ox, oy, ow, oh, o_cnts = raw_boxes[i]
            # Bounding box distance
            dx = max(0, max(cx, ox) - min(cx + cw, ox + ow))
            dy = max(0, max(cy, oy) - min(cy + ch, oy + oh))
            if dx <= 35 and dy <= 35:
                # Merge into unified bounding rectangle
                nx1 = min(cx, ox)
                ny1 = min(cy, oy)
                nx2 = max(cx + cw, ox + ow)
                ny2 = max(cy + ch, oy + oh)
                cx, cy, cw, ch = nx1, ny1, nx2 - nx1, ny2 - ny1
                c_cnts.extend(o_cnts)
                raw_boxes.pop(i)
                i = 0  # Re-scan remaining with updated unified box
            else:
                i += 1
        merged_candidates.append((cx, cy, cw, ch, c_cnts))

    # Sort merged candidates by total contour area (largest first)
    def _cand_total_area(cand_tuple):
        return sum(cv2.contourArea(c) for c in cand_tuple[4])

    candidates = sorted(merged_candidates, key=_cand_total_area, reverse=True)

    last_reason = "empty_box"
    last_msg = "No banknote detected. Please insert a note."

    for cx, cy, cw, ch, c_cnts in candidates[:4]:
        c_area = sum(float(cv2.contourArea(c)) for c in c_cnts)
        area_ratio = c_area / (box_area + 1e-5)

        # Dimension and area gates
        if cw < config.min_note_w or ch < config.min_note_h or c_area < config.min_note_area_px:
            last_reason = "object_too_small"
            last_msg = "Please adjust the note."
            continue

        if area_ratio > config.max_note_area_ratio:
            last_reason = "object_too_large_or_full_occlusion"
            last_msg = "Please adjust the note."
            continue

        aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            last_reason = "abnormal_aspect_ratio"
            last_msg = "No banknote detected. Please insert a note."
            continue

        # Upright bounding box in frame coordinates
        global_x = max(0, bx + cx)
        global_y = max(0, by + cy)
        global_w = min(fw - global_x, cw)
        global_h = min(fh - global_y, ch)
        note_bbox = (global_x, global_y, global_w, global_h)

        if global_w <= 0 or global_h <= 0:
            continue

        note_roi_bgr = frame[global_y:global_y + global_h, global_x:global_x + global_w]
        if note_roi_bgr.size == 0:
            continue

        # Canny edge density (engravings/text vs smooth paper/skin)
        note_gray = cv2.cvtColor(note_roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(note_gray, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / float(global_w * global_h + 1e-5)

        # Skin filter (YCrCb)
        ycrcb = cv2.cvtColor(note_roi_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(global_w * global_h + 1e-5)

        # Banknote vs Hand discrimination:
        # Banknotes (especially ₹10, ₹20, ₹200) have orange/brown ink that triggers YCrCb skin mask,
        # but they have dense fine printing (edge_density >= 0.022).
        # Smooth skin (bare hand, arm) has low edge density (< 0.022).
        if skin_ratio > config.max_skin_ratio and edge_density < 0.022:
            last_reason = "hand_or_skin_detected"
            last_msg = "Hand detected. Please remove hand."
            continue

        # Blank paper / plain surface rejection
        if edge_density < config.min_edge_density:
            last_reason = "blank_paper_or_plain_object"
            last_msg = "No banknote detected. Please insert a note."
            continue

        # Sharpness (only gate when surface has texture — not a plain blank object)
        note_std = float(np.std(note_gray))
        note_sharpness = float(cv2.Laplacian(note_gray, cv2.CV_64F).var())
        if note_std >= 5.0 and note_sharpness < config.min_sharpness:
            last_reason = "motion_blurred_or_defocused"
            last_msg = "Image unclear. Please wait."
            continue

        # All filters passed — this is a valid banknote candidate
        primary_cnt = c_cnts[0] if len(c_cnts) == 1 else np.concatenate([c.reshape(-1, 2) for c in c_cnts])
        return BanknoteDetectionResult(
            is_detected=True,
            note_bbox=note_bbox,
            contour=primary_cnt,
            aspect_ratio=aspect_ratio,
            area_ratio=area_ratio,
            skin_ratio=skin_ratio,
            edge_density=edge_density,
            sharpness=note_sharpness,
            reason="banknote_candidate_verified",
            user_message="Banknote detected.",
        )

    # No candidate passed
    return BanknoteDetectionResult(
        is_detected=False, note_bbox=None,
        reason=last_reason, user_message=last_msg,
    )


# ============================================================
# STAGE D: CROP AND CLASSIFY
# ============================================================
def crop_and_classify(
    model: torch.nn.Module,
    class_names: List[str],
    frame: np.ndarray,
    note_bbox: Tuple[int, int, int, int],
    device: torch.device,
    config: ValidationConfig = ValidationConfig(),
) -> ClassificationResult:
    """
    Crop detected banknote region, apply ImageNet normalization, run inference.
    Returns ClassificationResult with label, confidence, decision, and all class probs.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = note_bbox

    # 5% padding
    pad_w = max(4, int(w * 0.05))
    pad_h = max(4, int(h * 0.05))
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(fw, x + w + pad_w)
    y2 = min(fh, y + h + pad_h)

    crop_bgr = frame[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        crop_bgr = frame[y:y + h, x:x + w]
    if crop_bgr.size == 0:
        return ClassificationResult(
            label="unknown", confidence=0.0, decision="REJECT",
            target_bin="BIN_UNKNOWN", all_probs={cn: 0.0 for cn in class_names},
        )

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Banknotes are horizontal rectangular documents.
    # If the detected crop is vertical (taller than wide), orient it horizontally to match training distribution.
    if crop_rgb.shape[0] > crop_rgb.shape[1]:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE)

    pil_img = Image.fromarray(crop_rgb)
    tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probs, dim=0)

    conf_float = float(confidence.item())
    pred_label = class_names[int(index.item())]
    all_probs = {cn: float(p.item()) for cn, p in zip(class_names, probs)}

    # Decision
    if pred_label not in SUPPORTED_DENOMINATIONS:
        decision = "REJECT"
        target_bin = "BIN_UNKNOWN"
    elif conf_float >= config.accept_confidence:
        decision = "ACCEPT"
        target_bin = f"BIN_{pred_label}"
    elif conf_float >= config.review_confidence:
        decision = "REVIEW"
        target_bin = f"BIN_{pred_label}"
    else:
        decision = "REJECT"
        target_bin = f"BIN_{pred_label}"

    return ClassificationResult(
        label=pred_label,
        confidence=conf_float,
        decision=decision,
        target_bin=target_bin,
        all_probs=all_probs,
    )


# ============================================================
# STAGE E: DEDUPLICATION & MULTI-FRAME CONSENSUS TRACKER
# ============================================================
class BanknoteTracker:
    """
    State machine that handles:
      1. Consecutive-frame consensus (prevents noisy single-frame accept)
      2. Deduplication of the same physical note across frames
      3. Resetting once the note is discharged or box becomes empty
      4. Live prediction passthrough (does NOT suppress classification display)
      5. Unknown-object REVIEW tracking & deduplication via ReviewTracker

    NOTE: Live predictions are always returned even before consensus is reached.
    Only DONATION CONFIRMATION requires consensus_frames.
    """

    def __init__(
        self,
        config: ValidationConfig = ValidationConfig(),
        review_tracker: Optional[Any] = None,
    ):
        self.config = config
        self.review_tracker = review_tracker
        self.state = "WAITING_FOR_BOX"

        # Consensus tracking
        self.current_candidate_label: Optional[str] = None
        self.consensus_streak: int = 0
        self.confidence_accumulator: List[float] = []
        self.last_candidate_bbox: Optional[Tuple[int, int, int, int]] = None

        # Deduplication
        self.last_accepted_signature: Optional[str] = None
        self.empty_box_streak: int = 0
        self.is_locked: bool = False
        self.accepted_event_ready: bool = False
        self.latest_accepted_record: Optional[Dict[str, Any]] = None

    def _compute_note_signature(self, label: str, note_crop: np.ndarray) -> str:
        if note_crop is None or note_crop.size == 0:
            return f"sig_{label}_empty"
        hsv = cv2.cvtColor(note_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_sig = "_".join(f"{val:.2f}" for val in hist.flatten()[:8])
        return f"note_{label}_{hist_sig}"

    def process_frame(
        self,
        frame: np.ndarray,
        model: torch.nn.Module,
        class_names: List[str],
        previous_gray: Optional[np.ndarray],
        device: torch.device,
    ) -> "ValidationOutcome":
        """
        Process one frame through the complete 5-stage validation pipeline.

        Live prediction is always returned as soon as a banknote is detected
        (even before consensus is reached). Consensus only gates ACCEPT events.
        """
        self.accepted_event_ready = False

        # ── Stage A: Box Readiness ────────────────────────────────────────────
        box_res = validate_box_readiness(frame, self.config)
        if not box_res.is_ready:
            self.state = "WAITING_FOR_BOX"
            self.consensus_streak = 0
            self.current_candidate_label = None
            return ValidationOutcome(
                stage="BOX",
                passed=False,
                box_ready=False,
                quality_ok=False,
                note_detected=False,
                note_bbox=None,
                sharpness=0.0,
                predicted_label=None,
                confidence=0.0,
                all_probs={},
                decision="WAITING",
                user_message=box_res.user_message,
                details={"reason": box_res.reason, "mean_luminance": box_res.mean_luminance,
                         "dark_ratio": box_res.dark_ratio, **box_res.debug},
            )

        # ── Stage B: Movement & Quality (on box ROI) ─────────────────────────
        qual_res = assess_movement_and_quality(
            frame, previous_gray, box_res.box_bbox, self.config
        )
        if not qual_res.is_clear:
            self.state = "UNCLEAR_OR_MOVING"
            # Do NOT reset consensus — let a single blurry frame pass through
            return ValidationOutcome(
                stage="QUALITY",
                passed=False,
                box_ready=True,
                quality_ok=False,
                note_detected=False,
                note_bbox=None,
                sharpness=qual_res.sharpness,
                predicted_label=None,
                confidence=0.0,
                all_probs={},
                decision="WAITING",
                user_message=qual_res.user_message,
                details={"reason": qual_res.reason, "sharpness": qual_res.sharpness,
                         "motion_score": qual_res.motion_score},
            )

        # ── Stage C: Banknote Detection ───────────────────────────────────────
        det_res = detect_banknote(frame, box_res.box_bbox, self.config)
        if not det_res.is_detected:
            review_id = None
            is_review_event = False
            review_reason = ""

            if det_res.reason in ("empty_box", "object_too_small"):
                self.empty_box_streak += 1
                if self.empty_box_streak >= self.config.empty_reset_frames:
                    self.is_locked = False
                    self.last_accepted_signature = None
                    self.state = "EMPTY_BOX_READY"
                if self.review_tracker is not None:
                    self.review_tracker.reset()
                decision = "WAITING"
                msg = det_res.user_message
            else:
                self.empty_box_streak = 0
                review_reason = det_res.reason
                if self.review_tracker is not None:
                    is_review_event, review_id = self.review_tracker.should_create_event(
                        reason=det_res.reason,
                        candidate_label=None,
                        confidence=0.0,
                    )
                decision = "REVIEW" if (review_id or self.review_tracker) else "WAITING"

                if det_res.reason == "hand_or_skin_detected":
                    msg = "NON-BANKNOTE | NO DONATION"
                    if review_id:
                        msg += f" | {review_id}"
                elif det_res.reason == "blank_paper_or_plain_object":
                    msg = "UNKNOWN OBJECT | REVIEW"
                    if review_id:
                        msg += f" | {review_id}"
                elif det_res.reason == "abnormal_aspect_ratio":
                    msg = "UNKNOWN OBJECT | REVIEW"
                    if review_id:
                        msg += f" | {review_id}"
                else:
                    msg = det_res.user_message
                    if review_id:
                        msg += f" | {review_id}"

            self.consensus_streak = 0
            self.current_candidate_label = None
            return ValidationOutcome(
                stage="DETECTION",
                passed=False,
                box_ready=True,
                quality_ok=True,
                note_detected=False,
                note_bbox=det_res.note_bbox,
                sharpness=qual_res.sharpness,
                predicted_label=None,
                confidence=0.0,
                all_probs={},
                decision=decision,
                user_message=msg,
                details={"reason": det_res.reason, "aspect_ratio": det_res.aspect_ratio,
                         "skin_ratio": det_res.skin_ratio, "edge_density": det_res.edge_density},
                review_id=review_id,
                review_reason=review_reason,
                is_review_event=is_review_event,
            )

        # Note present — reset empty streak
        self.empty_box_streak = 0

        # ── Stage D: Crop & Classify ──────────────────────────────────────────
        clf = crop_and_classify(
            model=model, class_names=class_names, frame=frame,
            note_bbox=det_res.note_bbox, device=device, config=self.config,
        )

        # Deduplication check
        nx, ny, nw, nh = det_res.note_bbox
        note_crop = frame[ny:ny + nh, nx:nx + nw]
        signature = self._compute_note_signature(clf.label, note_crop)

        if self.is_locked and signature == self.last_accepted_signature:
            return ValidationOutcome(
                stage="CONSENSUS",
                passed=True,
                box_ready=True,
                quality_ok=True,
                note_detected=True,
                note_bbox=det_res.note_bbox,
                sharpness=det_res.sharpness,
                predicted_label=clf.label,
                confidence=clf.confidence,
                all_probs=clf.all_probs,
                decision="WAITING",
                user_message="Donation already recorded. Please clear box.",
                details={"reason": "duplicate_same_note_suppressed"},
            )

        # ── Stage E: Multi-frame Consensus (for DONATION CONFIRMATION only) ──
        # Live prediction is always returned immediately.
        # Consensus only controls when accepted_event_ready fires.
        if clf.decision == "ACCEPT":
            if self.current_candidate_label == clf.label:
                self.consensus_streak += 1
                self.confidence_accumulator.append(clf.confidence)
            else:
                self.current_candidate_label = clf.label
                self.consensus_streak = 1
                self.confidence_accumulator = [clf.confidence]

            if self.consensus_streak >= self.config.consensus_frames:
                # Donation confirmed
                avg_conf = float(np.mean(self.confidence_accumulator))
                self.is_locked = True
                self.last_accepted_signature = signature
                self.accepted_event_ready = True
                self.consensus_streak = 0
                self.current_candidate_label = None

                if self.review_tracker is not None:
                    self.review_tracker.reset()

                return ValidationOutcome(
                    stage="CONSENSUS",
                    passed=True,
                    box_ready=True,
                    quality_ok=True,
                    note_detected=True,
                    note_bbox=det_res.note_bbox,
                    sharpness=det_res.sharpness,
                    predicted_label=clf.label,
                    confidence=avg_conf,
                    all_probs=clf.all_probs,
                    decision="ACCEPT",
                    user_message=f"ACCEPTED: {DENOM_LABELS.get(clf.label, clf.label)} ({avg_conf * 100:.1f}%)",
                    details={"target_bin": clf.target_bin, "consensus_frames": self.config.consensus_frames},
                )
            else:
                # Accumulating — show live prediction immediately
                return ValidationOutcome(
                    stage="CONSENSUS",
                    passed=False,
                    box_ready=True,
                    quality_ok=True,
                    note_detected=True,
                    note_bbox=det_res.note_bbox,
                    sharpness=det_res.sharpness,
                    predicted_label=clf.label,
                    confidence=clf.confidence,
                    all_probs=clf.all_probs,
                    decision="WAITING",
                    user_message=(
                        f"NOTE DETECTED | {DENOM_LABELS.get(clf.label, clf.label)} | "
                        f"{clf.confidence * 100:.1f}% "
                        f"({self.consensus_streak}/{self.config.consensus_frames})"
                    ),
                    details={"streak": self.consensus_streak},
                )

        elif clf.decision == "REVIEW":
            self.consensus_streak = 0
            self.current_candidate_label = None

            review_id = None
            is_review_event = False
            if self.review_tracker is not None:
                is_review_event, review_id = self.review_tracker.should_create_event(
                    reason="low_confidence",
                    candidate_label=clf.label,
                    confidence=clf.confidence,
                )

            denom_text = DENOM_LABELS.get(clf.label, clf.label)
            if review_id:
                user_msg = f"LOW CONFIDENCE | REVIEW | {review_id} | {denom_text} ({clf.confidence * 100:.1f}%)"
            else:
                user_msg = f"LOW CONFIDENCE - REVIEW: {denom_text} ({clf.confidence * 100:.1f}%)"

            return ValidationOutcome(
                stage="CLASSIFICATION",
                passed=False,
                box_ready=True,
                quality_ok=True,
                note_detected=True,
                note_bbox=det_res.note_bbox,
                sharpness=det_res.sharpness,
                predicted_label=clf.label,
                confidence=clf.confidence,
                all_probs=clf.all_probs,
                decision="REVIEW",
                user_message=user_msg,
                details={"target_bin": clf.target_bin},
                review_id=review_id,
                review_reason="low_confidence",
                is_review_event=is_review_event,
            )

        else:  # REJECT
            self.consensus_streak = 0
            self.current_candidate_label = None

            review_id = None
            is_review_event = False
            if self.review_tracker is not None:
                is_review_event, review_id = self.review_tracker.should_create_event(
                    reason="not_a_banknote",
                    candidate_label=clf.label,
                    confidence=clf.confidence,
                )

            if review_id:
                user_msg = f"UNKNOWN OBJECT | REVIEW | {review_id}"
            else:
                user_msg = "Unrecognized note. Please adjust."

            return ValidationOutcome(
                stage="CLASSIFICATION",
                passed=False,
                box_ready=True,
                quality_ok=True,
                note_detected=True,
                note_bbox=det_res.note_bbox,
                sharpness=det_res.sharpness,
                predicted_label=clf.label,
                confidence=clf.confidence,
                all_probs=clf.all_probs,
                decision="REJECT",
                user_message=user_msg,
                details={"target_bin": clf.target_bin},
                review_id=review_id,
                review_reason="not_a_banknote",
                is_review_event=is_review_event,
            )
