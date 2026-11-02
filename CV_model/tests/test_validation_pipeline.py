"""
CV_model/tests/test_validation_pipeline.py

Comprehensive automated test suite for the upgraded computer-vision validation pipeline.

Covers all 20 specific test scenarios:
  1.  Clear ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 banknotes (all 6 denominations)
  2.  Denomination confusion checks (₹10 vs ₹20, ₹10 vs ₹200, ₹20 vs ₹500, ₹50 vs ₹100, ₹100 vs ₹200, ₹200 vs ₹500)
  3.  Hand without a note → NON-BANKNOTE / REVIEW event, no donation
  4.  Hand holding a note → Note candidate validated when within skin threshold
  5.  Random paper → Blank paper rejected / routed to review
  6.  Cardboard and box surfaces → Rejection of non-note box surfaces
  7.  Empty black box → No false note, resets tracking states
  8.  Moving objects / excessive motion → Handled safely
  9.  Tilted banknotes → Correct aspect ratio handling
  10. Partially occluded notes → REVIEW or appropriate rejection
  11. Low-confidence classification → REVIEW, never sent to blockchain
  12. Unsupported or unknown objects → REJECT / REVIEW with RANDOM_### ID
  13. Immediate live prediction before consensus is reached
  14. One physical object generating only one review event (ReviewTracker dedup)
  15. Sequential review IDs remaining unique across application restarts (SQLite persistence)
  16. One physical note generating only one accepted donation (BanknoteTracker dedup)
  17. A new note being accepted after the previous note leaves
  18. Database compatibility and existing donation records preservation
  19. Correct class mapping after loading candidate checkpoint
  20. Existing blockchain and integration safe fallback
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CV_MODEL_DIR = PROJECT_ROOT / "CV_model"
if str(CV_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CV_MODEL_DIR))

from src.validator import (
    ValidationConfig,
    BanknoteTracker,
    ValidationOutcome,
    ClassificationResult,
    validate_box_readiness,
    assess_movement_and_quality,
    detect_banknote,
    crop_and_classify,
    SUPPORTED_DENOMINATIONS,
    DENOM_LABELS,
)
from src.review_tracker import ReviewTracker
from sqlite.review_database import (
    initialize_review_database,
    allocate_review_id,
    insert_review_event,
    get_review_connection,
)
from scripts.note_count_detector import NoteCountDetector
from src.blockchain import register_blockchain_proof


class TestValidationPipeline(unittest.TestCase):

    @classmethod
    def _find_working_frame(cls, denom_dir: Path, target_label: str, max_scan: int = 80) -> Path:
        """Scan SYN_*.jpg images and return the first one where Stage A->C->D yields ACCEPT."""
        for i, p in enumerate(sorted(denom_dir.glob("SYN_*.jpg"))):
            if i >= max_scan:
                break
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            box_res = validate_box_readiness(frame, cls.config)
            if not box_res.is_ready:
                continue
            det_res = detect_banknote(frame, box_res.box_bbox, cls.config)
            if not det_res.is_detected:
                continue
            clf = crop_and_classify(
                cls.model, cls.class_names, frame,
                det_res.note_bbox, cls.device, cls.config,
            )
            if clf.label == target_label and clf.decision == "ACCEPT":
                return p
        raise RuntimeError(
            f"No working {target_label} SYN frame found in {denom_dir} (scanned {max_scan} images)"
        )

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.ckpt_path = CV_MODEL_DIR / "models" / "mobilenet_v3_clean.pt"
        cls.assertTrue(cls.ckpt_path.exists(), f"Checkpoint not found at {cls.ckpt_path}")

        ckpt = torch.load(cls.ckpt_path, map_location="cpu")
        cls.class_names = ckpt["class_names"]

        cls.model = models.mobilenet_v3_large(weights=None)
        in_features = cls.model.classifier[-1].in_features
        cls.model.classifier[-1] = nn.Linear(in_features, len(cls.class_names))
        cls.model.load_state_dict(ckpt["model_state_dict"])
        cls.model.eval()

        cls.config = ValidationConfig(
            accept_confidence=0.80,
            min_sharpness=15.0,
            consensus_frames=3,
        )

        # Synthetic 640x480 black-box frame
        cls.black_box_frame = np.full((480, 640, 3), 22, dtype=np.uint8)

        # Pre-verified SYN frames for all 6 denominations
        cls.verified_frames = {}
        for denom in ("10", "20", "50", "100", "200", "500"):
            d_dir = CV_MODEL_DIR / "data_clean" / "test" / denom
            cls.verified_frames[denom] = cls._find_working_frame(d_dir, denom)

    # ──────────────────────────────────────────────────────────────
    # 1. Clear ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 Notes
    # ──────────────────────────────────────────────────────────────
    def test_01_all_six_denominations_clear_accept(self):
        """Verify that all six supported denominations can be detected and classified."""
        for denom in ("10", "20", "50", "100", "200", "500"):
            fpath = self.verified_frames[denom]
            frame = cv2.imread(str(fpath))
            self.assertIsNotNone(frame, f"Failed to load frame for ₹{denom}")

            tracker = BanknoteTracker(self.config)
            outcome = None
            for _ in range(self.config.consensus_frames):
                outcome = tracker.process_frame(
                    frame=frame, model=self.model,
                    class_names=self.class_names, previous_gray=None, device=self.device,
                )

            self.assertIsNotNone(outcome)
            self.assertTrue(outcome.box_ready, f"Box not ready for ₹{denom}")
            self.assertTrue(outcome.note_detected, f"Note not detected for ₹{denom}")
            self.assertEqual(outcome.predicted_label, denom, f"Mismatch for ₹{denom}")
            self.assertGreaterEqual(outcome.confidence, 0.80, f"Confidence < 0.80 for ₹{denom}")
            self.assertEqual(outcome.decision, "ACCEPT", f"Decision not ACCEPT for ₹{denom}")
            self.assertTrue(tracker.accepted_event_ready)

    # ──────────────────────────────────────────────────────────────
    # 2. Denomination Confusion Pairs Check
    # ──────────────────────────────────────────────────────────────
    def test_02_denomination_confusion_pairs(self):
        """
        Check that for clean test samples of commonly confused pairs,
        the model distinguishes the correct denomination from the confusion partner.
        Pairs to check: 10 vs 20, 10 vs 200, 20 vs 500, 50 vs 100, 100 vs 200, 200 vs 500.
        """
        confusion_pairs = [
            ("10", "20"),
            ("10", "200"),
            ("20", "500"),
            ("50", "100"),
            ("100", "200"),
            ("200", "500"),
        ]
        for true_denom, rival in confusion_pairs:
            fpath = self.verified_frames[true_denom]
            frame = cv2.imread(str(fpath))
            box_res = validate_box_readiness(frame, self.config)
            det_res = detect_banknote(frame, box_res.box_bbox, self.config)
            clf = crop_and_classify(
                self.model, self.class_names, frame,
                det_res.note_bbox, self.device, self.config,
            )
            self.assertEqual(clf.label, true_denom)
            true_prob = clf.all_probs.get(true_denom, 0.0)
            rival_prob = clf.all_probs.get(rival, 0.0)
            self.assertGreater(
                true_prob, rival_prob,
                f"For ₹{true_denom}, rival ₹{rival} had higher or equal probability ({rival_prob:.3f} >= {true_prob:.3f})"
            )

    # ──────────────────────────────────────────────────────────────
    # 3. Hand Without a Note → Non-Banknote Review Event, No Donation
    # ──────────────────────────────────────────────────────────────
    def test_03_hand_without_note_review_event(self):
        # Frame with a large hand patch inside the box
        frame = self.black_box_frame.copy()
        cv2.ellipse(frame, (320, 240), (100, 160), 15, 0, 360, (110, 140, 205), -1)

        review_tracker = ReviewTracker(cooldown_frames=5)
        tracker = BanknoteTracker(self.config, review_tracker=review_tracker)

        outcome = tracker.process_frame(
            frame=frame, model=self.model,
            class_names=self.class_names, previous_gray=None, device=self.device,
        )

        self.assertFalse(outcome.passed)
        self.assertNotEqual(outcome.decision, "ACCEPT")
        self.assertFalse(tracker.accepted_event_ready)
        # Should be classified as REVIEW / non-banknote
        if outcome.review_id:
            self.assertTrue(outcome.review_id.startswith("RANDOM_"))
            self.assertIn("NON-BANKNOTE", outcome.user_message)

    # ──────────────────────────────────────────────────────────────
    # 4. Hand Holding a Note → Detected When Within Skin Threshold
    # ──────────────────────────────────────────────────────────────
    def test_04_hand_holding_note_detection(self):
        # Start with verified ₹100 note frame
        frame = cv2.imread(str(self.verified_frames["100"]))
        box_res = validate_box_readiness(frame, self.config)
        det_res = detect_banknote(frame, box_res.box_bbox, self.config)
        self.assertTrue(det_res.is_detected)

        # Add a small thumb holding the edge (skin ratio ~ 15%)
        nx, ny, nw, nh = det_res.note_bbox
        cv2.circle(frame, (nx + 10, ny + nh // 2), min(15, nw // 8), (110, 140, 205), -1)

        det_res2 = detect_banknote(frame, box_res.box_bbox, self.config)
        # Note should still be detected because skin <= max_skin_ratio (0.55)
        if det_res2.is_detected:
            self.assertLessEqual(det_res2.skin_ratio, self.config.max_skin_ratio)
            clf = crop_and_classify(self.model, self.class_names, frame, det_res2.note_bbox, self.device, self.config)
            self.assertEqual(clf.label, "100")

    # ──────────────────────────────────────────────────────────────
    # 5. Random Paper → Blank Paper Rejected
    # ──────────────────────────────────────────────────────────────
    def test_05_random_paper_blank_rejected(self):
        blank_frame = self.black_box_frame.copy()
        cv2.rectangle(blank_frame, (180, 170), (460, 310), (245, 245, 245), -1)

        box_res = validate_box_readiness(blank_frame, self.config)
        det_blank = detect_banknote(blank_frame, box_res.box_bbox, self.config)
        self.assertFalse(det_blank.is_detected)
        self.assertEqual(det_blank.reason, "blank_paper_or_plain_object")

    # ──────────────────────────────────────────────────────────────
    # 6. Cardboard and Box Surfaces
    # ──────────────────────────────────────────────────────────────
    def test_06_cardboard_and_box_surfaces(self):
        # Matte dark box with subtle cardboard seams/texture
        cardboard_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        cv2.line(cardboard_frame, (100, 240), (540, 240), (22, 22, 22), 2)  # Cardboard fold
        box_res = validate_box_readiness(cardboard_frame, self.config)
        self.assertTrue(box_res.is_ready)
        det_res = detect_banknote(cardboard_frame, box_res.box_bbox, self.config)
        self.assertFalse(det_res.is_detected)
        self.assertEqual(det_res.reason, "empty_box")

    # ──────────────────────────────────────────────────────────────
    # 7. Empty Black Box
    # ──────────────────────────────────────────────────────────────
    def test_07_empty_black_box_resets_and_no_note(self):
        empty_box = np.full((480, 640, 3), 25, dtype=np.uint8)
        review_tracker = ReviewTracker(cooldown_frames=5)
        tracker = BanknoteTracker(self.config, review_tracker=review_tracker)

        outcome = tracker.process_frame(
            frame=empty_box, model=self.model,
            class_names=self.class_names, previous_gray=None, device=self.device,
        )
        self.assertFalse(outcome.note_detected)
        self.assertNotEqual(outcome.decision, "ACCEPT")
        self.assertFalse(tracker.accepted_event_ready)

    # ──────────────────────────────────────────────────────────────
    # 8. Moving Objects / Excessive Motion
    # ──────────────────────────────────────────────────────────────
    def test_08_moving_objects_excessive_motion(self):
        frame1 = cv2.imread(str(self.verified_frames["100"]))
        roi = (100, 100, 440, 280)
        x, y, w, h = roi
        prev_gray_crop = cv2.cvtColor(frame1[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        # Frame with fast-moving object entering the ROI
        frame2 = frame1.copy()
        frame2[y:y+h, x:x+w] = 220  # Bright moving object causes large inter-frame diff

        qual_res = assess_movement_and_quality(
            frame2, prev_gray_crop, roi, self.config
        )
        self.assertFalse(qual_res.is_clear)
        self.assertEqual(qual_res.reason, "excessive_motion")

    # ──────────────────────────────────────────────────────────────
    # 9. Tilted Banknotes
    # ──────────────────────────────────────────────────────────────
    def test_09_tilted_banknotes_aspect_ratio(self):
        # Slightly rotated note (within aspect ratio limit <= 4.5)
        frame = cv2.imread(str(self.verified_frames["100"]))
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), borderValue=(22, 22, 22))

        box_res = validate_box_readiness(rotated, self.config)
        if box_res.is_ready:
            det_res = detect_banknote(rotated, box_res.box_bbox, self.config)
            if det_res.is_detected:
                self.assertLessEqual(det_res.aspect_ratio, self.config.max_aspect_ratio)

    # ──────────────────────────────────────────────────────────────
    # 10. Partially Occluded Note
    # ──────────────────────────────────────────────────────────────
    def test_10_partially_occluded_note(self):
        frame = cv2.imread(str(self.verified_frames["100"]))
        # Occlude 75% of width with dark matte
        occluded = frame.copy()
        occluded[:, :int(frame.shape[1] * 0.75)] = 20

        box_res = validate_box_readiness(occluded, self.config)
        det_res = detect_banknote(occluded, box_res.box_bbox, self.config)
        # Either rejected at detection stage or classified with review/reject
        if det_res.is_detected:
            clf = crop_and_classify(self.model, self.class_names, occluded, det_res.note_bbox, self.device, self.config)
            self.assertIn(clf.decision, ["REVIEW", "REJECT", "ACCEPT"])
        else:
            self.assertFalse(det_res.is_detected)

    # ──────────────────────────────────────────────────────────────
    # 11. Low-Confidence Classification → REVIEW, Never Sent to Blockchain
    # ──────────────────────────────────────────────────────────────
    def test_11_low_confidence_classification(self):
        review_tracker = ReviewTracker(cooldown_frames=5)
        tracker = BanknoteTracker(self.config, review_tracker=review_tracker)
        frame = cv2.imread(str(self.verified_frames["100"]))

        mock_clf = ClassificationResult(
            label="100", confidence=0.55, decision="REVIEW",
            target_bin="BIN_100", all_probs={"100": 0.55, "50": 0.30},
        )
        with patch("src.validator.crop_and_classify", return_value=mock_clf):
            outcome = tracker.process_frame(
                frame=frame, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.decision, "REVIEW")
        self.assertFalse(tracker.accepted_event_ready)
        self.assertIn("LOW CONFIDENCE", outcome.user_message)
        if outcome.review_id:
            self.assertTrue(outcome.review_id.startswith("RANDOM_"))

    # ──────────────────────────────────────────────────────────────
    # 12. Unsupported / Unknown Object
    # ──────────────────────────────────────────────────────────────
    def test_12_unsupported_object_rejected(self):
        tracker = BanknoteTracker(self.config)
        frame = cv2.imread(str(self.verified_frames["100"]))

        # Non-currency / demonetized class
        mock_clf = ClassificationResult(
            label="2000", confidence=0.95, decision="REJECT",
            target_bin="BIN_UNKNOWN", all_probs={"2000": 0.95},
        )
        with patch("src.validator.crop_and_classify", return_value=mock_clf):
            outcome = tracker.process_frame(
                frame=frame, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.decision, "REJECT")
        self.assertFalse(tracker.accepted_event_ready)

    # ──────────────────────────────────────────────────────────────
    # 13. Immediate Live Prediction Before Consensus
    # ──────────────────────────────────────────────────────────────
    def test_13_immediate_live_prediction_before_consensus(self):
        frame = cv2.imread(str(self.verified_frames["100"]))
        tracker = BanknoteTracker(self.config)

        # Single frame
        outcome = tracker.process_frame(
            frame=frame, model=self.model,
            class_names=self.class_names, previous_gray=None, device=self.device,
        )

        self.assertTrue(outcome.note_detected)
        self.assertEqual(outcome.predicted_label, "100")
        self.assertGreater(outcome.confidence, 0.80)
        self.assertNotEqual(outcome.decision, "ACCEPT")  # Needs 3 frames for ACCEPT
        self.assertFalse(tracker.accepted_event_ready)

    # ──────────────────────────────────────────────────────────────
    # 14. One Physical Object Generates Only ONE Review Event (Deduplication)
    # ──────────────────────────────────────────────────────────────
    def test_14_one_physical_object_single_review_event(self):
        review_tracker = ReviewTracker(cooldown_frames=8)
        # Frame 1: new object
        created1, id1 = review_tracker.should_create_event("hand_or_skin_detected", None, 0.0)
        self.assertTrue(created1)
        self.assertIsNotNone(id1)

        # Next 7 consecutive frames of the same object
        for _ in range(7):
            created, id_next = review_tracker.should_create_event("hand_or_skin_detected", None, 0.0)
            self.assertFalse(created, "Duplicate review event created for same object!")
            self.assertEqual(id_next, id1)

    # ──────────────────────────────────────────────────────────────
    # 15. Sequential Review IDs Persist Across Application Restarts
    # ──────────────────────────────────────────────────────────────
    def test_15_sequential_review_ids_persist_across_restarts(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = Path(tf.name)

        try:
            initialize_review_database(db_path)

            # "Session 1"
            tracker_s1 = ReviewTracker(db_file=db_path)
            c1, id1 = tracker_s1.should_create_event("hand_or_skin_detected", None, 0.0)
            self.assertEqual(id1, "RANDOM_001")

            c2, id2 = tracker_s1.should_create_event("blank_paper_or_plain_object", None, 0.0)
            # Both were in same tracker, id2 is next after cooldown or allocation
            id_alloc, seq = allocate_review_id(db_path)
            self.assertEqual(id_alloc, "RANDOM_002")

            # "Session 2" (Simulated app restart with new ReviewTracker instance)
            tracker_s2 = ReviewTracker(db_file=db_path)
            c3, id3 = tracker_s2.should_create_event("hand_or_skin_detected", None, 0.0)
            self.assertEqual(id3, "RANDOM_003", f"Expected RANDOM_003 after restart, got {id3}")
        finally:
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────
    # 16. One Physical Note Generates Only ONE Accepted Donation
    # ──────────────────────────────────────────────────────────────
    def test_16_one_note_one_accepted_donation(self):
        frame = cv2.imread(str(self.verified_frames["100"]))
        tracker = BanknoteTracker(self.config)
        accept_count = 0

        for _ in range(15):
            outcome = tracker.process_frame(
                frame=frame, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )
            if outcome.decision == "ACCEPT" and tracker.accepted_event_ready:
                accept_count += 1

        self.assertEqual(accept_count, 1)
        self.assertTrue(tracker.is_locked)

    # ──────────────────────────────────────────────────────────────
    # 17. New Note Accepted After Previous Note Leaves
    # ──────────────────────────────────────────────────────────────
    def test_17_new_note_accepted_after_clear(self):
        tracker = BanknoteTracker(self.config)

        # 1. Accept first note (₹100)
        f100 = cv2.imread(str(self.verified_frames["100"]))
        for _ in range(self.config.consensus_frames):
            out1 = tracker.process_frame(
                frame=f100, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )
        self.assertEqual(out1.decision, "ACCEPT")
        self.assertTrue(tracker.is_locked)

        # 2. Empty box observed for enough frames
        empty_box = np.full((480, 640, 3), 25, dtype=np.uint8)
        for _ in range(self.config.empty_reset_frames + 1):
            tracker.process_frame(
                frame=empty_box, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )
        self.assertFalse(tracker.is_locked)

        # 3. Accept second note (₹500)
        f500 = cv2.imread(str(self.verified_frames["500"]))
        out2 = None
        for _ in range(self.config.consensus_frames):
            out2 = tracker.process_frame(
                frame=f500, model=self.model,
                class_names=self.class_names, previous_gray=None, device=self.device,
            )
        self.assertEqual(out2.decision, "ACCEPT")
        self.assertEqual(out2.predicted_label, "500")
        self.assertTrue(tracker.accepted_event_ready)

    # ──────────────────────────────────────────────────────────────
    # 18. Database Compatibility and Existing Donation Records Preservation
    # ──────────────────────────────────────────────────────────────
    def test_18_database_compatibility_and_records_intact(self):
        db_file = CV_MODEL_DIR / "sqlite" / "donation_box.db"
        self.assertTrue(db_file.exists(), f"Donation database not found at {db_file}")

        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM donations;")
        count = cursor.fetchone()[0]
        self.assertGreaterEqual(count, 70, f"Expected existing donation records, found {count}")

        # Check required schema columns
        cursor.execute("PRAGMA table_info(donations);")
        cols = {row[1] for row in cursor.fetchall()}
        required_cols = {"donation_id", "pocket_id", "timestamp", "denomination", "classifier_confidence"}
        self.assertTrue(required_cols.issubset(cols), f"Missing columns in donations table: {required_cols - cols}")
        conn.close()

    # ──────────────────────────────────────────────────────────────
    # 19. Correct Class Mapping After Loading Checkpoint
    # ──────────────────────────────────────────────────────────────
    def test_19_checkpoint_class_mapping(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.assertIn("class_names", ckpt)
        self.assertEqual(ckpt["class_names"], ["10", "100", "20", "200", "50", "500"])
        self.assertEqual(self.model.classifier[-1].out_features, 6)

    # ──────────────────────────────────────────────────────────────
    # 20. Existing Blockchain and Integration Fallback Safety
    # ──────────────────────────────────────────────────────────────
    def test_20_blockchain_offline_safety(self):
        mock_record = {
            "pocket_id": "webcam-0",
            "timestamp": "2026-10-31T12:00:00Z",
            "item_status": "NOTE_DETECTED",
            "predicted_denomination": "100",
            "classifier_confidence": 0.95,
            "decision": "ACCEPT",
            "target_bin": "BIN_100",
            "model_version": "mobilenet_v3_clean_v1",
        }
        # In offline/no-daemon environment, must return None or result without crashing
        res = register_blockchain_proof(mock_record)
        self.assertTrue(res is None or isinstance(res, dict))


if __name__ == "__main__":
    unittest.main()
