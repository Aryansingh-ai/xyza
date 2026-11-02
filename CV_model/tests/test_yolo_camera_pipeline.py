"""
tests/test_yolo_camera_pipeline.py

Automated tests for the YOLO-first one-shot camera pipeline and associated
helpers in live_camera.py.

All tests use mocks or in-memory stubs for camera I/O and time so that:
  - No real camera is opened or required.
  - Tests never wait a real 7 seconds for timeout behavior.
  - Model checkpoints are not loaded.
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

# ── Ensure CV_model/ is on the path ─────────────────────────────────────────
CV_MODEL_DIR = Path(__file__).resolve().parent.parent
if str(CV_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CV_MODEL_DIR))

from scripts.live_camera import (  # noqa: E402
    clip_yolo_bbox,
    run_yolo_detection_window,
    select_yolo_detection,
    run_yolo_pipeline,
    append_review_event,
)
from src.review_tracker import ReviewTracker  # noqa: E402


# ── Shared fakes ─────────────────────────────────────────────────────────────

class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)


class FakeBoxes:
    """Mimics ultralytics Boxes object for a list of (xyxy, conf, cls) tuples."""
    def __init__(self, boxes):
        self.xyxy = [FakeTensor(box[0]) for box in boxes]
        self.conf = [FakeTensor(box[1]) for box in boxes]
        self.cls  = [FakeTensor(box[2]) for box in boxes]

    def __len__(self):
        return len(self.xyxy)


def _blank_frame(h=40, w=60):
    return np.zeros((h, w, 3), dtype=np.uint8)


class FakeCaptureGood:
    """Camera that always returns a valid frame."""
    def read(self):
        return True, _blank_frame()


class FakeCaptureBadFrame:
    """Camera that opens but immediately fails to read a frame."""
    def read(self):
        return False, None


class FakeYoloNoDetection:
    """YOLO model that never finds anything."""
    def predict(self, **_kwargs):
        return [SimpleNamespace(boxes=FakeBoxes([]))]


class FakeYoloWithDetection:
    """YOLO model that always returns one banknote (class 5, conf 0.90)."""
    def predict(self, **_kwargs):
        return [SimpleNamespace(boxes=FakeBoxes([
            ([5, 5, 30, 25], 0.90, 5),  # xyxy, conf, cls_id=5 (500 INR)
        ]))]


# ── Helper: common cv2 mock patches ──────────────────────────────────────────

_CV2_PATCHES = {
    "scripts.live_camera.cv2.imshow": None,
    "scripts.live_camera.cv2.waitKey": -1,   # no key pressed
    "scripts.live_camera.cv2.rectangle": None,
    "scripts.live_camera.cv2.putText": None,
}


def _patch_cv2(extra=None):
    patches = dict(_CV2_PATCHES)
    if extra:
        patches.update(extra)
    return [patch(k, return_value=v) for k, v in patches.items()]


# ═════════════════════════════════════════════════════════════════════════════
# clip_yolo_bbox
# ═════════════════════════════════════════════════════════════════════════════

class TestClipYoloBbox(unittest.TestCase):
    """Unit tests for the bbox clamping helper."""

    def test_rejects_nan_coordinate(self):
        self.assertIsNone(clip_yolo_bbox([0, 0, np.nan, 10], (40, 60, 3)))

    def test_rejects_inf_coordinate(self):
        self.assertIsNone(clip_yolo_bbox([0, 0, np.inf, 10], (40, 60, 3)))

    def test_rejects_degenerate_box_same_coords(self):
        self.assertIsNone(clip_yolo_bbox([20, 20, 10, 10], (40, 60, 3)))

    def test_clamps_negative_and_out_of_bounds(self):
        self.assertEqual(clip_yolo_bbox([-5, -2, 65, 50], (40, 60, 3)), (0, 0, 60, 40))

    def test_valid_box_unchanged(self):
        self.assertEqual(clip_yolo_bbox([5, 5, 30, 20], (40, 60, 3)), (5, 5, 30, 20))


# ═════════════════════════════════════════════════════════════════════════════
# select_yolo_detection
# ═════════════════════════════════════════════════════════════════════════════

class TestSelectYoloDetection(unittest.TestCase):
    """Tests for highest-confidence valid banknote selection."""

    def test_selects_highest_confidence_valid_banknote(self):
        result = SimpleNamespace(boxes=FakeBoxes([
            ([1, 2, 20, 20], 0.60, 8),   # class 8 → out of range, skipped
            ([2, 3, 25, 30], 0.80, 4),   # class 4, conf 0.80 → valid
            ([0, 0, 10, 10], 0.95, 8),   # class 8 → out of range, skipped
        ]))
        self.assertEqual(
            select_yolo_detection(result, (40, 60, 3), 0.25),
            (0.80, 4, (2, 3, 25, 30)),
        )

    def test_returns_none_when_no_boxes(self):
        result = SimpleNamespace(boxes=FakeBoxes([]))
        self.assertIsNone(select_yolo_detection(result, (40, 60, 3), 0.25))

    def test_returns_none_below_confidence_threshold(self):
        result = SimpleNamespace(boxes=FakeBoxes([
            ([0, 0, 20, 20], 0.10, 3),   # conf 0.10 < threshold 0.25
        ]))
        self.assertIsNone(select_yolo_detection(result, (40, 60, 3), 0.25))

    def test_no_boxes_attribute(self):
        result = SimpleNamespace()  # no 'boxes' attribute
        self.assertIsNone(select_yolo_detection(result, (40, 60, 3), 0.25))


# ═════════════════════════════════════════════════════════════════════════════
# run_yolo_detection_window — one-shot and timeout behaviour
# ═════════════════════════════════════════════════════════════════════════════

class TestRunYoloDetectionWindow(unittest.TestCase):
    """
    Tests for the detection window: single-shot stop, timeout, frame errors,
    and timeout-measured-from-start (not reset per frame).
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _call_detection(self, cap, yolo, window_seconds=10.0, review_log=None,
                        tracker=None, **cv2_overrides):
        if tracker is None:
            tracker = ReviewTracker(cooldown_frames=8)
        if review_log is None:
            review_log = Path(tempfile.mkdtemp()) / "review.jsonl"
        patches = [
            patch("scripts.live_camera.cv2.imshow"),
            patch("scripts.live_camera.cv2.waitKey", return_value=-1),
            patch("scripts.live_camera.cv2.rectangle"),
            patch("scripts.live_camera.cv2.putText"),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            return run_yolo_detection_window(
                cap, yolo, window_seconds, 0.25, "test",
                0, tracker, review_log,
            )

    # ── Detection stops immediately on first banknote ─────────────────────

    def test_returns_frame_and_bbox_when_banknote_found(self):
        event = self._call_detection(FakeCaptureGood(), FakeYoloWithDetection())
        self.assertIsNotNone(event)
        frame, bbox, class_id = event
        self.assertIsNotNone(frame)
        self.assertIsNotNone(bbox)        # bbox present → banknote found
        self.assertEqual(class_id, 5)    # class 5 = 500 INR

    def test_detection_bbox_is_xywh_format(self):
        """Return value must be (x, y, w, h) not xyxy."""
        event = self._call_detection(FakeCaptureGood(), FakeYoloWithDetection())
        _frame, (x, y, w, h), _cls = event
        # The fake detection is xyxy=[5,5,30,25]; expected xywh=(5,5,25,20)
        self.assertEqual((x, y, w, h), (5, 5, 25, 20))

    def test_stops_immediately_after_first_detection(self):
        """YOLO is called exactly once — the loop does not continue scanning."""
        yolo = FakeYoloWithDetection()
        yolo.predict = MagicMock(wraps=yolo.predict)
        self._call_detection(FakeCaptureGood(), yolo)
        self.assertEqual(yolo.predict.call_count, 1)

    def test_no_duplicate_processing_flag(self):
        """
        Verifies the function returns after finding one note so callers never
        receive a second positive result for the same physical scan.
        """
        call_count = 0

        class CountingYolo:
            def predict(self, **_kwargs):
                nonlocal call_count
                call_count += 1
                return [SimpleNamespace(boxes=FakeBoxes([
                    ([5, 5, 30, 25], 0.90, 5),
                ]))]

        event = self._call_detection(FakeCaptureGood(), CountingYolo())
        self.assertIsNotNone(event[1])    # positive result
        self.assertEqual(call_count, 1)  # only one predict call

    # ── Timeout behaviour ─────────────────────────────────────────────────

    def test_timeout_returns_none_bbox(self):
        """When window expires with no detection, bbox is None."""
        event = self._call_detection(FakeCaptureGood(), FakeYoloNoDetection(),
                                     window_seconds=0.0)
        self.assertIsNotNone(event)
        _frame, bbox, cls_id = event
        self.assertIsNone(bbox)
        self.assertIsNone(cls_id)

    def test_timeout_creates_one_review_event(self):
        with tempfile.TemporaryDirectory() as d:
            review_log = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)
            self._call_detection(FakeCaptureGood(), FakeYoloNoDetection(),
                                 window_seconds=0.0,
                                 review_log=review_log, tracker=tracker)
            lines = review_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1, "Expected exactly one review event on timeout")

    def test_timeout_measured_from_scan_start_not_reset_per_frame(self):
        """
        The deadline is set once at function entry. Frame processing time does
        not extend the deadline.  We simulate 3 slow frames by mocking
        time.monotonic() to advance 3 seconds per frame call.
        """
        calls = [0]      # mutable counter
        base  = time.monotonic()

        def fake_monotonic():
            t = base + calls[0] * 3.0    # each call advances by 3 s
            calls[0] += 1
            return t

        with tempfile.TemporaryDirectory() as d:
            review_log = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)
            with patch("scripts.live_camera.time.monotonic", side_effect=fake_monotonic), \
                 patch("scripts.live_camera.cv2.imshow"), \
                 patch("scripts.live_camera.cv2.waitKey", return_value=-1), \
                 patch("scripts.live_camera.cv2.rectangle"), \
                 patch("scripts.live_camera.cv2.putText"):
                # Window = 5 s; first monotonic call sets deadline = base+5.
                # Second call (inside loop) returns base+3 → still inside window.
                # Third call returns base+6 → past deadline → exits without
                # reading another frame.
                event = run_yolo_detection_window(
                    FakeCaptureGood(), FakeYoloNoDetection(),
                    5.0, 0.25, "test", 0, tracker, review_log,
                )
            self.assertIsNone(event[1])   # timed out, not a detection

    # ── Frame-read failure ────────────────────────────────────────────────

    def test_frame_read_failure_raises_runtime_error(self):
        """A camera that cannot supply frames raises RuntimeError, not timeout."""
        with self.assertRaises(RuntimeError) as ctx:
            self._call_detection(FakeCaptureBadFrame(), FakeYoloNoDetection(),
                                 window_seconds=10.0)
        self.assertIn("Failed to read frame", str(ctx.exception))

    # ── User quit ─────────────────────────────────────────────────────────

    def test_user_quit_returns_none(self):
        """Pressing 'q' (ord 113) returns None."""
        with patch("scripts.live_camera.cv2.imshow"), \
             patch("scripts.live_camera.cv2.waitKey", return_value=ord("q")), \
             patch("scripts.live_camera.cv2.rectangle"), \
             patch("scripts.live_camera.cv2.putText"):
            result = run_yolo_detection_window(
                FakeCaptureGood(), FakeYoloNoDetection(),
                10.0, 0.25, "test", 0,
                ReviewTracker(cooldown_frames=8),
                Path(tempfile.mkdtemp()) / "r.jsonl",
            )
        self.assertIsNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# run_yolo_pipeline — camera open/close and one-shot behavior
# ═════════════════════════════════════════════════════════════════════════════

class TestRunYoloPipeline(unittest.TestCase):
    """
    High-level tests for run_yolo_pipeline.  ML models are mocked so no
    checkpoint files are needed.
    """

    def _make_args(self, timeout=7.0, accept_confidence=0.80,
                   review_confidence=0.45, camera=0):
        return SimpleNamespace(
            timeout=timeout,
            yolo_confidence=0.25,
            accept_confidence=accept_confidence,
            review_confidence=review_confidence,
            camera=camera,
            continuous=False,
        )

    def _make_classification(self, decision="ACCEPT", label="500", conf=0.92):
        from src.validator import ClassificationResult
        return ClassificationResult(
            label=label, confidence=conf, decision=decision,
            target_bin=f"BIN_{label}",
            all_probs={label: conf},
        )

    def test_pipeline_processes_note_exactly_once_on_detection(self):
        """
        When YOLO detects a banknote, crop_and_classify is called exactly once
        and run_yolo_pipeline returns immediately.
        """
        args = self._make_args()
        fake_model = MagicMock()
        fake_class_names = ["10", "20", "50", "100", "200", "500"]

        with tempfile.TemporaryDirectory() as d:
            log_path    = Path(d) / "inference.jsonl"
            review_path = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)

            classify_mock = MagicMock(return_value=self._make_classification())

            with patch("scripts.live_camera.run_yolo_detection_window",
                       return_value=(_blank_frame(), (5, 5, 25, 20), 5)), \
                 patch("scripts.live_camera.crop_and_classify", classify_mock), \
                 patch("scripts.live_camera.append_log", return_value={"decision": "ACCEPT"}), \
                 patch("scripts.live_camera.register_blockchain_proof", return_value=None), \
                 patch("scripts.live_camera.cv2.imshow"), \
                 patch("scripts.live_camera.cv2.waitKey", return_value=-1), \
                 patch("scripts.live_camera.cv2.rectangle"), \
                 patch("scripts.live_camera.cv2.putText"):
                run_yolo_pipeline(
                    FakeCaptureGood(), FakeYoloWithDetection(),
                    fake_model, fake_class_names, "cpu",
                    args, tracker, log_path, review_path,
                )

            # crop_and_classify called exactly once
            self.assertEqual(classify_mock.call_count, 1)

    def test_pipeline_exits_on_timeout(self):
        """When no banknote is detected within timeout, pipeline returns without error."""
        args = self._make_args(timeout=7.0)

        with tempfile.TemporaryDirectory() as d:
            log_path    = Path(d) / "inference.jsonl"
            review_path = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)

            classify_mock = MagicMock()

            with patch("scripts.live_camera.run_yolo_detection_window",
                       return_value=(_blank_frame(), None, None)), \
                 patch("scripts.live_camera.crop_and_classify", classify_mock), \
                 patch("scripts.live_camera.cv2.imshow"), \
                 patch("scripts.live_camera.cv2.waitKey", return_value=-1):
                run_yolo_pipeline(
                    FakeCaptureGood(), FakeYoloNoDetection(),
                    MagicMock(), ["10", "500"], "cpu",
                    args, tracker, log_path, review_path,
                )

            # No classification should happen when timeout fires
            classify_mock.assert_not_called()

    def test_pipeline_exits_on_user_quit(self):
        """None from detection window (user quit) → pipeline returns cleanly."""
        args = self._make_args()
        classify_mock = MagicMock()

        with tempfile.TemporaryDirectory() as d:
            log_path    = Path(d) / "inference.jsonl"
            review_path = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)

            with patch("scripts.live_camera.run_yolo_detection_window",
                       return_value=None), \
                 patch("scripts.live_camera.crop_and_classify", classify_mock):
                run_yolo_pipeline(
                    FakeCaptureGood(), FakeYoloNoDetection(),
                    MagicMock(), ["500"], "cpu",
                    args, tracker, log_path, review_path,
                )

            classify_mock.assert_not_called()

    def test_pipeline_does_not_loop_after_detection_even_if_continuous(self):
        """
        The YOLO pipeline is one-shot.  Even if args.continuous is True, the
        pipeline must not run a second detection window after the first note is
        processed.
        """
        args = self._make_args()
        args.continuous = True    # should be ignored in YOLO pipeline

        detect_mock = MagicMock(return_value=(_blank_frame(), (5, 5, 25, 20), 5))
        classify_mock = MagicMock(return_value=self._make_classification())

        with tempfile.TemporaryDirectory() as d:
            log_path    = Path(d) / "inference.jsonl"
            review_path = Path(d) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)

            with patch("scripts.live_camera.run_yolo_detection_window", detect_mock), \
                 patch("scripts.live_camera.crop_and_classify", classify_mock), \
                 patch("scripts.live_camera.append_log", return_value={"decision": "ACCEPT"}), \
                 patch("scripts.live_camera.register_blockchain_proof", return_value=None), \
                 patch("scripts.live_camera.cv2.imshow"), \
                 patch("scripts.live_camera.cv2.waitKey", return_value=-1), \
                 patch("scripts.live_camera.cv2.rectangle"), \
                 patch("scripts.live_camera.cv2.putText"):
                run_yolo_pipeline(
                    FakeCaptureGood(), FakeYoloWithDetection(),
                    MagicMock(), ["10", "20", "50", "100", "200", "500"], "cpu",
                    args, tracker, log_path, review_path,
                )

            # Detection window opened exactly once, classification done exactly once
            self.assertEqual(detect_mock.call_count, 1)
            self.assertEqual(classify_mock.call_count, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Camera open / release
# ═════════════════════════════════════════════════════════════════════════════

class TestCameraOpenAndRelease(unittest.TestCase):
    """
    Verify that camera resource cleanup happens on every exit path from main().
    We monkey-patch cv2.VideoCapture and check that release() is always called.
    """

    def _run_main_with_mock_camera(self, camera_opens=True, frame_ok=True,
                                   extra_args=None):
        """Run main() with mocked camera and minimal mocked args."""
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = camera_opens
        cap_mock.read.return_value = (frame_ok, _blank_frame() if frame_ok else None)

        argv = ["live_camera.py", "--yolo_pipeline",
                "--yolo_checkpoint", "fake.pt",
                "--checkpoint", "fake_mobilenet.pt"]
        if extra_args:
            argv.extend(extra_args)

        with patch("sys.argv", argv), \
             patch("scripts.live_camera.cv2.VideoCapture", return_value=cap_mock), \
             patch("scripts.live_camera.cv2.namedWindow"), \
             patch("scripts.live_camera.cv2.resizeWindow"), \
             patch("scripts.live_camera.cv2.moveWindow"), \
             patch("scripts.live_camera.cv2.destroyAllWindows") as destroy_mock, \
             patch("scripts.live_camera.load_model",
                   return_value=(MagicMock(), ["10", "500"])), \
             patch("scripts.live_camera.load_yolo_model", return_value=MagicMock()), \
             patch("scripts.live_camera.run_yolo_pipeline"):
            from scripts.live_camera import main
            main()

        return cap_mock, destroy_mock

    def test_camera_released_on_normal_exit(self):
        cap_mock, destroy_mock = self._run_main_with_mock_camera()
        cap_mock.release.assert_called_once()
        destroy_mock.assert_called_once()

    def test_camera_not_opened_exits_without_processing(self):
        """If camera fails to open, main() should return without calling release on it."""
        cap_mock, destroy_mock = self._run_main_with_mock_camera(camera_opens=False)
        # release() must NOT be called if cap was never opened (unreliable resource)
        cap_mock.release.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# Existing tests kept intact
# ═════════════════════════════════════════════════════════════════════════════

class TestLegacyYoloCameraPipeline(unittest.TestCase):
    """
    Preserved versions of the original three tests so the test runner still
    passes them under their original names.
    """

    def test_clip_rejects_nonfinite_and_empty_boxes(self):
        self.assertIsNone(clip_yolo_bbox([0, 0, np.nan, 10], (40, 60, 3)))
        self.assertIsNone(clip_yolo_bbox([20, 20, 10, 10], (40, 60, 3)))
        self.assertEqual(clip_yolo_bbox([-5, -2, 65, 50], (40, 60, 3)), (0, 0, 60, 40))

    def test_selects_highest_confidence_valid_banknote(self):
        result = SimpleNamespace(boxes=FakeBoxes([
            ([1, 2, 20, 20], 0.60, 8),
            ([2, 3, 25, 30], 0.80, 4),
            ([0, 0, 10, 10], 0.95, 8),
        ]))
        self.assertEqual(
            select_yolo_detection(result, (40, 60, 3), 0.25),
            (0.80, 4, (2, 3, 25, 30)),
        )

    def test_no_detection_creates_one_review_event(self):
        with tempfile.TemporaryDirectory() as directory:
            review_log = Path(directory) / "review.jsonl"
            tracker = ReviewTracker(cooldown_frames=8)
            with patch("scripts.live_camera.cv2.imshow"), patch(
                "scripts.live_camera.cv2.waitKey", return_value=-1
            ), patch("scripts.live_camera.cv2.rectangle"), \
             patch("scripts.live_camera.cv2.putText"):
                event = run_yolo_detection_window(
                    FakeCaptureGood(), FakeYoloNoDetection(), 0.0, 0.25, "test",
                    0, tracker, review_log,
                )
            self.assertIsNotNone(event)
            self.assertIsNone(event[1])
            self.assertTrue(tracker.is_active())
            self.assertEqual(len(review_log.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
