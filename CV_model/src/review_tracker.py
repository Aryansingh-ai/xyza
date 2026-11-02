"""
CV_model/src/review_tracker.py

Deduplication and event management for REVIEW (unknown-object) events.

The same physical object may remain visible for many consecutive frames.
This tracker ensures one physical object generates ONE review event,
not hundreds.

Review IDs use the format RANDOM_### (e.g. RANDOM_001, RANDOM_007).
The sequential counter persists in SQLite across application restarts.

Usage:
    tracker = ReviewTracker(db_file=..., pocket_id="webcam-0")
    
    # On each frame where detection/classification indicates a non-banknote:
    should_create, review_id = tracker.should_create_event(reason, candidate_label, confidence)
    if should_create:
        # Log the review event
        print(f"New review event: {review_id}")
    
    # When the box becomes empty or a note is accepted:
    tracker.reset()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Reasons that trigger a REVIEW event (non-banknote detections)
REVIEW_TRIGGERING_REASONS = frozenset({
    "hand_or_skin_detected",
    "abnormal_aspect_ratio",
    "blank_paper_or_plain_object",
    "low_confidence",
    "not_a_banknote",
    "object_too_large_or_full_occlusion",
})

# These reasons mean "nothing there yet" — do NOT create a review event
NON_REVIEW_REASONS = frozenset({
    "empty_box",
    "object_too_small",
    "ambient_darkness_no_box",
    "scene_too_bright_no_box",
    "insufficient_dark_interior",
    "scene_not_box_interior",
    "quality_acceptable",
    "box_detected",
    "banknote_candidate_verified",
    "duplicate_same_note_suppressed",
    "",
})


@dataclass
class ReviewState:
    review_id: str = ""
    review_seq: int = 0
    reason: str = ""
    candidate_label: Optional[str] = None
    confidence: float = 0.0
    frame_count: int = 0          # Consecutive frames for this event
    created_at: str = ""


class ReviewTracker:
    """
    Manages deduplication of REVIEW events.
    
    State machine:
      IDLE        → waiting for a new unknown object
      ACTIVE      → tracking an unknown object (review event exists)
    
    Transitions:
      IDLE  + non-banknote detected → ACTIVE (create review event)
      ACTIVE + same/similar object  → stay ACTIVE (suppress duplicate)
      ACTIVE + empty box            → IDLE (reset)
      ACTIVE + note accepted        → IDLE (reset)
      ACTIVE + reason changes significantly → new ACTIVE (new review event)
    """

    def __init__(
        self,
        cooldown_frames: int = 8,
        db_file: Optional[Path] = None,
        pocket_id: str = "webcam-0",
    ):
        self.cooldown_frames = cooldown_frames
        self.pocket_id = pocket_id
        self._state: Optional[ReviewState] = None   # None = IDLE
        self._db_file = db_file

        # Lazy import to avoid circular deps
        self._db_available = False
        if db_file is not None:
            try:
                from sqlite.review_database import initialize_review_database, allocate_review_id, insert_review_event
                self._allocate = allocate_review_id
                self._insert = insert_review_event
                initialize_review_database(db_file)
                self._db_available = True
            except Exception as e:
                print(f"[ReviewTracker] Warning: review DB unavailable ({e}). Using in-memory counter.")
                self._db_available = False

        # In-memory fallback counter when DB is unavailable
        self._mem_counter = 1

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def should_create_event(
        self,
        reason: str,
        candidate_label: Optional[str],
        confidence: float,
    ) -> tuple:
        """
        Call on each frame where detection or classification indicates
        a non-banknote or uncertain result.

        Returns:
            (True,  "RANDOM_001") — new review event created, caller should log it
            (False, "RANDOM_001") — same event ongoing, caller should display ID but not re-log
            (False, None)         — reason is not review-triggering (empty box etc.)
        """
        if reason in NON_REVIEW_REASONS or reason not in REVIEW_TRIGGERING_REASONS:
            return False, None

        if self._state is None:
            # IDLE → ACTIVE: create new review event
            review_id, seq = self._allocate_id()
            self._state = ReviewState(
                review_id=review_id,
                review_seq=seq,
                reason=reason,
                candidate_label=candidate_label,
                confidence=confidence,
                frame_count=1,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._persist_event(review_id, seq, reason, candidate_label, confidence)
            return True, review_id

        # ACTIVE: same or similar object?
        if self._is_same_object(reason):
            self._state.frame_count += 1
            return False, self._state.review_id

        # Reason changed significantly → new review event
        if self._state.frame_count >= self.cooldown_frames:
            review_id, seq = self._allocate_id()
            self._state = ReviewState(
                review_id=review_id,
                review_seq=seq,
                reason=reason,
                candidate_label=candidate_label,
                confidence=confidence,
                frame_count=1,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._persist_event(review_id, seq, reason, candidate_label, confidence)
            return True, review_id

        # Still within cooldown — treat as same event
        self._state.frame_count += 1
        return False, self._state.review_id

    def reset(self):
        """
        Reset to IDLE. Call when:
        - Empty box detected for enough frames
        - A donation is accepted (note removed)
        - Manual tracker reset
        """
        self._state = None

    def get_current_review_id(self) -> Optional[str]:
        """Returns the active review ID, or None if idle."""
        return self._state.review_id if self._state else None

    def is_active(self) -> bool:
        return self._state is not None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _is_same_object(self, reason: str) -> bool:
        """
        Heuristic: if the new reason is in the same category as the current
        state reason, treat it as the same physical object.
        """
        if self._state is None:
            return False
        # Any non-banknote detection → same object category
        current_is_nonbanknote = self._state.reason in REVIEW_TRIGGERING_REASONS
        new_is_nonbanknote = reason in REVIEW_TRIGGERING_REASONS
        return current_is_nonbanknote and new_is_nonbanknote

    def _allocate_id(self) -> tuple:
        """Allocate the next review ID (from DB or in-memory fallback)."""
        if self._db_available and self._db_file is not None:
            try:
                from sqlite.review_database import allocate_review_id
                return allocate_review_id(self._db_file)
            except Exception as e:
                print(f"[ReviewTracker] DB allocate failed ({e}), using memory counter.")

        # In-memory fallback
        seq = self._mem_counter
        self._mem_counter += 1
        return f"RANDOM_{seq:03d}", seq

    def _persist_event(
        self,
        review_id: str,
        seq: int,
        reason: str,
        candidate_label: Optional[str],
        confidence: float,
    ):
        """Write the review event to the DB (best-effort)."""
        if self._db_available and self._db_file is not None:
            try:
                from sqlite.review_database import insert_review_event
                insert_review_event(
                    review_id=review_id,
                    review_seq=seq,
                    pocket_id=self.pocket_id,
                    reason=reason,
                    candidate_label=candidate_label,
                    classifier_confidence=confidence if confidence > 0 else None,
                    detector_stage=None,
                    user_message=None,
                    db_file=self._db_file,
                )
            except Exception as e:
                print(f"[ReviewTracker] DB persist failed ({e}).")

