"""
scripts/note_count_detector.py

Note-presence and note-count detector for the donation box ROI.

When a trained YOLO checkpoint (e.g. `models/yolo_note_detector.pt`) and `ultralytics`
are present, this class uses YOLO inference.

When YOLO weights or `ultralytics` are unavailable:
- Reports `is_operational = False` (never claims deep-learning detection is active).
- Provides a defensible computer-vision fallback using contour saliency and aspect ratio
  to estimate whether 0, 1, or multiple objects are present.
- Never crashes, fabricates weights, or fakes detection.
"""

import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class NoteCountDetector:
    def __init__(self, model_path="models/yolo_note_detector.pt", conf_thresh=0.5):
        """
        Initialize the note count detector.
        
        Args:
            model_path (str): Path to trained YOLO checkpoint.
            conf_thresh (float): Minimum confidence threshold.
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.model = None
        self.is_operational = False

        if not ULTRALYTICS_AVAILABLE:
            print("Notice: 'ultralytics' library is not installed. YOLO note detection disabled; fallback mode active.")
            return

        if not os.path.exists(self.model_path):
            print(f"Notice: YOLO model checkpoint not found at '{self.model_path}'. Fallback mode active.")
            return

        try:
            self.model = YOLO(self.model_path)
            self.is_operational = True
            print(f"Loaded YOLO note detector from: {self.model_path}")
        except Exception as err:
            print(f"Warning: Failed to load YOLO checkpoint ({err}). Fallback mode active.")
            self.model = None
            self.is_operational = False

    def detect_note_count(self, roi_bgr):
        """
        Detects the number of currency notes in the given ROI.

        Returns:
            dict: {
                "count": int,
                "status": "EMPTY" | "SINGLE" | "MULTIPLE",
                "detections": list,
                "detector_type": "yolo" | "cv_fallback" | "uninitialized"
            }
        """
        result_dict = {
            "count": 0,
            "status": "EMPTY",
            "detections": [],
            "detector_type": "uninitialized",
            "operational": self.is_operational,
        }

        if roi_bgr is None or roi_bgr.size == 0:
            return result_dict

        # Path 1: Trained YOLO is available and operational
        if self.model is not None and self.is_operational:
            try:
                results = self.model.predict(source=roi_bgr, conf=self.conf_thresh, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf
                        })

                count = len(detections)
                status = "EMPTY" if count == 0 else ("SINGLE" if count == 1 else "MULTIPLE")
                result_dict["count"] = count
                result_dict["status"] = status
                result_dict["detections"] = detections
                result_dict["detector_type"] = "yolo"
                return result_dict
            except Exception as err:
                print(f"Warning: YOLO prediction error ({err}), falling back to CV detection.")

        # Path 2: CV Saliency & Contour fallback (safe, honest fallback)
        result_dict["detector_type"] = "cv_fallback"
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        total_area = float(h * w)

        # Foreground separation against dark box
        median_lum = float(np.median(gray))
        _, thresh = cv2.threshold(gray, max(50, int(median_lum + 18)), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for c in contours:
            area = cv2.contourArea(c)
            # A valid note must occupy between 4% and 75% of ROI
            if (0.04 * total_area) <= area <= (0.75 * total_area):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                if 1.15 <= aspect <= 3.0:
                    valid_contours.append((c, area, [x, y, x + bw, y + bh]))

        count = len(valid_contours)
        status = "EMPTY" if count == 0 else ("SINGLE" if count == 1 else "MULTIPLE")
        detections = [{"bbox": bbox, "confidence": 0.80} for _, _, bbox in valid_contours]

        result_dict["count"] = count
        result_dict["status"] = status
        result_dict["detections"] = detections
        return result_dict


if __name__ == "__main__":
    detector = NoteCountDetector()
    print(f"Detector initialized. Operational: {detector.is_operational}")
