# DivyaKosh System Architecture & Workflow Specification

This document details the chronological software workflow of the **DivyaKosh** banknote recognition and donation verification system, based strictly on the inspected implementation in `CV_model/`.

---

## 1. Application Startup & Initialization

### Entry Point & Launch Command
The canonical production entry point is [`CV_model/scripts/live_camera.py`](file:///C:/Users/aryan/OneDrive/Documents/PROJECTS/Divyakosh-final/CV_model/scripts/live_camera.py).

**Production Command:**
```powershell
.\.venv\Scripts\python.exe CV_model\scripts\live_camera.py --yolo_pipeline --debug
```

### Command-Line Arguments & Default Values
Extracted directly from `live_camera.py:610-637`:

| CLI Flag | Type | Default Value | Description / Function |
| :--- | :--- | :--- | :--- |
| `--yolo_pipeline` | `bool` | `False` (flag) | Activates the two-stage YOLO-first one-shot banknote detector. When omitted, runs the legacy 5-stage heuristic loop. |
| `--timeout` | `float` | `7.0` | Inactivity timeout in seconds for banknote scanning. Measured continuously from scan start. |
| `--yolo_window_seconds`| `float` | `None` | Deprecated alias for `--timeout` (honored for backward compatibility). |
| `--camera` | `int` | `0` | Camera device index (`0` = default laptop/USB camera, `1` = secondary). |
| `--checkpoint` | `str` | `CV_model/models/mobilenet_v3_clean.pt` | Path to trained MobileNetV3-Large 6-class model. |
| `--yolo_checkpoint` | `str` | `CV_model/models/yolo26n_robust/train_seed42_20_20261101_203031/weights/best.pt` | Path to trained YOLO26n robust banknote detector. |
| `--accept_confidence` | `float` | `0.80` | Minimum classifier softmax probability required for an `ACCEPT` decision. |
| `--review_confidence` | `float` | `0.45` | Lower threshold below which an object is flagged for manual review (`REVIEW`). |
| `--yolo_confidence` | `float` | `0.25` | Minimum bounding-box confidence for YOLO detection. |
| `--sharpness_thresh` | `float` | `15.0` | Laplacian variance threshold to reject blurred crops. |
| `--consensus_frames` | `int` | `3` | Frames required for multi-frame consensus in heuristic mode (ignored in YOLO one-shot mode). |
| `--continuous` | `bool` | `False` (flag) | Continuous scanning mode. **Intentionally ignored** by `--yolo_pipeline` (YOLO mode is strictly one-shot). |
| `--debug` | `bool` | `False` (flag) | Enables per-frame console telemetry and visual diagnostic overlays on the OpenCV window. |
| `--box_roi` | `str` | `""` | Comma-separated calibrated cavity coordinates `x,y,w,h` (e.g. `150,90,720,420`). |
| `--log_path` | `str` | `CV_model/logs/inference.jsonl` | Append-only JSONL log path for confirmed donations. |
| `--review_log` | `str` | `CV_model/logs/review_events.jsonl` | Append-only JSONL log path for review events. |

### Module Loading & Device Selection Sequence
1. **Device Selection (`live_camera.py:640-647`):**
   Prioritizes Apple Silicon MPS (`torch.backends.mps.is_available()`) > NVIDIA CUDA (`torch.cuda.is_available()`) > CPU (`torch.device("cpu")`). On standard Windows/Linux Intel hardware without discrete NVIDIA GPUs, executes on `cpu`.
2. **MobileNetV3 Loading (`live_camera.py:118-134`):**
   Calls `load_model(checkpoint_path, device)`. Reads checkpoint dictionary (`class_names`, `model_state_dict`). Instantiates `models.mobilenet_v3_large(weights=None)`, alters classifier linear head:
   ```python
   in_features = model.classifier[-1].in_features
   model.classifier[-1] = nn.Linear(in_features, len(class_names))
   model.load_state_dict(ckpt["model_state_dict"])
   model.eval()
   ```
3. **YOLO Model Loading (`live_camera.py:136-146`):**
   If `--yolo_pipeline` is active, imports `from ultralytics import YOLO` and calls `load_yolo_model(yolo_checkpoint)`.
4. **Camera Initialization (`live_camera.py:657-662`):**
   Executes `cap = cv2.VideoCapture(args.camera)`. If `not cap.isOpened()`, prints an explicit error message and aborts startup immediately without resource leakage.
5. **Database Initialization (`live_camera.py:80-90`, `live_camera.py:669`):**
   Imports `sqlite.importer.import_single_record` and `sqlite.review_database.REVIEW_DB_FILE`. Initializes `ReviewTracker` pointing to `CV_model/sqlite/review_events.db`.
6. **GUI Window Setup (`live_camera.py:682-685`):**
   Creates an OpenCV window `"DaanDristi - Live Demo"` resized to 1280×720 at position (50, 50).

---

## 2. Banknote Detection & Camera Capture Lifecycle

### Camera Capture Mechanics
* **Supported Capture Resolution:** Depends on the connected hardware. Standard USB webcams negotiate natively at 1920×1080 (1080p Full HD) or 1280×720 (720p HD). OpenCV captures 3-channel BGR frames (`uint8`).
* **Hardware Trigger Status:** **None present in software.** Currently, scanning begins immediately upon Python process execution. There is no GPIO pin read, optical beam sensor, limit switch, or button listener implemented.
* **Camera Release Guarantee:** The main execution block is wrapped in a `try ... finally` clause (`live_camera.py:847-850`) ensuring `cap.release()` and `cv2.destroyAllWindows()` are executed on every exit path (successful donation, timeout, exception, or pressing `q`).

### 7.0-Second Inactivity Timeout
Implemented in `run_yolo_detection_window()` (`live_camera.py:209-260`):
* **Start Time Anchor:** `scan_start = time.monotonic()`.
* **Deadline:** `deadline = scan_start + window_seconds`.
* **Measurement Rule:** The deadline is computed **once** at scan initialization. It is **never reset per frame**.
* **Timeout Trigger:** If `time.monotonic() >= deadline` before a banknote is found:
  1. The loop terminates immediately.
  2. `review_tracker.should_create_event("not_a_banknote", None, 0.0)` is queried.
  3. A `REVIEW` record is logged to `CV_model/logs/review_events.jsonl` and `CV_model/sqlite/review_events.db`.
  4. Returns `(last_frame, None, None)`.
  5. The caller prints `No banknote detected within 7 seconds. Camera stopped.` and exits.

### Banknote Bounding Box Selection & Cropping
* **Inference per Frame:** Each frame is evaluated via `yolo_model.predict(source=frame, conf=0.25, verbose=False)`.
* **Selection Logic (`select_yolo_detection()`, `live_camera.py:164-178`):**
  Iterates over all predicted bounding boxes:
  - Discards boxes with confidence `< 0.25`.
  - Discards boxes with class IDs not in range `0..5`.
  - Clips and validates coordinates against the frame boundary via `clip_yolo_bbox()`.
  - If multiple valid banknotes are present in the frame, **selects the single candidate with the highest detection confidence**:
    ```python
    return max(candidates, key=lambda item: item[0])
    ```
* **Coordinate Clipping (`clip_yolo_bbox()`, `live_camera.py:148-161`):**
  Rejects coordinates containing `NaN`, `Inf`, or where `x2 <= x1` or `y2 <= y1`. Clamps `x1, x2` to `[0, frame_width]` and `y1, y2` to `[0, frame_height]`. Converts coordinates to `(x, y, w, h)`.
* **Stopping the Camera:** As soon as one valid banknote detection is obtained, `run_yolo_detection_window()` returns `(frame, (x, y, w, h), class_id)`. **Scanning stops immediately.** The camera loop does not continue.

---

## 3. Machine Learning Models Specification

### Model 1: YOLO26n Robust Banknote Detector
* **Model Checkpoint:** `CV_model/models/yolo26n_robust/train_seed42_20_20261101_203031/weights/best.pt`
* **Architecture:** Ultralytics YOLO26n (nano variant, single-stage anchor-free detector, 15.7 MB).
* **Supported Classes (6 classes):**
  ```python
  {0: '10 INR', 1: '20 INR', 2: '50 INR', 3: '100 INR', 4: '200 INR', 5: '500 INR'}
  ```
* **Runtime Role:** Detects the spatial location and presence of a banknote inside the camera field of view. Gated by `--yolo_confidence 0.25`.
* **Fundamental Limitation:** The YOLO model performs **visual object localization**. It confirms that an object matching the visual geometry and texture of an Indian currency note is present in the donation cavity. **It does NOT authenticate anti-counterfeiting features (watermarks, UV threads, micro-lettering, or magnetic ink).**

### Model 2: MobileNetV3-Large Denomination Classifier
* **Model Checkpoint:** `CV_model/models/mobilenet_v3_clean.pt`
* **Architecture:** Torchvision `models.mobilenet_v3_large` with custom linear classification head:
  `Linear(in_features=960, out_features=6)`. Checkpoint trained on leak-free dataset `data_clean/` (Val Acc: 93.26%, 17.0 MB).
* **Supported Classes (6 classes, sorted alphabetically):**
  `['10', '100', '20', '200', '50', '500']`
* **Preprocessing & Transform (`EVAL_TRANSFORM`, `src/validator.py:73-82`):**
  1. Input crop extracted from raw BGR frame: `crop = frame[y:y+h, x:x+w]`.
  2. Converted from OpenCV BGR to PIL RGB: `Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))`.
  3. Resized to fixed input tensor dimensions: `224 × 224`.
  4. Converted to tensor (`[0.0, 1.0]` float32).
  5. Normalized with ImageNet mean and standard deviation:
     `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
* **Output Classification:** Logits passed through Softmax: `probabilities = torch.softmax(logits, dim=1)`.
* **Classification Function:** [`crop_and_classify()`](file:///C:/Users/aryan/OneDrive/Documents/PROJECTS/Divyakosh-final/CV_model/src/validator.py#L592-L658).

---

## 4. Decision Rules & Validation Logic

Once MobileNetV3 computes softmax probabilities across the 6 classes, the result is evaluated by `crop_and_classify()` in `src/validator.py`:

```
Softmax Probability (top_prob)
  │
  ├── top_prob >= 0.80 (accept_confidence)
  │     │
  │     └── Outcome: ACCEPT
  │           Target Bin: BIN_<denomination> (e.g. BIN_500)
  │           Action: Confirmed donation → SQLite + JSONL + Blockchain
  │
  ├── 0.45 <= top_prob < 0.80 (review_confidence)
  │     │
  │     └── Outcome: REVIEW
  │           Reason: "low_confidence"
  │           Target Bin: REVIEW_BIN
  │           Action: Sequential Review Event (RANDOM_###) → SQLite + JSONL
  │
  └── top_prob < 0.45
        │
        └── Outcome: REJECT / REVIEW
              Reason: "unsupported_object_rejected"
              Target Bin: REVIEW_BIN
              Action: Sequential Review Event (RANDOM_###) → SQLite + JSONL
```

### Additional Quality Heuristics (Available in Validator Engine)
Defined in `src/validator.py` (and active during frame validation):
1. **Empty Crop Guard:** If `crop.size == 0`, immediately triggers `REVIEW` with reason `"low_confidence"` (`live_camera.py:534-542`).
2. **Sharpness Quality Gate:** Laplacian variance computed on grayscale note crop:
   `sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()`. If `sharpness < 15.0`, rejects frame as blurred.
3. **Skin & Hand Occlusion Filter:** HSV thresholding (`H: [0, 20], S: [48, 255], V: [80, 255]`). If skin pixel ratio `> 0.25`, triggers `REVIEW` with reason `"hand_or_skin_detected"`.
4. **Aspect Ratio Sanity Check:** Standard Indian notes have aspect ratios between 1.8 and 2.3. Ratios outside `[1.3, 2.8]` trigger `REVIEW` with reason `"abnormal_aspect_ratio"`.

---

## 5. Donation & Review State Transition Matrix

| Trigger / Event | Executing Function | State Transition | Generated Data / Fields | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| **User launches CLI** | `main()` (`live_camera.py:610`) | `OFFLINE` → `SCANNING` | CLI arguments parsed; models loaded into RAM/VRAM. | Camera opens, OpenCV window appears, 7s countdown starts. |
| **YOLO detects note** | `run_yolo_detection_window()` (`live_camera.py:244`) | `SCANNING` → `CLASSIFYING` | `note_bbox = (x, y, w, h)`, `class_id = 5`, `confidence = 0.89`. | Scanning stops immediately; camera freeze; note cropped. |
| **Inactivity timeout (7s)** | `run_yolo_detection_window()` (`live_camera.py:222`) | `SCANNING` → `TIMEOUT_REVIEW` | Review ID: `RANDOM_001`, reason: `"not_a_banknote"`, stage: `"YOLO_DETECTION"`. | Review event logged to SQLite/JSONL; camera released; app exits. |
| **MobileNet conf >= 0.80** | `crop_and_classify()` (`validator.py:656`) | `CLASSIFYING` → `ACCEPT` | `decision: "ACCEPT"`, `label: "500"`, `confidence: 0.9994`, `target_bin: "BIN_500"`. | `ReviewTracker.reset()`; donation written to JSONL & SQLite; proof posted to API. |
| **MobileNet conf < 0.80** | `crop_and_classify()` (`validator.py:619`) | `CLASSIFYING` → `REVIEW` | `decision: "REVIEW"`, `label: "50"`, `confidence: 0.58`, `reason: "low_confidence"`. | New `RANDOM_###` allocated; review event written to JSONL & SQLite. |
| **Empty or corrupt crop** | `run_yolo_pipeline()` (`live_camera.py:534`) | `CLASSIFYING` → `CROP_ERROR` | Review ID: `RANDOM_###`, reason: `"low_confidence"`, stage: `"YOLO_CROP"`. | Review event logged; exits safely without crash. |
| **Camera read error** | `run_yolo_detection_window()` (`live_camera.py:228`) | `SCANNING` → `ERROR` | `RuntimeError("Failed to read frame from camera during YOLO detection.")` | Propagated to caller; camera released; error printed. |
| **Camera open failure** | `main()` (`live_camera.py:658`) | `OFFLINE` → `ERROR` | None. | Console error printed; process returns immediately without GUI. |
| **User presses 'q'** | `run_yolo_detection_window()` (`live_camera.py:257`) | `SCANNING` → `ABORT` | None. | Returns `None`; camera released; graceful exit. |

---

## 6. Local Persistence & Audit Logging Specifications

### Sequence of Operations on Donation Confirmation (`ACCEPT`)
When a banknote is accepted, operations occur in strict sequence (`live_camera.py:570-581`):
1. **JSONL Audit Log Write:** Appends record to `CV_model/logs/inference.jsonl`.
2. **SQLite Database Write:** Calls `import_single_record()` to persist in `CV_model/sqlite/donation_box.db`.
3. **Blockchain API Submission:** Calls `register_blockchain_proof()` to send proof to `http://localhost:3000/register-proof`.

```
Classification ACCEPT
        │
        ▼
1. Append to inference.jsonl  (Guaranteed local disk write)
        │
        ▼
2. Insert to donation_box.db   (Atomic SQLite transaction)
        │
        ▼
3. POST to /register-proof     (HTTP call with offline fallback)
```

> [!IMPORTANT]
> **Transactional Boundary:** These three steps are **not** wrapped in a single distributed transaction. If the JSONL append succeeds and SQLite insert succeeds, but the blockchain API fails (e.g., node offline), **the donation remains fully recorded locally**. The blockchain error is caught (`requests.RequestException`), a warning is printed, and execution continues without crashing.

### SQLite Database 1: `CV_model/sqlite/donation_box.db`
Managed by `CV_model/sqlite/database.py` and `CV_model/sqlite/importer.py`.

#### Schema:
```sql
CREATE TABLE IF NOT EXISTS donations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    donation_id TEXT UNIQUE NOT NULL,
    pocket_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    item_status TEXT,
    denomination INTEGER NOT NULL,
    classifier_confidence REAL,
    confidence REAL,
    source TEXT,
    decision TEXT,
    target_bin TEXT,
    model_version TEXT,
    raw_record TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_donations_timestamp ON donations(timestamp);
CREATE INDEX IF NOT EXISTS idx_donations_denomination ON donations(denomination);
CREATE INDEX IF NOT EXISTS idx_donations_source ON donations(source);
```

#### Deterministic Idempotency Key (`donation_id`):
Generated by `make_donation_id()` in `sqlite/importer.py:32-47`:
```python
canonical_string = f"{timestamp}|{pocket_id}|{denomination}"
donation_id = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()
```
Because `donation_id` has a `UNIQUE` constraint, duplicate submissions of the exact same event are ignored (`INSERT ... ON CONFLICT DO NOTHING`), preventing duplicate database records.

### SQLite Database 2: `CV_model/sqlite/review_events.db`
Managed by `CV_model/sqlite/review_database.py`.

#### Schema:
```sql
CREATE TABLE IF NOT EXISTS review_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE NOT NULL,
    review_seq INTEGER NOT NULL,
    pocket_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    reason TEXT NOT NULL,
    candidate_label TEXT,
    classifier_confidence REAL,
    detector_stage TEXT,
    user_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_counter (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_seq INTEGER NOT NULL DEFAULT 0
);
```

#### Sequential ID Generator (`allocate_review_id()`):
Maintains an atomic counter in `review_counter`. Generates sequential human-readable strings formatted as `RANDOM_001`, `RANDOM_002`, `RANDOM_003`. These IDs persist across application restarts.

### JSONL Audit Log Records

#### Format 1: `CV_model/logs/inference.jsonl` (Confirmed Donations)
```json
{"pocket_id":"webcam-0","timestamp":"2026-11-02T12:45:00.123456+00:00","item_status":"NOTE_DETECTED","predicted_denomination":"500","classifier_confidence":0.9994,"confidence":0.9994,"source":"webcam","decision":"ACCEPT","target_bin":"BIN_500","model_version":"mobilenet_v3_clean_v1"}
```

#### Format 2: `CV_model/logs/review_events.jsonl` (Review / Timeout Events)
```json
{"event_type":"REVIEW","review_id":"RANDOM_007","pocket_id":"webcam-0","timestamp":"2026-11-02T12:45:10.654321+00:00","reason":"not_a_banknote","candidate_label":null,"classifier_confidence":0.0,"stage":"YOLO_DETECTION","user_message":"No banknote detected during the detection window."}
```

