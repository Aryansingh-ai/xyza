# DivyaKosh Integration Test Plan & Acceptance Matrix

This document provides a comprehensive test matrix for verifying the complete DivyaKosh smart donation system across hardware, computer vision, local database storage, and blockchain layers.

---

## 1. Test Tier Overview

Testing is split into two distinct tiers:

1. **Automated Software Test Suite (Already Implemented & Passing):**
   * Can be executed immediately on any development machine without physical hardware or real camera input.
   * `CV_model/tests/test_validation_pipeline.py` (20 unit/integration tests).
   * `CV_model/tests/test_yolo_camera_pipeline.py` (27 unit/mock tests).
   * `CV_model/sqlite/test_sqlite.py` (Database schema & reporting test).
   * `Blockchain/api/tests/envio-integration.test.js` (API & indexing tests).
   * `Blockchain/api/tamper-test.js` (Cryptographic tamper-detection verification).

2. **Physical Hardware & End-to-End Tests (To Be Executed by Integrator):**
   * Requires actual camera, lighting enclosure, physical banknotes, and hardware actuators.

---

## 2. Complete Integration Test Matrix

| Test ID | Category | Preconditions | Test Action | Expected ML / Software Result | Expected Local DB / Log Result | Expected Blockchain Result | Expected Hardware Result | Pass / Fail Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-01** | **Normal Scan (₹500 Note)** | System idle; camera active; API running. | Insert clean genuine ₹500 note into cavity; start scan. | YOLO detects note in <1.5s; MobileNet classifies `label="500"`, `conf >= 0.80`, `decision="ACCEPT"`. | Record appended to `inference.jsonl`; inserted into `donation_box.db` (`donations`). | Proof hash sent to `/register-proof`; tx mined on Monad; `txHash` returned. | Solenoid energizes for 500 ms; note drops into vault; LED turns Green. | Check SQLite: `SELECT * FROM donations WHERE denomination=500` returns new row. |
| **TC-02** | **Normal Scan (₹100 Note)** | System idle; camera active; API running. | Insert clean ₹100 note into cavity. | YOLO detects note; MobileNet classifies `label="100"`, `conf >= 0.80`, `decision="ACCEPT"`. | Record appended to `inference.jsonl`; inserted into `donation_box.db`. | Tx mined on Monad; `status="canonical-v1"`. | Solenoid actuates vault; note accepted; LED Green. | Row in `donations` with `denomination=100`. |
| **TC-03** | **Normal Scan (₹50 Note)** | System idle; camera active; API running. | Insert clean ₹50 note into cavity. | YOLO detects note; MobileNet classifies `label="50"`, `conf >= 0.80`, `decision="ACCEPT"`. | Record appended to `inference.jsonl`; inserted into `donation_box.db`. | Tx mined on Monad. | Solenoid actuates vault; note accepted; LED Green. | Row in `donations` with `denomination=50`. |
| **TC-04** | **Normal Scan (₹20, ₹10, ₹200)** | System idle; camera active; API running. | Repeat for remaining denominations individually. | Correct class predicted with `conf >= 0.80`; decision `ACCEPT`. | Each inserted into `donations` table. | Each anchored on-chain. | Accepted into cash vault. | Query database totals via `python sqlite/test_sqlite.py`. |
| **TC-05** | **Confusion Pair (₹500 vs ₹50)** | Lighting nominal. | Insert crumpled or tilted ₹500 note. | Cross-entropy logits clearly separate ₹500 from ₹50; if ambiguous, `conf < 0.80`. | If `conf < 0.80`, logged to `review_events.db` as `low_confidence`. | No blockchain tx submitted if `REVIEW`. | Diverter gate sends note to Review Chute; LED Amber. | Check `review_events.db`: `RANDOM_###` created with `candidate_label="500"`. |
| **TC-06** | **Inactivity Timeout (No Note)** | Empty box. | Start scan; place nothing in the cavity. | Timer runs for 7.0s; YOLO finds 0 boxes; stops at 7.0s. | Single review record written to `review_events.jsonl` & `review_events.db` (`reason="not_a_banknote"`). | No blockchain call. | Camera released; LED turns off; no mechanical movement. | Console prints: `No banknote detected within 7 seconds. Camera stopped.` |
| **TC-07** | **Foreign Object (Phone / Keys)** | System idle; camera active. | Place metal keys or smartphone in cavity; scan. | YOLO detects no banknote or aspect ratio fails (`aspect_ratio outside [1.3, 2.8]`); `decision="REVIEW"`. | Logged to `review_events.db` (`reason="not_a_banknote"` or `"abnormal_aspect_ratio"`). | No blockchain call. | Gate returns object to donor tray; buzzer beeps twice. | Review ID allocated; no change to `donations` table. |
| **TC-08** | **Hand / Fingers in Cavity** | System idle; camera active. | Place hand inside slot while holding a note. | Skin filter trips (`skin_ratio > 0.25`); `decision="REVIEW"`, `reason="hand_or_skin_detected"`. | Logged to `review_events.db`. | No blockchain call. | Gate returns note; display shows `"Hand detected. Remove hand."` | Check console/logs for `reason: hand_or_skin_detected`. |
| **TC-09** | **Camera Disconnected at Boot** | USB camera unplugged. | Run `live_camera.py --camera 0`. | Script detects `not cap.isOpened()`. | No DB or log records modified. | None. | Hardware LED blinks Red; host process exits cleanly. | Console output: `ERROR: Could not open camera index 0.` |
| **TC-10** | **Camera Unplugged Mid-Scan** | Camera active mid-scan. | Disconnect USB camera cable during 7.0s window. | `cap.read()` returns `False`; raises `RuntimeError("Failed to read frame...")`. | Handled in `try...finally`; camera released. | None. | Watchdog on MCU times out; gate stays closed; note returned. | Process terminates without unhandled crash or zombie camera lock. |
| **TC-11** | **Missing Model Checkpoint** | Rename `models/mobilenet_v3_clean.pt`. | Start `live_camera.py`. | Python raises `FileNotFoundError`. | None. | None. | System fails at boot; does not accept notes. | Clean error traceback specifying missing `.pt` file. |
| **TC-12** | **Offline Mode (Blockchain Down)** | API server stopped (`node index.js` killed). | Insert valid ₹100 note; scan. | `decision="ACCEPT"`. | Donation **successfully committed** to `inference.jsonl` and `donation_box.db`. | `requests.post` catches connection error; logs warning; skips Monad. | Solenoid drops note into vault (**Safe because local DB committed**). | Console prints: `Blockchain warning: registration failed (...). ML record remains safe in JSONL/SQLite.` |
| **TC-13** | **RPC Network Timeout** | API running, but internet/Monad RPC blocked. | Insert valid ₹500 note; scan. | `decision="ACCEPT"`. | Inserted into local SQLite. | API returns HTTP 500 (RPC timeout); Python catches safely. | Note accepted into vault; logged for future synchronization. | Local record safe; API test log records RPC retry timeout. |
| **TC-14** | **Tamper Detection Test** | 5 donations existing in SQLite. | Manually edit SQLite row 1: change denomination from `500` to `10`. | N/A (Offline attack). | Modified SQLite record. | Verification: `node Blockchain/api/tamper-test.js`. | N/A. | Script alerts: `TAMPER DETECTED: Hash mismatch between on-chain proof and SQLite row.` |
| **TC-15** | **Idempotency / Replay Guard** | Record already exists in SQLite. | Re-send same donation record to `import_single_record()`. | N/A. | SQLite `UNIQUE(donation_id)` triggers `ON CONFLICT DO NOTHING`; `inserted=0`. | If re-sent to API, Monad contract reverts with `"Error: Proof already exists onchain"`. | Actuator ignores duplicate signal. | `import_single_record` returns `inserted=0, skipped_duplicate=1`. |
| **TC-16** | **Power Loss Mid-Transaction** | Scan active. | Cut power immediately after classification. | Interrupted. | SQLite WAL journal automatically recovers uncommitted data on reboot. | Incomplete tx dropped. | Solenoid de-energizes; note drops into return tray. | Database integrity check: `PRAGMA integrity_check` returns `ok`. |

---

## 3. How to Run Automated Verification Commands

### 1. Execute All YOLO Camera Pipeline Tests (27 tests)
```powershell
.\.venv\Scripts\python.exe -m unittest CV_model/tests/test_yolo_camera_pipeline.py -v
```
*Verifies: Bounding box clipping, confidence thresholds, 7.0s timeout deadline measurement, one-shot stop on first note, camera release on all exit paths.*

### 2. Execute Full Validation Pipeline Tests (20 tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s CV_model/tests -p "test_validation_pipeline.py" -v
```
*Verifies: 6-class denomination recognition, confusion pairs, cardboard rejection, empty box detection, skin filter, partial occlusion, SQLite storage, and blockchain offline resilience.*

### 3. Verify SQLite Database & Query Engine
```powershell
.\.venv\Scripts\python.exe CV_model/sqlite/test_sqlite.py
```
*Verifies: Reads `inference.jsonl`, imports records into `donation_box.db`, checks duplicate deduplication, and reports total donation amounts and per-denomination counts.*

### 4. Verify Blockchain Tamper-Detection Engine
```powershell
cd Blockchain\api
node tamper-test.js
```
*Verifies: Computes proof hashes, checks Monad smart contract verification, tests intentional modification of donation fields, and confirms fraud alerts trigger properly.*

---

## 4. Safe Testing Procedure for Hardware Integrators

When setting up and testing the hardware rig, follow this protocol to avoid unnecessary gas spending or corrupting production data:

1. **Use Offline Mode for Initial Mechanical Tuning:**
   * Run `live_camera.py --yolo_pipeline --debug` **without** starting the Node.js API server.
   * This verifies camera focus, illumination, bounding-box cropping, and solenoid timing without sending transactions.
2. **Use Testnet Faucet for On-Chain Testing:**
   * Transactions use Monad Testnet `MON` tokens (free of cost from the testnet faucet).
   * Ensure `Blockchain/api/.env` points to `https://testnet-rpc.monad.xyz` and the deployed testnet contract `0xBCD966873f48563643Bde79760eEC98a567fd7e6`.
3. **Database Backup:**
   * Before running large physical batch tests, backup the SQLite files:
     ```powershell
     Copy-Item CV_model\sqlite\donation_box.db CV_model\sqlite\donation_box.db.bak
     Copy-Item CV_model\sqlite\review_events.db CV_model\sqlite\review_events.db.bak
     ```

