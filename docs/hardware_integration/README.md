# DivyaKosh Technical Handoff: Hardware, ML & Blockchain Architecture

Welcome to the technical handoff documentation for **DivyaKosh (DaanDristi)** — the AI-verified, tamper-evident smart donation box system.

This documentation suite is prepared specifically for teammates responsible for integrating the physical hardware (donation cavity, mechanical actuators, note slots, sensors, microcontroller/SBC) with the core software stack (computer vision pipeline, local database persistence, and blockchain provenance registry).

---

## Documentation Index

| Document | Focus & Content |
| :--- | :--- |
| [**`SYSTEM_WORKFLOW.md`**](SYSTEM_WORKFLOW.md) | **End-to-End System Architecture & Chronological Flow.** Detailed startup sequence, camera capture lifecycle, YOLO robust detection, MobileNetV3 classification, validation gating, state transitions, SQLite database schemas, and JSONL audit logging. |
| [**`BLOCKCHAIN_FLOW.md`**](BLOCKCHAIN_FLOW.md) | **Cryptographic Proof & Blockchain Architecture.** Canonical AI event serialization, SHA-256 model hashing, Monad testnet smart contract interaction (`DonationProvenanceRegistry.sol`), Express API endpoints, offline fallback guarantees, and audit verification. |
| [**`HARDWARE_INTEGRATION.md`**](HARDWARE_INTEGRATION.md) | **Hardware Boundary, Pinout & Interface Contract.** What is currently implemented vs missing, the proposed bidirectional hardware handshake protocol, actuator safety interlocks (preventing physical acceptance on unverified ML confidence), and power/fault recovery. |
| [**`INTEGRATION_TEST_PLAN.md`**](INTEGRATION_TEST_PLAN.md) | **Verification Matrix & Acceptance Criteria.** Test cases covering normal donations, denomination confusion, timeouts, camera faults, database failures, blockchain offline modes, tamper detection, and hardware timeouts. |
| [**`diagrams/`**](diagrams/) | **Mermaid Sequence & Architecture Diagrams.** Source `.mmd` files for Overall Architecture, Successful Donation Flow, Review/Timeout Flow, and Failure/Recovery Flow. |

---

## Teammate Quick-Start

### 1. Prerequisites & Environment Setup
* **Operating System:** Windows 10/11 or Linux (Ubuntu 22.04+ / Raspberry Pi OS 64-bit).
* **Python Environment:** Python 3.10 64-bit inside `.venv`.
  ```powershell
  # Windows
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
* **Node.js Environment:** Node.js v18+ or v20+ for Blockchain API and Dashboard.
  ```powershell
  cd Blockchain\api
  npm install
  cd ..\dashboard
  npm install
  ```

### 2. Service Launch Order
To test the complete end-to-end system locally:

#### Step A: Start the Blockchain API Server (Port 3000)
```powershell
cd Blockchain\api
node index.js
```
*Expected output:*
```text
Express server running on port 3000
Connected to contract at 0xBCD966873f48563643Bde79760eEC98a567fd7e6
```
*(Note: If no internet or RPC is available, the CV pipeline will still run safely with offline fallback).*

#### Step B: Launch the Live Camera Detection Pipeline
From the project root:
```powershell
.\.venv\Scripts\python.exe CV_model\scripts\live_camera.py --yolo_pipeline --debug
```
*Key parameters:*
* `--yolo_pipeline`: Enables YOLO one-shot banknote detection.
* `--timeout 7`: 7-second scan window before stopping (default 7.0s).
* `--camera 0`: Camera index (use `1` for external USB camera).
* `--debug`: Prints real-time bounding box coordinates, sharpness, and confidence telemetry.

#### Step C: (Optional) Launch the Audit Web Dashboard (Port 5173)
```powershell
cd Blockchain\dashboard
npm run dev
```

---

## Summary of Findings for Hardware Integrator

1. **What is Already Implemented & Production-Ready:**
   * One-shot camera scanning with automatic 7.0s inactivity timeout and clean exit.
   * Two-stage machine learning pipeline: YOLO26n robust banknote detector + MobileNetV3-Large 6-class denomination classifier (`10`, `20`, `50`, `100`, `200`, `500` INR).
   * Strict validation heuristics: aspect ratio bounds, skin/hand filtering, edge density, and bimodal luminance checks.
   * State management: Review deduplication (`ReviewTracker`) assigning persistent sequential IDs (`RANDOM_001`, `RANDOM_002`...).
   * Idempotent SQLite persistence (`donation_box.db` and `review_events.db`).
   * Blockchain proof generation (`CV_model/src/blockchain.py`) anchored to Monad Testnet (`DonationProvenanceRegistry.sol`) with graceful offline fallback.

2. **What Still Needs Hardware Work (Critical Integration Gaps):**
   * **Physical Note Trigger:** The software currently starts scanning immediately when the process is invoked. A physical trigger (beam-break IR optical sensor, limit switch, or user push button) must trigger the scan.
   * **Physical Note Acceptance/Rejection Actuator:** Currently, software says "ACCEPT" or "REVIEW", but no motor, solenoid, or dual-bin gate exists in code to route the physical note into the confirmed cash box or return/review slot.
   * **Hardware-to-Software Bridge:** A serial (UART/USB) or GPIO handshake module must be added to connect a microcontroller (Arduino / RP2040 / ESP32) or SBC pins to `live_camera.py`.
   * **Safety Interlock:** Physical acceptance must **never** be triggered by raw ML confidence alone; it must await confirmed local SQLite database insertion.

3. **Top 5 Integration Risks:**
   1. **Physical Jam / Premature Drop:** A donor releases a note into the cavity, but the camera times out or gets occluded before classification completes.
   2. **Lighting Fluctuations in Cavity:** Ambient light leaking into the black donation box can alter the dark-background luminance baseline required by Stage A box-presence detection.
   3. **Lack of Hardware Watchdog:** If the Python camera process hangs on OpenCV frame grab, the hardware must be able to timeout and safely return the note.
   4. **Camera Focus & Depth-of-Field:** A fixed-focus USB camera placed too close (<10 cm) to the banknote slot will produce blurry crops that fail the Laplacian sharpness gate (`min_sharpness=15.0`).
   5. **Offline Proof Queue:** In offline field scenarios, donations are persisted in SQLite, but transactions are skipped. A background synchronization worker is needed to post unanchored records once connectivity returns.

