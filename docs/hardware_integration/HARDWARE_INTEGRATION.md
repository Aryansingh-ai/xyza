# DivyaKosh Physical Hardware Integration Specification

This document defines the integration boundary between the **physical donation box hardware** (sensors, motors, solenoids, microcontrollers, indicators) and the **software processing stack** (computer vision, SQLite database, and blockchain API).

---

## 1. Hardware Interface Status & Gap Analysis

Based on an exhaustive audit of the `Divyakosh-final` repository, **no hardware driver code, GPIO pin manipulation, serial communication, or actuator control currently exists in the codebase.**

The table below delineates the software boundary as implemented versus the hardware requirements:

| Hardware / Interface | Existing Software Implementation | Required Signal / Data from Hardware | Expected Software Response | Implementation Status | Action Required from Hardware Team |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Camera Feed** | `cv2.VideoCapture(args.camera)` in `live_camera.py:657`. | Standard UVC USB video stream (1080p or 720p, >=15 FPS). | Opens video device; captures BGR frames; runs YOLO and MobileNet pipelines. | **IMPLEMENTED** | Mount USB camera rigidly 12–18 cm above note inspection stage with diffuse 5000K LED illumination. |
| **Note Insertion Sensor** | *None.* Software starts scanning immediately when `live_camera.py` starts. | Active-LOW digital signal (beam-break IR sensor, reflective optical gate, or roller limit switch). | Triggers camera scan window (initiates 7.0s timer). | **MISSING (GAP)** | Install IR beam-break sensor across note insertion slot; connect to microcontroller GPIO pin. |
| **Scan Trigger / Buttons** | *None.* CLI execution only. | Digital push button (e.g. "START / SCAN" button). | Launches detection window or resets tracker. | **MISSING (GAP)** | Add optional manual button on temple box panel for donors. |
| **Status LEDs / Display** | OpenCV GUI window (`cv2.imshow`) on host monitor only. | Serial/I2C/GPIO signals from host: `STATUS_SCANNING`, `STATUS_ACCEPT`, `STATUS_REVIEW`. | Software emits status code or updates local display. | **MISSING (GAP)** | Install RGB LED or I2C character display (16×2 LCD or 0.96" OLED) visible to donor. |
| **Accept / Review Gate** | *None.* Decision strings printed to console only. | Motor/solenoid driver signal: `ACTUATE_ACCEPT` (drop into cash vault) vs `ACTUATE_REVIEW` (return to donor / drop into review bin). | Software sends command only **after** SQLite commit confirmation. | **MISSING (GAP)** | Build dual-bin mechanical flapper/gate driven by 12V solenoid or stepper motor. |
| **Microcontroller / SBC Link** | *None.* No `pyserial` or `RPi.GPIO` in `requirements.txt`. | USB-UART (e.g. `/dev/ttyUSB0` or `COM3`) at 115200 baud, 8N1. | Exchanging JSON or ASCII command packets between Python and MCU. | **MISSING (GAP)** | Implement serial communication bridge in Python (`CV_model/src/hardware_bridge.py`). |
| **Power Management** | *None.* | Stable 12V DC input (for actuators) and 5V DC (for SBC/camera). | Safe shutdown on power loss; watchdog timer for hung frames. | **MISSING (GAP)** | Provide buck converters with power-fail sense line to trigger graceful shutdown. |

---

## 2. The Core Safety Interlock Rule

> [!CAUTION]
> **CRITICAL ARCHITECTURAL SAFETY PRINCIPLE:**
> **Physical banknote acceptance must NEVER be triggered by a high ML confidence score alone.**

### The Threat
If physical hardware actuates an acceptance solenoid as soon as MobileNet returns `conf = 0.9994`:
1. If the host PC crashes, loses power, or suffers a disk I/O failure immediately after classification, the note has dropped into the vault, but **no record exists in SQLite, JSONL, or the blockchain**.
2. A donor would have their money taken without receipt or verification.

### The Required Handshake Sequence
Physical banknote routing must strictly obey this chronological interlock:

```
[Physical Insertion]
        │  IR sensor triggers
        ▼
[Software Scan] ──► YOLO Detection ──► MobileNet Classification
                                            │
                                            ▼
                                   Decision == ACCEPT
                                            │
                                            ▼
                                   Local JSONL Appended
                                            │
                                            ▼
                                   SQLite Insert COMMITTED (donation_box.db)
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │ HARDWARE SAFETY GATE RELEASED │
                            └───────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         Host sends 'DROP_ACCEPT'                       Host invokes POST
         command over Serial to MCU                     to Blockchain API
                    │                                               │
                    ▼                                               ▼
         Solenoid opens vault;                       Proof anchored on Monad
         Note drops into cash bin                    (Offline fallback safe)
```

**Rule:** The hardware acceptance command (`DROP_ACCEPT`) is dispatched **only after** `import_single_record()` returns with `inserted >= 1` and the SQLite transaction has committed to disk.

---

## 3. Proposed Hardware Interface Contract (Serial Protocol)

To bridge the existing Python pipeline with the hardware microcontroller (e.g. Arduino Nano, Raspberry Pi Pico, or ESP32), implement a bidirectional serial handshake.

* **Physical Bus:** USB-UART (Microcontroller connected to host via USB).
* **Baud Rate:** `115200 bps`, 8 data bits, no parity, 1 stop bit (`8N1`).
* **Format:** Line-delimited JSON strings (`\n`).

### Microcontroller to Host (Sensors → Python)

#### 1. Note Inserted Event
Sent when the slot's optical beam-break sensor is tripped:
```json
{"event": "NOTE_INSERTED", "slot_id": 1, "timestamp_ms": 14205}
```
*Expected Software Action:* Starts `run_yolo_detection_window(timeout=7.0)`.

#### 2. Note Cancelled / Removed Event
Sent if the donor pulls the note back before scanning finishes:
```json
{"event": "NOTE_REMOVED", "timestamp_ms": 16400}
```
*Expected Software Action:* Aborts scanning immediately; releases camera.

#### 3. Hardware Status / Heartbeat
Sent every 2000 ms to confirm MCU health:
```json
{"event": "HEARTBEAT", "sensors_ok": true, "jam_detected": false, "solenoid_state": "IDLE"}
```

---

### Host to Microcontroller (Python → Actuators & LEDs)

#### 1. Status Update (LED & UI Control)
```json
{"cmd": "SET_STATUS", "state": "SCANNING", "color": "BLUE"}
```
*States:* `IDLE` (Amber/Dim), `SCANNING` (Pulsing Blue), `ACCEPTED` (Solid Green), `REVIEW` (Blinking Amber), `ERROR` (Red).

#### 2. Actuate Acceptance (Drop to Cash Vault)
Sent **only after** SQLite database insertion succeeds:
```json
{"cmd": "DISPENSE", "target": "ACCEPT_BIN", "denomination": 500, "donation_id": "47ab44db..."}
```
*MCU Action:* Energizes vault solenoid for 500 ms; verifies note has cleared via lower beam-break; returns confirmation.

#### 3. Actuate Rejection / Review (Return Note to Donor)
Sent when timeout expires, confidence is low, or an unknown object is detected:
```json
{"cmd": "DISPENSE", "target": "REVIEW_BIN", "reason": "not_a_banknote", "review_id": "RANDOM_007"}
```
*MCU Action:* Reverse rollers or energize diverter gate to return object to customer tray.

---

## 4. Mechanical & Optical Environment Specifications

To ensure the Computer Vision pipeline operates reliably in field deployment:

### 1. Cavity Geometry & Background
* **Interior Lining:** The inspection cavity interior must be lined with **matte black non-reflective material** (black anodized aluminum, matte black acrylic, or dense black flocking sheet).
* **Background Uniformity:** Stage A of `validator.py` (`_check_box_presence`) analyzes dark-pixel ratios. High-gloss black plastic that reflects LED hotspots will trigger false rejections.

### 2. Illumination
* **Color Temperature:** Use 5000K–6500K High-CRI (>90 CRI) white LED strip.
* **Diffuser:** Install milky-white acrylic diffusion film over LEDs to prevent specular glare on polymer threads or glossy new ₹500 notes.
* **Ambient Light Isolation:** The note slot must have a mechanical light baffle (rubber lip or angled chute) so external sunlight or fluorescent room lighting does not flood the camera sensor.

### 3. Camera Mounting & Focus
* **Focal Distance:** Rigidly fix the camera lens at a distance of **140 mm to 180 mm** from the note resting surface.
* **Focus Setting:** Manually adjust the camera focus ring to 160 mm and lock it with a dab of silicone adhesive. **Disable USB camera auto-focus** via OpenCV:
  ```python
  cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
  ```
  Auto-focus hunting during the 7.0-second window causes frame blurring that trips the `sharpness < 15.0` rejection gate.

---

## 5. Fault Recovery & Power Failure Sequence

| Fault Condition | Detection Mechanism | Immediate Hardware Action | Software Recovery Action |
| :--- | :--- | :--- | :--- |
| **Power Loss During Scan** | Hardware 12V rail drop sense. | Actuator springs default to **REVIEW / RETURN** position (normally-closed return chute). | On next reboot, SQLite WAL journal recovers uncommitted state automatically. |
| **Camera Unplugged / Disconnected** | `cap.read()` returns `False` or raises `RuntimeError`. | Status LED blinks RED; buzzer emits 3 error beeps. | `live_camera.py` catches error, releases resources, and logs camera fault to system journal. |
| **Mechanical Note Jam** | Lower optical sensor stays occluded > 2000 ms after solenoid pulse. | Solenoid de-energizes to prevent coil burnout; MCU sends `{"event": "JAM"}`. | Host logs `HARDWARE_FAULT` and pauses scanning until manual technician clear. |
| **Software Loop Hang / Freeze** | Host stops sending serial heartbeats for > 5000 ms. | Hardware watchdog timer on MCU resets gate to return chute; returns note to donor. | Process supervisor (systemd / Windows service) restarts `live_camera.py`. |

