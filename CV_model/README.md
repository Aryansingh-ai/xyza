# Daan Drishti - Prototype (ML Baseline)

Scope: 6-class denomination classifier (₹10 / ₹20 / ₹50 / ₹100 / ₹200 / ₹500)
with confidence-based ACCEPT/REVIEW decision logic. This is the software core
of the demo -- no YOLO, OCR, or hardware integration yet (those come later).

₹2000 is intentionally excluded (demonetized/rare, not relevant to temple
donations, not in the original spec's training class list).

## 1. Setup

```bash
cd daan-drishti
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download the Kaggle dataset:
https://www.kaggle.com/datasets/apoorvshekher/indian-currency-dataset

Unzip it so you end up with:
```
daan-drishti/data/raw/ten_new/*.jpg
daan-drishti/data/raw/ten_old/*.jpg
daan-drishti/data/raw/twenty_new/*.jpg
daan-drishti/data/raw/twenty_old/*.jpg
daan-drishti/data/raw/fifty_new/*.jpg
daan-drishti/data/raw/fifty_old/*.jpg
daan-drishti/data/raw/hundred_new/*.jpg
daan-drishti/data/raw/hundred_old/*.jpg
daan-drishti/data/raw/two_hundred/*.jpg
daan-drishti/data/raw/five_hundred/*.jpg
```
(the `two_thousand` folder can stay there too -- prepare_data.py ignores it
by default since it's out of scope.)

## 3. Prepare train/val/test splits

```bash
python src/prepare_data.py --raw_dir data/raw --out_dir data
```

**What this does:** groups images into "sessions" based on filename timestamp
proximity (images taken within 60 seconds are assumed to be the same physical
note under different backgrounds) and splits whole sessions 70/15/15 into
train/val/test. This avoids the same physical note leaking across splits,
which would give you a falsely high accuracy number.

**Expected output:** a summary printed per class showing total images, number
of sessions detected, and train/val/test image counts. Check that val/test
aren't empty for any class -- if a class has very few sessions, you may need
to lower `--session_gap_sec` or adjust ratios.

## 4. Train the baseline classifier

```bash
python src/train.py --data_dir data --epochs 15 --batch_size 16
```

**Expected output:** per-epoch train/val loss and accuracy, and at the end a
confusion matrix + classification report on the val set. The confusion
matrix matters more than raw accuracy here -- check specifically whether
100/200/500 get confused with each other (dangerous) vs. just having lower
confidence generally (safe, goes to REVIEW).

Best checkpoint is saved to `models/mobilenet_v3_baseline.pt`.

**If training is slow / you have no GPU:** reduce `--epochs` to 8-10 for a
first pass, or subsample the dataset. A CPU-only run on ~800 images at this
scale should still complete in a reasonable time, but don't be surprised if
it takes a while -- don't panic-tune the architecture, just let it run.

## 5. Test inference on a single image

```bash
python src/infer.py --image data/test/100/some_image.jpg --checkpoint models/mobilenet_v3_baseline.pt
```

Expected output (structured JSON):
```json
{
  "pocket_id": "webcam-0",
  "timestamp": "2026-09-09T14:32:10",
  "item_status": "NOTE_DETECTED",
  "predicted_denomination": "100",
  "classifier_confidence": 0.94,
  "decision": "ACCEPT",
  "target_bin": "BIN_100",
  "model_version": "mobilenet_v3_baseline_v0"
}
```

Image mode displays the input image with denomination, confidence, and the
ACCEPT/REVIEW decision. Add `--no_display` for headless testing. Every
intentional image inference is appended to `logs/inference.jsonl` by default;
change this location with `--log_path`.

## 6. Test with your phone camera (live, before committing to hardware)

```bash
python src/infer.py --webcam --checkpoint models/mobilenet_v3_baseline.pt --camera_index 0
```

Press SPACE to capture a frame and classify it, ESC to quit. If your phone
is connected as a webcam via USB/app, you may need to try `--camera_index 1`
or `2` instead of `0`.

The webcam remains live and overlays the latest Space-triggered prediction.
Only Space-triggered captures are logged; regular preview frames are never
stored. The default `--threshold 0.90` is a provisional conservative starting
point, not a calibrated threshold, and it does not detect unknown objects.

**Do this test with a plain background**, not a decorative one, even though
the training data has decorative backgrounds. If accuracy drops badly here
compared to val accuracy, that confirms the background-bias risk flagged
earlier -- the fix is adding stronger crop/occlusion augmentation and
re-training, not panicking. Flag it to me with the numbers if it happens.

## 7. Known limitations at this stage (say these out loud to judges, don't hide them)

- Notes-only classification for all 6 planned denominations is implemented;
  physical-content validation is not.
- No YOLO validity/contamination check yet -- assumes a single note is
  already correctly placed in frame.
- No OCR secondary verification yet -- decision relies on MobileNet
  confidence alone.
- Confidence threshold (0.90 default) is a provisional starting point, not yet tuned
  against this dataset's actual confidence distribution -- next step is to
  look at correct vs. incorrect prediction confidences on the val set and
  pick a threshold that actually separates them.
- No hardware integration yet -- this is the software core only.

## Next steps (Day 2)

1. Look at the val-set confidence distribution to properly tune the
   REVIEW threshold instead of using the 0.90 default.
2. Wire ESP32 <-> Python serial protocol for capture-trigger and
   discharge based on the `decision` field above.
3. (Optional, if time allows) Add PaddleOCR as secondary evidence per the
   fusion logic described in the original spec.
