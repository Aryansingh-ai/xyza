"""
scripts/evaluate_test.py  â€” one-shot test evaluation, no training.
Loads models/mobilenet_v3_clean.pt and evaluates on data_clean/test/.
Outputs: models/mobilenet_v3_clean_test_results.json
"""
import json, sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
)
from tqdm import tqdm

DIVYAKOSH  = Path(__file__).resolve().parent.parent
CKPT_PATH  = DIVYAKOSH / "models/mobilenet_v3_clean.pt"
TEST_DIR   = DIVYAKOSH / "data_clean/test"
OUT_JSON   = DIVYAKOSH / "models/mobilenet_v3_clean_test_results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# â”€â”€ Load checkpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ckpt        = torch.load(CKPT_PATH, map_location=device)
class_names = ckpt["class_names"]   # ['10','100','20','200','50','500']
num_classes = len(class_names)
print(f"Checkpoint epoch : {ckpt['epoch']}")
print(f"Checkpoint val_acc: {ckpt['val_acc']:.4f}")
print(f"Classes: {class_names}")

# â”€â”€ Rebuild model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# â”€â”€ Test dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
# ImageFolder sorts classes alphabetically â€” must match checkpoint class_names
assert test_ds.classes == class_names, (
    f"Class mismatch! checkpoint={class_names}, folder={test_ds.classes}"
)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Per-class image counts
class_counts = Counter()
for _, label in test_ds.samples:
    class_counts[class_names[label]] += 1

print(f"\nTest images per class:")
for name in class_names:
    print(f"  {name:>4}: {class_counts[name]}")
print(f"  TOTAL: {len(test_ds)}")

# â”€â”€ Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
acc = accuracy_score(all_labels, all_preds)
cm  = confusion_matrix(all_labels, all_preds)
report_str = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)

prec, rec, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(num_classes))
)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro"
)

print(f"\n=== TEST RESULTS ===")
print(f"Test accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Macro precision  : {macro_p:.4f}")
print(f"Macro recall     : {macro_r:.4f}")
print(f"Macro F1         : {macro_f1:.4f}")
print(f"\nPer-class metrics:")
print(f"  {'Class':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
for i, name in enumerate(class_names):
    print(f"  {name:>6}  {prec[i]:>10.4f}  {rec[i]:>8.4f}  {f1[i]:>8.4f}  {support[i]:>8}")

print(f"\nConfusion matrix (rows=true, cols=predicted):")
print(f"Classes: {class_names}")
print(cm)

print(f"\nFull classification report:")
print(report_str)

# â”€â”€ Save JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results = {
    "checkpoint": str(CKPT_PATH),
    "checkpoint_epoch": ckpt["epoch"],
    "checkpoint_val_acc": ckpt["val_acc"],
    "test_dir": str(TEST_DIR),
    "test_accuracy": acc,
    "macro_precision": macro_p,
    "macro_recall": macro_r,
    "macro_f1": macro_f1,
    "class_names": class_names,
    "per_class": {
        class_names[i]: {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }
        for i in range(num_classes)
    },
    "confusion_matrix": cm.tolist(),
    "test_images_total": len(test_ds),
    "test_images_per_class": {k: int(v) for k, v in class_counts.items()},
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {OUT_JSON}")

