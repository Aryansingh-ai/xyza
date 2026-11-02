"""
Fine-tunes MobileNetV3-Large on the leakage-safe data_clean/ split.

6-class: 10, 20, 50, 100, 200, 500

Changes vs src/train.py:
  - Uses data_clean/ (leak-free, source-grouped split)
  - 6 classes instead of 3
  - Class-weighted CrossEntropyLoss (computed from train set)
  - Patience-based early stopping
  - JSON training history saved alongside model
  - num_workers=0 (Windows CPU stability)
  - Explicit random seed
  - Output: models/mobilenet_v3_clean.pt
  - Does NOT overwrite models/mobilenet_v3_baseline.pt

Architecture, transforms, optimizer, and epoch logic are preserved
from src/train.py.
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DIVYAKOSH = Path(
    str(Path(__file__).resolve().parent.parent)
)

DATA_DIR = DIVYAKOSH / "data_clean"
OUT_PATH = DIVYAKOSH / "models/mobilenet_v3_clean.pt"
HISTORY_PATH = DIVYAKOSH / "models/mobilenet_v3_clean_history.json"

EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4
PATIENCE = 5
SEED = 42


# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True


# â”€â”€ Transforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 1.5)
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.15)
    ),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

train_ds = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_tf
)

val_ds = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_tf
)

class_names = train_ds.classes
# Alphabetical:
# ['10', '100', '20', '200', '50', '500']

num_classes = len(class_names)


# â”€â”€ Class weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class_counts = [0] * num_classes

for _, label in train_ds.samples:
    class_counts[label] += 1


# Inverse-frequency class weights, normalized so mean = 1
weights = [
    sum(class_counts) / (num_classes * c)
    for c in class_counts
]

weights_tensor = torch.tensor(
    weights,
    dtype=torch.float32
)


# â”€â”€ Training configuration output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("=== TRAINING CONFIGURATION ===")
print("Device           : CPU (no CUDA in installed build)")
print(
    "Architecture     : MobileNetV3-Large "
    "(ImageNet IMAGENET1K_V2 pretrained)"
)
print("Input resolution : 224x224")
print(f"Classes ({num_classes})      : {class_names}")
print(f"Train images     : {len(train_ds)}")
print(f"Val images       : {len(val_ds)}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Learning rate    : {LR}")
print(f"Epochs (max)     : {EPOCHS}")
print(f"Early stop pat.  : {PATIENCE}")
print("Optimizer        : Adam")
print("Loss             : CrossEntropyLoss (class-weighted)")
print(f"Seed             : {SEED}")
print()

print("Class counts (train):")

for i, name in enumerate(class_names):
    print(
        f"  {name:>4}: "
        f"{class_counts[i]:>6}  "
        f"weight={weights[i]:.4f}"
    )

print()


# â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights_tensor = weights_tensor.to(device)


# â”€â”€ DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)


# â”€â”€ Loss + Optimizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

criterion = nn.CrossEntropyLoss(
    weight=weights_tensor
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    train
):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    phase = "Training" if train else "Validation"

    with torch.set_grad_enabled(train):

        progress = tqdm(
            loader,
            desc=phase,
            unit="batch",
            leave=True
        )

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if train:
                loss.backward()
                optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

            preds = outputs.argmax(dim=1)

            correct += (
                (preds == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    return (
        total_loss / total,
        correct / total
    )


# â”€â”€ Main training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

best_val_acc = 0.0
epochs_no_imp = 0
history = []

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


print("=== TRAINING ===")

for epoch in range(1, EPOCHS + 1):

    print()
    print(f"Epoch {epoch:02d}/{EPOCHS}")

    # Training
    train_loss, train_acc = run_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        train=True
    )

    # Validation
    val_loss, val_acc = run_epoch(
        model,
        val_loader,
        criterion,
        optimizer,
        train=False
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f}",
        end=""
    )

    history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "train_acc": round(train_acc, 6),
        "val_loss": round(val_loss, 6),
        "val_acc": round(val_acc, 6),
    })


    # â”€â”€ Save best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        epochs_no_imp = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc,
            "epoch": epoch,
            "class_weights": weights,
            "class_counts": class_counts,
        }, OUT_PATH)

        print(
            f"  -> SAVED "
            f"(best val_acc={val_acc:.4f})"
        )

    else:

        epochs_no_imp += 1

        print(
            f"  (no improvement "
            f"{epochs_no_imp}/{PATIENCE})"
        )

        if epochs_no_imp >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}."
            )

            break


# â”€â”€ Save training history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with open(HISTORY_PATH, "w") as f:

    json.dump(
        {
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "class_counts": class_counts,
            "class_weights": weights,

            "config": {
                "epochs_max": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "patience": PATIENCE,
                "seed": SEED,
                "architecture": "MobileNetV3-Large",
                "input_resolution": "224x224",
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss (class-weighted)",
            },

            "history": history,
        },
        f,
        indent=2
    )


# â”€â”€ Final confusion matrix on validation set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print()
print(f"Best val accuracy: {best_val_acc:.4f}")

print(
    "Loading best checkpoint "
    "for final confusion matrix..."
)

ckpt = torch.load(
    OUT_PATH,
    map_location=device
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()


all_preds = []
all_labels = []


with torch.no_grad():

    for images, labels in tqdm(
        val_loader,
        desc="Final validation",
        unit="batch"
    ):

        images = images.to(device)

        preds = (
            model(images)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())


# â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cm = confusion_matrix(
    all_labels,
    all_preds
)

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names
)


print()
print(
    "Confusion matrix "
    "(rows=true, cols=predicted):"
)

print(class_names)
print(cm)

print()
print("Classification report:")
print(report)

print()
print(f"Model saved: {OUT_PATH}")
print(f"History saved: {HISTORY_PATH}")
