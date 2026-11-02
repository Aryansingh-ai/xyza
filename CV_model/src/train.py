"""
train.py

Fine-tunes MobileNetV3-Large (ImageNet-pretrained) as the denomination
classifier for the 3-class MVP: 100, 200, 500.

AUGMENTATION NOTE:
We deliberately do NOT use horizontal/vertical flips. Flipping a currency
note produces a mirror image of printed text/portraits, which is not a
physically realistic sample (a real note in a pocket is never seen
mirrored) and would teach the model on unrealistic data, per the project's
augmentation rule. We use rotation, brightness/contrast, and blur instead,
which do reflect real handling conditions (skewed placement, poor lighting,
motion blur).

USAGE:
    python src/train.py --data_dir data --epochs 15 --batch_size 16

OUTPUT:
    models/mobilenet_v3_baseline.pt   (best checkpoint by val accuracy)
    Console: per-epoch train/val loss & accuracy, final confusion matrix
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        # RandomResizedCrop simulates partial occlusion (hand covering part
        # of the note) and background variation (crowded/rushed handling),
        # without ever showing the model an unrealistic mirrored note.
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # RandomErasing (applied after ToTensor) blacks out a random patch,
        # further simulating occlusion/damage without altering note geometry.
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_confusion(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    return cm, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_path", type=str, default="models/mobilenet_v3_baseline.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(Path(args.data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(args.data_dir) / "val", transform=eval_tf)

    class_names = train_ds.classes  # alphabetical order, e.g. ['100', '200', '500']
    print(f"Classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
            }, args.out_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Loading best checkpoint for final confusion matrix...")
    checkpoint = torch.load(args.out_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cm, report = evaluate_confusion(model, val_loader, device, class_names)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(class_names)
    print(cm)
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()
