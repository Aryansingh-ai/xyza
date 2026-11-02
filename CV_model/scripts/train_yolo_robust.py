"""Train or evaluate YOLO26n on the prepared robust banknote dataset.

Training uses only ``train`` and ``val`` from the dataset YAML. The held-out
``test`` split is available only through the explicit ``--eval-test`` mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "CV_model" / "Data-yolo-robust" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "CV_model" / "models" / "yolo26n_robust"
EXPECTED_NAMES = {
    0: "10 INR",
    1: "20 INR",
    2: "50 INR",
    3: "100 INR",
    4: "200 INR",
    5: "500 INR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Pretrained YOLO26n weights for a new run.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training data to use; validation/test remain unchanged.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision; leave disabled for the current CPU environment.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--resume", type=Path, metavar="LAST_PT",
                        help="Resume explicitly from an existing last.pt.")
    parser.add_argument("--eval-test", action="store_true",
                        help="Evaluate --weights on the held-out test split only.")
    parser.add_argument("--weights", type=Path,
                        help="Checkpoint used with --eval-test.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate dataset paths and class mapping without loading YOLO.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to validate the dataset YAML.") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read dataset YAML: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Dataset YAML must contain a mapping: {path}")
    return value


def validate_dataset(data_path: Path) -> Path:
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    config = load_yaml(data_path)
    dataset_root = Path(config.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()

    names = config.get("names")
    normalized_names = (
        {int(key): str(value) for key, value in names.items()}
        if isinstance(names, dict)
        else {index: str(value) for index, value in enumerate(names)}
        if isinstance(names, list)
        else None
    )
    if normalized_names != EXPECTED_NAMES:
        raise RuntimeError(
            f"Dataset class mapping does not match the required mapping: {normalized_names}"
        )

    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not isinstance(split_value, str):
            raise RuntimeError(f"Dataset YAML is missing the {split!r} path.")
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.is_dir():
            raise FileNotFoundError(f"{split} directory not found: {split_path.resolve()}")

    return data_path


def unique_run_name(prefix: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def train(args: argparse.Namespace, data_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install it in the existing .venv before training."
        ) from exc

    args.project.mkdir(parents=True, exist_ok=True)
    if args.resume:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        run_name = checkpoint.parent.parent.name
        run_project = checkpoint.parent.parent.parent
        resume = True
    else:
        model = YOLO(args.model)
        run_name = unique_run_name("train_seed42_20")
        run_project = args.project
        resume = False

    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        fraction=args.fraction,
        amp=args.amp,
        project=str(run_project),
        name=run_name,
        # An explicit resume is the only mode allowed to reopen an existing run.
        exist_ok=resume,
        resume=resume,
        save=True,
        plots=True,
        val=True,
        # The prepared dataset already contains deterministic augmentation.
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        auto_augment=None,
    )


def evaluate_test(args: argparse.Namespace, data_path: Path) -> None:
    if not args.weights:
        raise RuntimeError("--weights CHECKPOINT is required with --eval-test.")
    checkpoint = args.weights.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install it in the existing .venv before evaluation."
        ) from exc

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=unique_run_name("test_seed42"),
        exist_ok=False,
        plots=True,
    )


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 1:
        raise ValueError("--epochs, --batch, and --imgsz must be positive.")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and no greater than 1.")
    data_path = validate_dataset(args.data)
    if args.check_only:
        print(f"Dataset configuration valid: {data_path}")
        print("Verified train, val, and test directories and the exact six-class mapping.")
        return 0
    if args.eval_test:
        evaluate_test(args, data_path)
    else:
        train(args, data_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
