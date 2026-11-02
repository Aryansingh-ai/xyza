"""
scripts/prepare_yolo_note_dataset.py

Prepare a YOLO-format dataset for a single-class object detector ("note").
The YOLO model will be used to count the number of notes in an ROI (0, 1, or 2+).
It does NOT classify denominations.

Required YOLO structure:
yolo_note_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

NOTE:
Currently, the existing dataset (e.g., data_clean/) consists of whole-image 
classification samples. There are NO bounding-box annotations (*.txt) available 
for the physical notes within these images. 
This script sets up the target directory structure and the data.yaml file, 
but will stop before copying images because we cannot generate YOLO labels 
without ground-truth bounding boxes.
"""

import os
import shutil
import yaml
from pathlib import Path

def setup_yolo_directories(base_dir="yolo_note_dataset"):
    """Creates the YOLO directory skeleton."""
    base_path = Path(base_dir)
    
    # Define subdirectories
    dirs_to_create = [
        base_path / "images" / "train",
        base_path / "images" / "val",
        base_path / "images" / "test",
        base_path / "labels" / "train",
        base_path / "labels" / "val",
        base_path / "labels" / "test",
    ]
    
    print(f"Creating YOLO dataset structure in: {base_path.absolute()}")
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")

def create_data_yaml(base_dir="yolo_note_dataset"):
    """Creates the data.yaml configuration file for a single class ('note')."""
    yaml_path = Path(base_dir) / "data.yaml"
    
    # Absolute or relative paths can be used in data.yaml. 
    # Using relative paths here based on YOLO conventions.
    data_config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {
            0: "note"
        }
    }
    
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data_config, f, sort_keys=False)
        
    print(f"Created YOLO config file: {yaml_path}")
    print("Content:")
    print(yaml.dump(data_config, sort_keys=False).strip())

def check_for_annotations(source_dir="data_clean"):
    """
    Checks if there are any existing bounding-box annotations (e.g., XML, JSON, or TXT).
    In this project, data_clean/ contains subdirectories by denomination, 
    but no bounding box labels.
    """
    source_path = Path(source_dir)
    print(f"\nChecking source directory '{source_path}' for bounding box annotations...")
    
    if not source_path.exists():
        print(f"  Source directory '{source_path}' does not exist.")
        return False
        
    # Search for typical annotation files
    txt_files = list(source_path.rglob("*.txt"))
    xml_files = list(source_path.rglob("*.xml"))
    json_files = list(source_path.rglob("*.json"))
    
    if len(txt_files) == 0 and len(xml_files) == 0 and len(json_files) == 0:
        print("  NO bounding box annotations found (no .txt, .xml, or .json files).")
        return False
        
    print(f"  Found potential annotations: {len(txt_files)} TXT, {len(xml_files)} XML, {len(json_files)} JSON.")
    return True

def main():
    print("=== YOLO Note-Count Dataset Preparer ===\n")
    
    base_dir = "yolo_note_dataset"
    source_dir = "data_clean"
    
    # 1. Setup structure
    setup_yolo_directories(base_dir)
    
    # 2. Create data.yaml
    create_data_yaml(base_dir)
    
    # 3. Check for existing annotations
    has_annotations = check_for_annotations(source_dir)
    
    print("\n=== STATUS ===")
    if not has_annotations:
        print("CRITICAL REQUIREMENT MISSING: No bounding-box annotations exist.")
        print("We cannot automatically convert classification images into a YOLO dataset")
        print("because we do not know where the physical note is located in each image.")
        print("\nNEXT STEPS:")
        print("1. Select a subset of images (e.g., from data_clean/).")
        print("2. Manually annotate them by drawing bounding boxes around the 'note'.")
        print("3. Save the annotations in YOLO format (txt files with: <class=0> <x_center> <y_center> <width> <height>).")
        print("4. Place the annotated images and labels into the yolo_note_dataset/ structure.")
    else:
        print("Annotations found! You can proceed with copying/parsing them into YOLO format.")
        
    print("\nDone. No existing datasets were modified.")

if __name__ == "__main__":
    main()
