"""
scripts/generate_yolo_note_scenes.py

Generates synthetic scenes containing 1, 2, or 3 overlapping physical notes
using the existing single-note "black_box" images. 

It generates YOLO format labels (class 0 = note) based on the visible 
extent of each note after accounting for occlusion.

Expected output structure:
yolo_note_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
"""

import os
import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
NUM_SINGLE = 1000
NUM_DOUBLE = 1500
NUM_TRIPLE = 1000

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
CANVAS_SIZE = (640, 640)
SEED = 42

def setup_directories(base_dir="yolo_note_dataset"):
    base_path = Path(base_dir)
    splits = ["train", "val", "test"]
    for split in splits:
        (base_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    return base_path

def create_yaml(base_dir):
    data_config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "note"}
    }
    with open(Path(base_dir) / "data.yaml", "w") as f:
        yaml.dump(data_config, f, sort_keys=False)

def get_source_images():
    """Finds all black_box images to use as source material."""
    source_dir = Path("synthetic_dataset/black_box")
    if not source_dir.exists():
        print(f"Warning: {source_dir} not found. Attempting to fallback to data_clean/train")
        source_dir = Path("data_clean/train")
        if not source_dir.exists():
            raise FileNotFoundError("Could not find source images.")
    
    images = list(source_dir.rglob("*.jpg"))
    return images

def extract_note(image_path):
    """
    Reads an image and extracts the note and its mask.
    Assumes a black background (black_box dataset).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    # Create mask by thresholding (background is near black)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Optional: morphology to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Crop to the bounding box of the note to make transformations easier
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_img, cropped_mask

def transform_note(img, mask, canvas_size):
    """Applies random rotation, scaling, and translation to a note."""
    ch, cw = canvas_size
    h, w = img.shape[:2]
    
    # Random scale between 0.5 and 1.2
    scale = random.uniform(0.5, 1.2)
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    mask_resized = cv2.resize(mask, (new_w, new_h))
    
    # Random rotation
    angle = random.uniform(0, 360)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate bounding box of rotated image to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    rot_w = int((new_h * sin) + (new_w * cos))
    rot_h = int((new_h * cos) + (new_w * sin))
    
    M[0, 2] += (rot_w / 2) - center[0]
    M[1, 2] += (rot_h / 2) - center[1]
    
    img_rotated = cv2.warpAffine(img_resized, M, (rot_w, rot_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    mask_rotated = cv2.warpAffine(mask_resized, M, (rot_w, rot_h), flags=cv2.INTER_NEAREST, borderValue=0)
    
    # Random translation (ensure it fits within the canvas roughly)
    max_x = max(1, cw - rot_w)
    max_y = max(1, ch - rot_h)
    
    # Allow partial cutoffs by going slightly negative or beyond canvas
    start_x = random.randint(int(-rot_w * 0.2), int(cw - rot_w * 0.8))
    start_y = random.randint(int(-rot_h * 0.2), int(ch - rot_h * 0.8))
    
    # Place on full canvas
    full_img = np.zeros((ch, cw, 3), dtype=np.uint8)
    full_mask = np.zeros((ch, cw), dtype=np.uint8)
    
    # Compute intersection of note and canvas
    x1, y1 = max(0, start_x), max(0, start_y)
    x2, y2 = min(cw, start_x + rot_w), min(ch, start_y + rot_h)
    
    img_x1, img_y1 = max(0, -start_x), max(0, -start_y)
    img_x2, img_y2 = img_x1 + (x2 - x1), img_y1 + (y2 - y1)
    
    if x1 < x2 and y1 < y2:
        full_img[y1:y2, x1:x2] = img_rotated[img_y1:img_y2, img_x1:img_x2]
        full_mask[y1:y2, x1:x2] = mask_rotated[img_y1:img_y2, img_x1:img_x2]
        
    return full_img, full_mask

def add_noise(image):
    """Add mild noise/blur for augmentation."""
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image

def generate_scene(source_images, num_notes, canvas_size):
    """Generates a single scene with multiple notes and resolves occlusions for labels."""
    canvas_img = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    # To resolve occlusion, we keep track of the notes placed.
    # We will composite them bottom-to-top.
    notes_data = []
    
    for _ in range(num_notes):
        src_path = random.choice(source_images)
        img, mask = extract_note(src_path)
        if img is None:
            continue
            
        trans_img, trans_mask = transform_note(img, mask, canvas_size)
        notes_data.append((trans_img, trans_mask))
        
    # Composite bottom-to-top
    for img, mask in notes_data:
        idx = mask > 127
        canvas_img[idx] = img[idx]
        
    # Calculate visible masks and bounding boxes (YOLO format)
    labels = []
    ch, cw = canvas_size
    
    # To find visible mask of note i, we start with its mask, and subtract
    # the masks of all notes j > i (which are placed ON TOP of it).
    for i, (_, mask_i) in enumerate(notes_data):
        visible_mask = mask_i.copy()
        for j in range(i + 1, len(notes_data)):
            _, mask_j = notes_data[j]
            visible_mask[mask_j > 127] = 0
            
        # Find bounding box of the visible portion
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Combine all contours into one bounding box
            all_pts = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            
            # If visible area is too small, skip (e.g., highly occluded)
            if cv2.countNonZero(visible_mask) > 500:
                # YOLO format: class x_center y_center width height (normalized)
                xc = (x + w / 2) / cw
                yc = (y + h / 2) / ch
                nw = w / cw
                nh = h / ch
                labels.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                
    # Augment final canvas
    canvas_img = add_noise(canvas_img)
    
    return canvas_img, labels

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    base_dir = setup_directories()
    create_yaml(base_dir)
    
    print("Gathering source images...")
    source_images = get_source_images()
    print(f"Found {len(source_images)} source images.")
    
    if len(source_images) == 0:
        print("No source images found. Exiting.")
        return
        
    # Plan distribution
    total_scenes = NUM_SINGLE + NUM_DOUBLE + NUM_TRIPLE
    scene_plans = ([1] * NUM_SINGLE) + ([2] * NUM_DOUBLE) + ([3] * NUM_TRIPLE)
    random.shuffle(scene_plans)
    
    splits = []
    num_train = int(total_scenes * SPLIT_RATIOS["train"])
    num_val = int(total_scenes * SPLIT_RATIOS["val"])
    num_test = total_scenes - num_train - num_val
    
    splits.extend(["train"] * num_train)
    splits.extend(["val"] * num_val)
    splits.extend(["test"] * num_test)
    random.shuffle(splits)
    
    print(f"Generating {total_scenes} synthetic YOLO scenes...")
    print(f"  {NUM_SINGLE} Single-note")
    print(f"  {NUM_DOUBLE} Double-note")
    print(f"  {NUM_TRIPLE} Triple-note")
    
    for idx, (num_notes, split) in enumerate(tqdm(zip(scene_plans, splits), total=total_scenes)):
        img, labels = generate_scene(source_images, num_notes, CANVAS_SIZE)
        
        # Save image
        img_name = f"syn_{idx:05d}.jpg"
        img_path = base_dir / "images" / split / img_name
        cv2.imwrite(str(img_path), img)
        
        # Save labels
        lbl_name = f"syn_{idx:05d}.txt"
        lbl_path = base_dir / "labels" / split / lbl_name
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))
            
    print("\nDataset generation complete!")
    print(f"Generated dataset structure in: {base_dir.absolute()}")

if __name__ == "__main__":
    main()
