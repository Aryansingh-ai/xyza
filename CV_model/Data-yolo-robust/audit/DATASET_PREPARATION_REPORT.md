# Robust YOLO Dataset Preparation Report

- Created: 2026-11-01T12:05:15.432036+00:00
- Random seed: `42`
- Dataset: `C:\Users\aryan\OneDrive\Documents\PROJECTS\Divyakosh-final\CV_model\Data-yolo-robust`
- YAML: `C:\Users\aryan\OneDrive\Documents\PROJECTS\Divyakosh-final\CV_model\Data-yolo-robust\data.yaml`

## Source usage
- Processed YOLO images used: **1312**
- Synthetic black-box images used: **22918**
- Additional data_clean ₹500 synthetic images used: **2780**
- data_clean unresolved images: **309**

## Final split counts
| Split | Images |
|---|---:|
| train | 20042 |
| val | 3769 |
| test | 3199 |

| Denomination | Train | Val | Test |
|---|---:|---:|---:|
| 10 INR | 4907 | 970 | 840 |
| 20 INR | 4357 | 700 | 620 |
| 50 INR | 3595 | 940 | 700 |
| 100 INR | 2627 | 430 | 400 |
| 200 INR | 2166 | 430 | 340 |
| 500 INR | 2390 | 299 | 299 |

## Annotation and augmentation
- Valid original annotations: **27010**
- Invalid/rejected annotations: **2**
- Augmented training images: **20042**
- Augmentation is applied only after grouped splitting and only to training samples.
- Geometric augmentation does not create physically realistic folded or crumpled notes.

## Limitations
- Exact duplicate checks use SHA-256; near-duplicate candidates use 64-bit DCT perceptual hashes.
- Physical-note grouping is available for synthetic families through the clean manifest; processed YOLO images have no shared physical-note metadata.
- No retained masks were present in the synthetic output, so boxes were derived from the black-background foreground and must be visually reviewed.
- Folded/crumpled-note presence was not established automatically; real-camera samples were unresolved.
- Empty/unrelated-object negatives are not included in this positive detection dataset.
- Original datasets and checkpoints were not modified.
