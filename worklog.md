
---
Task ID: 2-a
Agent: data-pipeline builder
Task: Data pipeline module (retinaedge.data.*) per contract v1.0

Work Log:
- src/retinaedge/data/ben_graham.py: fundus mask crop -> radius rescale (default 300) -> CLAHE on LAB L-channel; pure cv2/numpy, deterministic, degenerate-image safe
- src/retinaedge/data/dataset.py:
  - build_dataset(cfg, split, transform=None) for synthetic|aptos|eyepacs|ddr|folder|hf
  - build_train_val_transforms(cfg): CLAHE(p=clahe_prob) + LongestMaxSize(256) + RandomResizedCrop/HFlip/VFlip/RandRotate90/ColorJitter + Normalize(ImageNet) + ToTensorV2; val = CLAHE + LongestMaxSize + CenterCrop(pad_if_needed) + Normalize
  - SyntheticDRDataset: grades drawn from default_rng(seed + split_offset), dist [0.45,0.20,0.15,0.10,0.10]; per-index image rendering is index-seeded (deterministic under any shuffle/worker layout); split offsets train=0/val=1e6/test=2e6
  - FileListDataset: images/ + labels.csv (image,grade); __getitem__ -> (float CHW ImageNet-normalized, int grade)
  - split_tags(): deterministic md5-hash train/val/test assignment (val_fraction 0.15, test_fraction 0.0, optional split_salt) — no split files stored
- src/retinaedge/data/download_kaggle.py: kagglehub CLI (--handle/--dest/--type auto), stages into <dest>/raw, hardlink-first, idempotent
- src/retinaedge/data/download_hf.py: huggingface_hub snapshot_download CLI (--repo/--dest/--revision)
- src/retinaedge/data/prepare.py: normalizes aptos (train.csv+train_images), eyepacs (trainLabels/retinopathy_solution + train/), ddr+folder (grade-subdir trees), hf (grade trees / csv / embedded-image parquet best-effort); auto-extracts zips incl. multi-part (EyePACS train.zip.001...); writes labels.csv + per-dataset README (creates repo data/README.md only if missing — did not overwrite existing)
- tests/test_data.py: 17 network-free tests (ben-graham geometry, split determinism, synthetic contract+dist, transforms, file datasets, prepare CLIs incl. multipart zip, error paths)
- Verified: smoke.yaml -> train 240 / val 80 datasets, DataLoader batch (16,3,64,64) float32; all 3 CLIs run via python -m; ruff clean

Stage Summary:
- Contract notes/deviations:
  1. Splitting: contract does not fix a split mechanism -> deterministic name-hash split via data.val_fraction/data.test_fraction/data.split_salt (documented in module + per-dataset README)
  2. Ben-Graham is opt-in via data.ben_graham (default false) + data.ben_graham_radius (300); applies to all splits when enabled for aptos/eyepacs/ddr/folder
  3. prepare writes per-dataset README at data/<name>/README.md and only creates data/README.md if absent (main agent's placeholder preserved)
  4. Synthetic split offsets defined as {train:0, val:1e6, test:2e6} (contract left them open)
- docs/DATASETS.md already present (other agent) — verified consistent with implemented CLIs; typo there (`".f]"`) left for its owner
- Full pytest has failures in OTHER agents' modules (metrics/model_build/train_pipeline) — out of scope; tests/test_data.py + test_ordinal_ops.py + test_loss-related data paths pass
