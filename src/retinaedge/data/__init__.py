"""Data pipeline: datasets, transforms, preprocessing and download/prepare CLIs.

Modules:
    ben_graham     -- fundus crop + radius rescale + CLAHE preprocess
    dataset        -- build_dataset / build_train_val_transforms (contract v1.0)
    download_kaggle-- CLI: fetch Kaggle competitions/datasets via kagglehub
    download_hf    -- CLI: fetch Hugging Face dataset snapshots
    prepare        -- CLI: normalise sources into images/ + labels.csv
"""

__all__ = ["ben_graham", "dataset", "download_hf", "download_kaggle", "prepare"]
