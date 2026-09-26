# Datasets

How RetinaEdge-DR sources, normalises and preprocesses fundus image data.
All paths below are relative to the repository root. `data/` is **gitignored** — never commit
images or label files.

## 1. Supported sources

`cfg["data"]["dataset"]` selects the builder in `retinaedge.data.dataset`:

| Name | Source | Files on disk? | Notes |
|---|---|---|---|
| `synthetic` | procedural generator | no | CI/laptop smoke runs; class dist ≈ `[0.45, 0.20, 0.15, 0.10, 0.10]` |
| `aptos` | Kaggle *APTOS 2019 Blindness Detection* | yes | 3,662 labelled train images, primary benchmark |
| `eyepacs` | Kaggle *Diabetic Retinopathy Detection* (EyePACS) | yes | ~35k train images, noisy labels, large-scale pretraining |
| `ddr` | DDR (DeepDRiD-related public set) | yes | graded lesions + images; check licence before use |
| `folder` | any local folder you arrange yourself | yes | escape hatch for private/hospital data |
| `hf` | any Hugging Face dataset with an image+label pair | via `datasets` | needs the `hf` extra |

## 2. Normalised on-disk layout

`prepare` converts every source into one canonical shape, so trainers/eval never care about origin:

```
data/<name>/
├── images/          # all images, any raster format opencv can read
│   ├── 000c1434d8d7.png
│   └── ...
└── labels.csv       # columns: image,grade   (image path RELATIVE to images/)
```

- `grade` is an integer **0–4** in ICDRSS order — this scheme is fixed in Python, Kotlin and docs
  (see [INTERFACES.md](INTERFACES.md#grade-scheme-fixed-everywhere-python-kotlin-docs)).
- `prepare` also (re)writes `data/README.md` documenting provenance and licence of each source
  directory.
- `labels.txt` for export/Android lists the five labels in the same order:
  `No DR, Mild, Moderate, Severe, Proliferative DR`.

## 3. Downloading

```bash
# Kaggle (kagglehub; creds from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)
python -m retinaedge.data.download_kaggle --handle aptos2019-blindness-detection --dest data/aptos
python -m retinaedge.data.download_kaggle --handle diabetic-retinopathy-detection --dest data/eyepacs

# Hugging Face (needs `pip install -e ".[hf]"`)
python -m retinaedge.data.download_hf --repo <hf-dataset-id> --dest data/hf_<name>

# Normalise into images/ + labels.csv
python -m retinaedge.data.prepare --config configs/data/aptos.yaml
```

Kaggle one-time step: accept the competition rules on the competition page, otherwise the download
returns 403.

## 4. Preprocessing

### Ben-Graham illumination — `retinaedge.data.ben_graham`

```python
preprocess_ben_graham(rgb: np.ndarray, radius: int = 300) -> np.ndarray
```

The classic fundus normalisation (Ben-Graham et al., 2015):

1. **Mask & crop** the fundus circle (threshold on brightness), removing black borders.
2. **Scale** so the fundus radius == `radius` px — normalises camera/zoom differences.
3. **CLAHE** on the L-channel of LAB space — recovers peripheral lesion detail.

Pure `cv2`/`numpy`, no extra dependencies. Applied to `aptos | eyepacs | ddr | folder` datasets
(**not** synthetic).

### Train-time augmentation (albumentations)

| Stage | Train | Val/Test |
|---|---|---|
| CLAHE | with probability `data.clahe_prob` | same fixed prob |
| Resize | longest side → 256 | longest side → 256 |
| Crop | `RandomResizedCrop(img_size, scale=(0.8, 1.0))` | `CenterCrop(img_size)` |
| Geometric | h/v flip + `RandomRotate90` | — |
| Photometric | `ColorJitter` | — |
| Normalise | ImageNet mean/std, `ToTensorV2` | same |

Why random CLAHE (`clahe_prob: 0.5` on real configs)? The Android app deliberately **skips CLAHE
on-device** (saves CPU/battery). Training with randomised CLAHE makes the model robust *with and
without* it. Keep `clahe_prob: 0.0` for synthetic data.

### Dataset contract

`__getitem__` returns `(float Tensor CHW ImageNet-normalised, int grade)` — nothing else, no dict,
no path. Whoever breaks this breaks trainer, eval, demo and export in one stroke.

## 5. Class imbalance

Real DR cohorts are heavily skewed toward grade 0 (APTOS ≈ 49% / 33% / 12% / 5% / 1%):

- `train.sampler: true` enables the class-balanced weighted sampler (resample by inverse frequency).
- `train.loss.focal_gamma > 0` down-weights easy ordinal binary tasks.
- The referable head (grade ≥ 2) is the clinically relevant binary view and is tracked separately
  (AUC/sensitivity/specificity at Youden threshold) — see [TRAINING.md](TRAINING.md#4-metrics).

## 6. Licensing & responsible use

| Dataset | Access | Licence / terms |
|---|---|---|
| APTOS 2019 | Kaggle competition | competition terms; **research only**, no redistribution of images |
| EyePACS | Kaggle competition | competition terms; research only |
| DDR | project site | request/accept their terms; verify before use |
| Your own data (`folder`) | you | you are the data controller — de-identify, follow local law |

Rules of thumb:

1. Never commit or re-upload dataset images (this repo gitignores `data/` for a reason).
2. Cite the original source in any derived publication.
3. De-identify any clinical data you add under `folder/`.
4. This pipeline is research software — see the [Model Card](MODEL_CARD.md) before any real-world
   evaluation.
