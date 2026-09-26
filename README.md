# RetinaEdge-DR

**Diabetic Retinopathy (DR) grading at the mobile edge** — a fully open-source pipeline from
PyTorch training to an int8-quantized TFLite model that runs on Android, entirely on CPU-friendly
hardware.

- **Ordinal grading** (CORAL-style cumulative-link head) over the 5-grade ICDRSS scheme instead of
  plain softmax classification — the model knows that *Moderate* is closer to *Mild* than to
  *Proliferative*.
- **Auxiliary referable-DR head**: `P(grade >= 2)` is the clinically actionable output
  (referable DR = "needs an ophthalmologist referral").
- **Calibration built-in**: temperature scaling on the validation split, ECE reported everywhere.
- **Edge-first export**: `float32 (1,3,H,W)` → `float32 (1,5)` probs, ONNX and TFLite
  (full-integer int8), verified parity vs. PyTorch before anything ships.

> ⚠️ **Research / educational software.** RetinaEdge-DR is **not** a medical device and has **not**
> received regulatory clearance. Do not use it for clinical decision-making. See the
> [Model Card](docs/MODEL_CARD.md).

## Grade scheme (fixed across Python, Kotlin and docs)

| Grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Label | No DR | Mild | Moderate | Severe | Proliferative DR |
| Referable | | | ✅ | ✅ | ✅ |

## Pipeline

```
 data (APTOS / EyePACS / DDR / synthetic)          Android app (TFLite)
          │                                               ▲
          ▼  Ben-Graham · CLAHE · albumentations          │ file:///android_asset/models/
 ┌─────────────────┐   DrNet (timm backbone)     ┌────────┴───────┐
 │  train.trainer  │──▶ best.pt ──▶ Inference-  ─▶│ model.tflite   │
 │  QWK · AUC · ECE│        │       Wrapper       │ labels.txt     │
 └─────────────────┘        ▼                     └────────────────┘
                    ┌───────────────┐
                    │ export: ONNX  │──▶ benchmark ──▶ metadata (labels.txt,
                    │ TFLite int8   │                   model_info.json)
                    └───────────────┘
```

## Quickstart

```bash
# 1. Install (editable, dev tooling included)
make setup          # = pip install -e ".[dev]"

# 2. Sanity-check the environment
python scripts/verify_env.py

# 3. Prove the whole pipeline in ~1 minute, no dataset download:
#    train (synthetic) -> eval -> ONNX export -> benchmark
make smoke

# 4. Launch the Gradio demo (falls back to untrained weights if no checkpoint exists)
pip install -e ".[demo]"
make demo
```

Full walkthrough: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

## Repository map

| Path | Purpose |
|---|---|
| `src/retinaedge/data/` | datasets, Ben-Graham preprocess, Kaggle/HF download, prepare |
| `src/retinaedge/models/` | DrNet builder (timm), ordinal ops, loss |
| `src/retinaedge/train/` | trainer, QWK/metrics tracking |
| `src/retinaedge/eval/` | evaluation, temperature calibration |
| `src/retinaedge/export/` | ONNX / TFLite export, benchmark, metadata |
| `configs/` | data + train + export YAML configs |
| `demo/` | Gradio web demo |
| `notebooks/` | executable walkthroughs (quickstart, calibration & export) |
| `docs/` | this documentation set — start at [GETTING_STARTED](docs/GETTING_STARTED.md) |
| `android/` | Android client consuming the export contract |
| `tests/`, `scripts/`, `.github/` | pytest, env check, CI/workflows |

## Documentation index

| Doc | Contents |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | install, smoke run, CLI + Makefile reference, troubleshooting |
| [docs/DATASETS.md](docs/DATASETS.md) | APTOS / EyePACS / DDR, normalized layout, Ben-Graham, licensing |
| [docs/TRAINING.md](docs/TRAINING.md) | config reference, loss, metrics (QWK/AUC/ECE), calibration, artifacts |
| [docs/EXPORT_DEPLOY.md](docs/EXPORT_DEPLOY.md) | export graph contract, ONNX, TFLite int8, benchmark, Android integration |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | intended use, limitations, ethics, responsible-use notice |
| [docs/INTERFACES.md](docs/INTERFACES.md) | binding module contract (v1.0) for contributors |

## Notebooks

| Notebook | Contents |
|---|---|
| [notebooks/01_quickstart_synthetic.ipynb](notebooks/01_quickstart_synthetic.ipynb) | ordinal ops → synthetic data → train tiny DrNet → QWK/AUC |
| [notebooks/02_calibration_and_export.ipynb](notebooks/02_calibration_and_export.ipynb) | temperature scaling → ECE before/after → ONNX export → parity + latency |

## Status

- ✅ Core utils (config / seed / logging), ordinal ops, smoke config, interface contract
- ✅ Docs, demo, notebooks (this delivery)
- 🚧 Data pipeline, DrNet builder, trainer, eval/export modules (built against contract v1.0)
- 🚧 Android client, GitHub workflows

## License

MIT — see [LICENSE](LICENSE). Datasets (APTOS, EyePACS, DDR) carry their own licences and terms of
use; see [docs/DATASETS.md](docs/DATASETS.md) before redistributing anything.
