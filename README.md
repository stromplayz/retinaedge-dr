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

## The 97% campaign (research plan + continuous improvement)

The road to **97% referable-DR accuracy** is formalized in
[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md): medical physics (Retinex/Ben-Graham/CLAHE),
CORAL ordinal objective, the ten-item loophole playbook (data maximalism, patient-aware
splits, threshold/cut-point fitting, TTA, model soup, ensembling, progressive resizing,
pseudo-labeling, CLAHE invariance, calibration), and a Wilson-CI-gated definition of
"reached".

Continuous improvement runs **inside GitHub** via
[`.github/workflows/improve.yml`](.github/workflows/improve.yml) (manual dispatch or
every 3 days): it pulls the 18,383-image Kaggle blend (DDR+APTOS+Messidor-2+IDRiD) with
the `KAGGLE_API_TOKEN` secret, advances the resumable escalation ladder
(`retinaedge.train.auto_improve`), applies the full loophole stack, and commits
`runs/improvement_log.md` + `runs/improve_state.json` back to the repo.

```bash
# locally:
python3 -m retinaedge.train.auto_improve --config configs/train/kaggle_blend.yaml --budget small
# or: make improve CFG=configs/train/kaggle_blend.yaml BUDGET=small
```

## Documentation index

| Doc | Contents |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | install, smoke run, CLI + Makefile reference, troubleshooting |
| [docs/DATASETS.md](docs/DATASETS.md) | APTOS / EyePACS / DDR, normalized layout, Ben-Graham, licensing |
| [docs/TRAINING.md](docs/TRAINING.md) | config reference, loss, metrics (QWK/AUC/ECE), calibration, artifacts |
| [docs/EXPORT_DEPLOY.md](docs/EXPORT_DEPLOY.md) | export graph contract, ONNX, TFLite int8, benchmark, Android integration |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | intended use, limitations, ethics, responsible-use notice |
| [docs/INTERFACES.md](docs/INTERFACES.md) | binding module contract (v1.0) for contributors |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 97% campaign: team, physics, formulas, loophole playbook, milestones |

## Notebooks

| Notebook | Contents |
|---|---|
| [notebooks/01_quickstart_synthetic.ipynb](notebooks/01_quickstart_synthetic.ipynb) | ordinal ops → synthetic data → train tiny DrNet → QWK/AUC |
| [notebooks/02_calibration_and_export.ipynb](notebooks/02_calibration_and_export.ipynb) | temperature scaling → ECE before/after → ONNX export → parity + latency |

## Status

- ✅ Full pipeline: data → training → eval → calibration → ONNX export (parity-gated) → Android app
- ✅ **Real-data pilot trained inside GitHub Actions** (run #2): APTOS 2019 (3,662 images) resolved
  at runtime from the Hugging Face Hub — val **QWK 0.503**, referable-DR **AUC 0.846**, sensitivity
  **0.916** @ spec 0.638, calibrated ECE **0.088**; ONNX 6.11 MB, parity 1.8e-07. Details and honest
  caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), provenance in `provenance.json`.
- ✅ CI green on `main` (lint + **160 tests** + smoke pipeline on every push)
- ✅ int8 TFLite conversion for Android (1.97 MB dynamic-range + 1.83 MB full-integer, CI-built)
- ✅ **97% campaign live** and resumable: escalation ladder executed s1→s5 in GitHub Actions on the
  18,383-image Kaggle blend (DDR+APTOS+Messidor-2+IDRiD, patient-aware splits). Current best
  (`s5-fusion`, soup+tta): referable-DR accuracy **0.890** (Wilson 95% CI [0.875, 0.904], n=1843),
  **QWK 0.845**, sens 0.855 @ spec 0.917 — progress log: `runs/improvement_log.md`.
- ✅ **v0.2.0-mobile additions**: KGAT-token Kaggle fetcher + 7-dataset max-coverage catalog
  (up to the 22 GB EyePACS+APTOS+Messidor union, ~88.7k imgs), knowledge distillation to
  `mobilenetv3_small_050`, ONNX int8 dynamic quantization with drift gates + mobile-fit verdict
- 🚧 Remaining gap to honest-97 (~8 pt): data maximalism rung (catalog priority 1–2), GPU rung,
  distill round, pseudo-label round on EyePACS, external Messidor-2 proof

## License

MIT — see [LICENSE](LICENSE). Datasets (APTOS, EyePACS, DDR) carry their own licences and terms of
use; see [docs/DATASETS.md](docs/DATASETS.md) before redistributing anything.
