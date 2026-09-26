# Getting Started

Everything you need to go from a fresh clone to a trained, exported, benchmarked model — on a
laptop CPU. For the binding module contract see [INTERFACES.md](INTERFACES.md).

## 1. Prerequisites

- Python **3.10–3.12** (developed on 3.12)
- ~2 GB free disk (PyTorch CPU wheel + artifacts)
- Optional: free **Kaggle** account (for APTOS / EyePACS) and/or **Hugging Face** account
- No GPU required anywhere in this project; AMP is opt-in and simply no-ops on CPU.

## 2. Install

```bash
git clone https://github.com/stromplayz/retinaedge-dr   # private until release
cd retinaedge-dr

make setup                    # pip install -e ".[dev]"
```

Extras:

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff, onnx, onnxruntime | tests, lint, ONNX export/parity |
| `demo` | gradio | `demo/gradio_app.py` web UI |
| `hf` | datasets, huggingface_hub | `dataset: hf` source |
| `export` | onnx2tf, onnx-graphsurgeon, ai-edge-torch | direct TFLite export path |

Then verify the stack:

```bash
python scripts/verify_env.py  # prints versions; MISSING entries tell you what to fix
```

## 3. Smoke run — the 60-second end-to-end proof

`make smoke` executes the full chain on **procedural synthetic data** (nothing is downloaded):

```bash
make smoke
# 1) python -m retinaedge.train.trainer   --config configs/train/smoke.yaml
#    -> artifacts/smoke/{best.pt,last.pt,metrics.json,history.csv}
# 2) python -m retinaedge.eval.evaluate  --config configs/train/smoke.yaml \
#        --ckpt artifacts/smoke/best.pt --split val   -> eval.json + metrics table
# 3) python -m retinaedge.export.export_onnx --config configs/train/smoke.yaml \
#        --ckpt artifacts/smoke/best.pt --out artifacts/smoke/model.onnx --img-size 64
#    -> parity check vs PyTorch (atol 1e-3)
# 4) python -m retinaedge.export.benchmark   --model artifacts/smoke/model.onnx \
#        --backend onnxruntime --img-size 64 --runs 30
```

Expected outcome: 2 epochs of `mobilenetv3_small_100` at 64 px finish in about a minute on 2 CPU
cores, an ONNX file of a few MB is written, and the benchmark prints mean/p95 latency. Exact metric
values vary by machine; the point is that the plumbing works.

## 4. Train for real

```bash
# Kaggle credentials: export KAGGLE_USERNAME=... KAGGLE_KEY=...  (or ~/.kaggle/kaggle.json)
python -m retinaedge.data.download_kaggle --handle aptos2019-blindness-detection --dest data/aptos
python -m retinaedge.data.prepare --config configs/data/aptos.yaml

python -m retinaedge.train.trainer --config configs/train/aptos.yaml
# every value is overridable on the CLI (dotted, JSON-parsed):
python -m retinaedge.train.trainer --config configs/train/aptos.yaml \
    train.lr=3e-4 train.epochs=40 train.amp=true data.batch_size=64
```

Config anatomy, loss and metric definitions: [TRAINING.md](TRAINING.md).
Dataset preparation and licensing: [DATASETS.md](DATASETS.md).

## 5. Evaluate, calibrate, export

```bash
python -m retinaedge.eval.evaluate     --config configs/train/aptos.yaml --ckpt artifacts/aptos/best.pt --split val
python -m retinaedge.eval.calibration  --config configs/train/aptos.yaml --ckpt artifacts/aptos/best.pt --out artifacts/aptos/temperature.json

python -m retinaedge.export.export_onnx --config configs/train/aptos.yaml --ckpt artifacts/aptos/best.pt \
    --out artifacts/aptos/model.onnx --img-size 224 --dynamic-batch
python -m retinaedge.export.export_tflite --direct --config configs/train/aptos.yaml \
    --ckpt artifacts/aptos/best.pt --out artifacts/aptos/dr_model.tflite --int8
python -m retinaedge.export.benchmark --model artifacts/aptos/dr_model.tflite --backend tflite --img-size 224 --runs 50
python -m retinaedge.export.metadata  --model artifacts/aptos/dr_model.tflite \
    --out-dir artifacts/aptos/android --temperature 1.07
```

Export details and the Android integration contract: [EXPORT_DEPLOY.md](EXPORT_DEPLOY.md).

## 6. Demo

```bash
pip install -e ".[demo]"
make demo                       # = python demo/gradio_app.py
# explicitly pick weights:
python demo/gradio_app.py --ckpt artifacts/smoke/best.pt
python demo/gradio_app.py --onnx artifacts/smoke/model.onnx --img-size 64
```

The demo runs with **whatever it finds**: `--onnx` (onnxruntime) → `--ckpt` (PyTorch) →
`artifacts/smoke/*` → untrained fallback weights (clearly flagged in the UI). It is a visual
sanity check, not a clinical tool.

## 7. Makefile targets

| Target | Command |
|---|---|
| `make setup` | `pip install -e ".[dev]"` |
| `make lint` / `make format` | `ruff check/format src tests scripts demo` |
| `make test` | `pytest -q` |
| `make smoke` | synthetic train → eval → ONNX → benchmark |
| `make train CFG=...` | `python -m retinaedge.train.trainer --config $(CFG)` |
| `make eval CFG=... CKPT=...` | evaluate a checkpoint |
| `make export-onnx CFG=... CKPT=... OUT=...` | ONNX export @ 224 |
| `make bench MODEL=... BACKEND=...` | latency benchmark |
| `make demo` | Gradio app |
| `make clean` | remove artifacts / caches / build dirs |

## 8. Tests

```bash
pytest -q                  # fast, network-free, no pretrained weights needed
pytest -m "not slow" -q    # skip anything marked slow
```

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install -e .` fails on README | ensure `README.md` exists at repo root (it does since docs delivery) |
| timm tries to reach the Hub and hangs | keep `model.pretrained: false` for smoke; real configs degrade gracefully to `pretrained=False` when the Hub is unreachable |
| `kagglehub` 401/403 | set `KAGGLE_USERNAME`/`KAGGLE_KEY`, accept the competition rules on kaggle.com once |
| `cv2` import error | `pip install opencv-python-headless` (already in base deps) |
| ONNX parity check fails | you probably evaluated with a different `--img-size` than the checkpoint config — keep them consistent |
| Gradio port busy | `python demo/gradio_app.py --server-port 7861` |
