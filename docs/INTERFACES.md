# RetinaEdge-DR — Interface Contract v1.0

> **READ THIS FULLY BEFORE WRITING ANY CODE.** This document is the binding contract
> between all modules/agents. If you believe a deviation is required, DO NOT deviate
> silently — flag it in the worklog and your final report.

Conventions:
- Python **3.10+**, full **type hints**, docstrings on every public function/class.
- Ruff: line-length 100, target py310. Package lives at `src/retinaedge` (editable install: `pip install -e ".[dev]"`).
- All CLIs must be runnable as `python -m retinaedge.<module> ...` AND expose a `main()` entry point.
- No network access at import time. Tests must pass **without** network and **without** pretrained weights.
- Deterministic: every randomness source flows through `seed_everything(cfg)`.
- Never commit data or model binaries (`data/`, `artifacts/` are gitignored).

## Grade scheme (fixed everywhere: Python, Kotlin, docs)
`0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR` (ICDRSS 5-grade).
**Referable DR** = grade >= 2 (any DR). Labels list order in `labels.txt` must match this.

## Config system — `retinaedge.utils.config` (IMPLEMENTED, do not rewrite)
```python
load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict
```
YAML load + dotted overrides `"train.lr=3e-4"` (values parsed via `json.loads`, fallback str).
Train configs are **self-contained**: they embed the full `data:` section.

## Seed — `retinaedge.utils.seed.seed_everything(seed: int) -> None` (IMPLEMENTED)
Seeds random/numpy/torch(+cuda), sets cudnn deterministic.

## Logging — `retinaedge.utils.logging_utils.get_logger(name: str) -> logging.Logger` (IMPLEMENTED)

## Ordinal ops — `retinaedge.models.ordinal_ops` (IMPLEMENTED, import — never re-implement)
```python
ordinal_probs(ordinal_logits: Tensor) -> Tensor   # (B, K-1) -> (B, K) grade probs, rows sum to 1
expected_grade(probs: Tensor) -> Tensor           # float E[Y], shape (B,)
hard_grade(probs: Tensor) -> Tensor               # argmax long, shape (B,)
referable_prob(probs: Tensor) -> Tensor           # P(grade>=2) = probs[:,2:].sum(-1), shape (B,)
```
Formulation: cumulative-link ordinal (CORAL-style, Shi et al. 2021). Head outputs K-1=4 logits,
each logit `g_k` is a binary "grade > k" task trained with BCE-with-logits.

## Model — `retinaedge.models.build` (owner: agent 2-b)
```python
build_model(cfg: dict) -> DrNet          # cfg["model"]: backbone (timm name), pretrained: bool, dropout: float, num_grades: int = 5
```
`DrNet(nn.Module)` must expose:
- `forward(imgs) -> dict{"ordinal_logits": (B,4) float, "refer_logits": (B,1) float}`
- `embed_dim: int` (timm backbone feature dim, `num_classes=0`)
- `predict_probs(imgs) -> (B,5) probs` (@torch.no_grad(), applies stored temperature)
- `set_temperature(t: float) -> None` (temperature scaling for calibration; default 1.0)
Backbone via `timm.create_model(name, pretrained, num_classes=0, global_pool="avg")`.
`pretrained=True` must degrade gracefully to `pretrained=False` if HF hub is unreachable (try/except).

## Loss — `retinaedge.models.loss` (owner: agent 2-b)
```python
build_loss(cfg: dict) -> DrLoss   # cfg["train"]["loss"]: {"ordinal_weight": 1.0, "refer_weight": 0.3, "focal_gamma": 0.0}
DrLoss.__call__(outputs: dict, targets: Tensor) -> tuple[Tensor, dict[str, float]]
```
Ordinal part: BCEWithLogits over the 4 cumulative tasks (`targets >= k`), mean over tasks+batch.
Refer part: BCEWithLogits on the auxiliary referable head (`targets >= 2`).
`focal_gamma > 0` switches the ordinal BCE to focal weighting.

## Datasets — `retinaedge.data.dataset` (owner: agent 2-a)
```python
build_dataset(cfg: dict, split: str, transform=None) -> torch.utils.data.Dataset
build_train_val_transforms(cfg: dict) -> tuple[train_t, val_t]
```
- `split` in {"train", "val", "test"}.
- Dataset names via `cfg["data"]["dataset"]`: `"synthetic" | "aptos" | "eyepacs" | "ddr" | "folder" | "hf"`.
- **Contract:** `__getitem__` returns `(float Tensor CHW normalized, int grade)`. Internally: load RGB uint8
  HWC -> (optionally Ben-Graham preprocess for aptos/eyepacs/ddr/folder) -> albumentations transform -> ToTensorV2.
- Synthetic dataset is procedural (no files on disk): `cfg["data"]["synthetic"] = {n_train, n_val, size, seed}`;
  class distribution ~[0.45, 0.20, 0.15, 0.10, 0.10]; draws from `numpy.random.default_rng(seed + split_offset)`.
- Transform pipelines (albumentations): train = CLAHE(prob=clahe_prob) + Resize(longest 256) +
  RandomResizedCrop(img_size, scale=(0.8,1.0)) + h/v flip + RandomRotate90 + ColorJitter + Normalize(ImageNet) + ToTensorV2;
  val = CLAHE(prob=clahe_prob) + Resize(longest 256) + CenterCrop(img_size) + Normalize + ToTensorV2.
  `img_size` from `cfg["data"]["img_size"]`.

## Ben-Graham preprocess — `retinaedge.data.ben_graham` (owner: 2-a)
```python
preprocess_ben_graham(rgb: np.ndarray, radius: int = 300) -> np.ndarray
```
Fundus mask crop -> scale to `radius` -> CLAHE on LAB L-channel (cv2). Pure cv2/numpy, no deps beyond opencv.

## Data download/prepare — `retinaedge.data.{download_kaggle, download_hf, prepare}` (owner: 2-a)
```python
# CLI: python -m retinaedge.data.download_kaggle --handle aptos2019-blindness-detection --dest data/aptos
#      (kagglehub; creds from env KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)
# CLI: python -m retinaedge.data.download_hf --repo <hf dataset id> --dest data/hf_<name>
# CLI: python -m retinaedge.data.prepare --config configs/data/aptos.yaml
```
`prepare` normalizes every source into `data/<name>/{images/, labels.csv}` with `labels.csv` columns
`image,grade` (paths relative to `images/`). Adds `data/README.md` documenting sources/licences.

## Trainer — `retinaedge.train.trainer` (owner: 2-b)
```python
# CLI: python -m retinaedge.train.trainer --config configs/train/smoke.yaml [dotted.overrides...]
#      overrides passed to load_config as positional args
def main(argv: Sequence[str] | None = None) -> int
```
Artifacts written to `cfg["train"]["save_dir"]`: `best.pt`, `last.pt`, `metrics.json`, `history.csv`.
`best.pt` payload: `{"state_dict", "cfg", "val_qwk", "temperature", "epoch"}`.
Features: AdamW + cosine schedule with linear warmup, grad clip, AMP (opt-in), EMA (opt-in),
optional class-balanced weighted sampler (toggle `train.sampler`), early stopping on val QWK
(`train.patience`), deterministic seeding, CSV+JSON logging, QWK computed via `retinaedge.train.metrics.QWKTracker`.

## Metrics — `retinaedge.train.metrics` (owner: 2-b)
```python
class QWKTracker:  # accumulates over batches
    def update(self, probs: np.ndarray, targets: np.ndarray) -> None
    def result(self) -> dict  # {qwk, auc_refer, sens, spec, threshold, ece, n}
```
QWK = sklearn `cohen_kappa_score(weights="quadratic")` on argmax grades. AUC/sens/spec on referable
prob with Youden-optimal threshold on the accumulated set. ECE = 15-bin expected calibration error on
referable prob.

## Evaluate — `retinaedge.eval.evaluate` (owner: 2-b)
```python
# CLI: python -m retinaedge.eval.evaluate --config <train cfg> --ckpt best.pt [--split val]
def main(argv=None) -> int   # writes eval.json next to the ckpt, prints table
```

## Calibration — `retinaedge.eval.calibration` (owner: 2-b)
```python
# CLI: python -m retinaedge.eval.calibration --config <train cfg> --ckpt best.pt --out temperature.json
```
Fit temperature on val split (NLL minimisation, LBFGS), write `{"temperature": t, "ece_before", "ece_after"}`.

## Export — `retinaedge.export` (owner: 2-c)
```python
# wrappers.py
class InferenceWrapper(nn.Module):
    """Wraps DrNet (+ temperature) -> forward(x) -> probs (B,5) float32. Export-safe (no dict output)."""
    def __init__(self, model: DrNet): ...
# CLI: python -m retinaedge.export.export_onnx --config configs/train/smoke.yaml \
#        --ckpt artifacts/smoke/best.pt --out artifacts/smoke/model.onnx [--img-size 64] [--opset 17] [--dynamic-batch]
#   Verifies parity vs torch (atol 1e-3) when onnxruntime importable; prints size + latency summary.
# CLI: python -m retinaedge.export.export_tflite --direct --config <cfg> --ckpt <pt> --out <tflite> [--int8] [--representative-dir d]
#      python -m retinaedge.export.export_tflite --onnx <onnx> --out <tflite> [--int8]   (onnx2tf fallback path)
# CLI: python -m retinaedge.export.benchmark --model <onnx|tflite> --backend {onnxruntime,tflite} --img-size 224 --runs 50
# CLI: python -m retinaedge.export.metadata --model <tflite|onnx> --out-dir artifacts/... [--temperature t]
#      writes labels.txt + model_info.json (input size, normalization, outputs, referable threshold)
```
Export graph contract (Android depends on this): input `float32 (1,3,H,W)` normalized with
ImageNet mean `(0.485,0.456,0.406)` std `(0.229,0.224,0.225)`; output `float32 (1,5)` grade probs.
int8 models: full-integer preferred (uint8 in/out) — the Android reader must handle BOTH uint8 and float32 IO.

## Android (owner: 2-d) — consumes the export contract above
App id `com.retinaedge.app`. Loads `file:///android_asset/models/dr_model.tflite` + `labels.txt`.
Postprocess: `grade = argmax(probs)`, `referable = probs[2]+probs[3]+probs[4] >= 0.5`.
On-device preprocessing: crop fundus circle, resize 224, **no CLAHE on device** (model is trained with
`clahe_prob=0.5` so it is robust to both), ImageNet normalize, quantize if input tensor is uint8.

## GitHub workflows (owner: 2-e)
`ci.yml` (lint + pytest + smoke train), `train.yml` (dispatch, Kaggle secrets), `export.yml`
(dispatch, consumes train artifact), `release.yml` (tag v*). Runner python 3.11, torch from the
CPU wheel index. Paths must match this contract exactly.

## Repo root map (do not create files outside your listed scope)
```
src/retinaedge/{data,models,train,eval,export,utils}
configs/{data,train,export}   tests/   scripts/   docs/   demo/   notebooks/   android/   .github/
```
