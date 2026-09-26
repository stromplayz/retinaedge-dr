# Benchmarks

All numbers below come from **real runs of this repository's own pipelines** (GitHub
Actions "Train online" runs and local smoke runs). Nothing is estimated from papers.
Each table states the data source, config and caveats — do not compare across tables.

> Research prototype — **not a medical device**. These numbers do not imply clinical
> validity. See `docs/CLINICAL_NOTES.md` for the mandatory validation ladder.

## 1. Online pilot — real APTOS 2019 data, trained inside GitHub Actions

| | |
|---|---|
| Run | `train-online.yml` run [#2](https://github.com/stromplayz/retinaedge-dr/actions/runs/36259645931) (GitHub-hosted runner, CPU, total ~24 min wall) |
| Data | APTOS 2019 mirror `sngsfydy/aptos_gaussian_filtered` @ revision `d9cae240`, 3,662 fundus photos, resolved at runtime by `retinaedge.data.resolve_hf`, grades repaired to ICDRSS (`icdrss5_aptos_permutation_repaired` — mirror stored alphabetical indices), Ben-Graham baked, CLAHE p=0.5 |
| Config | `configs/train/online_pilot.yaml` — MobileNetV3-Small (1,522,981 params, ImageNet init), 160px, batch 32, AdamW 1e-3, focal ordinal loss + referable head, class-balanced sampler, 15 epochs budget, early stop @ 9 |
| Split | 80/15/5 by md5 salt `online_v2` (2,942 train / 557 val) |

### Results (best checkpoint, epoch 3/9)

| Metric (val split, n=557) | Value |
|---|---|
| Quadratic weighted kappa | **0.5034** (best val QWK 0.5058 @ epoch 3) |
| Referable-DR AUC (P(grade≥2)) | **0.8457** |
| Sensitivity @ Youden threshold 0.455 | **0.9163** |
| Specificity @ same threshold | 0.6384 |
| ECE before calibration | 0.1603 |
| ECE after temperature scaling (T=0.442) | **0.0876** |

Export (CI, parity-gated): ONNX 6.11 MB, opset 17, PyTorch parity max diff 1.79e-07;
latency p50 1.15 ms on the Actions runner CPU; **p50 1.47 ms / 323 fps on a 2-core
sandbox CPU @160px** (onnxruntime, 40 runs).

### Honest caveats (read before quoting these numbers)

- 9 CPU epochs at 160px is a *pilot*, not a finished model. The confusion matrix shows the
  model currently concentrates mass on grades 0 and 4 (ordinal probabilities under-trained);
  QWK 0.50 and AUC 0.85 come mostly from the referable separation. Published APTOS systems
  reach QWK ≈ 0.85+ with 224–380px inputs, 30–100 GPU epochs, TTA and heavy preprocessing.
- The data source is a community mirror with pre-applied gaussian/ben-graham-style filtering;
  provenance (repo id + revision + repair scheme) is recorded in `provenance.json` and must
  be quoted with any result.
- Next steps for quality: more epochs at 224px, 5-fit ensembling, TTA on the server CLI, and
  external validation on Messidor-2 / IDRiD (see `docs/CLINICAL_NOTES.md`).

## 2. Smoke pipeline (synthetic data, CI gate only)

| Metric | Value |
|---|---|
| Purpose | Prove train → eval → export → benchmark wiring; **not** a model quality number |
| Params | 1.52 M (MobileNetV3-Small, ordinal + referable heads) |
| ONNX | 6.11 MB, opset 17, parity max diff 7.45e-08 |
| ONNX CPU latency (2-core sandbox) | p50 0.63 ms, p95 0.76 ms @ 64px (~1,377 fps) |
| Val QWK | ~0 (random synthetic labels — expected) |

## 3. Edge budget targets (design goals, from docs/ARCHITECTURE.md)

| Budget item | Target | Status |
|---|---|---|
| int8 TFLite size | ≤ 5 MB | pending first int8 export |
| On-device latency (mid-range Android, 4 threads) | ≤ 150 ms @ 224px | pending on-device bench |
| Server ONNX latency (2-core CPU) | ≤ 30 ms @ 224px | measured post-pilot |
