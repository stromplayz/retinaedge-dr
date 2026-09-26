# Benchmarks

All numbers below come from **real runs of this repository's own pipelines** (GitHub
Actions "Train online" runs and local smoke runs). Nothing is estimated from papers.
Each table states the data source, config and caveats — do not compare across tables.

> Research prototype — **not a medical device**. These numbers do not imply clinical
> validity. See `docs/CLINICAL_NOTES.md` for the mandatory validation ladder.

## 1. Online pilot — real APTOS 2019 data, trained inside GitHub Actions

| | |
|---|---|
| Run | `train-online.yml` (GitHub-hosted runner, CPU) |
| Data | APTOS 2019 mirror `sngsfydy/aptos_gaussian_filtered` (3,662 fundus photos), resolved at runtime by `retinaedge.data.resolve_hf`, grades repaired to ICDRSS via distribution-matching (`icdrss5_aptos_permutation_repaired`), Ben-Graham baked, CLAHE p=0.5 train-time |
| Config | `configs/train/online_pilot.yaml` — MobileNetV3-Small (1.52M params, ImageNet init), 160px, batch 32, AdamW 1e-3, focal ordinal loss + referable head, class-balanced sampler |
| Split | 80/15/5 by md5 salt `online_v2` (2,922 train / 589 val) |

*Table filled from `artifacts/online/{metrics,eval}.json` after the run completes.*

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
