# Workflows, Secrets & Codespaces

## Pipeline overview

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR → `main` | ruff lint → pytest (network-free) → **full smoke pipeline** (train → eval → export → parity → benchmark) on 2-core CPU; uploads `model.onnx` + metrics as artifacts |
| `train.yml` | manual dispatch | downloads Kaggle data with your secrets → prepares it → trains the selected config → evaluates → calibrates → uploads `artifacts/<run>` |
| `export.yml` | manual dispatch | takes a training run artifact (or a fresh smoke run), exports ONNX **with parity check**, converts to int8 TFLite (`ai-edge-torch`, `onnx2tf` fallback), benchmarks, writes labels + metadata |
| `release.yml` | tag `v*` | packages model binaries + metadata and publishes a GitHub Release |

## Required repository secrets (Settings → Secrets and variables → Actions)

| Secret | Where to get it | Used by |
|---|---|---|
| `KAGGLE_USERNAME` | kaggle.com → Account → API | `train.yml` |
| `KAGGLE_KEY` | kaggle.com → Account → Create New Token | `train.yml` |

Or from the CLI:

```bash
gh secret set KAGGLE_USERNAME --body "<username>"
gh secret set KAGGLE_KEY --body "<key>"
```

No secrets are needed for `ci.yml` — the smoke pipeline runs on procedural synthetic data.

## Running real training on CI

Actions → **train** → Run workflow:
- `config`: `configs/train/aptos_mobilenetv3.yaml` (default)
- `overrides`: extra dotted args, e.g. `train.epochs=5 data.img_size=160` for a cheap real-data pilot
- `gpu`: `true` only if you have a self-hosted GPU runner labelled `[self-hosted, gpu]` —
  otherwise train on Kaggle/Colab (see `notebooks/02_train_colab.ipynb` or Kaggle Notebooks;
  public runners are CPU-only and too slow for 224px full runs).

Artifacts appear under the run page (`artifacts/aptos_mv3`) — download `best.pt` for local export,
or dispatch **export** pointing at the run.

## Running export on CI

Actions → **export** → Run workflow:
- `train_run_id`: the numeric id of a completed `train` run (optional — falls back to smoke)
- The workflow exports ONNX (with PyTorch parity gate), int8 TFLite, benchmarks both, and
  uploads `model.onnx` / `dr_model.tflite` / `labels.txt` / `model_info.json`.

## Cutting a release

```bash
git tag v0.2.0 -m "second model drop"
git push origin v0.2.0            # release.yml takes over
```

## Codespaces

The repo ships a devcontainer (Python 3.11 + JDK 17 + Gradle): **Code → Codespaces →
Create codespace on main**. Post-create runs `pip install -e ".[dev]"` and `verify_env`.
From there you can run `make smoke`, the Gradio demo (port 7860 is forwarded), and even the
Android Gradle build (`./gradlew :app:assembleDebug` inside `android/`) — no local install
needed. Kaggle credentials can be forwarded from your machine via the `remoteEnv` mapping
(they are never persisted in the container).

## Notes & limits

- Public GitHub runners: 2 cores / 7 GB RAM / 14 GB disk. The smoke CI job uses ~3 minutes.
- Private-repo Actions consume free-tier minutes; `ci.yml` is kept deliberately lean.
- Wheels for `ai-edge-torch`/TensorFlow are installed only in `export.yml` (single-use),
  keeping CI fast and the dependency tree clean.
- If Kaggle throttles a competition download, re-running the workflow resumes from cache
  inside the same job only — for repeated experiments prefer a self-hosted runner with a
  persistent `data/` cache.
