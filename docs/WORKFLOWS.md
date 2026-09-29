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
Training *inside* the Codespace (2-core free machine, ~60 free core-hours/month):

```bash
# 1. Full synthetic pipeline proof (~2 min on the 2-core machine)
make smoke

# 2. Real-data training inside the Codespace (recommended for CPU pilots)
export KAGGLE_USERNAME=... KAGGLE_KEY=...          # optional, for Kaggle sources
python -m retinaedge.data.resolve_hf --out data/online_dr \
    --bake-ben-graham --json provenance.json       # pulls a real DR dataset from HF Hub
python -m retinaedge.train.trainer --config configs/train/online_pilot.yaml
python -m retinaedge.eval.evaluate --config configs/train/online_pilot.yaml \
    --ckpt artifacts/online/best.pt --split val
python -m retinaedge.export.export_onnx --config configs/train/online_pilot.yaml \
    --ckpt artifacts/online/best.pt --out artifacts/online/model.onnx --img-size 160
```

The same flow runs unattended via **Actions → "Train online (real data on runner)"** —
that is the recommended path (artifacts are uploaded automatically and `export.yml`
can consume them for TFLite conversion). Stop or delete idle Codespaces to conserve
the free storage quota. The Gradio demo (port 7860) is forwarded automatically.

## Notes & limits

- Public GitHub runners: 2 cores / 7 GB RAM / 14 GB disk. The smoke CI job uses ~3 minutes.
- Private-repo Actions consume free-tier minutes; `ci.yml` is kept deliberately lean.
- Wheels for `ai-edge-torch`/TensorFlow are installed only in `export.yml` (single-use),
  keeping CI fast and the dependency tree clean.
- If Kaggle throttles a competition download, re-running the workflow resumes from cache
  inside the same job only — for repeated experiments prefer a self-hosted runner with a
  persistent `data/` cache.

## Continuous improve — every-3-hours cadence (v0.4.0)

`improve.yml` is the autonomous campaign loop. It runs on a cron schedule
**every 3 hours** (`17 */3 * * *` UTC) and on manual dispatch, and each
invocation performs five phases:

1. **Restore continuity** — downloads the `improve-run` artifact from the last
   successful run and reinstates `runs/*.pt` so model soup and the s7-distill
   teacher survive ephemeral runners.
2. **Resolve data** — scheduled runs rotate the Kaggle dataset handle across
   the top-priority validated catalog entries (`scripts/rotate_dataset.py`,
   rotation index = campaign history length), with automatic fallback to the
   default handle if a mirror fails. Manual dispatch keeps the handle you type.
3. **Train + loophole stack** — resumes `runs/improve_state.json`, trains up to
   `max_stages` (default 3, bounded for the 3-hour cadence) ladder stages, and
   evaluates base / TTA / soup / soup+TTA variants with Wilson-CI honesty.
4. **Decide a release** — `scripts/bump_release.py` maps the campaign onto a
   semantic version: goal reached (97% referable accuracy, CI-gated when
   `--require-ci`) becomes **v1.0.0**; otherwise a champion improvement of
   >= 0.005 since the last release cuts a minor bump (v0.4.0, v0.5.0, ...);
   post-1.0 improvements cut patch bumps. The decision ledger lives in
   `runs/version_state.json` (committed).
5. **Persist + publish** — state/log/ledger are committed back to `main`, and
   on a bump the workflow tags `vX.Y.Z` and publishes a GitHub Release whose
   notes carry the metrics table and whose assets include the champion
   checkpoint.

**Rounds:** when a full ladder round finishes without the goal, the campaign
escalates the budget (small -> medium -> full) and starts the next round —
champion preserved, `--max-rounds` caps the escalation. A dispatched
`--budget` seeds fresh campaigns only; it can never downgrade an escalated one.

**Minutes budget:** on private repos, 8 runs/day x ~30-60 min consumes roughly
4,000-9,000 Actions minutes/month. If the free tier (2,000 min) is exhausted,
either make the repo public (unlimited standard-runner minutes), raise the
cadence back to 6-12 hours, or attach a larger Actions quota.
