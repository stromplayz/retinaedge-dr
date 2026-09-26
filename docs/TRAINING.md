# Training

Reference for `retinaedge.train.trainer`, its config surface, loss and metrics.
Everything here follows the frozen contract in [INTERFACES.md](INTERFACES.md) v1.0.

## 1. Launch

```bash
python -m retinaedge.train.trainer --config configs/train/smoke.yaml
python -m retinaedge.train.trainer --config configs/train/aptos.yaml train.lr=3e-4 train.epochs=40
# (or the console script: retinaedge-train --config ...)
```

Overrides are `dotted.key=value` pairs parsed by
`retinaedge.utils.config.load_config` (JSON-parsed, so `true`, `3e-4`, `[1,2]` all work).

Artifacts land in `cfg["train"]["save_dir"]`:

| File | Contents |
|---|---|
| `best.pt` | `{"state_dict", "cfg", "val_qwk", "temperature", "epoch"}` — best val QWK |
| `last.pt` | same payload, last epoch (resume/inspection) |
| `metrics.json` | final best-epoch metric block |
| `history.csv` | one row per epoch: losses, lr, val metrics |

## 2. Config anatomy

Complete annotated example (values from `configs/train/smoke.yaml`):

```yaml
data:                      # train configs are SELF-CONTAINED: they embed data
  dataset: synthetic       # synthetic | aptos | eyepacs | ddr | folder | hf
  img_size: 64             # square; export must use the same value
  batch_size: 16
  num_workers: 0           # >0 on real runs (linux); 0 keeps Windows/simple envs happy
  clahe_prob: 0.0          # 0.5 on real fundus data; 0.0 for synthetic
  synthetic:               # only read when dataset == synthetic
    n_train: 240
    n_val: 80
    size: 64
    seed: 0

model:
  backbone: mobilenetv3_small_100   # any timm classification name
  pretrained: false                 # true on real runs; degrades to False offline
  dropout: 0.1
  num_grades: 5                     # fixed by the grade scheme — do not change

train:
  epochs: 2
  lr: 0.002               # AdamW peak lr
  weight_decay: 0.01
  warmup_epochs: 0        # linear lr warmup before cosine decay
  grad_clip: 5.0          # global-norm clip; null to disable
  amp: false              # mixed precision (opt-in; no-op on CPU)
  ema: false              # exponential moving average of weights (opt-in)
  sampler: false          # class-balanced weighted sampler (opt-in)
  patience: 5             # early stopping on val QWK (epochs without improvement)
  seed: 0                 # single source of randomness -> seed_everything(cfg)
  save_dir: artifacts/smoke
  loss:
    ordinal_weight: 1.0   # weight of the 4-task cumulative BCE
    refer_weight: 0.3     # weight of the auxiliary referable-DR BCE
    focal_gamma: 0.0      # >0 switches ordinal BCE to focal weighting

eval:
  batch_size: 32
```

Reproducibility: every randomness source (init, shuffling, augmentation, synthetic generator)
flows through `retinaedge.utils.seed.seed_everything(cfg["train"]["seed"])`. Two runs with the same
config on the same hardware are bit-identical; a fixed seed in CI keeps the smoke metrics stable.

## 3. Model & loss

**DrNet** (built by `retinaedge.models.build.build_model`):

- timm backbone, `num_classes=0`, `global_pool="avg"` → feature vector (`embed_dim`).
- Ordinal head: **K−1 = 4** logits, one binary task each: `g_k = P(grade > k)`.
- Refer head: 1 logit for `P(grade ≥ 2)`.
- `predict_probs(imgs)` returns `(B, 5)` grade probabilities with the stored temperature applied;
  `set_temperature(t)` swaps it after calibration.

**DrLoss** = `ordinal_weight · BCE_coral + refer_weight · BCE_refer`

- Ordinal part: for each of the 4 cumulative tasks the binarised target is `y > k`, trained with
  `BCEWithLogits`, averaged over tasks and batch. This is the CORAL formulation — it encodes grade
  *order* and empirically reduces far-misgrading vs. plain softmax.
- Refer part: plain BCE on `y ≥ 2`.
- `focal_gamma > 0` replaces the ordinal BCE with focal weighting `(1−p)^γ` to emphasise hard
  examples on imbalanced cohorts.

Grade probabilities come from the shared, tested ops in
`retinaedge.models.ordinal_ops` (`ordinal_probs`, `expected_grade`, `hard_grade`,
`referable_prob`) — import, never re-implement.

## 4. Metrics

`retinaedge.train.metrics.QWKTracker` accumulates `(probs, targets)` over batches and reports:

| Metric | Definition | Why it matters |
|---|---|---|
| `qwk` | quadratic-weighted Cohen's κ on argmax grades (sklearn, `weights="quadratic"`) | primary selection metric — punishes far-off grades harder |
| `auc_refer` | ROC-AUC of `P(grade ≥ 2)` vs. binary referable truth | screening-utility view |
| `sens` / `spec` | sensitivity / specificity at `threshold` | operating point |
| `threshold` | Youden-optimal (`sens + spec − 1` maximiser) on the accumulated val set | best trade-off for *this* cohort |
| `ece` | 15-bin expected calibration error on `P(grade ≥ 2)` | can we trust the number shown to the user? |
| `n` | accumulated samples | sanity |

Trainer selects `best.pt` by **val QWK**; early stopping (`train.patience`) counts epochs without
QWK improvement.

## 5. Calibration

After training, temperature scaling rescales the 4 cumulative logits (`logits / T`) to minimise
validation NLL (LBFGS, single scalar):

```bash
python -m retinaedge.eval.calibration --config configs/train/aptos.yaml \
    --ckpt artifacts/aptos/best.pt --out artifacts/aptos/temperature.json
# -> {"temperature": t, "ece_before": ..., "ece_after": ...}
```

The fitted `t` is baked into checkpoints/exports (DrNet `set_temperature`, export metadata) —
inference code never has to know about it. See
[notebooks/02_calibration_and_export.ipynb](../notebooks/02_calibration_and_export.ipynb) for a
worked example with reliability diagrams.

## 6. Practical defaults

| Setting | Smoke (64 px) | Real (APTOS/EyePACS 224 px) |
|---|---|---|
| backbone | `mobilenetv3_small_100`, `pretrained: false` | same, `pretrained: true` |
| epochs / lr | 2 / 2e-3 | 25–40 / 3e-4 (backbone lr ×0.1 is a good start) |
| warmup | 0 | 1–2 epochs |
| `sampler` / `focal_gamma` | off / 0 | try `true` / `2.0` |
| `amp` | off (CPU) | on (GPU) |
| early stop | patience 5 | patience 5–8 |

**Never** change `num_grades`, the grade ordering, or the export I/O contract — the Android client
and `labels.txt` depend on them byte-for-byte.
