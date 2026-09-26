# Model Card — RetinaEdge-DR

**Model**: DrNet — ordinal (CORAL-style) Diabetic Retinopathy grader, mobile-edge deployment.
**Version**: 0.1.0 · **Date**: 2025 · **License**: MIT (code); datasets under their own terms.

> ⚠️ **RetinaEdge-DR is research/educational software. It is NOT a medical device, has no
> regulatory clearance (no FDA/CE/CDSCO approval), and must never be used for clinical
> decision-making, diagnosis, or triage of real patients.**

## 1. Model details

| Property | Value |
|---|---|
| Task | 5-grade DR grading from a single colour fundus photograph (ICDRSS 0–4) |
| Architecture | timm CNN backbone (default `mobilenetv3_small_100`) + 4 cumulative-logit heads + 1 referable-DR head |
| Paradigm | ordinal regression (CORAL / cumulative-link), auxiliary binary referable head |
| Inputs | RGB fundus crop, 224×224, ImageNet-normalised; Ben-Graham illumination normalisation at train time |
| Outputs | `(1,5)` grade probabilities; `P(referable) = p2+p3+p4`; temperature-calibrated |
| Calibration | temperature scaling on validation NLL; ECE reported before/after |
| Format | PyTorch → ONNX (opset 17) / TFLite (float32 + full-integer int8) |

## 2. Intended use

- **In scope**: education on edge-AI medical imaging pipelines; research on ordinal grading,
  calibration and quantization; offline triage *prototypes* on de-identified, consented images.
- **Out of scope**: clinical diagnosis; use on patients without ethics approval; use as the sole
  basis for referral decisions; paediatric or non-fundus imagery; use after image compression or
  editing that breaks the preprocessing assumptions.

## 3. Training data

| Dataset | Role | Size | Notes |
|---|---|---|---|
| APTOS 2019 (Kaggle) | primary fine-tune/eval | 3,662 labelled fundus images | India, Aravind Eye Hospital camps |
| EyePACS (Kaggle) | large-scale pretraining option | ~35k labelled | noisy grades (US telemedicine) |
| DDR | additional eval | per project terms | |
| synthetic | plumbing tests only | procedural | **never** a clinical proxy |

Known skew: grade 0/1 dominate (APTOS ≈ 49%/33%); severe grades are rare → sensitivity on
grade ≥ 2 is the metric to watch, not accuracy. Mitigations available: class-balanced sampler,
focal loss, referable-head focus (see [TRAINING.md](TRAINING.md)).

## 4. Metrics & evaluation

Report on a held-out test split, per cohort, at minimum:

- QWK (quadratic-weighted κ) on grades 0–4
- referable-DR AUC, sensitivity, specificity at the Youden threshold
- ECE before/after temperature scaling
- per-grade confusion matrix (far-misgrading is the dangerous failure mode)
- on-device: latency and int8-vs-float32 QWK drift

> **Status: placeholders.** The numbers below must be filled by the training agent once real runs
> finish; smoke/synthetic runs are pipeline checks only and carry no clinical meaning.

| Metric | APTOS val (float32) | APTOS val (int8) |
|---|---|---|
| QWK | _TBD_ | _TBD_ |
| Referable AUC | _TBD_ | _TBD_ |
| Sensitivity @ Youden | _TBD_ | _TBD_ |
| Specificity @ Youden | _TBD_ | _TBD_ |
| ECE (after calibration) | _TBD_ | _TBD_ |

## 5. Limitations & risks

1. **Population shift** — camera, ethnicity, grading protocol and prevalence differ across
   cohorts; performance measured on APTOS does not transfer automatically.
2. **Ungradable images** — poor focus, cataract, tiny FOV. The current head has no quality/
   gradability gate; treat very low-confidence outputs (`max prob` small) as "reject/retake".
3. **Ordinal confusion** — a soft output is not a safety mechanism; audit the confusion matrix.
4. **Calibration drifts under quantization** — re-measure ECE on the int8 artefact.
5. **Label noise** — EyePACS grades are known noisy; do not treat as ground truth.
6. **Overtrust** — a confident UI number invites over-reliance; the demo therefore always shows
   the full distribution and the disclaimer.

## 6. Ethical considerations

- Screening programs must keep a **human-in-the-loop** ophthalmologist pathway.
- Test for **differential performance** across sex/age/ethnicity subgroups before any deployment
  discussion; report them, do not average them away.
- Use only **de-identified, consented** data; the repo's datasets forbid image redistribution —
  respect the licences ([DATASETS.md](DATASETS.md#6-licensing--responsible-use)).
- Failure transparency: the app displays all five class probabilities, not a single verdict.

## 7. Maintenance

- Contract: [INTERFACES.md](INTERFACES.md) v1.0 (grade order and I/O are frozen).
- Retrain triggers: new backbone, dataset refresh, calibration ECE > 0.05, QWK drop > 2 pts.
- Owners recorded in the repository worklog; changes to this card accompany any retrain.
