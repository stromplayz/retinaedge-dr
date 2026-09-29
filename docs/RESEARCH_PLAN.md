# RetinaEdge-DR Research Plan — The Road to an Honest 97%

**Version:** 2.0 · **Date:** 2026-09-29 · **Status:** Active program
**Scope:** medical rationale, imaging physics, mathematical formulations, and
the milestone ladder that governs every training/improvement round in this
repository. The plan binds the *medical research team model* to concrete
formulas and to the automation in `.github/workflows/improve.yml`.

---

## 1. Clinical Problem and Why Accuracy Percentages Are Not All Equal

Diabetic retinopathy (DR) is the leading cause of preventable blindness in
working-age adults. Of the estimated 537 million people living with diabetes
worldwide, roughly one third show some degree of DR and about one in ten has
vision-threatening DR (VTDR: severe non-proliferative or proliferative
stages, or macular edema). Screening programs succeed because DR has a long
asymptomatic window: laser therapy and anti-VEGF treatment preserve vision
when referral happens early. The clinical barrier is *grading capacity* —
trained retina graders are scarce, so automated grading directly converts
into extra screening coverage.

That is why this project fixes on two numbers, not one:

1. **Accuracy ≥ 97% (honest, see §5)** — screening triage must be right far
   more often than not, or human review queues become noise.
2. **Referable-DR sensitivity ≥ 92% at ≥ 80% specificity** — the guideline
   constraint from the ICDRSS screening literature; a model that is accurate
   on average but misses diseased eyes is unusable clinically.

All reporting in this repo therefore pairs accuracy with Wilson confidence
bounds, QWK, and the sensitivity/specificity operating point (see
`docs/CLINICAL_NOTES.md`). Percentages without intervals are treated as
marketing, not research.

## 2. The Medical Research Team Model

The program is steered by six standing roles; every milestone must be signed
off against the role's acceptance criterion before the next round starts.

| Role | Mandate | Acceptance criterion |
|---|---|---|
| Chief ophthalmologist (PI) | Clinical endpoints, referral policy | Referable sens ≥ 92%, spec ≥ 80% on held-out data |
| Retinal imaging scientist | Image physics, preprocessing fidelity | Ben-Graham crop + CLAHE validated on 500-image audit sample |
| ML research lead | Architecture, training, the accuracy ladder | Every claimed point of accuracy reproducible from a config + seed |
| Biostatistician | Statistical honesty, leakage police | All headline numbers carry Wilson 95% CI; splits salted per §5 |
| Edge deployment engineer | Mobile budget guardian | Artifact < 5 MB, p50 latency < 120 ms on mid-tier ARM |
| Regulatory & ethics officer | Intended-use discipline | Model card updated; "screening aid, not diagnosis" everywhere |

Decisions are logged in git (configs, reports, thresholds) so any clinician
can audit exactly which data, formulas, and cut-points produced a release.

## 3. Physics of Fundus Imaging — What the Preprocessing Actually Encodes

Fundus cameras image the retina through the dilated pupil. The illuminating
ring and the imaging optics share the cornea/pupil aperture (separated by the
corneal reflex geometry), which is why off-axis pupils produce *vignetting*
— a dark annulus at the image border — and why fundus photographs are
circular. The Ben-Graham preprocessing (`retinaedge.data.ben_graham`)
encodes exactly this physics:

- **Circular crop** to radius 300 px (tunable): removes the vignette border,
  leaving the retina disc the camera actually resolved. The Airy-disk
  resolution limit of the camera (`d ≈ 1.22·λ/NA`, λ ≈ 540 nm green channel
  where hemoglobin contrast peaks) is what justifies treating the inside of
  the circle as the signal support.
- **Contrast normalization**: microaneurysms are 30–60 µm structures whose
  contrast against the retinal background can be under 5% of dynamic range
  after media-opacity attenuation (lens yellowing increases with age and
  diabetes). Per-image scaling to the 90th-percentile circle brightness
  restores a comparable energy to the network regardless of acquisition
  camera.
- **CLAHE** (clip limit 2–3, tile 8×8): local histogram equalization with
  slope clipping. Formally, per tile `t` the mapping amplifies pixel `i` by
  `w_t(i) = clip(h_t(i) / N_t, c)` — the clip constant `c` bounds noise
  amplification, which matters because DR's earliest signs (microaneurysms,
  dot hemorrhages) sit one noise-sigma above background. Randomizing
  `clahe_prob=0.5` at train time makes the network robust across cameras
  (APTOS vs EyePACS vs Messidor optics differ measurably).
- **Sampling theorem discipline**: at 224 px input covering a ~45° field,
  a microaneurysm spans 1–2 px; multi-scale TTA (`scales 1.0/1.15`) gives
  the conv stack a second sampling phase, which is precisely why TTA helps
  most on small-lesion classes (§6).

## 4. Mathematical Model and Every Formula the Program Uses

**Ordinal model (CORAL-style).** The network `f` produces `K-1 = 4`
cumulative logits, one per binary task "grade > j":

```
P(y > j | x) = σ(f_j(x) − b_j),   j = 0..3
p_k(x) = P(y > k−1 | x) − P(y > k | x),  p_0 = 1 − P(y > 0), p_4 = P(y > 3)
```

Training uses BCE-with-logits on those cumulative tasks (focal weighting
optional, γ = 2), plus an auxiliary binary referable head
`P(y ≥ 2 | x)` with BCE.

**Expected grade.** `E[Y] = Σ_k k · p_k(x)` — the scalar the cut-point
search (§6) discretizes. Temperature calibration maps logits through
`σ(z/T)` with `T` fit post-hoc (ECE minimization); the pilot run reduced
ECE 0.160 → 0.088 with T = 0.442.

**Distillation (architectural training).** With teacher logits `z_t`,
student logits `z_s`, distillation temperature `T_kd`:

```
L = α · T_kd² · KL( softmax(z_s/T_kd) ‖ softmax(z_t/T_kd) ) + (1−α) · CE(p_s, y)
```

implemented in `retinaedge.train.distill` over the grade-probability
simplex, so the student inherits the teacher's ordinal structure, not just
its argmax.

**Metrics.** QWK with weight matrix `w_ij = (i−j)²/(K−1)²`; referable
operating point chosen by max-accuracy threshold with Youden J reported
alongside; McNemar's paired test (exact binomial under n < 25 discordant
pairs, else χ² with continuity correction `p = erfc(√(stat/2))`) for
model-vs-model claims.

**Honest accuracy (the 97% definition).** With `k` correct of `n`:

```
Wilson 95% CI = [ (p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² )) / (1 + z²/n) ]
```

The program claims "97%" **only** when `lower bound ≥ 0.97` with `n ≥ 1000`
held-out graded images — at n = 1000 that requires 985+ correct, not 970.

## 5. Data Strategy for Maximum Accuracy

The validated Kaggle catalog (`configs/data/kaggle_dr_catalog.yaml`,
authenticated by the repo's KGAT token) is ordered by expected information
gain: the 22 GB EyePACS+APTOS+Messidor union (~88.7k images) at priority 1,
the 10.8 GB five-source union (adds DDR/IDIRD) at priority 2, down to the
447 MB Gaussian-filtered APTOS used for CI calibration. Grading discipline:
all sources map to ICDRSS 0–4 through `resolve_hf` / provenance records;
`salt_salt` stays fixed per program phase so train/val/test assignment is
deterministic across rounds (a different salt resets the split honestly —
never reuse a test set that tuned any hyper-parameter). Pseudo-labeling
(`retinaedge.train.pseudo_label`, triple-consistency filter at τ = 0.92)
converts unlabeled fundus pools into extra training rows only when
confidence, argmax-vs-round(E[Y]), and referable-head agreement all hold.

## 6. The Accuracy Ladder — Ordered "Loopholes" with Expected Yields

Each rung adds evaluation-time or data-time leverage **without inflating the
mobile graph**, in the order the engine (`retinaedge.train.auto_improve`)
executes them:

| Rung | Mechanism | Expected gain | Cost |
|---|---|---|---|
| 0 baseline | argmax decode | — | — |
| 1 TTA | hflip + multi-scale prob. averaging | +0.2–0.4 pt | 4× inference (eval only) |
| 2 cut-points | coordinate ascent on E[Y] cuts, searched on train | +0.5–1.5 pt acc | zero (frozen thresholds) |
| 3 model soup | weight averaging of same-basin checkpoints | +0.2–1.0 pt | zero (one graph) |
| 4 ensemble | probability averaging across models | +0.3–1.5 pt | eval only |
| 5 distill | teacher → mobile student (KD) | recovers 97–100% of teacher QWK at ~40% params | one training run |
| 6 pseudo-label | self-training on unlabeled pool | +0.3–1.0 pt | one training run |
| 7 more data | catalog priorities 1–2 (up to ~150k images) | +1–3 pt | scheduled run |

Cut-points and calibration constants are frozen into the exported artifact
(`thresholds.json`) so shipped behavior is deterministic; the engine stops
early the moment the Wilson lower bound meets 0.97 at n ≥ 1000.

## 7. Mobile Architecture Program

The deliverable is a single graph that satisfies the edge budget:

- **Backbones:** `mobilenetv3_small_100` (1.5 M params, current pilot) and
  `mobilenetv3_small_050` (~0.24 M, distillation student,
  `configs/train/mobile_student.yaml`).
- **Quantization:** ONNX int8 dynamic (`retinaedge.export.quantize_onnx`)
  with automated drift check (max |Δp| < 0.05 gate) and mobile-fit verdict
  (≤ 5 MB, p50 latency ≤ 120 ms); full-integer TFLite via
  `retinaedge.export.export_tflite` for NNAPI/Hexagon paths.
- **Distillation transfer:** the small_050 student trains against the
  small_100 teacher on the *same split salt*, so val comparisons are paired
  and McNemar-testable.

## 8. Continuous Improvement Engine and Governance

`.github/workflows/improve.yml` runs the resumable escalation ladder on every
dispatch (and on a 3-day cadence). It pulls the 18,383-image Kaggle blend
(DDR+APTOS+Messidor-2+IDRiD, patient-aware splits) using the
`KAGGLE_API_TOKEN` secret, advances `retinaedge.train.auto_improve`
(s1-baseline → s2-longer-ema → s3-sharper → s4-backbone → s5-fusion, each
rung keeping every previous winning trick), evaluates the full loophole stack
per run (base / TTA / soup / soup+tta × cut-point decoding × referable
threshold), and commits `runs/improvement_log.md` +
`runs/improve_state.json` back to the repo — so every improvement round is
auditable in git history and the loop resumes exactly where it stopped.

**Live status (2026-09-29):** s1→s5 executed on CPU runners; best variant
`soup+tta` at s5-fusion — referable accuracy **0.890** (Wilson 95%
[0.875, 0.904], n = 1843), QWK **0.845**, sens 0.855 @ spec 0.917. Gap to the
honest gate: ≈ 8 pt, to be closed by the data-maximalism rungs below. The
loop's exit criterion remains §5's Wilson-lower-bound test; until then each
round's next move is data expansion (22 GB union), distillation, and
pseudo-labeling — the "train continuously and improve towards the goal"
requirement, formalized.

## 9. Milestones

| ID | Milestone | Acceptance | Status |
|---|---|---|---|
| M0 | Pipeline + pilot training inside GitHub Actions | APTOS 3.6k, QWK 0.50, ECE 0.088, TFLite 1.83 MB | ✅ done |
| M1 | Kaggle blend + resumable escalation ladder | 18,383-img blend, patient-aware splits, state committed to git | ✅ done |
| M2 | Escalation rungs s1→s5 on the blend | s5-fusion: acc_refer 0.890 (CI [0.875, 0.904]), QWK 0.845 | ✅ done (CPU) |
| M3 | Mobile architecture program | distill to small_050 + int8 ONNX drift gate + mobile-fit verdict | ✅ code shipped (round pending) |
| M4 | Data maximalism | catalog priority 1–2 resolved (~88.7k imgs), retrain + GPU rung | 🚧 next |
| M5 | Honest 97% gate | Wilson LB ≥ 0.97 at n ≥ 1000 + referable sens ≥ 92% | 🎯 target |
| M6 | v1.0 clinical-audit release | model card, calibration report, thresholds frozen, Android demo | gated on M5 |

## 10. Risks and Safeguards

- **Leakage:** salted splits, provenance JSON per resolved dataset, cuts
  searched on train only.
- **Label noise:** pseudo-label triple filter; APTOS permutation repair from
  the earlier pilot is now part of the resolver.
- **Domain shift:** multi-source catalog + CLAHE randomization + TTA.
- **Over-claiming:** Wilson lower-bound gate; McNemar for paired deltas;
  bootstrap cross-check.
- **Calibration drift after quantization:** drift gate in
  `quantize_onnx.py`; int8 artifacts ship with their own eval JSON.
- **Ethics:** screening triage aid; referral decisions remain clinician's;
  model card carries intended-use limits (see `docs/MODEL_CARD.md`).
