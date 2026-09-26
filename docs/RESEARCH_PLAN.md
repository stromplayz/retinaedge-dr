# RetinaEdge-DR Research Plan — Road to 97% Referable-DR Accuracy

**Version:** 1.0 · **Status:** Active campaign · **Owner:** RetinaEdge research team
**Companion engine:** `python -m retinaedge.train.auto_improve` (the escalation ladder)
**Continuous loop:** `.github/workflows/improve.yml` (scheduled every 3 days, resumable)

---

## 1. Mission and success criteria

The mission is a clinically credible diabetic-retinopathy (DR) grading algorithm that runs
on mobile edge hardware, trained on the maximum available public data, and improved
continuously and autonomously until it reaches the accuracy goal. The goal is stated
precisely, because "97% accuracy" is meaningless until the metric, the operating point,
and the uncertainty are pinned down.

**Primary goal (the 97% target).** Referable-DR binary accuracy
`Acc_ref = (TP + TN) / N ≥ 0.97` on the held-out validation split of the Kaggle
multi-source blend, where *referable* means ICDRSS grade ≥ 2 (moderate NPDR or worse —
the level at which referral is indicated). The goal is only declared reached when the
**Wilson 95% confidence lower bound** clears 0.97 at `n ≥ 1,000` validation images, not
merely the point estimate. A point estimate of 97% on 1,000 images has a CI lower bound
near 95.8%, which is not 97% in any defensible sense.

**Secondary goals (tracked, not gated).**

| Metric | Formula | Realistic target | Ceiling evidence |
|---|---|---|---|
| QWK (5-grade) | `κ = 1 − Σ p_ij w_ij / Σ p_i· w_ij p_·j`, `w_ij = (i−j)²/(K−1)²` | ≥ 0.85 | Kaggle APTOS winners ≈ 0.905 |
| Referable AUC | Mann–Whitney U of referable scores | ≥ 0.985 | Google 2018 (EyePACS+M2) |
| Sensitivity @ goal | `TP/(TP+FN)` | ≥ 0.90 | FDA EyeArt trial: 0.955 |
| Specificity @ goal | `TN/(TN+FP)` | ≥ 0.95 | FDA EyeArt trial: 0.892 |
| ECE (calibration) | `Σ_b (n_b/N)·|acc(b) − conf(b)|` | ≤ 0.03 | post temperature scaling |

**Why binary accuracy can reach 97% when 5-grade accuracy cannot.** Five-grade
categorical accuracy is bounded by inter-grader agreement: published ophthalmologist
κ for DR severity sits around 0.65–0.85, so even a perfect clone of one grader scores
~80–90% against another on the 5-level scale. The binary referable/no-referable decision
collapses the noisy adjacent-grade boundaries (0↔1 and 1↔2 disagreement is the bulk of
human inter-rater error) into one clinically actionable question, which is why
97% is ambitious but physically attainable there. We are explicit about this so the
"loopholes" below never drift into metric gaming: every number is reported with its
definition, its n, and its CI.

---

## 2. The medical research team (simulated, with accountability)

Real clinical ML is a team sport; the plan assigns every risk to a named role even
though agents fill them.

| Role | Mandate | Deliverables |
|---|---|---|
| **Retinal specialist** | Grading protocol, referable definition, adjudication rules, clinical plausibility of errors | Grade-scheme doc; error taxonomy review per milestone |
| **Medical physicist** | Image-formation model, preprocessing physics, quality gates, domain-shift characterization | Preprocessing pipeline; quality-gate thresholds; external-set drift report |
| **ML scientist** | Objective functions, ordinal decoding, ensembling, self-training | Loss/decoding specs; ablation table per milestone |
| **Data engineer** | Ingestion, provenance, patient-aware splits, dedup | `manifest_import` outputs; provenance.json per dataset |
| **Clinical evaluator** | Honest statistics: Wilson CIs, McNemar tests, external validation | `eval/stats.py` reports; external-test scorecards |
| **MLOps engineer** | Continuous-training loop, resumable state, artifact/version hygiene | `auto_improve` ladder; `improve.yml` workflow; releases |

Decision rule: any accuracy claim without a CI, or fitted on test data, is rejected by
the clinical evaluator. This is the anti-loophole that keeps the other loopholes honest.

---

## 3. Clinical definitions and grading protocol

- **Scheme:** ICDRSS 5-level — `0` No DR, `1` Mild NPDR, `2` Moderate NPDR, `3` Severe
  NPDR, `4` Proliferative. One model, one scheme, everywhere (training, Android app,
  model card).
- **Referable DR:** grade ≥ 2. This matches the screening convention (any moderate NPDR,
  severe NPDR, or PDR is referred; mild NPDR is typically re-screened).
- **Ordinality is real:** the grades are ordered, adjacent confusion is clinically mild,
  and non-adjacent confusion (0↔4) is severe. The whole objective design (Section 5)
  exists to exploit this structure.

---

## 4. Imaging physics and preprocessing (medical physicist's section)

### 4.1 Image formation

A fundus camera illuminates the retina through the pupil and photographs the reflected
light. The captured intensity at pixel *x* follows the Retinex-style decomposition:

```
I(x) = R(x) · L(x)      →      log I(x) = log R(x) + log L(x)
```

where `R` is retinal reflectance (the diagnostically meaningful term: vessels,
microaneurysms, exudates, neovascularization) and `L` is the slowly varying illumination
field (flash falloff, vignetting, media opacity). Most preprocessing amounts to
suppressing `log L` while preserving `log R`.

Lesion visibility is wavelength-dependent: hemoglobin absorbs green strongly, so vessel
contrast peaks in the green channel; the red channel saturates on the optic disc; the
blue channel carries almost no retinal signal and mostly autofocus noise. Hence the
classical green-channel analysis and the reason color jitter is applied conservatively.

### 4.2 Field-of-view crop (Ben-Graham)

The usable retina is a bright circular field on a near-black background. The Ben-Graham
crop estimates the field, rescales it to a fixed radius, and masks the surround:

1. blur the image, threshold at ~15/255 of the peak, keep the largest contour;
2. enclose it in a circle, rescale so the circle radius = 300 px (configurable);
3. mask everything outside the circle.

This removes meaningless black borders and normalizes lesion scale across cameras —
a 40-px microaneurysm from a 60°-field camera and a 20-px one from a 45°-field camera
become comparable.

### 4.3 CLAHE (contrast-limited adaptive histogram equalization)

CLAHE equalizes `log I` locally on a tile grid, with a clip limit that redistributes
excess histogram mass uniformly across the tile to prevent noise amplification. For a
tile with `N` pixels and clip limit `c` (in units of 256), the surplus mass per bin is

```
surplus = Σ_bins max(0, h(bin) − c·N/256)      (redistributed uniformly)
```

Training applies CLAHE stochastically (`clahe_prob = 0.5`), making the model
approximately CLAHE-invariant; the Android pipeline then runs *without* CLAHE (saving
CPU) because the model tolerates both regimes. This is a deliberate "train-time physics
augmentation" loophole: on-device cost stays at the edge budget while accuracy is kept.

### 4.4 Quality gates (before a pixel reaches training)

| Gate | Measure | Threshold |
|---|---|---|
| Focus | variance of Laplacian `Σ|∇²I|` | reject bottom ~1% of corpus |
| Illumination | mean green-channel luminance inside field | percentile fence [1, 99] |
| Field size | Ben-Graham circle radius | reject if < 0.3 × image side |

Blurred or unilluminated fundi inject label noise the model cannot learn around; the
gates are applied at import time and recorded in `provenance.json` so every exclusion is
auditable.

---

## 5. Model and objective (ML scientist's section)

### 5.1 Ordinal head (CORAL-style cumulative links)

For grades `k = 0 … K−2` the model emits cumulative logits

```
P(y > k | x) = σ(f_k(x) + b_k)
```

with a shared feature `f(x)` and per-link biases `b_k` (monotonicity by construction).
Training minimizes the sum of `K−1` binary cross-entropies over the cumulative targets
`1[y > k]`; an auxiliary referable head `P(y ≥ 2 | x)` is trained jointly with weight
0.3. Total loss:

```
L = L_CORAL + λ_ref · BCE_refer ,         λ_ref = 0.3
L_CORAL = −Σ_i Σ_k [ c_ik log p_ik + (1 − c_ik) log(1 − p_ik) ]
```

with an optional focal modulation `(1 − p_t)^γ` (γ = 2) on the referable term for the
imbalanced tail. Class imbalance in sampling is handled by an inverse-frequency
`WeightedRandomSampler`, and class-balanced weighting `w_c = (1−β)/(1−β^{n_c})` remains
available as an override.

### 5.2 Decoding: from probabilities to grades

Grade probabilities `p_j` are obtained by first differences of the cumulative
probabilities (with clamping and renormalization), then decoded two ways:

- **argmax:** `ŷ = argmax_j p_j` (baseline);
- **expected-grade cut points:** `E[Y] = Σ_j j·p_j`, then `ŷ = #{k : E[Y] > c_k}` with
  `K−1 = 4` ordered cut points `c_0 > c_1 > c_2 > c_3`.

The cut points are fitted on validation by coordinate ascent maximizing QWK (tie-break:
accuracy), typically worth +1–3 QWK points on imbalanced DR data because the cut points
absorb class-prior skew that argmax cannot.

### 5.3 Referable decision

`p_ref = Σ_{j≥2} p_j` (cross-checked against the auxiliary head). The operating
threshold `t* = argmax_t Acc_ref(t)` is fitted on validation and frozen; sensitivity and
specificity at `t*` are reported next to it, and the Youden optimum
`J = sens + spec − 1` is reported for clinical context. The Android artifact keeps the
default 0.5 for safety, with `t*` shipped in `model_info.json` for screening-mode use.

---

## 6. The loophole playbook (accuracy extracted without cheating)

Each entry is a legitimate, published technique; the "loophole" framing is that they
convert *already-trained* capacity into decisions better, at zero or near-zero training
cost. Expected gains are cumulative and empirically typical for DR corpora.

| # | Loophole | Where | Expected gain |
|---|---|---|---|
| 1 | **Data maximalism** — 18,383-image 4-source blend (DDR+APTOS+M2+IDRiD) instead of 3,662-image APTOS alone | `data/kaggle_blend` | +3–6 acc pts, biggest robustness win |
| 2 | **Patient-aware splits** — ship-and-honor `split` column; zero patient overlap | `dataset.py` | removes optimistic bias (an *anti*-loophole: protects the 97% claim) |
| 3 | **Threshold/cut-point fitting** — argmax → fitted cuts + referable threshold | `eval/threshold_search.py` | +1–3 QWK pts, +0.5–1.5 acc_ref pts |
| 4 | **TTA** — horizontal-flip (+ multi-scale) probability averaging | `models/tta.py` | +0.2–1.0 pt AUC/acc |
| 5 | **EMA + model soup** — average best/last (or EMA) weights | `train.ema`, `models/soup.py` | +0.3–1.0 pt, free at inference |
| 6 | **Output ensembling** — average probabilities across ladder members | `models/ensemble.py` | +0.5–1.5 pts (evaluation/reporting path) |
| 7 | **Pretrained + progressive resizing** — ImageNet init, 160→224→256 curriculum | ladder stages s3/s5 | faster convergence, +0.5–1 pt |
| 8 | **Pseudo-labeling (self-training)** — triple-agreement filter (`max p ≥ τ` ∧ `argmax = round(E[Y])` ∧ referable-head agrees) on unlabeled EyePACS | `train/pseudo_label.py` | +0.5–1.5 pts when unlabeled pool is large |
| 9 | **CLAHE-invariance training** — stochastic CLAHE at train, none at deploy | `clahe_prob=0.5` | keeps edge cost low without acc loss |
| 10 | **Calibration** — temperature scaling post-hoc | `eval/calibration.py` | no acc change; ECE ↓ (trustworthy confidence) |

**Forbidden moves** (the clinical evaluator rejects them): fitting thresholds or cuts on
test; tuning on the external set; hash splits that leak patients; reporting the better
of val/test after peeking; augmenting the validation set; training on Messidor-2 and
claiming it external.

---

## 7. Statistical honesty protocol (clinical evaluator's section)

1. **Every headline number ships as `value (95% Wilson CI, n)`** using
   `eval/stats.py::wilson_ci`; the goal gate reads the CI lower bound
   (`accuracy_ci_report(...).target_reached`).
2. **Model comparisons** use McNemar's test on paired validation predictions
   (`mcnemar_test`); a "better" model must win with `p < 0.05` and not lose elsewhere.
3. **Bootstrapped CIs** (`bootstrap_accuracy_ci`) accompany QWK, which has no closed
   form interval.
4. **External validation** is Messidor-2 (1,744 images, google-brain 5-level adjudicated
   grades) at `data/messidor2_ext` — touched exactly once per milestone, never for
   tuning, reported with the same CI discipline.
5. **Human ceiling framing:** 5-grade QWK is contextualized against inter-grader
   agreement; referable-binary numbers are contextualized against FDA-trial devices
   (sens 0.855–0.955, spec 0.82–0.90). Beating the trial numbers *in distribution* is
   the honest reading of "97%".

---

## 8. Data strategy (data engineer's section)

| Corpus | Size | Grades | Role |
|---|---|---|---|
| **Kaggle blend** (`jin0507/…-256-x-256`): DDR + APTOS + Messidor-2 + IDRiD | 18,383 | 0–4 | train/val/test (patient-aware splits from manifest) |
| Messidor-2 external (`google-brain` grades + preprocessed images) | 1,744 | 0–4 | external validation only |
| APTOS 2019 (sovitrath gaussian-filtered mirror) | 3,662 | 0–4 | augmentation of APTOS view; excluded from val/test to avoid duplication with the blend |
| Hugging Face fundus (auto-resolved) | variable | mapped | fallback / additional diversity via `resolve_hf` |
| EyePACS unlabeled pool | ~35k | — | pseudo-labeling reservoir (loophole #8) |

Grade distribution of the blend: 0: 9,217 · 1: 1,292 · 2: 5,979 · 3: 588 · 4: 1,307
(referable share 42.8% — unusually rich in referable cases versus APTOS's 26%, which
directly helps the binary goal). Sources are recorded per-image in
`provenance.json`; source-stratified error analysis is part of every milestone review
(the blend must not be dominated by one camera system).

---

## 9. The continuous improvement loop (MLOps engineer's section)

The campaign is a **resumable escalation ladder** driven by
`retinaedge.train.auto_improve`. Each rung trains with `retinaedge.train.trainer`, then
applies the loophole stack in-process (TTA → soup → cut points → referable threshold →
Wilson CI), writes a markdown table to `runs/improvement_log.md`, and updates
`runs/improve_state.json`. The ladder:

```
s1-baseline  →  s2-longer-ema  →  s3-sharper  →  s4-backbone  →  s5-fusion
(epochs e1)     (+EMA, e2)        (+resolution)   (+effnet_lite0)  (+max res + EMA)
```

Budget profiles scale the knobs: `small` (CPU runner, e2=4, 224px) · `medium`
(e2=14) · `full` (GPU, e2=40, 320px). The loop stops early the moment the goal gate
fires (`goal_reached`), and each GitHub Actions invocation resumes from
`improve_state.json` — so the every-3-days scheduled run (`improve.yml`) is genuinely
*continuous*: download Kaggle blend → advance one or more rungs → commit the log and
state back to the repo → upload artifacts. Escalation beyond the ladder (ensembles,
pseudo-label rounds, bigger corpora) is added as new rungs rather than ad-hoc runs, so
every accuracy jump in git history has a reproducible config behind it.

---

## 10. Milestones and gates

| Milestone | Scope | Gate to pass |
|---|---|---|
| **M1 — Real-data baseline** (done) | blend ingested, patient splits, ladder s1 on runner | val QWK ≥ 0.75 pipeline proof; provenance complete |
| **M2 — Decision-rule tuning** | thresholds/cuts + TTA + EMA-soup on s1/s2 weights | val `Acc_ref` ≥ 0.90, QWK ≥ 0.80 |
| **M3 — Scale + backbone** | s3/s4 rungs at 224–256px, soup of top rungs | val `Acc_ref` ≥ 0.94, QWK ≥ 0.84, sens ≥ 0.88 @ spec ≥ 0.93 |
| **M4 — Self-training** | pseudo-label round on EyePACS pool, re-run s4/s5 | val `Acc_ref` ≥ 0.96, QWK ≥ 0.86 |
| **M5 — External proof** | one-shot Messidor-2 evaluation, calibration, model card refresh | external referable AUC ≥ 0.95; ECE ≤ 0.03; **goal: val `Acc_ref` ≥ 0.97 with Wilson LB ≥ 0.97** |
| **M6 — Edge release** | int8 TFLite parity, Android E2E, release | parity ≤ 1e-2 probs; on-device p50 ≤ 250 ms; release v1.0 |

---

## 11. Risks and mitigations

- **Source dominance in the blend** (DDR > 2/3 of images) → source-stratified metrics;
  per-source sampling weights as an override.
- **Grade-scheme heterogeneity** across sources → the blend's harmonization is audited
  against the canonical APTOS multiset (1,805/370/999/194/294) and spot-checked by the
  retinal-specialist role.
- **Val overfitting via repeated threshold fitting** → thresholds are refit per rung but
  the final gate is the untouched **test split** + one-shot external set.
- **Pseudo-label drift** → triple-agreement filter, τ ≥ 0.92, ≤ 30% of train mix per
  round, and the previous round's val QWK must not regress (McNemar-checked).
- **CPU-runner ceilings** → the ladder's `small` profile keeps the loop alive on
  Actions; `full` profile is dispatched manually when GPU (Kaggle/Colab) is wired in.
- **Metric gaming pressure near the goal** → the CI-lower-bound gate and the forbidden
  list in Section 6 are binding.

---

## 12. Key references

1. Ben-Cohen et al. / Ben-Graham — lesion-based CLAHE/crop preprocessing for DR grading.
2. Cao, Z., et al. "Rank consistent ordinal regression for neural networks with application to age estimation" (CORAL) — the cumulative-link head.
3. Wortsman et al. 2022 — Model soups.
4. Wang et al. 2019 (DDR dataset) — large-scale Chinese DR grading corpus.
5. Google (Gulshan et al. 2016; IDx-DR trial 2018; EyeArt 2021) — referable-DR operating points in regulatory trials.
6. Abramson et al. 2023 (Google AMIE-style reporting discipline) — honest reporting norms for medical AI.
7. Wilson 1927 — the score interval used for every gate.
