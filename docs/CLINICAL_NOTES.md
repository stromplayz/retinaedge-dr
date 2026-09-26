# Clinical Notes & Safety

> **Read this before using any artifact produced by this repository.**

## Status

RetinaEdge-DR is a **research prototype**. It is **not** a medical device, has no CE mark,
no FDA 510(k)/De Novo clearance, no CDSCO approval, and is not listed in any medical device
register. The phrase "clinical-grade-oriented" in this repository describes *engineering
practices borrowed from clinical ML* (ordinal targets, calibration, external-validation
scaffolding, model cards) — **not** a claim of clinical validity.

## What referable DR means here

- Grades follow ICDRSS: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative.
- **Referable DR = grade ≥ 2** (International Council of Ophthalmology / ICDR standards):
  the patient should be examined by an eye-care professional.
- The app renders a *referral suggestion banner* from `P(grade ≥ 2) ≥ 0.5`. This threshold
  is a placeholder; a deployed screening program would tune it on a validated cohort for a
  **minimum sensitivity** target (literature programs such as EyeArt / IDx-DR operated at
  ~85-96% sensitivity with ~70-90% specificity for referable DR), accepting lower specificity.

## Mandatory validation ladder before any real-world claim

1. **Internal validation** — stratified val split of the training dataset (APTOS or EyePACS).
   Report QWK, AUC-referable, sensitivity/specificity at the chosen threshold, ECE.
2. **External validation** — datasets the model never saw, from *different cameras and
   populations*: Messidor-2, IDRiD, EyePACS test. Report the same metrics **per dataset**
   and per subgroup (camera type, image quality, ethnicity where available).
3. **Robustness analysis** — low-quality/ungradable images, different FOVs, compression
   artifacts. A quality gate ("image ungradable" output) is standard in deployed systems and
   is on our roadmap (see `docs/ARCHITECTURE.md` §7).
4. **Calibration check** — ECE on every external set; refit temperature per deployment site
   if needed.
5. **Prospective evaluation** — only then, and only with clinical partners, IRB oversight
   and regulatory counsel.

## Non-negotiable usage constraints

- **Do not** use any output to withhold or delay care.
- **Do not** deploy to patients without a licensed clinician in the loop.
- **Do not** treat on-device predictions as diagnoses; they are *referral triage hints*
  for research.
- Fundus photographs alone under-diagnose some conditions (e.g. macular edema severity);
  OCT and clinical exam remain necessary.

## Privacy

The Android app runs inference **fully on-device** — images never leave the phone, no
analytics, no network permission. Training datasets remain on the machine/runner that
downloaded them and are never committed. If you extend the app, preserve these properties.

## Incident reporting

If you find a safety-relevant behaviour (e.g. systematic under-grading on a population),
please open a GitHub issue tagged `safety` with de-identified example characteristics.
See `SECURITY.md` for security-relevant disclosure.

## Key references

- Gulshan et al., JAMA 2016;316(22):2402-2410.
- Ting et al., JAMA 2017;318(22):2211-2223.
- Abràmoff et al., *Pivotal trial of an autonomous AI diagnostic system (IDx-DR)*, npj Digital
  Medicine 1:39 (2018).
- International Council of Ophthalmology guidelines for diabetic eye care (2017).
- NHS Diabetic Eye Screening programme: grading definitions and referral thresholds.
