# Security Policy

## Reporting a vulnerability

Open a **private security advisory** (GitHub → Security → Advisories → New draft security
advisory) or email the owner. Do not open a public issue for exploitable flaws.

Scope: anything that would let an attacker execute code via crafted model files
(`.onnx`/`.tflite` loading paths), the Gradio demo upload surface, the GitHub Actions
workflows (e.g. workflow-injection through dispatch inputs), or the Android app's
model loading.

## Handling of credentials

- The GitHub PAT used to bootstrap this repository was used **out-of-band** (by the
  repository owner) and is **never** stored in the repo, workflows, or logs.
- Kaggle credentials enter CI only via repository **secrets**; `data/` and credential files
  are gitignored.
- If you accidentally paste a token into an issue/PR: revoke it immediately at
  github.com/settings/tokens, then clean history if needed (git filter-repo) and force-push.

## Model supply chain

- Model artifacts are produced by the workflows in this repo; treat externally obtained
  `.pt/.onnx/.tflite` files as untrusted input.
- The export pipeline runs a **numerical parity check** against the source checkpoint —
  a mismatch fails the build rather than shipping.
- Dependencies are pinned by version range in `pyproject.toml`; Dependabot watches pip and
  GitHub Actions weekly.

## Research-use disclaimer

Nothing in this repository is approved for clinical use. Safety-relevant model behaviour
(systematic mis-grading for a population) should be reported via issues tagged `safety`
(see `docs/CLINICAL_NOTES.md`).
