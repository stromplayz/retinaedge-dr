# Contributing

Thanks for helping make edge DR screening better. This repo is research software — quality
matters more than volume.

## Ground rules

1. **Read `docs/INTERFACES.md` first** — it is the binding contract between modules
   (grade scheme, export graph, CLI signatures). PRs that break it will be rejected unless
   the contract is updated in the same PR.
2. **Never commit** patient data, model binaries (>1 MB), tokens or Kaggle credentials.
   `data/`, `artifacts/`, `*.pt`, `*.onnx`, `*.tflite` are gitignored; binaries belong in
   GitHub Releases or Git LFS.
3. Every PR must keep `make lint` and `make test` green. The smoke pipeline
   (`make smoke`) must stay under 5 minutes on a 2-core CPU runner.

## Workflow

```bash
git clone https://github.com/stromplayz/retinaedge-dr && cd retinaedge-dr
make setup                     # pip install -e ".[dev]"
python scripts/verify_env.py   # sanity check
git checkout -b feat/my-change
# ... code ...
make lint && make test && make smoke
git commit -m "feat(scope): change"   # conventional commits preferred
git push -u origin feat/my-change     # open a PR on GitHub
```

## Conventions

- Python 3.10+, full type hints, docstrings on public API. Ruff (`E,F,W,I,UP,B`) enforced.
- Configs are YAML; every new tunable goes behind a config key, never a hardcoded constant.
- Tests are network-free and deterministic: no dataset downloads, no pretrained weight
  fetches, seeds via `seed_everything`. Mark slow tests `@pytest.mark.slow`.
- Kotlin: keep the Android module binary-free (no committed icons/binaries), unit-testable
  postprocessing (pure-JVM `DrPostprocess`), and match the export contract exactly.
- Medical/clinical claims must cite literature in the PR description; remember this code is
  research-only (see `docs/CLINICAL_NOTES.md`).

## CI expectations

- `ci.yml` runs lint + tests + the full smoke pipeline on every push/PR.
- New workflows need a dry-run note in `docs/WORKFLOWS.md` and must pin action versions.
