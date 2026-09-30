
---
Task ID: 2-a
Agent: data-pipeline builder
Task: Data pipeline module (retinaedge.data.*) per contract v1.0

Work Log:
- src/retinaedge/data/ben_graham.py: fundus mask crop -> radius rescale (default 300) -> CLAHE on LAB L-channel; pure cv2/numpy, deterministic, degenerate-image safe
- src/retinaedge/data/dataset.py:
  - build_dataset(cfg, split, transform=None) for synthetic|aptos|eyepacs|ddr|folder|hf
  - build_train_val_transforms(cfg): CLAHE(p=clahe_prob) + LongestMaxSize(256) + RandomResizedCrop/HFlip/VFlip/RandRotate90/ColorJitter + Normalize(ImageNet) + ToTensorV2; val = CLAHE + LongestMaxSize + CenterCrop(pad_if_needed) + Normalize
  - SyntheticDRDataset: grades drawn from default_rng(seed + split_offset), dist [0.45,0.20,0.15,0.10,0.10]; per-index image rendering is index-seeded (deterministic under any shuffle/worker layout); split offsets train=0/val=1e6/test=2e6
  - FileListDataset: images/ + labels.csv (image,grade); __getitem__ -> (float CHW ImageNet-normalized, int grade)
  - split_tags(): deterministic md5-hash train/val/test assignment (val_fraction 0.15, test_fraction 0.0, optional split_salt) — no split files stored
- src/retinaedge/data/download_kaggle.py: kagglehub CLI (--handle/--dest/--type auto), stages into <dest>/raw, hardlink-first, idempotent
- src/retinaedge/data/download_hf.py: huggingface_hub snapshot_download CLI (--repo/--dest/--revision)
- src/retinaedge/data/prepare.py: normalizes aptos (train.csv+train_images), eyepacs (trainLabels/retinopathy_solution + train/), ddr+folder (grade-subdir trees), hf (grade trees / csv / embedded-image parquet best-effort); auto-extracts zips incl. multi-part (EyePACS train.zip.001...); writes labels.csv + per-dataset README (creates repo data/README.md only if missing — did not overwrite existing)
- tests/test_data.py: 17 network-free tests (ben-graham geometry, split determinism, synthetic contract+dist, transforms, file datasets, prepare CLIs incl. multipart zip, error paths)
- Verified: smoke.yaml -> train 240 / val 80 datasets, DataLoader batch (16,3,64,64) float32; all 3 CLIs run via python -m; ruff clean

Stage Summary:
- Contract notes/deviations:
  1. Splitting: contract does not fix a split mechanism -> deterministic name-hash split via data.val_fraction/data.test_fraction/data.split_salt (documented in module + per-dataset README)
  2. Ben-Graham is opt-in via data.ben_graham (default false) + data.ben_graham_radius (300); applies to all splits when enabled for aptos/eyepacs/ddr/folder
  3. prepare writes per-dataset README at data/<name>/README.md and only creates data/README.md if absent (main agent's placeholder preserved)
  4. Synthetic split offsets defined as {train:0, val:1e6, test:2e6} (contract left them open)
- docs/DATASETS.md already present (other agent) — verified consistent with implemented CLIs; typo there (`".f]"`) left for its owner
- Full pytest has failures in OTHER agents' modules (metrics/model_build/train_pipeline) — out of scope; tests/test_data.py + test_ordinal_ops.py + test_loss-related data paths pass

---
Task ID: 7
Agent: main agent (cadence loop)
Task: Every-3-hour improvement cron + version-bump automation toward 1.0

Work Log:
- auto_improve.py: campaign rounds (ladder exhausted + goal open -> budget escalates small->medium->full, champion preserved); --max-rounds; history entries carry round; persisted budget wins over CLI seed
- scripts/bump_release.py: deterministic semver decision (goal -> 1.0.0, >= +0.005 acc_refer -> minor, post-1.0 -> patch), release notes + ledger writer, GITHUB_OUTPUT emitter
- scripts/rotate_dataset.py: scheduled runs rotate top-priority Kaggle catalog handles by campaign history length
- improve.yml: cron 17 */3 * * *; restore champion ckpts from previous improve-run artifact (soup/distill continuity); dataset rotation w/ fallback; bump + tag + release steps; max_stages 3
- runs/version_state.json ledger seeded from v0.3.0-arch3x champion; .gitignore exception added
- tests: TestRoundEscalation (6) + test_bump_release.py (13) — all green locally; ruff clean
- docs/WORKFLOWS.md: cadence section incl. minutes-budget guidance

Stage Summary:
- Cron every 3h trains, escalates, and self-releases: v0.4.0..v0.9.0 on +0.5% steps, v1.0.0 when the 97% goal (CI-gated when require_ci) is hit
- First run dispatched and verified in_progress (run 36617918655); stale pre-push run cancelled (its commit could not have pushed)
- KAGGLE_API_TOKEN secret confirmed present in repo

---
Task ID: 8
Agent: main agent (public flip + next-level release)
Task: Make repo public, next-model-level ladder, data maximalism, detailed release pages

Work Log:
- Secret-scanned all tracked files (clean) -> PATCH /repos -> visibility public (unlimited Actions minutes)
- Diagnosed failed run 36617918655: restore step 403 — permissions block lacked actions:read; fixed
- Ladder: added s8-reinforce (focal_gamma 3.0, refer_weight 0.5 — reward-shaped) and s9-dataset-learn (mixup 0.4, smoothing 0.1, balanced sampler)
- rotate_dataset.py: parses handle+bytes pairs, --max-bytes 9GB disk guard; workflow rotates top-4 fitting handles (APTOS-2019 8.6GB, 224px blend, Messidor-2, labels)
- bump_release.py: release pages now include champion metrics, campaign leaderboard, training-data provenance, try-the-champion links, run permalink
- Tests: updated ladder tests for 9-stage ladder, new test_rotate_dataset.py (8), notes assertions; 40 green locally (CPU torch), ruff clean
- Enabled GitHub Pages (build_type workflow) and dispatched pages.yml; dispatched improve.yml again

Stage Summary:
- Repo public: https://github.com/stromplayz/retinaedge-dr — Actions quota pressure gone
- Site target: https://stromplayz.github.io/retinaedge-dr/ (deploy in flight)
- 3h loop now: restores ckpts (unblocked), rotates size-fitting datasets, trains s8/s9 next-level stages, ships detailed cadence releases toward v1.0.0

---
Task ID: 8b
Agent: main agent (live-run debugging)
Task: Fix the two bugs the first real 3h-cadence run exposed

Work Log:
- Bug 1: max_stages bounded the ladder HEAD, not the PENDING window -> with s1-s5 done and max_stages=3 the run trained nothing yet reported success. Fixed: pending = [s for s in ladder if not done]; to_run = pending[:max_stages]; dry-run plan prints [run ]/[next] queues
- Bug 2: permissions lacked actions:read -> gh run list 403 aborted the restore step. Fixed: contents:write + actions:read
- CI fixes: test_loophole_stack ladder length 7->9; pending-window test assertions (an edit/commit race had shipped the stale version); ruff format repo-wide
- Verified live: run 36663311223 restore step passes, dataset resolved, ladder step TRAINING (s6/s7/s8) for 25+ min (vs instant no-op before); CI green on 1953359

Stage Summary:
- The 3h loop is now genuinely self-training and self-releasing; both bugs have pinned regression tests
- Lesson recorded: never batch Edit + git commit in one message (race); verify file state before commit
