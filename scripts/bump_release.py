#!/usr/bin/env python3
"""Map an improvement-campaign result onto a semantic version + release notes.

Reads ``runs/improve_state.json`` (written by the auto_improve ladder) and
``runs/version_state.json`` (the released-version ledger), decides whether the
campaign improved enough to justify a release, and — when it did — bumps the
version:

* goal reached (97% referable-DR accuracy, per the campaign's own gating,
  which honors ``--require-ci`` when the ladder ran with it): **1.0.0** — the
  long-standing maximum-goal release — then patch bumps afterwards;
* otherwise: a minor bump when the champion improved by at least
  ``--min-delta`` over the last released accuracy (v0.4.0, v0.5.0, ...).

The decision is deterministic: the same campaign state always yields the same
version. The updated ledger and the release notes are written to disk, and
``should_release`` / ``tag`` / ``ckpt`` are emitted as GitHub Actions outputs
when ``GITHUB_OUTPUT`` is set (the workflow owns all git/tag/release ops).

Usage:
    python scripts/bump_release.py \
        --state runs/improve_state.json \
        --version-state runs/version_state.json \
        --out runs/release_notes.md \
        --min-delta 0.005
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_SEMVER_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")

ONE = (1, 0, 0)


def parse_semver(version: str) -> tuple[int, int, int] | None:
    """Extract the leading ``X.Y.Z`` triple from a version-ish string."""
    match = _SEMVER_RE.search(str(version).strip().lstrip("v"))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def decide_version(
    last_version: str,
    acc: float,
    last_acc: float,
    goal_reached: bool,
    min_delta: float = 0.005,
) -> str | None:
    """Next version string, or ``None`` when no release is warranted.

    Rules (monotonic, never downgrades):
      * goal reached before 1.0.0            -> ``1.0.0``
      * accuracy improved by >= ``min_delta`` before 1.0.0 -> minor bump
      * accuracy improved after 1.0.0        -> patch bump (post-goal upkeep)
      * otherwise                            -> ``None``
    """
    last = parse_semver(last_version) or (0, 0, 0)
    if bool(goal_reached) and last < ONE:
        return "1.0.0"
    if (float(acc) - float(last_acc)) < float(min_delta):
        return None
    if last >= ONE:
        return f"{last[0]}.{last[1]}.{last[2] + 1}"
    return f"{last[0]}.{last[1] + 1}.0"


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _leaderboard(history: list, k: int = 5) -> list[str]:
    """Top-k ladder stages by referable accuracy — the campaign scoreboard."""
    rows = sorted(history, key=lambda h: float(h.get("acc_refer", 0.0)), reverse=True)[:k]
    lines = [
        "## Campaign leaderboard (top stages)",
        "",
        "| rank | stage | variant | acc_refer | QWK |",
        "|---|---|---|---|---|",
    ]
    for i, h in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{h.get('stage', '?')}` | {h.get('best_variant', '?')} | "
            f"{float(h.get('acc_refer', 0.0)):.4f} | {float(h.get('qwk', 0.0)):.4f} |"
        )
    return lines


def _run_url() -> str:
    """Permalink to the workflow run that produced this release (or '')."""
    server, repo, run_id = (
        os.environ.get("GITHUB_SERVER_URL"),
        os.environ.get("GITHUB_REPOSITORY"),
        os.environ.get("GITHUB_RUN_ID"),
    )
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def _fmt_notes(
    tag: str,
    campaign: dict,
    ledger: dict,
    min_delta: float,
    provenance: dict | None = None,
    site_url: str = "",
) -> str:
    """Markdown release notes: champion metrics, leaderboard, data, usage."""
    best = campaign.get("best") or {}
    target = float(campaign.get("target", 0.97))
    acc = float(best.get("acc_refer", 0.0))
    last_acc = float(ledger.get("last_acc", 0.0))
    delta = acc - last_acc
    goal = bool(campaign.get("goal_reached"))
    history = campaign.get("history") or []
    round_no = campaign.get("round", 1)

    lines = [
        f"# RetinaEdge-DR {tag} — automated cadence release",
        "",
        f"Goal: **{target:.0%} referable-DR accuracy** | campaign round {round_no} "
        f"| budget `{campaign.get('budget', '?')}` | ladder stages completed: {len(history)}",
        "",
        "## Champion metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| stage | `{best.get('stage', '?')}` |",
        f"| variant | `{best.get('variant', '?')}` |",
        f"| acc_refer | **{acc:.4f}** |",
        f"| 95% Wilson CI | [{best.get('ci95_lo', 0.0):.4f}, {best.get('ci95_hi', 0.0):.4f}] (n={best.get('n', '?')}) |",
        f"| QWK | {best.get('qwk', 0.0):.4f} |",
        f"| referable sens / spec | {best.get('sens_refer', 0.0):.4f} / {best.get('spec_refer', 0.0):.4f} |",
        f"| vs last release | {delta:+.4f} (release gate: >= +{min_delta:.3f} or goal) |",
        f"| goal status | {'REACHED — 1.0.0 shipped' if goal else 'not reached yet'} |",
        "",
        "Training pipeline: supervised ordinal heads (CORAL) + reward-shaped focal "
        "loss (reinforcement-style emphasis on referable misses) + EMA/model-soup "
        "weight averaging + TTA + cut-point decoding + self-distillation + "
        "class-balanced dataset learning — all trained inside GitHub Actions.",
        "",
    ]
    lines += _leaderboard(history)

    prov = provenance or {}
    manifest = prov.get("manifest") or prov.get("dataset") or "?"
    images = prov.get("images") or prov.get("total_images") or "?"
    lines += [
        "",
        "## Training data",
        "",
        f"- resolved inside the run from the validated Kaggle catalog: `{manifest}`",
        f"- images: {images} (patient-aware splits; external Messidor-2 kept for eval only)",
        "- scheduled runs rotate the catalog (size-guarded) for data maximalism",
        "",
    ]

    run_url = _run_url()
    site = (
        site_url
        or f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'stromplayz')}.github.io/{(os.environ.get('GITHUB_REPOSITORY', 'stromplayz/retinaedge-dr').split('/')[-1])}/"
    )
    lines += [
        "## Try the champion",
        "",
        f"- Web playground (no install): {site}",
        "- Android: exported INT8 TFLite — see `docs/EXPORT_DEPLOY.md`",
        "- Python:",
        "",
        "```python",
        "from retinaedge.train.trainer import load_checkpoint",
        "payload = load_checkpoint('champion-checkpoint.pt', map_location='cpu')  # attached asset",
        "```",
        "",
    ]
    if run_url:
        lines += [f"Produced by workflow run: {run_url}", ""]
    lines += [
        "Full auditable history: `runs/improvement_log.md`. Next cadence run in 3h "
        "continues the campaign until the 97% goal ships v1.0.0.",
        "",
    ]
    return "\n".join(lines)


def _emit(should_release: bool, tag: str, ckpt: str) -> None:
    out_file = os.environ.get("GITHUB_OUTPUT")
    payload = f"should_release={'true' if should_release else 'false'}\ntag={tag}\nckpt={ckpt}\n"
    print(payload, end="")
    if out_file:
        with open(out_file, "a", encoding="utf-8") as handle:
            handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bump_release", description="Decide a cadence release from campaign state."
    )
    parser.add_argument("--state", default="runs/improve_state.json")
    parser.add_argument("--version-state", default="runs/version_state.json")
    parser.add_argument("--out", default="runs/release_notes.md")
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument(
        "--default-version",
        default="0.3.0",
        help="seed version when the ledger file is missing",
    )
    parser.add_argument(
        "--default-acc",
        type=float,
        default=0.0,
        help="seed last-released accuracy when the ledger file is missing",
    )
    parser.add_argument(
        "--provenance",
        default="data/kaggle_blend/provenance.json",
        help="dataset provenance JSON written by manifest_import (optional)",
    )
    parser.add_argument(
        "--site-url",
        default="",
        help="web playground URL for the release notes (default: Pages URL)",
    )
    args = parser.parse_args(argv)

    campaign = _load_json(args.state)
    best = campaign.get("best") or {}
    if "acc_refer" not in best:
        print("bump_release: no campaign champion yet — nothing to release")
        _emit(False, "", "")
        return 0

    ledger = _load_json(args.version_state)
    if not ledger:
        ledger = {"last_version": args.default_version, "last_acc": args.default_acc}

    acc = float(best["acc_refer"])
    last_acc = float(ledger.get("last_acc", 0.0))
    tag = decide_version(
        str(ledger.get("last_version", args.default_version)),
        acc,
        last_acc,
        bool(campaign.get("goal_reached")),
        args.min_delta,
    )

    if tag is None:
        print(
            f"bump_release: no release — acc_refer {acc:.4f} vs released {last_acc:.4f} "
            f"(gate >= +{args.min_delta:.3f}), goal_reached={bool(campaign.get('goal_reached'))}"
        )
        _emit(False, "", "")
        return 0

    ledger.update(
        {
            "last_version": tag,
            "last_acc": acc,
            "released_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "campaign": {
                "stage": best.get("stage"),
                "variant": best.get("variant"),
                "acc_refer": acc,
                "qwk": best.get("qwk"),
                "n": best.get("n"),
                "goal_reached": bool(campaign.get("goal_reached")),
                "round": campaign.get("round", 1),
            },
        }
    )
    ledger_path = Path(args.version_state)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(ledger_path)

    notes = _fmt_notes(
        tag,
        campaign,
        ledger,
        args.min_delta,
        provenance=_load_json(args.provenance) or None,
        site_url=args.site_url,
    )
    notes_path = Path(args.out)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(notes, encoding="utf-8")

    ckpt = str(best.get("ckpt", "") or "")
    print(f"bump_release: releasing {tag} (champion acc_refer {acc:.4f}, ckpt {ckpt or 'n/a'})")
    _emit(True, tag, ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
