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


def _fmt_notes(tag: str, campaign: dict, ledger: dict, min_delta: float) -> str:
    """Markdown release notes summarizing the champion and the campaign."""
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
        f"| budget `{campaign.get('budget', '?')}`",
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
        f"| goal status | {'REACHED' if goal else 'not reached'} |",
        "",
        f"Ladder stages completed so far: {len(history)}. Full auditable history: "
        "`runs/improvement_log.md` (committed to the repo by the improve workflow).",
        "",
        "The champion checkpoint is attached for the web playground and mobile "
        "pipelines. Data: Kaggle/HF fundus blends resolved inside GitHub Actions "
        "(no local training).",
        "",
    ]
    return "\n".join(lines)


def _emit(should_release: bool, tag: str, ckpt: str) -> None:
    import os

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

    notes = _fmt_notes(tag, campaign, ledger, args.min_delta)
    notes_path = Path(args.out)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(notes, encoding="utf-8")

    ckpt = str(best.get("ckpt", "") or "")
    print(f"bump_release: releasing {tag} (champion acc_refer {acc:.4f}, ckpt {ckpt or 'n/a'})")
    _emit(True, tag, ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
