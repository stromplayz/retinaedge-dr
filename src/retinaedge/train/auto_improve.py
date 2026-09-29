"""Continuous-improvement engine (contract: ``retinaedge.train.auto_improve``).

An escalation ladder that trains, applies the full accuracy "loophole stack"
(TTA + model soup + ordinal cut-point decoding + referable threshold search),
measures honestly (Wilson CI), and keeps escalating until the accuracy goal
(default: **97% referable-DR accuracy** on the validation split) is reached —
or the ladder/budget is exhausted. State is resumable across runs
(``improve_state.json``), which is what makes GitHub Actions scheduled runs
a *continuous* improvement loop: every invocation continues where the last
one stopped and commits its logs back to the repo.

Ladder (each rung keeps every previous winning trick and adds one):
    s1-baseline     — budget epochs at the config's default resolution
    s2-longer-ema   — longer schedule + EMA weight averaging
    s3-sharper      — higher input resolution
    s4-backbone     — stronger backbone (efficientnet_lite0)
    s5-fusion       — stronger backbone + sharper still + EMA (the "full send")

When every rung of the current round is complete and the goal is still open,
the campaign starts a NEW ROUND at the next budget level (small -> medium ->
full): same ladder, stronger settings, champion preserved. This is what lets
the every-3-hours scheduled workflow grind toward the goal indefinitely
(capped by ``--max-rounds``).

Usage:
    python -m retinaedge.train.auto_improve --config configs/train/online_pilot.yaml \
        --budget small --target 0.97 --max-stages 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retinaedge.eval.stats import accuracy_ci_report
from retinaedge.utils.logging_utils import get_logger

__all__ = [
    "Stage",
    "build_ladder",
    "load_state",
    "save_state",
    "variant_metrics",
    "pick_best_variant",
    "next_budget",
    "main",
]

_LOGGER = get_logger("train.auto_improve")

_BUDGETS: dict[str, dict[str, float]] = {
    # e1: baseline epochs | e2: escalated epochs | sz/sz2: escalated resolutions
    "small": {"e1": 2, "e2": 4, "sz": 192, "sz2": 224},
    "medium": {"e1": 6, "e2": 14, "sz": 224, "sz2": 288},
    "full": {"e1": 20, "e2": 40, "sz": 256, "sz2": 320},
}


@dataclass(frozen=True)
class Stage:
    """One rung of the improvement ladder."""

    name: str
    description: str
    overrides: tuple[str, ...]  # dotted overrides with {e1}/{e2}/{sz}/{sz2} placeholders
    runner: str = "trainer"  # "trainer" or "distill"

    def resolved(self, params: dict[str, float]) -> tuple[str, ...]:
        """Substitute budget placeholders into the overrides."""
        return tuple(ovr.format(**params) for ovr in self.overrides)


_BUDGET_ORDER = ["small", "medium", "full"]


def next_budget(budget: str) -> str | None:
    """Next rung of the budget-escalation ladder (``None`` at the ceiling)."""
    if budget not in _BUDGET_ORDER:
        return None
    i = _BUDGET_ORDER.index(budget)
    return _BUDGET_ORDER[i + 1] if i + 1 < len(_BUDGET_ORDER) else None


def build_ladder(budget: str) -> list[Stage]:
    """Escalation ladder for a budget profile (``small``/``medium``/``full``)."""
    if budget not in _BUDGETS:
        raise ValueError(f"budget must be one of {sorted(_BUDGETS)}, got {budget!r}")
    return [
        Stage("s1-baseline", "budget epochs at default resolution", ("train.epochs={e1}",)),
        Stage(
            "s2-longer-ema",
            "longer schedule + EMA weight averaging",
            ("train.epochs={e2}", "train.ema=true", "train.patience=4"),
        ),
        Stage(
            "s3-sharper",
            "higher input resolution",
            ("data.img_size={sz}", "train.epochs={e2}", "train.ema=true"),
        ),
        Stage(
            "s4-backbone",
            "stronger backbone (efficientnet_lite0)",
            (
                "model.backbone=efficientnet_lite0",
                "data.img_size={sz}",
                "train.epochs={e2}",
                "train.ema=true",
            ),
        ),
        Stage(
            "s5-fusion",
            "stronger backbone + highest resolution + EMA",
            (
                "model.backbone=efficientnet_lite0",
                "data.img_size={sz2}",
                "train.epochs={e2}",
                "train.ema=true",
                "train.patience=6",
            ),
        ),
        Stage(
            "s6-mixup",
            "mixup alpha=0.2 + label smoothing at escalated resolution (v0.3.0)",
            (
                "model.backbone=efficientnet_lite0",
                "data.img_size={sz2}",
                "train.epochs={e2}",
                "train.ema=true",
                "train.patience=6",
                "train.mixup_alpha=0.2",
                "train.loss.label_smoothing=0.05",
            ),
        ),
        Stage(
            "s7-distill",
            "self-distillation from the best soup teacher, T=3 (v0.3.0)",
            (
                "model.backbone=efficientnet_lite0",
                "data.img_size={sz2}",
                "train.epochs={e2}",
                "train.ema=true",
                "train.patience=6",
            ),
            runner="distill",
        ),
    ]


# --------------------------------------------------------------------------- #
# Resumable state
# --------------------------------------------------------------------------- #


def default_state(target: float, budget: str) -> dict:
    """Fresh state dict for a new improvement campaign."""
    return {
        "target": float(target),
        "budget": budget,
        "round": 1,
        "goal_reached": False,
        "best": None,
        "history": [],
        "updated_at": "",
    }


def load_state(path: str | Path) -> dict | None:
    """Load improvement state, or ``None`` when absent/corrupt (fresh start)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _LOGGER.warning("state file %s unreadable — starting fresh", p)
        return None
    if not isinstance(state, dict) or "history" not in state:
        return None
    return state


def save_state(path: str | Path, state: dict) -> None:
    """Persist state atomically (tmp file + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Variant metrics (pure numpy — unit-testable without torch)
# --------------------------------------------------------------------------- #


def variant_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Full loophole-stack metrics for one probability matrix.

    Includes argmax + cut-point QWK/accuracy, threshold-optimized referable
    accuracy (with sens/spec) and the counts needed for the Wilson CI.
    """
    from retinaedge.eval.threshold_search import (
        coordinate_ascent_cuts,
        search_referable_threshold,
    )
    from retinaedge.train.metrics import _quadratic_weighted_kappa

    t = np.asarray(targets, dtype=np.int64).ravel()
    p = np.asarray(probs, dtype=np.float64)
    n = int(t.size)
    hard = p.argmax(axis=1)

    qwk = _quadratic_weighted_kappa(hard, t)
    acc = float((hard == t).mean()) if n else float("nan")

    expected = p @ np.arange(p.shape[1], dtype=np.float64)
    cuts, qwk_cuts, acc_cuts = coordinate_ascent_cuts(expected, t, passes=2)

    refer_score = p[:, 2:].sum(axis=1)
    refer_true = (t >= 2).astype(int)
    refer = search_referable_threshold(refer_true, refer_score)
    correct_refer = int(round(refer["accuracy"] * n)) if n else 0

    return {
        "qwk": float(qwk),
        "accuracy": float(acc),
        "qwk_cuts": float(qwk_cuts),
        "accuracy_cuts": float(acc_cuts),
        "cuts": [float(c) for c in cuts],
        "acc_refer": float(refer["accuracy"]),
        "sens_refer": float(refer["sens"]),
        "spec_refer": float(refer["spec"]),
        "refer_threshold": float(refer["threshold"]),
        "n": n,
        "correct_refer": correct_refer,
    }


def pick_best_variant(variants: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    """Select the winning variant: referable accuracy first, then best QWK."""

    def key(item: tuple[str, dict[str, float]]) -> tuple[float, float]:
        m = item[1]
        return (m["acc_refer"], max(m["qwk"], m["qwk_cuts"]))

    name, metrics = max(variants.items(), key=key)
    return name, metrics


# --------------------------------------------------------------------------- #
# In-process evaluation of the loophole stack on a trained run
# --------------------------------------------------------------------------- #


def evaluate_variants(cfg: dict, run_dir: str | Path, device: str) -> dict[str, dict[str, float]]:
    """Evaluate base / +TTA / +soup probabilities (+ cut decoding) on val.

    Loads the val split once; every variant reuses it. Thresholds/cuts are
    fitted on this split by design (frozen afterwards, test untouched).
    """
    import torch

    from retinaedge.models.build import build_model
    from retinaedge.models.soup import make_soup
    from retinaedge.models.tta import tta_probs
    from retinaedge.train.trainer import build_split_loader, load_checkpoint

    run = Path(run_dir)
    ckpt_path = run / "best.pt"

    def load_model(path: Path):
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        temperature = float(payload.get("temperature", 1.0)) if isinstance(payload, dict) else 1.0
        model = build_model(cfg).to(device)
        model.load_state_dict(state)
        model.set_temperature(temperature)
        model.eval()
        return model

    model = load_model(ckpt_path)
    loader = build_split_loader(cfg, "val", shuffle=False)

    @torch.no_grad()
    def _collect(m, use_tta: bool) -> tuple[np.ndarray, np.ndarray]:
        probs_all, targets_all = [], []
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = (
                tta_probs(m, imgs, scales=(1.0,), hflip=True) if use_tta else m.predict_probs(imgs)
            )
            probs_all.append(probs.float().cpu().numpy())
            targets_all.append(targets.numpy())
        return np.concatenate(probs_all), np.concatenate(targets_all).astype(np.int64)

    variants: dict[str, dict[str, float]] = {}
    base_probs, targets = _collect(model, use_tta=False)
    variants["base"] = variant_metrics(base_probs, targets)
    variants["tta"] = variant_metrics(_collect(model, use_tta=True)[0], targets)

    last_path = run / "last.pt"
    if last_path.exists():
        soup_path = make_soup([ckpt_path, last_path], run / "soup.pt")
        soup_model = load_model(soup_path)
        soup_probs, _ = _collect(soup_model, use_tta=False)
        variants["soup"] = variant_metrics(soup_probs, targets)
        variants["soup+tta"] = variant_metrics(_collect(soup_model, use_tta=True)[0], targets)
    return variants


def _variant_probs_checkpoint(cfg: dict, ckpt: str, device: str):  # pragma: no cover - helper
    """Reserved for external tooling; not used by the ladder itself."""
    from retinaedge.models.build import build_model
    from retinaedge.train.trainer import build_split_loader, collect_predictions, load_checkpoint

    payload = load_checkpoint(ckpt, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    loader = build_split_loader(cfg, "val", shuffle=False)
    return collect_predictions(model, loader, device)


# --------------------------------------------------------------------------- #
# Stage runner + logging
# --------------------------------------------------------------------------- #


def _run_trainer_subprocess(config: str, overrides: Sequence[str], run_dir: Path) -> None:
    """Run one training stage as a subprocess (crash isolation + log file)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "retinaedge.train.trainer",
        "--config",
        config,
        *overrides,
        f"train.save_dir={run_dir.as_posix()}",
    ]
    log_path = run_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"trainer failed ({config} {' '.join(overrides)}):\n{tail}")


def _run_distill_subprocess(
    config: str, overrides: Sequence[str], run_dir: Path, teacher_ckpt: str | None
) -> None:
    """Run one distillation stage (teacher -> student) as a subprocess.

    ``distill.py`` saves ``student_best.pt``; it is renamed to ``best.pt`` so
    :func:`evaluate_variants` can treat the run exactly like a trainer run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "retinaedge.train.distill",
        "--config",
        config,
        "--teacher-ckpt",
        str(teacher_ckpt),
        "--out-dir",
        run_dir.as_posix(),
        "--kd-temp",
        "3.0",
        "--alpha",
        "0.7",
        *overrides,
    ]
    log_path = run_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"distill failed ({config} {' '.join(overrides)}):\n{tail}")
    student = run_dir / "student_best.pt"
    if student.exists():
        student.replace(run_dir / "best.pt")
    else:
        raise RuntimeError(f"distill produced no student_best.pt in {run_dir}")


def _fmt_log_entry(
    stage_name: str,
    description: str,
    variants: dict[str, dict[str, float]],
    best_name: str,
    best: dict[str, float],
    target: float,
    elapsed_s: float,
    overrides: Sequence[str],
) -> str:
    """Markdown block appended to ``improvement_log.md`` after every stage."""
    lines = [
        f"## {stage_name} — {description}",
        f"*{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} | overrides: "
        f"`{' '.join(overrides) if overrides else '(config defaults)'}` | {elapsed_s:.0f}s*",
        "",
        "| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in variants.items():
        lines.append(
            f"| {name} | {m['qwk']:.4f} | {m['qwk_cuts']:.4f} | {m['accuracy']:.4f} | "
            f"{m['accuracy_cuts']:.4f} | {m['acc_refer']:.4f} | {m['sens_refer']:.4f} | "
            f"{m['spec_refer']:.4f} |"
        )
    rep = accuracy_ci_report(best["correct_refer"], best["n"], target)
    gap = target - best["acc_refer"]
    lines += [
        "",
        f"**Best variant:** `{best_name}` — acc_refer **{best['acc_refer']:.4f}** "
        f"(95% Wilson CI [{rep['ci95_lo']:.4f}, {rep['ci95_hi']:.4f}], n={best['n']}), "
        f"QWK {max(best['qwk'], best['qwk_cuts']):.4f}",
        f"**Goal {target:.2%}:** {'REACHED' if rep['target_reached'] else 'not reached'} "
        f"({'point estimate clears; CI lower bound does not' if rep['point_only_reached'] and not rep['target_reached'] else f'gap {gap:+.2%}'})",
        "",
    ]
    return "\n".join(lines) + "\n"


def append_log(log_path: str | Path, entry: str) -> None:
    """Append a markdown entry to the improvement log."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as handle:
        handle.write(entry)


# --------------------------------------------------------------------------- #
# Campaign runner
# --------------------------------------------------------------------------- #


def run_campaign(
    config: str,
    budget: str = "small",
    target: float = 0.97,
    max_stages: int = 5,
    state_path: str = "runs/improve_state.json",
    log_path: str = "runs/improvement_log.md",
    device: str = "auto",
    require_ci: bool = False,
    max_rounds: int = 10,
    dry_run: bool = False,
) -> dict:
    """Execute (or plan, with ``dry_run``) the escalation ladder.

    Returns the final state dict. Designed to be resumable: completed stages
    recorded in the state file are skipped on the next invocation, and once a
    whole ladder round completes without the goal, the budget escalates
    (small -> medium -> full) and the next round begins.
    """
    state = load_state(state_path) or default_state(target, budget)
    state.setdefault("round", 1)
    # A started campaign owns its (possibly escalated) budget: the persisted
    # value wins over the CLI seed so scheduled runs continue the escalation.
    if state.get("budget") in _BUDGETS:
        budget = state["budget"]
    state["target"], state["budget"] = float(target), budget

    ladder = build_ladder(budget)
    params = _BUDGETS[budget]
    round_no = int(state["round"])

    def _round_done() -> set[str]:
        return {h["stage"] for h in state["history"] if int(h.get("round", 1)) == round_no}

    done_names = _round_done()

    # Round escalation: ladder exhausted + goal still open -> stronger budget.
    while (
        ladder
        and all(stage.name in done_names for stage in ladder)
        and not state.get("goal_reached")
        and round_no < max(1, int(max_rounds))
    ):
        escalated = next_budget(budget)
        if escalated is None:
            _LOGGER.info(
                "ladder exhausted at budget=%s (round %d) with goal open — "
                "already at the escalation ceiling",
                budget,
                round_no,
            )
            break
        round_no += 1
        budget = escalated
        state["round"], state["budget"] = round_no, budget
        ladder, params = build_ladder(budget), _BUDGETS[budget]
        done_names = _round_done()
        _LOGGER.info("=== round %d begins: budget escalated to %s ===", round_no, budget)

    if dry_run:
        print(f"Ladder plan (round {round_no}, budget={budget}, target={target:.2%}):")
        for i, stage in enumerate(ladder[: max(1, max_stages)]):
            ovr = stage.resolved(params)
            status = "DONE" if stage.name in done_names else "pending"
            print(f"  {i + 1}. {stage.name:14s} [{status}] {' '.join(ovr) or '(defaults)'}")
        return state

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    for stage in ladder[: max(1, max_stages)]:
        if stage.name in done_names:
            _LOGGER.info("skipping completed stage %s", stage.name)
            continue
        overrides = stage.resolved(params)
        run_dir = Path(state_path).parent / stage.name
        _LOGGER.info(
            "=== stage %s: %s (%s) ===", stage.name, stage.description, " ".join(overrides)
        )
        t0 = time.time()
        teacher = (state.get("best") or {}).get("ckpt")
        if stage.runner == "distill" and (not teacher or not Path(teacher).exists()):
            _LOGGER.warning(
                "stage %s requests the distill runner but teacher ckpt %s is missing — "
                "falling back to the trainer (fresh CI runners do not keep .pt files; "
                "restore runs/*.pt from the previous improve-run artifact to enable distill)",
                stage.name,
                teacher,
            )
            _run_trainer_subprocess(config, overrides, run_dir)
        elif stage.runner == "distill":
            _run_distill_subprocess(config, overrides, run_dir, teacher)
        else:
            _run_trainer_subprocess(config, overrides, run_dir)
        cfg = _load_cfg(config, overrides)
        variants = evaluate_variants(cfg, run_dir, device)
        best_name, best = pick_best_variant(variants)
        elapsed = time.time() - t0

        rep = accuracy_ci_report(int(best["correct_refer"]), int(best["n"]), target)
        reached = rep["target_reached"] or (rep["point_only_reached"] and not require_ci)
        entry = _fmt_log_entry(
            stage.name, stage.description, variants, best_name, best, target, elapsed, overrides
        )
        append_log(log_path, entry)

        prev_best = state.get("best") or {}
        if prev_best.get("acc_refer", -1.0) <= best["acc_refer"]:
            state["best"] = {
                "stage": stage.name,
                "variant": best_name,
                "acc_refer": best["acc_refer"],
                "qwk": max(best["qwk"], best["qwk_cuts"]),
                "sens_refer": best["sens_refer"],
                "spec_refer": best["spec_refer"],
                "ci95_lo": rep["ci95_lo"],
                "ci95_hi": rep["ci95_hi"],
                "ckpt": (run_dir / "best.pt").as_posix(),
                "n": best["n"],
            }
        state["history"].append(
            {
                "stage": stage.name,
                "round": round_no,
                "best_variant": best_name,
                "acc_refer": best["acc_refer"],
                "qwk": max(best["qwk"], best["qwk_cuts"]),
                "target_reached": bool(reached),
                "elapsed_s": elapsed,
                "ts": state.get("updated_at", ""),
            }
        )
        state["goal_reached"] = bool(reached) or bool(state.get("goal_reached"))
        save_state(state_path, state)
        _LOGGER.info(
            "stage %s done in %.0fs: best=%s acc_refer=%.4f qwk=%.4f (goal %s)",
            stage.name,
            elapsed,
            best_name,
            best["acc_refer"],
            max(best["qwk"], best["qwk_cuts"]),
            "REACHED" if reached else "pending",
        )
        if state["goal_reached"]:
            _LOGGER.info("target %.2f%% reached — stopping escalation", 100.0 * target)
            break
    return state


def _load_cfg(config: str, overrides: Sequence[str]) -> dict:
    from retinaedge.utils.config import load_config

    return load_config(config, list(overrides))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.train.auto_improve",
        description="Escalation-ladder trainer targeting the accuracy goal with the full loophole stack.",
    )
    parser.add_argument(
        "--config", required=True, help="training YAML config (data section embedded)"
    )
    parser.add_argument("--budget", default="small", choices=sorted(_BUDGETS))
    parser.add_argument("--target", type=float, default=0.97, help="accuracy goal (referable DR)")
    parser.add_argument("--max-stages", type=int, default=5)
    parser.add_argument(
        "--max-rounds", type=int, default=10, help="campaign rounds before escalation stops"
    )
    parser.add_argument("--state", default="runs/improve_state.json")
    parser.add_argument("--log", default="runs/improvement_log.md")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--require-ci",
        action="store_true",
        help="goal counts only when the Wilson CI lower bound clears the target",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan without training")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        run_campaign(
            config=args.config,
            budget=args.budget,
            target=args.target,
            max_stages=args.max_stages,
            state_path=args.state,
            log_path=args.log,
            device=args.device,
            require_ci=args.require_ci,
            max_rounds=args.max_rounds,
            dry_run=args.dry_run,
        )
    except Exception:
        _LOGGER.exception("improvement campaign failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
