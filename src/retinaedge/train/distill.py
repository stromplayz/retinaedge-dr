"""Knowledge distillation — architectural training for mobile-fit accuracy.

Teacher (large/backbone-rich, already trained) transfers *dark knowledge* —
the full inter-grade probability distribution — into a smaller student that
must fit mobile latency/size budgets.  The student optimizes:

    L = alpha * T_kd^2 * KL( softmax(z_s/T_kd) || softmax(z_t/T_kd) )
        + (1 - alpha) * CE( p_student, y )          [ordinal + referable]

where the KL term operates on the grade-probability simplex produced by
:func:`retinaedge.models.ordinal_ops.ordinal_probs` (CORAL-consistent), and
the hard term reuses the production :class:`DrLoss`.  Typical outcome on
fundus grading: a student with ~40-60% of teacher parameters recovers
97-100% of teacher QWK.

Pipeline position: ``auto_improve`` stage 5 — run after TTA/cuts/soup have
been exhausted, producing the next-generation mobile checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from retinaedge.models.build import DrNet, build_model
from retinaedge.models.ordinal_ops import ordinal_probs
from retinaedge.train.metrics import _quadratic_weighted_kappa
from retinaedge.train.trainer import (
    _resolve_device,
    build_split_loader,
    collect_predictions,
    load_checkpoint,
)

__all__ = ["distillation_loss", "distill_train"]


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    alpha: float = 0.7,
    kd_temp: float = 3.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined KD + supervised ordinal loss.

    Args:
        student_logits: Raw ``ordinal_logits`` ``(B, K-1)`` from the student.
        teacher_logits: Raw ``ordinal_logits`` ``(B, K-1)`` from the teacher.
        targets: Long grades ``(B,)``.
        alpha: Weight of the soft-KL term.
        kd_temp: Distillation temperature for both sides.

    Returns:
        ``(loss, parts)`` with detached floats for logging.
    """
    if kd_temp <= 0:
        raise ValueError("kd_temp must be > 0")
    s_soft = ordinal_probs(student_logits / kd_temp)
    t_soft = ordinal_probs(teacher_logits / kd_temp)
    kd = F.kl_div(
        s_soft.clamp_min(1e-7).log(),
        t_soft.detach(),
        reduction="batchmean",
        log_target=False,
    ) * (kd_temp * kd_temp)

    # hard term: supervised CE on the T=1 grade distribution + referable BCE
    s_probs = ordinal_probs(student_logits)
    n_classes = s_probs.shape[1]
    ce = F.nll_loss(s_probs.clamp_min(1e-7).log(), targets.long())

    loss = alpha * kd + (1.0 - alpha) * ce
    parts = {
        "loss": float(loss.detach()),
        "loss_kd": float(kd.detach()),
        "loss_ce": float(ce.detach()),
        "n_classes": n_classes,
    }
    return loss, parts


@torch.no_grad()
def _evaluate(model: DrNet, loader, device: str) -> dict[str, float]:
    probs, targets = collect_predictions(model, loader, device)
    preds = probs.argmax(axis=1)
    qwk = _quadratic_weighted_kappa(preds, targets)
    acc = float((preds == targets).mean())
    return {"qwk": float(qwk), "accuracy": acc, "n": int(len(targets))}


def distill_train(
    teacher_ckpt: str | Path,
    cfg: dict,
    student_cfg: dict | None = None,
    out_dir: str | Path = "artifacts/distill",
    epochs: int | None = None,
    alpha: float = 0.7,
    kd_temp: float = 3.0,
    lr: float | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Run knowledge distillation and save the best student checkpoint.

    Args:
        teacher_ckpt: Path to the frozen teacher checkpoint.
        cfg: Config providing the **data** section (loaders) and defaults.
        student_cfg: Overrides for ``cfg["model"]`` (e.g. a smaller backbone).
        out_dir: Where checkpoints/history are written.
        epochs: Override of ``cfg["train"]["epochs"]``.
        alpha: KD weight (see :func:`distillation_loss`).
        kd_temp: Distillation temperature.
        lr: Optional learning-rate override.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``.

    Returns:
        Report dict (best QWK, history, student path).
    """
    dev = _resolve_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- teacher (frozen) -------------------------------------------------
    t_payload = load_checkpoint(teacher_ckpt)
    teacher = build_model(t_payload.get("cfg") or cfg)
    teacher.load_state_dict(t_payload["state_dict"])
    teacher.set_temperature(float(t_payload.get("temperature", 1.0)))
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- student ----------------------------------------------------------
    s_cfg = json.loads(json.dumps(cfg))  # deep copy
    if student_cfg:
        s_cfg.setdefault("model", {}).update(student_cfg)
    s_cfg.setdefault("train", {})["epochs"] = int(epochs or s_cfg["train"].get("epochs", 5))
    if lr is not None:
        s_cfg["train"]["lr"] = float(lr)
    epochs_n = s_cfg["train"]["epochs"]

    student = build_model(s_cfg)
    student.to(dev).train()

    train_loader = build_split_loader(s_cfg, "train", shuffle=True)
    val_loader = build_split_loader(s_cfg, "val", shuffle=False)

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=float(s_cfg["train"].get("lr", 1e-3)),
        weight_decay=float(s_cfg["train"].get("weight_decay", 0.02)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs_n * max(1, len(train_loader)))
    )

    history: list[dict[str, Any]] = []
    best_qwk = -1e9
    best_path = out / "student_best.pt"
    teacher_t = float(t_payload.get("temperature", 1.0))

    for epoch in range(1, epochs_n + 1):
        student.train()
        ep_parts: dict[str, float] = {}
        n_batches = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(dev), targets.to(dev)
            with torch.no_grad():
                t_out = teacher(imgs)
                t_logits = t_out["ordinal_logits"] / teacher_t
            s_out = student(imgs)
            loss, parts = distillation_loss(
                s_out["ordinal_logits"],
                t_logits,
                targets,
                alpha=alpha,
                kd_temp=kd_temp,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), float(s_cfg["train"].get("grad_clip", 5.0))
            )
            opt.step()
            sched.step()
            for k, v in parts.items():
                ep_parts[k] = ep_parts.get(k, 0.0) + v
            n_batches += 1

        student.eval()
        val = _evaluate(student, val_loader, dev)
        row = {
            "epoch": epoch,
            **{k: v / max(1, n_batches) for k, v in ep_parts.items() if k != "n_classes"},
            **val,
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)
        print(f"[distill] epoch {epoch}: val_qwk={val['qwk']:.4f} acc={val['accuracy']:.4f}")

        if val["qwk"] > best_qwk:
            best_qwk = val["qwk"]
            payload = {
                "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                "cfg": s_cfg,
                "val_qwk": float(best_qwk),
                "temperature": 1.0,
                "epoch": epoch,
                "distilled_from": str(teacher_ckpt),
                "kd": {"alpha": alpha, "temp": kd_temp},
            }
            torch.save(payload, best_path)

    report = {
        "teacher": str(teacher_ckpt),
        "teacher_val_qwk": float(t_payload.get("val_qwk", 0.0)),
        "student_backbone": s_cfg.get("model", {}).get("backbone", "?"),
        "best_val_qwk": float(best_qwk),
        "student_ckpt": str(best_path),
        "history": history,
    }
    (out / "distill_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge distillation (teacher -> student)")
    parser.add_argument("--config", default="configs/train/online_pilot.yaml")
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--out-dir", default="artifacts/distill")
    parser.add_argument("--student-backbone", default=None, help="timm name override")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--kd-temp", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=None)
    args, overrides = parser.parse_known_args(argv)

    # Dotted overrides (e.g. data.img_size=224 train.ema=true) — same mechanism
    # as the trainer CLI, which is what the auto_improve ladder passes through.
    from retinaedge.utils.config import load_config

    cfg = load_config(args.config, overrides)

    student_cfg = {"backbone": args.student_backbone} if args.student_backbone else None
    report = distill_train(
        args.teacher_ckpt,
        cfg,
        student_cfg=student_cfg,
        out_dir=args.out_dir,
        epochs=args.epochs,
        alpha=args.alpha,
        kd_temp=args.kd_temp,
        lr=args.lr,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
