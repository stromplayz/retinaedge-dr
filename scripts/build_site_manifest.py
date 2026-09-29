#!/usr/bin/env python3
"""Build the model-registry manifest for the RetinaEdge-DR web playground.

Scans a models directory for known exported ONNX artifacts, attaches metrics
from the campaign / eval JSON reports, and writes ``manifest.json`` — the
single source of truth consumed by BOTH the GitHub Pages static site
(``site/``) and the local Next.js sandbox preview (``public/models/``).

The ONNX weights themselves are NOT committed to git (repo stays lean); the
``pages.yml`` workflow downloads them from the GitHub release assets at deploy
time and re-runs this script. Locally, point ``--models-dir`` at a directory
that already contains the files (e.g. copied from ``artifacts/release/``).

Usage:
    python scripts/build_site_manifest.py \
        --models-dir site/models --out site/models/manifest.json \
        [--metrics-dir artifacts/release] [--pilot-dir artifacts/online] \
        [--release-tag v0.2.0-mobile] [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = "stromplayz/retinaedge-dr"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GRADES = [
    {"id": 0, "name": "No DR", "short": "NoDR"},
    {"id": 1, "name": "Mild", "short": "Mild"},
    {"id": 2, "name": "Moderate", "short": "Mod"},
    {"id": 3, "name": "Severe", "short": "Sev"},
    {"id": 4, "name": "Proliferative DR", "short": "PDR"},
]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def _model_entry(
    *,
    mid: str,
    label: str,
    file: Path,
    models_dir: Path,
    stage: str,
    variant: str,
    quant: str,
    img_size: int,
    resize_longest: int,
    metrics: dict,
    recommended: bool = False,
    notes: str = "",
    qdq_fallback: bool = False,
) -> dict | None:
    """One registry entry; ``None`` when the ONNX file is missing.

    With ``qdq_fallback=True`` (web int8 entries) the entry prefers a
    ``*.qdq.onnx`` sibling — ORT-web's wasm EP lacks the ConvInteger kernel
    that dynamic (QOperator) int8 exports produce. If neither file exists the
    entry is omitted entirely so the web registry never offers a model the
    browser cannot run.
    """
    path = models_dir / file
    if qdq_fallback:
        stem = path.stem[: -len(".int8")] if path.stem.endswith(".int8") else path.stem
        qdq = path.with_name(stem + ".qdq.onnx")
        if qdq.exists():
            path = qdq
            quant = "int8-qdq"
            notes = (notes + " " if notes else "") + "QDQ int8 for browser WASM compat"
    if not path.exists():
        return None
    return {
        "id": mid,
        "label": label,
        "stage": stage,
        "variant": variant,
        "quant": quant,
        "file": f"models/{path.name}",
        "size_mb": _mb(path),
        "input": {
            "h": img_size,
            "w": img_size,
            "resize_longest": resize_longest,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "metrics": metrics,
        "recommended": recommended,
        "notes": notes,
    }


def build_manifest(models_dir: Path, metrics_dir: Path, pilot_dir: Path, release_tag: str) -> dict:
    s5 = _load_json(metrics_dir / "s5fusion_metrics.json")
    state = _load_json(metrics_dir / "improve_state.json") or _load_json(
        Path("runs") / "improve_state.json"
    )
    best = state.get("best") or {}
    pilot_eval = _load_json(pilot_dir / "eval.json")
    pilot_metrics = _load_json(pilot_dir / "metrics.json")

    campaign = {}
    if best:
        campaign = {
            "variant": best.get("variant"),
            "stage": best.get("stage"),
            "acc_refer": round(float(best.get("acc_refer", 0.0)), 4),
            "qwk": round(float(best.get("qwk", 0.0)), 4),
            "sens_refer": round(float(best.get("sens_refer", 0.0)), 4),
            "spec_refer": round(float(best.get("spec_refer", 0.0)), 4),
            "ci95_lo": round(float(best.get("ci95_lo", 0.0)), 4),
            "ci95_hi": round(float(best.get("ci95_hi", 0.0)), 4),
            "n": best.get("n"),
        }

    def s5_metrics(extra_note: str = "") -> dict:
        fv = s5.get("final_val") or {}
        m = {
            "val_qwk": round(float(fv.get("qwk", s5.get("best_val_qwk", 0.0))), 4),
            "auc_refer": round(float(fv.get("auc", 0.0)), 4),
            "ece": round(float(fv.get("ece", 0.0)), 4),
            "n_parameters": s5.get("n_parameters"),
            "n_val": s5.get("n_val"),
        }
        if campaign:
            m["campaign_best"] = campaign
        if extra_note:
            m["note"] = extra_note
        return m

    def pilot_metrics_() -> dict:
        return {
            "val_qwk": round(float(pilot_metrics.get("qwk", pilot_eval.get("qwk", 0.0))), 4),
            "auc_refer": round(float(pilot_eval.get("auc_refer", 0.0)), 4),
            "ece": round(float(pilot_eval.get("ece", 0.0)), 4),
            "n_val": pilot_eval.get("n"),
            "note": "v0.1.1 data-pilot (mobilenetv3_small_100 @160, APTOS 3.6k) — first real-data model",
        }

    entries = []

    e = _model_entry(
        mid="s5fusion-soup-int8",
        label="S5-Fusion · Soup · INT8",
        file=Path("dr_s5fusion_soup.int8.onnx"),
        models_dir=models_dir,
        stage="s5-fusion",
        variant="soup",
        quant="int8",
        img_size=224,
        resize_longest=256,
        metrics=s5_metrics("Flagship: weight-averaged soup, int8-quantized (3.7x smaller)"),
        recommended=True,
        notes="Best mobile pick — 3.4MB, fits the <5MB mobile budget; campaign-best accuracy variant",
        qdq_fallback=True,
    )
    if e:
        entries.append(e)

    e = _model_entry(
        mid="s5fusion-soup-fp32",
        label="S5-Fusion · Soup · FP32",
        file=Path("dr_s5fusion_soup.onnx"),
        models_dir=models_dir,
        stage="s5-fusion",
        variant="soup",
        quant="fp32",
        img_size=224,
        resize_longest=256,
        metrics=s5_metrics("Weight-averaged soup (same basin), highest-stability variant"),
        recommended=False,
        notes="Reference accuracy for the soup; use INT8 for mobile",
    )
    if e:
        entries.append(e)

    e = _model_entry(
        mid="s5fusion-best-int8",
        label="S5-Fusion · Best · INT8",
        file=Path("dr_s5fusion_best.int8.onnx"),
        models_dir=models_dir,
        stage="s5-fusion",
        variant="best",
        quant="int8",
        img_size=224,
        resize_longest=256,
        metrics=s5_metrics("Single best epoch (EMA weights), int8"),
        recommended=False,
        notes="",
        qdq_fallback=True,
    )
    if e:
        entries.append(e)

    e = _model_entry(
        mid="s5fusion-best-fp32",
        label="S5-Fusion · Best · FP32",
        file=Path("dr_s5fusion_best.onnx"),
        models_dir=models_dir,
        stage="s5-fusion",
        variant="best",
        quant="fp32",
        img_size=224,
        resize_longest=256,
        metrics=s5_metrics("Single best epoch (EMA weights)"),
        recommended=False,
        notes="",
    )
    if e:
        entries.append(e)

    e = _model_entry(
        mid="v011-pilot-fp32",
        label="v0.1.1 Pilot · FP32",
        file=Path("model.onnx"),
        models_dir=models_dir,
        stage="online-pilot",
        variant="best",
        quant="fp32",
        img_size=160,
        resize_longest=256,
        metrics=pilot_metrics_(),
        recommended=False,
        notes="First real-data model trained inside GitHub Actions (APTOS pilot)",
    )
    if e:
        entries.append(e)

    return {
        "schema": "retinaedge/site-manifest@1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": REPO,
        "release": release_tag,
        "contract": {
            "input": "float32 (1,3,H,W) ImageNet-normalized",
            "output": "(1,5) grade probabilities (softmax embedded)",
            "grade_scheme": "ICDRSS 0-4",
            "referable_rule": "grade >= 2 or sum(p[2:]) > 0.5",
        },
        "grades": GRADES,
        "models": entries,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="site/models")
    ap.add_argument("--out", default="site/models/manifest.json")
    ap.add_argument("--metrics-dir", default="artifacts/release")
    ap.add_argument("--pilot-dir", default="artifacts/online")
    ap.add_argument("--release-tag", default="v0.2.0-mobile")
    ap.add_argument("--strict", action="store_true", help="fail if zero models found")
    args = ap.parse_args(argv)

    models_dir = Path(args.models_dir)
    out_path = Path(args.out)
    manifest = build_manifest(
        models_dir, Path(args.metrics_dir), Path(args.pilot_dir), args.release_tag
    )

    if not manifest["models"]:
        msg = f"no ONNX models found in {models_dir}"
        if args.strict:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1
        print(f"WARN: {msg} — writing manifest with empty model list", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"manifest: {out_path} — {len(manifest['models'])} models, "
        f"total {sum(m['size_mb'] for m in manifest['models']):.1f} MB"
    )
    for m in manifest["models"]:
        print(f"  - {m['id']:22s} {m['size_mb']:7.2f} MB  {m['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
