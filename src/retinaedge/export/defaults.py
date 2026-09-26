"""Shared export settings and their resolution order.

Precedence (highest wins):
    1. explicit CLI flags (passed in as ``cli_overrides`` with ``None`` filtered out)
    2. the train config's optional ``export:`` block (``--config <train.yaml>``)
    3. an export profile YAML (``configs/export/default.yaml``, via ``--export-config``)
    4. :data:`DEFAULTS` (code constants — the last line of defence)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULTS", "resolve_export_settings"]

#: Canonical defaults for the export CLIs. Mirrors ``configs/export/default.yaml``.
DEFAULTS: dict[str, Any] = {
    "img_size": None,  # None -> fall back to cfg["data"]["img_size"] or 224
    "opset": 17,
    "dynamic_batch": False,
    "verify": True,
    "verify_atol": 1e-3,
    "latency_runs": 30,
    "warmup": 5,
    "int8": False,
    "representative_dir": None,
    "representative_samples": 200,
    "referable_threshold": 0.5,
}

_KEYS = frozenset(DEFAULTS)


def _load_export_block(path: str | Path) -> dict[str, Any]:
    """Load a YAML profile and return its ``export:`` block (empty if absent)."""
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Export profile not found: {profile_path}")
    with open(profile_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    block = data.get("export", data if isinstance(data, dict) else {})
    return block if isinstance(block, dict) else {}


def resolve_export_settings(
    cfg: dict | None = None,
    profile_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge export settings from all sources.

    Args:
        cfg: Train config (its optional ``export:`` block is consulted).
        profile_path: Optional YAML profile (e.g. ``configs/export/default.yaml``).
        cli_overrides: Explicit flag values; ``None`` entries are ignored so that
            unset flags fall through to the lower-precedence sources.

    Returns:
        Fully-resolved settings dict (a copy — callers may mutate freely).
    """
    settings = dict(DEFAULTS)
    if profile_path is not None:
        settings.update({k: v for k, v in _load_export_block(profile_path).items() if k in _KEYS})
    if cfg:
        cfg_block = cfg.get("export", {})
        if isinstance(cfg_block, dict):
            settings.update({k: v for k, v in cfg_block.items() if k in _KEYS})
    if cli_overrides:
        settings.update({k: v for k, v in cli_overrides.items() if k in _KEYS and v is not None})
    return settings
