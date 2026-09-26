"""Config loading with YAML + dotted CLI overrides.

Example:
    cfg = load_config("configs/train/smoke.yaml", ["train.lr=3e-4", "data.img_size=64"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["load_config", "parse_scalar"]


def parse_scalar(raw: str) -> Any:
    """Parse a CLI override value: JSON first, plain string as fallback."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: dict = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Load a YAML config and apply ``a.b.c=value`` dotted overrides.

    Args:
        path: Path to the YAML config file.
        overrides: Sequence of ``dotted.key=value`` strings (values JSON-parsed).

    Returns:
        Plain dict config. Nested dicts are shared structures — callers must not
        mutate shared nested objects across configs.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)!r} in {cfg_path}")
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"Invalid override {override!r} — expected dotted.key=value")
        _set_dotted(cfg, key.strip(), parse_scalar(raw))
    return cfg
