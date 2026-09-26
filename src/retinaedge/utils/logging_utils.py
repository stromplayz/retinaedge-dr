"""Small logging helper shared by all CLIs."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s: %(message)s"
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger("retinaedge")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under ``retinaedge`` with a single handler."""
    _configure_root()
    if not name.startswith("retinaedge"):
        name = f"retinaedge.{name}"
    return logging.getLogger(name)
