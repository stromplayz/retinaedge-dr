"""Training package: trainer loop/CLI (:mod:`retinaedge.train.trainer`) and
metric accumulators (:mod:`retinaedge.train.metrics`)."""

from retinaedge.train.metrics import QWKTracker
from retinaedge.train.trainer import build_loaders, train

__all__ = ["QWKTracker", "build_loaders", "train"]
