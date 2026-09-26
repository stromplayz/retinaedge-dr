"""Evaluation package: checkpoint evaluation (:mod:`retinaedge.eval.evaluate`)
and temperature calibration (:mod:`retinaedge.eval.calibration`)."""

from retinaedge.eval.calibration import calibrate, fit_temperature
from retinaedge.eval.evaluate import evaluate

__all__ = ["calibrate", "evaluate", "fit_temperature"]
