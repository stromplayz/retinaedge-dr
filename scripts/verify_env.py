"""Environment sanity check: prints versions of the core dependency stack.

Usage: python scripts/verify_env.py
Exit code 0 always (diagnostic only).
"""

from __future__ import annotations

import importlib
import platform
import sys


def _version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 - diagnostic must not crash
        return f"MISSING ({type(exc).__name__})"


def main() -> int:
    print(f"python  : {platform.python_version()} ({sys.executable})")
    packages = [
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "cv2",
        "albumentations",
        "PIL",
        "pandas",
        "sklearn",
        "yaml",
        "onnx",
        "onnxruntime",
        "kagglehub",
        "pytest",
    ]
    for pkg in packages:
        print(f"{pkg:<14}: {_version(pkg)}")
    try:
        import torch

        print(f"cuda     : {torch.cuda.is_available()}")
        print(f"threads  : {torch.get_num_threads()}")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
