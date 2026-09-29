"""Kaggle dataset fetcher — authenticated via the new KGAT bearer tokens.

Downloads Kaggle datasets with `KAGGLE_API_TOKEN` (the modern ``KGAT_…``
format; falls back to legacy ``KAGGLE_USERNAME``/``KAGGLE_KEY``) and unpacks
them into the standard ``images/ + labels.csv``-agnostic raw layout used by
the data pipeline.  A ``--max-bytes`` probe mode lets CI and local runs
verify access without pulling multi-GB archives.

Catalog of validated fundus/DR datasets (see configs/data/kaggle_dr_catalog.yaml):
  ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy  ~22 GB, 88k images
  sehastrajits/fundus-aptosddridirdeyepacsmessidor       ~10.8 GB, 5 sources
  mariaherrerot/aptos2019                               ~8.6 GB, APTOS raw
  sovitrath/diabetic-retinopathy-224x224-gaussian-filtered  ~447 MB, 3662
  mariaherrerot/messidor2preprocess                     ~399 MB, Messidor-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests

__all__ = ["kaggle_auth_headers", "download_dataset", "dataset_info"]

_API = "https://www.kaggle.com/api/v1"


def kaggle_auth_headers() -> dict[str, str]:
    """Build auth headers from the environment (KGAT bearer preferred)."""
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user, key = os.environ.get("KAGGLE_USERNAME", ""), os.environ.get("KAGGLE_KEY", "")
    if user and key:
        import base64

        b64 = base64.b64encode(f"{user}:{key}".encode()).decode()
        return {"Authorization": f"Basic {b64}"}
    raise RuntimeError(
        "Kaggle credentials missing: set KAGGLE_API_TOKEN (KGAT_…) or KAGGLE_USERNAME/KAGGLE_KEY"
    )


def dataset_info(handle: str) -> dict[str, Any]:
    """Fetch dataset metadata (title, size, license) for provenance."""
    r = requests.get(
        f"{_API}/datasets/list",
        params={"search": handle},
        headers=kaggle_auth_headers(),
        timeout=30,
    )
    r.raise_for_status()
    owner, slug = handle.split("/")
    for d in r.json():
        if d.get("ref") == handle or d.get("urlNullable", "").endswith(slug):
            return {
                "handle": handle,
                "title": d.get("titleNullable"),
                "total_bytes": d.get("totalBytesNullable"),
                "license": d.get("licenseNameNullable"),
                "usability": d.get("usabilityRatingNullable"),
            }
    return {"handle": handle, "title": None, "total_bytes": None, "license": None}


def download_dataset(
    handle: str,
    out_dir: str | Path,
    max_bytes: int | None = None,
    keep_zip: bool = False,
) -> dict[str, Any]:
    """Download and unpack a Kaggle dataset into ``out_dir``.

    Args:
        handle: ``owner/slug`` dataset reference.
        out_dir: Destination directory (created; existing files kept).
        max_bytes: Probe mode — fetch only the first N bytes of the archive
            and report validity (no unpacking).
        keep_zip: Keep the downloaded archive next to the unpacked files.

    Returns:
        Dict with paths, sizes, and (probe) validity info.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = f"{_API}/datasets/download/{handle}"
    headers = kaggle_auth_headers()

    if max_bytes is not None:
        with requests.get(
            url, headers=headers, stream=True, timeout=120, params={"maxSize": None}
        ) as r:
            r.raise_for_status()
            head = b""
            for chunk in r.iter_content(chunk_size=1 << 20):
                head += chunk
                if len(head) >= max_bytes:
                    break
        valid_zip = head[:4] == b"PK\x03\x04"
        return {
            "handle": handle,
            "probe_bytes": len(head),
            "zip_magic_valid": valid_zip,
            "mode": "probe",
        }

    zip_path = out / f"{handle.replace('/', '__')}.zip"
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0) or 0)
        done = 0
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (100 << 20) < (1 << 22):
                    print(f"  downloaded {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)

    extract_dir = out / handle.split("/")[-1]
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    if not keep_zip:
        zip_path.unlink()

    n_files = sum(1 for p in extract_dir.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in extract_dir.rglob("*") if p.is_file())
    info = {
        "handle": handle,
        "out_dir": str(extract_dir),
        "files": n_files,
        "bytes": size,
        "mode": "full",
    }
    (extract_dir / "kaggle_provenance.json").write_text(
        json.dumps({**info, **dataset_info(handle)}, indent=2), encoding="utf-8"
    )
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset (KGAT token auth)")
    parser.add_argument("--handle", required=True, help="owner/slug Kaggle dataset reference")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-bytes", type=int, default=None, help="probe mode: fetch only the first N bytes"
    )
    parser.add_argument("--keep-zip", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    result = download_dataset(
        args.handle, args.out, max_bytes=args.max_bytes, keep_zip=args.keep_zip
    )
    print(json.dumps(result, indent=2))
    if result.get("mode") == "probe" and not result.get("zip_magic_valid"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
