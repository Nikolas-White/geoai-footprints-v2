"""Environment diagnostic. Run this and send me the full output.

    python -m src.common.check_env
"""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

from src.common.config import REPO_ROOT, human_bytes

PACKAGES = [
    "numpy", "pandas", "yaml", "cv2", "skimage", "PIL",
    "rasterio", "shapely", "geopandas", "pyproj", "fiona",
    "torch", "torchvision", "segmentation_models_pytorch", "timm",
    "matplotlib", "folium", "py7zr", "requests", "tqdm",
]

VERSION_ATTRS = ["__version__", "version", "VERSION"]


def version_of(mod) -> str:
    for attr in VERSION_ATTRS:
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def main() -> int:
    print("=" * 68)
    print("SYSTEM")
    print("=" * 68)
    print(f"platform     : {platform.platform()}")
    print(f"python       : {sys.version.splitlines()[0]}")
    print(f"executable   : {sys.executable}")
    print(f"repo root    : {REPO_ROOT}")

    total, used, free = shutil.disk_usage(REPO_ROOT)
    print(f"disk (repo)  : {human_bytes(free)} free of {human_bytes(total)}")

    print()
    print("=" * 68)
    print("PACKAGES")
    print("=" * 68)
    missing = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:32s} {version_of(mod)}")
        except Exception as exc:  # noqa: BLE001
            missing.append(name)
            print(f"  {name:32s} MISSING ({type(exc).__name__})")

    print()
    print("=" * 68)
    print("GPU")
    print("=" * 68)
    try:
        import torch

        print(f"torch.cuda.is_available : {torch.cuda.is_available()}")
        print(f"torch CUDA build        : {torch.version.cuda}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  device {i}: {props.name}")
                print(f"    VRAM          : {human_bytes(props.total_memory)}")
                print(f"    capability    : {props.major}.{props.minor}")
                print(f"    multiprocs    : {props.multi_processor_count}")
        else:
            print("  No CUDA device visible to PyTorch.")
    except ImportError:
        print("  torch not installed")

    print()
    print("=" * 68)
    print("DATA")
    print("=" * 68)
    for rel in [
        "data/archive",
        "data/raw/AerialImageDataset/train/images",
        "data/raw/AerialImageDataset/train/gt",
        "data/processed/tiles/train/images",
        "data/processed/tiles/val/images",
    ]:
        p = REPO_ROOT / rel
        if p.is_dir():
            n = sum(1 for _ in p.iterdir())
            print(f"  {rel:44s} {n} entries")
        else:
            print(f"  {rel:44s} (absent)")

    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        return 1
    print("\n[OK] Environment looks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
