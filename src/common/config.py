"""Configuration loading and shared path helpers.

Every script in this repo takes ``--config configs/default.yaml`` and resolves
all paths relative to the repository root, so behaviour does not depend on the
directory you happen to launch from.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# repo root = .../src/common/config.py -> up three levels
REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """Dict with attribute access, so cfg.data.tile_size works."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run scripts from the repository root."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply --set key.path=value overrides in place.

    This is what lets one runner script drive the whole ablation ladder without
    maintaining seven near-identical YAML files that inevitably drift apart.
    Values are parsed as YAML, so 0.5, [0.5, 2.0], true and vienna all work.
    """
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"unknown config path: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = value
    return cfg


def experiment_settings(cfg: Config) -> Dict[str, Any]:
    """Resolve the active experiment mode into concrete settings."""
    mode = cfg["experiment"]["mode"]
    if mode not in cfg["experiment"]:
        raise KeyError(f"experiment.mode={mode!r} has no matching block")
    block = dict(cfg["experiment"][mode])
    block["mode"] = mode
    block.setdefault("oodval_cities", [])
    return block


def resolve(path_like: str | Path) -> Path:
    """Turn a config-relative path into an absolute path under the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"
