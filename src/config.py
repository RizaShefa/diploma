"""Configuration loading and global seeding.

Configs are plain JSON so the project needs no YAML dependency. A config is a
nested dict; `load_config` merges an experiment config on top of `configs/base.json`
so experiments only declare what differs. That is what makes the model comparison
fair: every model inherits the same data, preprocessing and training settings
unless it explicitly overrides them.
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(name_or_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load `configs/base.json`, optionally merged with an experiment config.

    `name_or_path` may be a bare name ("resnet50"), a filename ("resnet50.json")
    or a full path. Returns the merged config with a `_config_name` key added.
    """
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    config["_config_name"] = "base"

    if name_or_path is None:
        return config

    path = Path(name_or_path)
    if not path.exists():
        candidate = CONFIG_DIR / path.name
        if not candidate.suffix:
            candidate = CONFIG_DIR / f"{path.name}.json"
        path = candidate
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {name_or_path}. Looked in {CONFIG_DIR}."
        )

    with open(path, "r", encoding="utf-8") as fh:
        override = json.load(fh)
    merged = _deep_merge(config, override)
    merged["_config_name"] = path.stem
    return merged


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow.

    Called at the start of every training and evaluation run so results are
    reproducible. Note that full determinism on GPU also requires
    TF_DETERMINISTIC_OPS, which is set here but only takes effect if set before
    TensorFlow initialises its kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - allows non-TF unit tests
        pass


def resolve_path(config: Dict[str, Any], *keys: str) -> Path:
    """Resolve a (possibly relative) path from config against the project root."""
    node: Any = config
    for key in keys:
        node = node[key]
    path = Path(node)
    return path if path.is_absolute() else PROJECT_ROOT / path
