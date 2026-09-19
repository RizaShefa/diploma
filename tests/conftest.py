"""Shared fixtures.

Tests avoid depending on the real dataset wherever possible: synthetic images
are generated on the fly so the suite runs on a clean checkout, before BUSI has
been downloaded. Tests that genuinely need a trained model are skipped with a
clear reason rather than failing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_bgr(rng) -> np.ndarray:
    """A deterministic 'ultrasound-like' BGR image: dark, speckled, with a blob."""
    image = rng.normal(60, 18, size=(180, 240, 3)).clip(0, 255)
    yy, xx = np.mgrid[0:180, 0:240]
    blob = np.exp(-(((yy - 90) ** 2) / (2 * 25 ** 2) + ((xx - 120) ** 2) / (2 * 30 ** 2)))
    image += (blob * 90)[..., None]
    return image.clip(0, 255).astype(np.uint8)


@pytest.fixture
def png_bytes(synthetic_bgr) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", synthetic_bgr)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def synthetic_dataset(tmp_path, rng):
    """A two-class dataset on disk, with deliberate near-duplicate groups.

    Each 'source lesion' is written out several times with small perturbations,
    mimicking the augmented dataset that caused the original leakage. The
    splitter must keep every such group inside a single split.
    """
    import cv2

    from src.data.loader import Sample

    root = tmp_path / "busi"
    samples = []
    for label, class_name in enumerate(["benign", "malignant"]):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for source in range(12):           # 12 distinct sources per class
            base = rng.normal(70 + label * 25, 20, size=(96, 96)).clip(0, 255)
            for variant in range(3):       # 3 near-duplicate variants each
                image = base + rng.normal(0, 2, size=base.shape)
                path = class_dir / f"{class_name}_{source}_{variant}.png"
                cv2.imwrite(str(path), image.clip(0, 255).astype(np.uint8))
                samples.append(Sample(
                    path=str(path), class_name=class_name, label=label,
                    stem=path.stem, mask_paths=[], group_id=f"{class_name}_{source}",
                ))
    return {"root": root, "samples": samples, "classes": ["benign", "malignant"]}


@pytest.fixture
def tiny_model():
    """A minimal compiled 2-class CNN with a named conv layer for Grad-CAM."""
    from keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(4, (3, 3), name="conv_block1_conv"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), name="conv_block3_conv"),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(2, activation="softmax", name="predictions"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with history redirected to a temporary database."""
    monkeypatch.setenv("BCD_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    for module in [m for m in list(sys.modules) if m == "app"]:
        del sys.modules[module]
    import app as application

    application.app.config["TESTING"] = True
    return application.app.test_client()
