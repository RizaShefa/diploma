"""Image preprocessing -- the single source of truth for training AND serving.

WHY THIS MODULE EXISTS
----------------------
The original implementation preprocessed images differently in training
(`MainTrain.py`) and in serving (`app.py`):

    training : cv2.imread -> cvtColor(BGR2RGB) -> resize(64) -> keras normalize(axis=1)
    serving  : cv2.imread ->        (no cvtColor)-> resize(64) -> (no normalize)

Measured on 150 held-out images with the shipped model, that train/serve skew
flipped 9/150 predictions (6.0%) and shifted mean |p| by 0.068. Any accuracy
figure obtained offline therefore did not describe what the web app actually did.

Every consumer -- the trainer, the evaluator, the Grad-CAM explainer and the
Flask predictor -- now calls `preprocess_array` from here. A model's profile is
recorded in its metadata at training time and replayed at inference time, so the
two cannot drift apart. `tests/test_preprocessing.py` asserts this invariant.

PROFILES
--------
legacy_64      Reproduces MainTrain.py exactly (64x64, RGB, L2 row-normalised).
               Kept so the pre-existing BreastCancer10Epochs.h5 keeps working --
               and, because serving now uses the *same* steps as training, it is
               applied correctly for the first time.
standard_224   224x224 RGB scaled to [0,1]. Default for newly trained models;
               large enough for a meaningful Grad-CAM feature map.
resnet50_224   224x224 with keras.applications.resnet50.preprocess_input.
efficientnet_224  224x224 raw [0,255]; EfficientNet rescales inside the graph.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

# Profile registry: name -> (image_size, normalization key)
PROFILES: Dict[str, Dict[str, Any]] = {
    "legacy_64": {"image_size": (64, 64), "normalization": "legacy_l2"},
    "standard_224": {"image_size": (224, 224), "normalization": "rescale_01"},
    "resnet50_224": {"image_size": (224, 224), "normalization": "resnet50"},
    "efficientnet_224": {"image_size": (224, 224), "normalization": "efficientnet"},
}

DEFAULT_PROFILE = "standard_224"


class PreprocessingError(ValueError):
    """Raised when an image cannot be decoded or preprocessed."""


def get_profile(name: str) -> Dict[str, Any]:
    if name not in PROFILES:
        raise KeyError(
            f"Unknown preprocessing profile {name!r}. Known: {sorted(PROFILES)}"
        )
    return PROFILES[name]


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image from disk as BGR uint8, raising on failure.

    cv2.imread returns None rather than raising for unreadable/corrupt files,
    which previously produced a confusing downstream crash.
    """
    path = Path(path)
    if not path.exists():
        raise PreprocessingError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(f"Could not decode image: {path.name}")
    return image


def decode_image_bgr(payload: bytes) -> np.ndarray:
    """Decode raw bytes (e.g. an HTTP upload) into a BGR uint8 array."""
    if not payload:
        raise PreprocessingError("Empty image payload.")
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError("Could not decode image data.")
    return image


def _normalize(batch: np.ndarray, kind: str) -> np.ndarray:
    """Apply the profile's normalization to a float32 RGB batch in [0,255]."""
    if kind == "legacy_l2":
        # Reproduces keras.utils.normalize(x, axis=1) as used by MainTrain.py.
        # This is an L2 row-normalisation, NOT a [0,1] rescale -- an unusual
        # choice, but reproducing it exactly is what keeps the legacy model valid.
        from keras.utils import normalize as keras_normalize

        return keras_normalize(batch, axis=1)
    if kind == "rescale_01":
        return batch / 255.0
    if kind == "resnet50":
        from keras.applications.resnet50 import preprocess_input

        return preprocess_input(batch.copy())
    if kind == "efficientnet":
        # EfficientNet in Keras includes a Rescaling layer, so it wants [0,255].
        return batch
    raise KeyError(f"Unknown normalization {kind!r}")


def preprocess_array(
    image_bgr: np.ndarray,
    profile: str = DEFAULT_PROFILE,
    *,
    as_batch: bool = True,
) -> np.ndarray:
    """Convert a BGR uint8 image into a model-ready float32 array.

    Steps, in order, identical for training and serving:
      1. BGR -> RGB   (cv2 loads BGR; the models are trained on RGB)
      2. resize to the profile's image_size (INTER_AREA: correct for downscaling)
      3. cast to float32
      4. profile-specific normalization

    Returns shape (1, H, W, 3) when `as_batch`, else (H, W, 3).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise PreprocessingError("Empty image array.")
    spec = get_profile(profile)

    if image_bgr.ndim == 2:  # grayscale -> 3 channels
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:  # drop alpha
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, spec["image_size"], interpolation=cv2.INTER_AREA)
    batch = resized.astype(np.float32)[np.newaxis, ...]
    batch = _normalize(batch, spec["normalization"])
    return batch if as_batch else batch[0]


def preprocess_path(path: str | Path, profile: str = DEFAULT_PROFILE, **kw) -> np.ndarray:
    """Convenience wrapper: load from disk then preprocess."""
    return preprocess_array(load_image_bgr(path), profile, **kw)


def preprocess_batch_paths(
    paths, profile: str = DEFAULT_PROFILE
) -> np.ndarray:
    """Preprocess many paths into one stacked batch (used by training/eval)."""
    arrays = [preprocess_array(load_image_bgr(p), profile, as_batch=False) for p in paths]
    if not arrays:
        raise PreprocessingError("No images to preprocess.")
    return np.stack(arrays).astype(np.float32)


def display_image_rgb(image_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize for display/overlay WITHOUT normalization (stays uint8 RGB).

    Grad-CAM overlays must be drawn on the un-normalised image, otherwise the
    heatmap is composited onto values that are not viewable pixels.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
