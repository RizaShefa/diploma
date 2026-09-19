"""Preprocessing tests.

The central test here is `test_training_and_serving_paths_are_identical`. It is
the regression test for the defect that flipped 6% of live predictions: training
applied BGR->RGB conversion and normalisation, serving applied neither. Both
paths now route through `preprocess_array`, and this asserts they cannot drift.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    PROFILES,
    DEFAULT_PROFILE,
    PreprocessingError,
    decode_image_bgr,
    display_image_rgb,
    get_profile,
    load_image_bgr,
    preprocess_array,
    preprocess_batch_paths,
    preprocess_path,
)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_profile_produces_expected_shape(synthetic_bgr, profile):
    batch = preprocess_array(synthetic_bgr, profile)
    height, width = get_profile(profile)["image_size"]
    assert batch.shape == (1, height, width, 3)
    assert batch.dtype == np.float32


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_training_and_serving_paths_are_identical(tmp_path, synthetic_bgr, profile):
    """REGRESSION: the file-based (training) and byte-based (serving) paths must agree.

    Training reads from disk via `preprocess_batch_paths`; the Flask endpoint
    decodes uploaded bytes and calls `preprocess_array`. If these ever diverge,
    offline metrics stop describing live behaviour.
    """
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), synthetic_bgr)

    from_disk = preprocess_batch_paths([str(path)], profile)
    from_bytes = preprocess_array(decode_image_bgr(path.read_bytes()), profile)

    np.testing.assert_allclose(from_disk[0], from_bytes[0], rtol=0, atol=0)


def test_bgr_to_rgb_conversion_actually_happens(tmp_path):
    """A pure-red BGR image must come out red in channel 0, not channel 2."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 2] = 255  # red in BGR ordering
    batch = preprocess_array(image, "standard_224")[0]
    assert batch[..., 0].mean() > batch[..., 2].mean(), "BGR->RGB conversion missing"


def test_legacy_profile_matches_original_training_code(tmp_path, synthetic_bgr):
    """`legacy_64` must reproduce MainTrain.py's preprocessing exactly.

    Original: cv2.imread -> cvtColor(BGR2RGB) -> PIL resize(64,64) -> normalize(axis=1).
    Reproducing it is what keeps the shipped BreastCancer10Epochs.h5 valid.
    """
    from keras.utils import normalize
    from PIL import Image

    path = tmp_path / "legacy.png"
    cv2.imwrite(str(path), synthetic_bgr)

    original = cv2.imread(str(path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = np.array(Image.fromarray(original).resize((64, 64)))
    original = normalize(np.expand_dims(original, 0).astype(np.float32), axis=1)

    ours = preprocess_path(path, "legacy_64")
    assert ours.shape == original.shape
    # PIL and cv2 use different resampling kernels, so allow a small tolerance;
    # what must match is the pipeline (channel order, size, normalisation).
    assert np.corrcoef(ours.ravel(), original.ravel())[0, 1] > 0.98


def test_rescale_01_is_in_unit_range(synthetic_bgr):
    batch = preprocess_array(synthetic_bgr, "standard_224")
    assert batch.min() >= 0.0 and batch.max() <= 1.0


def test_efficientnet_profile_keeps_raw_range(synthetic_bgr):
    """EfficientNet rescales inside the graph, so input must stay in [0,255]."""
    batch = preprocess_array(synthetic_bgr, "efficientnet_224")
    assert batch.max() > 1.5


def test_grayscale_and_alpha_inputs_are_handled():
    grayscale = np.full((40, 40), 120, dtype=np.uint8)
    assert preprocess_array(grayscale, "standard_224").shape == (1, 224, 224, 3)
    with_alpha = np.full((40, 40, 4), 120, dtype=np.uint8)
    assert preprocess_array(with_alpha, "standard_224").shape == (1, 224, 224, 3)


def test_display_image_is_uint8_and_unnormalised(synthetic_bgr):
    """Grad-CAM overlays must be drawn on viewable pixels, not normalised floats."""
    display = display_image_rgb(synthetic_bgr, (224, 224))
    assert display.dtype == np.uint8
    assert display.shape == (224, 224, 3)
    assert display.max() > 1


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        get_profile("does_not_exist")


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(PreprocessingError, match="not found"):
        load_image_bgr(tmp_path / "nope.png")


def test_corrupt_bytes_raise(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is definitely not a png")
    with pytest.raises(PreprocessingError, match="Could not decode"):
        load_image_bgr(path)
    with pytest.raises(PreprocessingError):
        decode_image_bgr(b"nonsense")


def test_empty_inputs_raise():
    with pytest.raises(PreprocessingError):
        decode_image_bgr(b"")
    with pytest.raises(PreprocessingError):
        preprocess_array(np.zeros((0, 0, 3), dtype=np.uint8), DEFAULT_PROFILE)


def test_empty_batch_raises():
    with pytest.raises(PreprocessingError):
        preprocess_batch_paths([], DEFAULT_PROFILE)
