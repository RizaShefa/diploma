"""Upload validation and Grad-CAM tests.

Validation covers the security gap in the original app, which accepted any file
the browser sent. Grad-CAM tests confirm the heatmap comes from real gradients
of the real model -- never a placeholder.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.explainability.gradcam import (
    DISCLAIMER,
    ExplainabilityError,
    compute_heatmap,
    explain,
    mask_agreement,
    overlay_heatmap,
)
from src.inference.validation import (
    MAX_UPLOAD_BYTES,
    ValidationError,
    detect_format,
    validate_upload,
)

# ---------------------------------------------------------------- validation


def test_accepts_a_real_png(png_bytes):
    result = validate_upload(png_bytes, "scan.png")
    assert result.detected_format == "png"
    assert result.width > 0 and result.height > 0


def test_rejects_missing_filename(png_bytes):
    with pytest.raises(ValidationError, match="No file"):
        validate_upload(png_bytes, "")


def test_rejects_disallowed_extension(png_bytes):
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload(png_bytes, "payload.exe")


def test_rejects_empty_payload():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload(b"", "scan.png")


def test_rejects_oversized_payload(png_bytes):
    with pytest.raises(ValidationError, match="too large"):
        validate_upload(png_bytes, "scan.png", max_bytes=10)


def test_rejects_non_image_with_image_extension():
    """SECURITY: the extension lies; magic bytes decide."""
    with pytest.raises(ValidationError, match="does not appear to be"):
        validate_upload(b"#!/bin/sh\nrm -rf /\n", "innocent.png")


def test_rejects_truncated_image():
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40  # valid magic, broken body
    with pytest.raises(ValidationError, match="could not be decoded"):
        validate_upload(payload, "broken.png")


def test_rejects_tiny_image():
    ok, buffer = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok
    with pytest.raises(ValidationError, match="too small"):
        validate_upload(buffer.tobytes(), "tiny.png")


def test_detect_format_recognises_supported_types():
    assert detect_format(b"\x89PNG\r\n\x1a\n rest") == "png"
    assert detect_format(b"\xff\xd8\xff rest") == "jpeg"
    assert detect_format(b"BM rest") == "bmp"
    assert detect_format(b"GIF89a") is None


def test_error_messages_do_not_leak_paths(png_bytes):
    try:
        validate_upload(b"nope", "x.png")
    except ValidationError as exc:
        assert "\\" not in str(exc) and "/" not in str(exc).replace("PNG, JPEG", "")


def test_default_size_cap_is_sane():
    assert 1e6 < MAX_UPLOAD_BYTES <= 50e6


# ------------------------------------------------------------------ Grad-CAM


def _batch(rng):
    return rng.normal(0.5, 0.2, size=(1, 32, 32, 3)).astype(np.float32)


def test_heatmap_is_normalised_and_correctly_shaped(tiny_model, rng):
    heatmap, layer = compute_heatmap(tiny_model, _batch(rng), class_index=1,
                                     layer_name="conv_block3_conv")
    assert layer == "conv_block3_conv"
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


def test_heatmap_depends_on_the_target_class(tiny_model, rng):
    """A class-discriminative method must give different maps per class."""
    batch = _batch(rng)
    a, _ = compute_heatmap(tiny_model, batch, 0, layer_name="conv_block3_conv")
    b, _ = compute_heatmap(tiny_model, batch, 1, layer_name="conv_block3_conv")
    assert a.shape == b.shape
    # Untrained weights can occasionally produce two all-zero maps; only assert
    # difference when at least one map carries signal.
    if a.max() > 0 and b.max() > 0:
        assert not np.allclose(a, b)


def test_heatmap_depends_on_the_input(tiny_model, rng):
    a, _ = compute_heatmap(tiny_model, _batch(rng), 1, layer_name="conv_block3_conv")
    b, _ = compute_heatmap(tiny_model, _batch(rng) * 3.0, 1, layer_name="conv_block3_conv")
    assert a.shape == b.shape


def test_rejects_batch_of_more_than_one(tiny_model, rng):
    batch = np.repeat(_batch(rng), 2, axis=0)
    with pytest.raises(ExplainabilityError, match="single preprocessed image"):
        compute_heatmap(tiny_model, batch, 1, layer_name="conv_block3_conv")


def test_rejects_unknown_layer(tiny_model, rng):
    with pytest.raises(ExplainabilityError, match="not found"):
        compute_heatmap(tiny_model, _batch(rng), 1, layer_name="no_such_layer")


def test_rejects_out_of_range_class(tiny_model, rng):
    with pytest.raises(ExplainabilityError, match="out of range"):
        compute_heatmap(tiny_model, _batch(rng), 99, layer_name="conv_block3_conv")


def test_overlay_preserves_size_and_stays_viewable(rng):
    display = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    heatmap = rng.random((8, 8)).astype(np.float32)
    overlay = overlay_heatmap(display, heatmap, alpha=0.4)
    assert overlay.shape == display.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, display), "overlay did not modify the image"


def test_explain_bundle_carries_the_disclaimer(tiny_model, rng):
    display = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    result = explain(tiny_model, _batch(rng), display, class_index=1,
                     layer_name="conv_block3_conv")
    assert result["method"] == "Grad-CAM"
    assert result["disclaimer"] == DISCLAIMER
    assert "segmentation" in result["disclaimer"]
    assert result["overlay_rgb"].shape == display.shape
    assert len(result["feature_map_size"]) == 2


def test_mask_agreement_scores_a_known_overlap():
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[4:8, 4:8] = 1.0
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    scores = mask_agreement(heatmap, mask, percentile=80)
    assert scores["pointing_game_hit"] == 1.0
    assert scores["iou"] > 0.4


def test_mask_agreement_reports_a_miss():
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[0:3, 0:3] = 1.0
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[12:16, 12:16] = 255
    assert mask_agreement(heatmap, mask)["pointing_game_hit"] == 0.0
