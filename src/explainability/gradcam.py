"""Grad-CAM (Selvaraju et al., 2017) for the ultrasound classifiers.

WHY GRAD-CAM AND NOT SHAP/LIME
------------------------------
Grad-CAM weights the final convolutional feature maps by the gradient of the
target class score, producing a class-discriminative spatial heatmap from a
single backward pass. For breast ultrasound that answers the question that
actually matters clinically: did the model look at the lesion, or at the depth
ruler, the burned-in scanner text, or the black background?

LIME's superpixel segmentation is unstable on speckle-textured ultrasound and
varies run to run. KernelSHAP is prohibitively slow per image and DeepSHAP's
pixel attributions are noisy on low-contrast grayscale. Neither is a better fit
here, and both cost far more compute.

RESOLUTION CAVEAT (important for the thesis)
--------------------------------------------
The heatmap's spatial precision is bounded by the final feature map, not by the
display size. The original 64x64 model ends with a 6x6 map -- upsampled, that is
a 36-cell blur, too coarse to be clinically meaningful. At 224x224 the custom
CNN reaches 26x26 and ResNet50/EfficientNet reach 7x7. `explain` reports
`feature_map_size` so this limitation is stated rather than hidden.

WHAT THIS IS NOT
----------------
The heatmap is a visualisation of model attention. It is not a segmentation, not
a lesion boundary, and not a medically validated localisation. The UI must say
so, and `explain` returns that disclaimer alongside the image.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..models.registry import find_last_conv_layer


class ExplainabilityError(RuntimeError):
    """Raised when a Grad-CAM heatmap cannot be produced."""


def _locate_layer(model, layer_name: str):
    """Find a layer by name, descending into a nested backbone if needed.

    Returns (layer, container) where `container` is the model that owns it.
    Transfer-learning models wrap the whole backbone as a single Sequential
    layer, so the target conv layer often lives one level down.
    """
    try:
        return model.get_layer(layer_name), model
    except ValueError:
        pass
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(layer_name), layer
            except ValueError:
                continue
    raise ExplainabilityError(
        f"Layer {layer_name!r} not found in model {getattr(model, 'name', '?')!r}."
    )


def compute_heatmap(
    model,
    input_batch: np.ndarray,
    class_index: int,
    layer_name: Optional[str] = None,
    architecture: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Return a normalised [0,1] Grad-CAM heatmap and the layer used.

    `input_batch` must already be preprocessed by `src.preprocessing` with the
    SAME profile used at training time, otherwise the gradients describe a model
    input that never occurs in practice.
    """
    import tensorflow as tf
    from keras import models as kmodels

    if input_batch.ndim != 4 or input_batch.shape[0] != 1:
        raise ExplainabilityError(
            f"Expected a single preprocessed image of shape (1,H,W,C), got {input_batch.shape}."
        )

    name = layer_name or find_last_conv_layer(model, architecture)
    target_layer, container = _locate_layer(model, name)

    # Build a graph from the model input to (conv activations, predictions).
    # When the conv layer sits inside a nested backbone, the backbone is first
    # re-expressed as a two-output model so activations remain reachable.
    if container is model:
        grad_model = kmodels.Model(model.inputs, [target_layer.output, model.output])
    else:
        inner = kmodels.Model(container.inputs, [target_layer.output, container.output])
        inputs = model.inputs
        conv_out, backbone_out = inner(inputs)
        x = backbone_out
        passed = False
        for layer in model.layers:
            if layer is container:
                passed = True
                continue
            if passed:
                x = layer(x)
        grad_model = kmodels.Model(inputs, [conv_out, x])

    tensor = tf.convert_to_tensor(input_batch.astype(np.float32))
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor, training=False)
        tape.watch(conv_outputs)
        if predictions.shape[-1] <= class_index:
            raise ExplainabilityError(
                f"class_index {class_index} out of range for {predictions.shape[-1]} outputs."
            )
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ExplainabilityError(
            "Gradients did not reach the target layer; cannot produce Grad-CAM."
        )

    # Channel importance = spatially averaged gradient (Selvaraju et al. eq. 1)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_outputs[0]
    heatmap = tf.reduce_sum(conv * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)  # only evidence FOR the class

    heatmap = heatmap.numpy().astype(np.float32)
    peak = float(heatmap.max())
    if peak <= 1e-12:
        # All-zero map: the class score does not increase with any spatial
        # region. Return zeros rather than dividing by ~0 and amplifying noise.
        return np.zeros_like(heatmap), name
    return heatmap / peak, name


def overlay_heatmap(
    display_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Composite a heatmap onto an un-normalised uint8 RGB image."""
    if display_rgb.dtype != np.uint8:
        display_rgb = np.clip(display_rgb, 0, 255).astype(np.uint8)
    height, width = display_rgb.shape[:2]
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    coloured = cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(display_rgb, 1.0 - alpha, coloured, alpha, 0)


DISCLAIMER = (
    "Grad-CAM shows which image regions most influenced this model's output. "
    "It is a visualisation of model attention, not a medically validated lesion "
    "segmentation or boundary."
)


def explain(
    model,
    input_batch: np.ndarray,
    display_rgb: np.ndarray,
    class_index: int,
    *,
    layer_name: Optional[str] = None,
    architecture: Optional[str] = None,
    alpha: float = 0.4,
) -> Dict[str, Any]:
    """Full explanation bundle: heatmap, overlay, and honest metadata.

    Raises ExplainabilityError on failure; callers are expected to degrade
    gracefully (return the prediction without an explanation) rather than fail
    the whole request.
    """
    heatmap, used_layer = compute_heatmap(
        model, input_batch, class_index, layer_name, architecture
    )
    overlay = overlay_heatmap(display_rgb, heatmap, alpha=alpha)
    return {
        "method": "Grad-CAM",
        "reference": "Selvaraju et al., 2017 (arXiv:1610.02391)",
        "layer": used_layer,
        "feature_map_size": list(heatmap.shape),
        "class_index": int(class_index),
        "heatmap": heatmap,
        "overlay_rgb": overlay,
        "is_degenerate": bool(heatmap.max() <= 0.0),
        "disclaimer": DISCLAIMER,
    }


def mask_agreement(heatmap: np.ndarray, mask: np.ndarray, percentile: float = 80.0) -> Dict[str, float]:
    """Overlap between the heatmap's hottest region and a ground-truth mask.

    BUSI ships lesion masks, so Grad-CAM output can be scored rather than merely
    displayed. Reports the 'pointing game' hit (does the single hottest pixel
    fall inside the lesion?) and the IoU of the thresholded heatmap.

    NOTE: this is provided for offline analysis only and is NOT wired into the
    prediction endpoint, since uploaded images have no ground-truth mask.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary_mask = (mask > 127).astype(np.uint8)
    resized = cv2.resize(heatmap, (binary_mask.shape[1], binary_mask.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    peak_y, peak_x = np.unravel_index(int(np.argmax(resized)), resized.shape)
    hit = bool(binary_mask[peak_y, peak_x])

    # Exclude zero-valued cells from the "hot" region. Grad-CAM maps are ReLU'd,
    # so a focused heatmap is mostly exact zeros; without this guard the
    # percentile cutoff lands on 0 and selects the ENTIRE image, collapsing IoU
    # towards the mask's area fraction regardless of where the model looked.
    cutoff = np.percentile(resized, percentile)
    hot = ((resized >= cutoff) & (resized > 0)).astype(np.uint8)
    if hot.sum() == 0:  # wholly degenerate map: nothing to compare
        return {
            "pointing_game_hit": float(hit),
            "iou": 0.0,
            "mask_coverage": float(binary_mask.mean()),
            "percentile": float(percentile),
            "degenerate": True,
        }
    intersection = int((hot & binary_mask).sum())
    union = int((hot | binary_mask).sum())
    return {
        "pointing_game_hit": float(hit),
        "iou": float(intersection / union) if union else 0.0,
        "mask_coverage": float(binary_mask.mean()),
        "percentile": float(percentile),
    }
