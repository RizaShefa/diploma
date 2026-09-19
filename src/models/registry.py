"""Model architectures and the registry that keeps the comparison fair.

Every architecture is built through `build_model`, which applies the SAME
classification head (global pooling -> dropout -> softmax), the same number of
output classes, and the same optimiser/loss settings from config. Only the
feature extractor differs. That is what makes the comparison in
`scripts/compare_models.py` an architecture comparison rather than an
accident of differing training recipes.

Each entry also declares `last_conv_layer`, the layer Grad-CAM reads activations
from. Declaring it here -- rather than guessing at explain time -- is what lets
one Grad-CAM implementation serve all three architectures correctly.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "custom_cnn": {
        "description": "The thesis's original 3-block CNN, retrained under the corrected protocol.",
        "pretrained": False,
        "last_conv_layer": "conv_block3_conv",
        "default_profile": "standard_224",
    },
    "resnet50": {
        "description": "ImageNet-pretrained ResNet50 (He et al. 2016) with a new head.",
        "pretrained": True,
        "last_conv_layer": "conv5_block3_out",
        "default_profile": "resnet50_224",
    },
    "efficientnetb0": {
        "description": "ImageNet-pretrained EfficientNet-B0 (Tan & Le 2019) with a new head.",
        "pretrained": True,
        "last_conv_layer": "top_activation",
        "default_profile": "efficientnet_224",
    },
}


def available() -> Dict[str, Dict[str, Any]]:
    return ARCHITECTURES


def get_spec(architecture: str) -> Dict[str, Any]:
    if architecture not in ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture {architecture!r}. Known: {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[architecture]


def _build_custom_cnn(
    input_shape, dropout: float, head: str = "gap", batch_norm: bool = True
):
    """The original architecture from MainTrain.py, with optional corrections.

    Preserved by default: three Conv/ReLU/MaxPool blocks with 32/32/64 filters,
    Dense(64), dropout, softmax output -- so the comparison includes the thesis's
    own model rather than a substitute.

    Two deviations, both switchable so the original remains reproducible:

    `head="flatten"` reproduces the original Flatten -> Dense(64) head exactly.
    `head="gap"` (default) uses GlobalAveragePooling2D instead. At 224x224 the
    original head would flatten a 26x26x64 map into a 2.7M-parameter dense layer
    -- roughly 4,000 parameters per training image, which overfits badly on 647
    images. Pooling also gives every architecture in the comparison the same
    head, so the comparison isolates the feature extractor.

    `batch_norm=True` (default) adds BatchNormalization after each convolution.
    The original had none, despite the thesis claiming it in Ch. 10.2; enabling
    it makes that claim true and stabilises training at the larger input size.

    Convolutions are explicitly named so Grad-CAM can locate the final feature
    map deterministically rather than guessing.
    """
    from keras import layers, models

    model = models.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=input_shape))
    for block, filters in enumerate([32, 32, 64], start=1):
        model.add(
            layers.Conv2D(
                filters,
                (3, 3),
                kernel_initializer="he_uniform",
                name=f"conv_block{block}_conv",
            )
        )
        if batch_norm:
            model.add(layers.BatchNormalization(name=f"conv_block{block}_bn"))
        model.add(layers.Activation("relu", name=f"conv_block{block}_relu"))
        model.add(layers.MaxPooling2D((2, 2), name=f"conv_block{block}_pool"))

    if head == "flatten":
        model.add(layers.Flatten(name="flatten"))
    elif head == "gap":
        model.add(layers.GlobalAveragePooling2D(name="gap"))
    else:
        raise KeyError(f"Unknown head {head!r}; expected 'gap' or 'flatten'.")

    model.add(layers.Dense(64, activation="relu", name="head_dense"))
    model.add(layers.Dropout(dropout, name="head_dropout"))
    return model


def _build_pretrained(architecture: str, input_shape, dropout: float,
                      freeze_backbone: bool, fine_tune_from: Optional[int]):
    """Build an ImageNet-pretrained backbone with a fresh pooled head."""
    from keras import layers, models

    if architecture == "resnet50":
        from keras.applications import ResNet50 as Backbone
    elif architecture == "efficientnetb0":
        from keras.applications import EfficientNetB0 as Backbone
    else:  # pragma: no cover - guarded by get_spec
        raise KeyError(architecture)

    backbone = Backbone(include_top=False, weights="imagenet", input_shape=input_shape)

    if freeze_backbone:
        backbone.trainable = False
    elif fine_tune_from is not None:
        # Freeze early generic filters, fine-tune the deeper, more specialised
        # layers. With only a few hundred training images, fine-tuning the whole
        # network tends to overfit.
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
    else:
        backbone.trainable = True

    model = models.Sequential(name=architecture)
    model.add(layers.Input(shape=input_shape))
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D(name="gap"))
    model.add(layers.Dropout(dropout, name="head_dropout"))
    return model


def build_model(config: Dict[str, Any], n_classes: int):
    """Construct and compile a model from a merged config."""
    from keras import layers, optimizers

    from ..preprocessing import get_profile

    model_cfg = config["model"]
    architecture = model_cfg["architecture"]
    spec = get_spec(architecture)
    dropout = float(model_cfg.get("dropout", 0.5))

    profile = get_profile(config["preprocessing"]["profile"])
    height, width = profile["image_size"]
    input_shape = (height, width, 3)

    if architecture == "custom_cnn":
        model = _build_custom_cnn(
            input_shape,
            dropout,
            head=str(model_cfg.get("head", "gap")),
            batch_norm=bool(model_cfg.get("batch_norm", True)),
        )
    else:
        model = _build_pretrained(
            architecture,
            input_shape,
            dropout,
            bool(model_cfg.get("freeze_backbone", False)),
            model_cfg.get("fine_tune_from"),
        )

    model.add(layers.Dense(n_classes, activation="softmax", name="predictions"))

    training = config["training"]
    optimizer_name = str(training.get("optimizer", "adam")).lower()
    lr = float(training.get("learning_rate", 1e-4))
    if optimizer_name == "adam":
        optimizer = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise KeyError(f"Unsupported optimizer {optimizer_name!r}")

    model.compile(
        optimizer=optimizer,
        loss=training.get("loss", "categorical_crossentropy"),
        metrics=["accuracy"],
    )
    return model


def find_last_conv_layer(model, architecture: Optional[str] = None) -> str:
    """Resolve the Grad-CAM target layer.

    Prefers the registry's declared layer. Falls back to the last 4-D output
    layer, searching inside a nested backbone if the model wraps one. Raising a
    clear error beats silently explaining the wrong layer.
    """
    if architecture:
        declared = get_spec(architecture).get("last_conv_layer")
        if declared:
            try:
                model.get_layer(declared)
                return declared
            except ValueError:
                for layer in model.layers:
                    if hasattr(layer, "layers"):
                        try:
                            layer.get_layer(declared)
                            return declared
                        except ValueError:
                            continue

    def _scan(container) -> Optional[str]:
        for layer in reversed(container.layers):
            if hasattr(layer, "layers"):
                nested = _scan(layer)
                if nested:
                    return nested
            output_shape = getattr(layer, "output_shape", None)
            if isinstance(output_shape, tuple) and len(output_shape) == 4:
                return layer.name
        return None

    found = _scan(model)
    if not found:
        raise ValueError(
            "No 4-D convolutional output found; Grad-CAM cannot be generated "
            f"for model {getattr(model, 'name', '<unnamed>')!r}."
        )
    return found
