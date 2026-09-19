"""Training-time data augmentation.

SCIENTIFIC CONSTRAINT
---------------------
Augmentation is applied to the TRAINING SPLIT ONLY, on the fly, after splitting.
Augmenting before splitting is precisely what corrupted the dataset originally
shipped with this project: copies of one lesion ended up in both train and test,
inflating reported accuracy to 99.6%.

`build_train_generator` therefore takes already-split arrays, and
`build_eval_generator` deliberately offers no augmentation at all.

ORIENTATION NOTE
----------------
Vertical flip is disabled by default. Breast ultrasound has a fixed acquisition
geometry -- the skin line is at the top of the frame and depth increases
downward -- so a vertical flip produces an anatomically impossible image and
teaches the model an invariance that does not exist in the data it will see.
Horizontal flip is safe (left/right breast, probe orientation).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def build_train_generator(config: Dict[str, Any]):
    """Create a Keras ImageDataGenerator from the augmentation config.

    Returns None when augmentation is disabled, which lets the trainer run the
    no-augmentation arm of the ablation without a separate code path.
    """
    aug = config.get("augmentation", {})
    if not aug.get("enabled", False):
        return None

    from keras.preprocessing.image import ImageDataGenerator

    kwargs: Dict[str, Any] = {
        "horizontal_flip": bool(aug.get("horizontal_flip", False)),
        "vertical_flip": bool(aug.get("vertical_flip", False)),
        "rotation_range": float(aug.get("rotation_range", 0)),
        "zoom_range": float(aug.get("zoom_range", 0.0)),
        "width_shift_range": float(aug.get("width_shift_range", 0.0)),
        "height_shift_range": float(aug.get("height_shift_range", 0.0)),
        "fill_mode": aug.get("fill_mode", "nearest"),
    }
    brightness = aug.get("brightness_range")
    if brightness:
        kwargs["brightness_range"] = tuple(brightness)
    return ImageDataGenerator(**kwargs)


def describe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable summary of the active augmentation, recorded in metadata."""
    aug = config.get("augmentation", {})
    if not aug.get("enabled", False):
        return {"enabled": False, "transforms": []}
    transforms = []
    if aug.get("horizontal_flip"):
        transforms.append("horizontal_flip")
    if aug.get("vertical_flip"):
        transforms.append("vertical_flip")
    if aug.get("rotation_range"):
        transforms.append(f"rotation +/-{aug['rotation_range']} deg")
    if aug.get("zoom_range"):
        transforms.append(f"zoom +/-{aug['zoom_range']}")
    if aug.get("width_shift_range"):
        transforms.append(f"width_shift +/-{aug['width_shift_range']}")
    if aug.get("height_shift_range"):
        transforms.append(f"height_shift +/-{aug['height_shift_range']}")
    if aug.get("brightness_range"):
        transforms.append(f"brightness {aug['brightness_range']}")
    return {"enabled": True, "transforms": transforms}


def augment_examples(
    images: np.ndarray, config: Dict[str, Any], n: int = 5, seed: int = 42
) -> Optional[np.ndarray]:
    """Generate example augmented variants of one image, for the dataset report.

    Returns None if augmentation is disabled. Used by the dataset-analysis
    figure so the thesis can SHOW the augmentation rather than assert it.
    """
    generator = build_train_generator(config)
    if generator is None:
        return None
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    iterator = generator.flow(images, batch_size=1, seed=seed)
    return np.concatenate([next(iterator) for _ in range(n)], axis=0)
