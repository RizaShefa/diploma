"""Training orchestration: config in, trained model + metadata out.

PROTOCOL CORRECTIONS vs. the original MainTrain.py
--------------------------------------------------
    original                              now
    ------------------------------------  ------------------------------------
    validation_data = test set            separate val split; test untouched
    shuffle=False (class-ordered batches) shuffled every epoch
    epochs=10, fixed                      early stopping on val_loss
    no class weighting                    balanced weights (BUSI is 437/210)
    no callbacks                          EarlyStopping + ReduceLROnPlateau
    no seeding                            seeded from config
    nothing persisted but the .h5         model + metadata.json + history.json

The test split is loaded here but never passed to `fit`. It is evaluated once,
by `scripts/evaluate.py`, after training has finished.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..augmentation import build_train_generator, describe as describe_augmentation
from ..config import set_global_seeds
from ..data.loader import Sample
from ..models.registry import build_model, find_last_conv_layer, get_spec
from ..preprocessing import preprocess_batch_paths


@dataclass
class TrainingArtifacts:
    """Everything a training run produces."""

    model: Any
    history: Dict[str, List[float]]
    metadata: Dict[str, Any]
    model_path: Path
    metadata_path: Path


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def compute_class_weights(labels: np.ndarray, mode: Any) -> Optional[Dict[int, float]]:
    """Balanced class weights, or None when disabled.

    BUSI is imbalanced (437 benign vs 210 malignant). Without weighting, the
    loss is dominated by the benign class and the model under-predicts
    malignancy -- the clinically costly direction.
    """
    if not mode or mode in ("none", False):
        return None
    if isinstance(mode, dict):
        return {int(k): float(v) for k, v in mode.items()}
    if mode != "balanced":
        raise KeyError(f"Unsupported class_weight {mode!r}")
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}


def build_callbacks(config: Dict[str, Any]) -> List[Any]:
    from keras import callbacks as kc

    training = config["training"]
    out: List[Any] = []
    early = training.get("early_stopping", {})
    if early.get("enabled", False):
        out.append(
            kc.EarlyStopping(
                monitor=early.get("monitor", "val_loss"),
                patience=int(early.get("patience", 10)),
                restore_best_weights=bool(early.get("restore_best_weights", True)),
                verbose=1,
            )
        )
    reduce = training.get("reduce_lr", {})
    if reduce.get("enabled", False):
        out.append(
            kc.ReduceLROnPlateau(
                monitor=reduce.get("monitor", "val_loss"),
                factor=float(reduce.get("factor", 0.5)),
                patience=int(reduce.get("patience", 5)),
                min_lr=float(reduce.get("min_lr", 1e-7)),
                verbose=1,
            )
        )
    return out


def load_split_arrays(
    samples: Sequence[Sample], indices: Sequence[int], profile: str
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess one split into (X, y) using the shared preprocessing module."""
    paths = [samples[i].path for i in indices]
    labels = np.array([samples[i].label for i in indices], dtype=np.int64)
    features = preprocess_batch_paths(paths, profile)
    return features, labels


def train(
    config: Dict[str, Any],
    samples: Sequence[Sample],
    split: Dict[str, List[int]],
    *,
    output_dir: Path,
    class_names: Sequence[str],
    verbose: int = 1,
) -> TrainingArtifacts:
    """Train one model under the protocol described in this module's docstring."""
    set_global_seeds(int(config["seed"]))
    profile = config["preprocessing"]["profile"]
    n_classes = len(class_names)

    x_train, y_train = load_split_arrays(samples, split["train"], profile)
    x_val, y_val = load_split_arrays(samples, split["val"], profile)

    model = build_model(config, n_classes)
    class_weight = compute_class_weights(y_train, config["training"].get("class_weight"))

    y_train_oh = _one_hot(y_train, n_classes)
    y_val_oh = _one_hot(y_val, n_classes)

    generator = build_train_generator(config)
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    callbacks = build_callbacks(config)

    started = time.time()
    if generator is not None:
        flow = generator.flow(
            x_train, y_train_oh, batch_size=batch_size, seed=int(config["seed"])
        )
        history = model.fit(
            flow,
            steps_per_epoch=max(1, len(x_train) // batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val_oh),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        history = model.fit(
            x_train,
            y_train_oh,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val_oh),
            class_weight=class_weight,
            callbacks=callbacks,
            shuffle=True,  # the original used shuffle=False on class-ordered data
            verbose=verbose,
        )
    train_seconds = time.time() - started

    model_name = config["model"]["name"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.keras"
    model.save(model_path)

    metadata = {
        "model_name": model_name,
        "architecture": config["model"]["architecture"],
        "description": get_spec(config["model"]["architecture"])["description"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_names": list(class_names),
        "preprocessing_profile": profile,
        "gradcam_layer": find_last_conv_layer(model, config["model"]["architecture"]),
        "n_parameters": int(model.count_params()),
        "epochs_requested": epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "train_seconds": round(train_seconds, 2),
        "class_weight": class_weight,
        "augmentation": describe_augmentation(config),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": _tf_version(),
        },
    }
    metadata_path = output_dir / f"{model_name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (output_dir / f"{model_name}.history.json").write_text(
        json.dumps(history_dict, indent=2), encoding="utf-8"
    )

    return TrainingArtifacts(model, history_dict, metadata, model_path, metadata_path)


def _tf_version() -> str:
    try:
        import tensorflow as tf

        return tf.__version__
    except ImportError:  # pragma: no cover
        return "unavailable"
