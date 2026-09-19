"""Model loading and prediction, with confidence and optional Grad-CAM.

KEY GUARANTEE
-------------
A model is always paired with the preprocessing profile recorded in its metadata
at training time. Serving therefore cannot drift from training -- the defect
that flipped 6% of predictions in the original app, where training used
BGR->RGB + normalisation and serving used neither.

Models are discovered from `models/`:

    models/<name>.keras           the model
    models/<name>.metadata.json   profile, class names, Grad-CAM layer, metrics

The legacy `BreastCancer10Epochs.h5` at the project root is registered
automatically with a synthesised metadata record so the app keeps working before
anything new has been trained. Its entry is explicitly flagged `legacy: true`
and carries a warning, because it was trained on a leaked split.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import PROJECT_ROOT
from ..explainability.gradcam import ExplainabilityError, explain
from ..preprocessing import display_image_rgb, get_profile, preprocess_array

LEGACY_MODEL_FILE = "BreastCancer10Epochs.h5"
LEGACY_WARNING = (
    "This model was trained on a dataset with substantial train/test overlap "
    "(77.3% of test images had a near-duplicate in training). Its reported "
    "99.6% accuracy is not a valid generalisation estimate. Retained only so "
    "the application runs before a corrected model is trained."
)


class ModelNotAvailableError(RuntimeError):
    """Raised when no usable model can be loaded."""


@dataclass
class ModelEntry:
    """One registered model and its metadata."""

    name: str
    path: Path
    metadata: Dict[str, Any]
    legacy: bool = False
    _model: Any = field(default=None, repr=False)

    @property
    def profile(self) -> str:
        return self.metadata.get("preprocessing_profile", "standard_224")

    @property
    def class_names(self) -> List[str]:
        return list(self.metadata.get("class_names", ["benign", "malignant"]))


class ModelRegistry:
    """Lazily loads and caches models. Thread-safe for Flask's threaded server."""

    def __init__(self, models_dir: Optional[Path] = None, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or PROJECT_ROOT)
        self.models_dir = Path(models_dir or self.project_root / "models")
        self._entries: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self) -> None:
        """Rescan the models directory. Cheap; safe to call on each request."""
        entries: Dict[str, ModelEntry] = {}
        if self.models_dir.exists():
            for model_path in sorted(self.models_dir.glob("*.keras")):
                metadata_path = model_path.with_suffix("").with_suffix(".metadata.json")
                if not metadata_path.exists():
                    metadata_path = self.models_dir / f"{model_path.stem}.metadata.json"
                metadata: Dict[str, Any] = {}
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        metadata = {}
                metadata.setdefault("model_name", model_path.stem)
                entries[model_path.stem] = ModelEntry(model_path.stem, model_path, metadata)

        legacy_path = self.project_root / LEGACY_MODEL_FILE
        if legacy_path.exists():
            entries["legacy_cnn"] = ModelEntry(
                name="legacy_cnn",
                path=legacy_path,
                legacy=True,
                metadata={
                    "model_name": "legacy_cnn",
                    "architecture": "custom_cnn",
                    "description": "Original model shipped with the thesis (10 epochs, 64x64).",
                    "preprocessing_profile": "legacy_64",
                    "class_names": ["benign", "malignant"],
                    "n_parameters": 176290,
                    "legacy": True,
                    "warning": LEGACY_WARNING,
                    "test_metrics": None,
                },
            )

        with self._lock:
            for name, entry in entries.items():
                existing = self._entries.get(name)
                if existing is not None and existing.path == entry.path:
                    entry._model = existing._model  # preserve cache across refreshes
            self._entries = entries

    def names(self) -> List[str]:
        return sorted(self._entries)

    def describe_all(self) -> List[Dict[str, Any]]:
        """Metadata for every registered model (no weights loaded)."""
        out = []
        for name in self.names():
            entry = self._entries[name]
            out.append({
                "name": name,
                "legacy": entry.legacy,
                "preprocessing_profile": entry.profile,
                "class_names": entry.class_names,
                **{k: v for k, v in entry.metadata.items() if k != "config"},
            })
        return out

    def default_name(self) -> str:
        """Prefer a properly trained model; fall back to legacy."""
        non_legacy = [n for n in self.names() if not self._entries[n].legacy]
        if non_legacy:
            return sorted(non_legacy)[0]
        if self._entries:
            return self.names()[0]
        raise ModelNotAvailableError(
            "No models found. Train one with `python scripts/train.py --config custom_cnn`, "
            f"or place {LEGACY_MODEL_FILE} in the project root."
        )

    def get(self, name: Optional[str] = None) -> ModelEntry:
        name = name or self.default_name()
        entry = self._entries.get(name)
        if entry is None:
            raise ModelNotAvailableError(
                f"Unknown model {name!r}. Available: {self.names() or 'none'}."
            )
        if entry._model is None:
            with self._lock:
                if entry._model is None:
                    entry._model = self._load(entry.path)
        return entry

    @staticmethod
    def _load(path: Path):
        from keras.models import load_model

        try:
            return load_model(path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
            raise ModelNotAvailableError(
                f"Failed to load model from {path.name}: {exc}"
            ) from exc


def confidence_band(probability: float) -> str:
    """Coarse verbal band for the UI.

    Deliberately coarse: these are RAW softmax outputs, which are typically
    overconfident. Presenting '87.3% confident' implies a calibration that has
    not been demonstrated. See `metrics.expected_calibration_error`.
    """
    confidence = max(probability, 1.0 - probability)
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "low"


def predict(
    registry: ModelRegistry,
    image_bgr: np.ndarray,
    *,
    model_name: Optional[str] = None,
    with_explanation: bool = True,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run inference and (optionally) Grad-CAM on one image.

    Explainability failure never fails the prediction: the result carries
    `explanation: None` plus `explanation_error`, and the caller renders the
    prediction without a heatmap.
    """
    entry = registry.get(model_name)
    model = entry._model
    profile = entry.profile
    class_names = entry.class_names

    batch = preprocess_array(image_bgr, profile)

    started = time.perf_counter()
    probabilities = model.predict(batch, verbose=0)[0]
    inference_ms = (time.perf_counter() - started) * 1000.0

    positive_index = (
        class_names.index("malignant") if "malignant" in class_names else len(class_names) - 1
    )
    positive_probability = float(probabilities[positive_index])

    decision_threshold = (
        float(threshold)
        if threshold is not None
        else float(entry.metadata.get("operating_threshold", 0.5))
    )
    predicted_index = positive_index if positive_probability >= decision_threshold else (
        1 - positive_index if len(class_names) == 2 else int(np.argmax(probabilities))
    )

    result: Dict[str, Any] = {
        "model": {
            "name": entry.name,
            "architecture": entry.metadata.get("architecture"),
            "legacy": entry.legacy,
            "preprocessing_profile": profile,
            "n_parameters": entry.metadata.get("n_parameters"),
            "warning": entry.metadata.get("warning"),
            "test_metrics": entry.metadata.get("test_metrics"),
        },
        "prediction": {
            "label": class_names[predicted_index],
            "class_index": int(predicted_index),
            "probabilities": {
                name: float(probabilities[i]) for i, name in enumerate(class_names)
            },
            "positive_class": class_names[positive_index],
            "probability_positive": positive_probability,
            "threshold": decision_threshold,
            "confidence": float(max(probabilities)),
            "confidence_band": confidence_band(positive_probability),
            "calibrated": False,
        },
        "timing": {"inference_ms": round(inference_ms, 2)},
        "explanation": None,
        "explanation_error": None,
    }

    if with_explanation:
        try:
            display = display_image_rgb(image_bgr, get_profile(profile)["image_size"])
            explanation = explain(
                model, batch, display,
                class_index=int(predicted_index),
                layer_name=entry.metadata.get("gradcam_layer"),
                architecture=entry.metadata.get("architecture"),
            )
            result["explanation"] = {
                "method": explanation["method"],
                "reference": explanation["reference"],
                "layer": explanation["layer"],
                "feature_map_size": explanation["feature_map_size"],
                "is_degenerate": explanation["is_degenerate"],
                "disclaimer": explanation["disclaimer"],
                "overlay_rgb": explanation["overlay_rgb"],
                "heatmap": explanation["heatmap"],
            }
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            # Covers ExplainabilityError plus any backend failure. A missing
            # heatmap must never cost the user their prediction.
            result["explanation_error"] = str(exc)

    return result
