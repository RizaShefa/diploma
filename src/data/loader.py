"""Dataset discovery and manifest building for BUSI.

Expected layout after `scripts/prepare_data.py` (BUSI = Al-Dhabyani et al. 2020):

    data/busi/
      benign/     benign (1).png, benign (1)_mask.png, ...
      malignant/  malignant (1).png, malignant (1)_mask.png, ...
      normal/     normal (1).png, ...

Mask files (`*_mask*.png`) are catalogued alongside their image but never used as
training inputs. They are retained because BUSI's masks give ground-truth lesion
locations, which can later be used to check whether Grad-CAM attends to the lesion.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_PATTERN = re.compile(r"_mask(_\d+)?$", re.IGNORECASE)


@dataclass
class Sample:
    """One dataset item. `group_id` is what splitting must not break apart."""

    path: str
    class_name: str
    label: int
    stem: str
    mask_paths: List[str]
    group_id: str  # reassigned by integrity clustering; defaults to the stem
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_mask(path: Path) -> bool:
    return bool(MASK_PATTERN.search(path.stem))


def _read_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read image dimensions without decoding full pixel data where possible."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def discover_samples(
    root: str | Path,
    classes: Sequence[str],
    *,
    read_dimensions: bool = False,
) -> List[Sample]:
    """Walk `root/<class>/` and build the sample manifest.

    Raises FileNotFoundError with an actionable message if a class directory is
    missing. This is exactly the failure the original MainTrain.py hit when
    `dataset/malignant/` was absent, except it now names the fix.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Run `python scripts/prepare_data.py --source <path-to-BUSI>` first. "
            "See README.md > Dataset setup."
        )

    samples: List[Sample] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory missing: {class_dir}\n"
                f"Expected one directory per class in {list(classes)}."
            )

        masks: Dict[str, List[str]] = {}
        images: List[Path] = []
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if _is_mask(path):
                base = MASK_PATTERN.sub("", path.stem)
                masks.setdefault(base, []).append(str(path))
            else:
                images.append(path)

        for path in images:
            sample = Sample(
                path=str(path),
                class_name=class_name,
                label=label,
                stem=path.stem,
                mask_paths=sorted(masks.get(path.stem, [])),
                group_id=f"{class_name}/{path.stem}",
            )
            if read_dimensions:
                sample.width, sample.height = _read_dimensions(path)
            samples.append(sample)

    if not samples:
        raise FileNotFoundError(f"No images found under {root} for classes {list(classes)}.")
    return samples


def class_distribution(samples: Sequence[Sample]) -> Dict[str, int]:
    """Count samples per class name."""
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.class_name] = counts.get(sample.class_name, 0) + 1
    return counts


def labels_array(samples: Sequence[Sample]):
    """Integer label vector for the manifest (used by splitting/evaluation)."""
    import numpy as np

    return np.array([sample.label for sample in samples], dtype=np.int64)
