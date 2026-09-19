"""Ingest the BUSI dataset into the layout this project expects.

BUSI (Breast Ultrasound Images), Al-Dhabyani et al. 2020, Data in Brief 28:104863
    780 images: 437 benign, 210 malignant, 133 normal, with lesion masks.

Obtain it from Kaggle ("Breast Ultrasound Images Dataset") or the authors' page,
then point this script at the extracted folder:

    python scripts/prepare_data.py --source "C:/Downloads/Dataset_BUSI_with_GT"

WHY BUSI RATHER THAN THE PREVIOUS DATASET
-----------------------------------------
The 5,000-image set previously used here was a pre-augmented derivative: 82% of
its images had a near-duplicate, collapsing to roughly 891 distinct sources.
Because the augmentation predated the split, 77.3% of test images had a
near-duplicate in training. BUSI is the original, un-augmented collection the
thesis already cites, so it removes the leakage at its root and makes the
citation accurate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402

EXPECTED_CLASSES = ("benign", "malignant", "normal")
BUSI_REFERENCE = {
    "name": "BUSI (Breast Ultrasound Images Dataset)",
    "citation": (
        "Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound "
        "images. Data in Brief. 2020 Feb;28:104863."
    ),
    "doi": "10.1016/j.dib.2019.104863",
    "expected_counts": {"benign": 437, "malignant": 210, "normal": 133},
}


def find_class_dirs(source: Path) -> dict:
    """Locate the class directories, tolerating one level of nesting."""
    found = {}
    for class_name in EXPECTED_CLASSES:
        candidate = source / class_name
        if candidate.is_dir():
            found[class_name] = candidate
            continue
        matches = [p for p in source.rglob(class_name) if p.is_dir()]
        if matches:
            found[class_name] = sorted(matches, key=lambda p: len(p.parts))[0]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Extracted BUSI folder")
    parser.add_argument("--dest", default=None, help="Destination (default: config data.root)")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="Copy files (default). Originals are never modified.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing destination")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"ERROR: source folder not found: {source}", file=sys.stderr)
        return 1

    config = load_config()
    dest = Path(args.dest) if args.dest else PROJECT_ROOT / config["data"]["root"]
    dest = dest.resolve()

    class_dirs = find_class_dirs(source)
    if not class_dirs:
        print(f"ERROR: no BUSI class folders ({', '.join(EXPECTED_CLASSES)}) under {source}",
              file=sys.stderr)
        return 1

    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"ERROR: destination {dest} already exists and is not empty. Use --force.",
              file=sys.stderr)
        return 1

    print(f"Source     : {source}")
    print(f"Destination: {dest}\n")

    manifest = {"source": str(source), "reference": BUSI_REFERENCE, "classes": {}}
    for class_name, class_dir in sorted(class_dirs.items()):
        target = dest / class_name
        target.mkdir(parents=True, exist_ok=True)
        images = masks = 0
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            shutil.copy2(path, target / path.name)
            if "_mask" in path.stem.lower():
                masks += 1
            else:
                images += 1
        expected = BUSI_REFERENCE["expected_counts"].get(class_name)
        flag = ""
        if expected is not None and images != expected:
            flag = f"  <-- expected {expected}; verify the download"
        print(f"  {class_name:<10} {images:>4} images, {masks:>4} masks{flag}")
        manifest["classes"][class_name] = {"images": images, "masks": masks,
                                           "expected": expected}

    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(c["images"] for c in manifest["classes"].values())
    print(f"\nTotal: {total} images. Manifest written to {manifest_path}")
    print("\nNext: python scripts/analyze_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
