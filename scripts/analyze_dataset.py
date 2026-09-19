"""Dataset analysis: distribution, integrity audit, shortcut check, split report.

Produces `results/dataset_report.json` plus figures in `results/figures/`, which
the /dataset page renders. Run before training -- the split it writes is the one
training and evaluation both consume, so every model sees identical data.

    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --legacy-dataset   # audit the OLD dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import PROJECT_ROOT, load_config, set_global_seeds  # noqa: E402
from src.data import analysis, integrity  # noqa: E402
from src.data.loader import Sample, class_distribution, discover_samples  # noqa: E402
from src.data.splitting import split_report, train_val_test_split  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"


def audit_legacy_dataset(threshold: float) -> dict:
    """Audit the original pre-augmented dataset, if it is still present.

    This is the evidence behind the leakage claim in the thesis. It runs on
    whatever class folders remain under `dataset/`.
    """
    legacy_root = PROJECT_ROOT / "dataset"
    if not legacy_root.exists():
        return {"available": False, "reason": "dataset/ not present."}

    samples = []
    for class_dir in sorted(p for p in legacy_root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.glob("*.png")):
            samples.append(Sample(path=str(path), class_name=class_dir.name, label=0,
                                  stem=path.stem, mask_paths=[], group_id=str(path)))
    if not samples:
        return {"available": False, "reason": "No images under dataset/."}

    print(f"  auditing {len(samples)} legacy images (this takes a moment)...")
    report = integrity.audit(samples, threshold=threshold, assign_groups=False)

    # Reproduce the original naive split to measure the leakage it produced.
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=0)
    report["naive_split_leakage"] = integrity.measure_split_leakage(
        samples, train_idx, test_idx
    )
    report["note"] = (
        "Leakage under the ORIGINAL splitting strategy "
        "(train_test_split(test_size=0.2, random_state=0), no grouping)."
    )
    return {"available": True, **report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config name (default: base)")
    parser.add_argument("--legacy-dataset", action="store_true",
                        help="Also audit the original pre-augmented dataset/")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seeds(int(config["seed"]))
    data_cfg = config["data"]
    root = PROJECT_ROOT / data_cfg["root"]

    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "classes": list(data_cfg["classes"]),
        "figures": {},
    }

    print(f"Loading dataset from {root} ...")
    try:
        samples = discover_samples(root, data_cfg["classes"], read_dimensions=True)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if args.legacy_dataset:
            print("\nRunning the legacy audit only.\n")
            report["legacy_dataset_audit"] = audit_legacy_dataset(
                float(data_cfg["near_duplicate_threshold"])
            )
            report["main_dataset"] = {"available": False, "reason": str(exc)}
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "dataset_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print(f"Partial report written to {RESULTS / 'dataset_report.json'}")
            return 0
        return 1

    print(f"  {len(samples)} images: {class_distribution(samples)}")
    report["summary"] = analysis.summarize(samples)

    print("Checking for class-correlated artifacts ...")
    report["shortcut_check"] = analysis.class_correlated_artifacts(samples)
    if report["shortcut_check"].get("any_concern"):
        print("  WARNING: a trivial image property separates the classes (|d| >= 0.5).")
        print("           Reported accuracy may reflect an acquisition shortcut.")
    else:
        print("  No strong class-correlated artifact detected.")

    print("Running integrity audit (duplicates / near-duplicates) ...")
    report["integrity"] = integrity.audit(
        samples, threshold=float(data_cfg["near_duplicate_threshold"])
    )
    print(f"  {report['integrity']['n_clusters']} distinct groups from "
          f"{len(samples)} images (reduction {report['integrity']['reduction_factor']:.2f}x)")

    print("Building group-aware split ...")
    split = train_val_test_split(
        samples,
        train=data_cfg["split"]["train"], val=data_cfg["split"]["val"],
        test=data_cfg["split"]["test"], seed=int(config["seed"]),
    )
    report["split"] = split_report(samples, split)
    for name, info in report["split"].items():
        print(f"  {name:<6} n={info['n']:<5} groups={info['n_groups']:<5} {info['per_class']}")

    print("Measuring residual leakage in the split ...")
    report["split_leakage"] = integrity.measure_split_leakage(
        samples, split["train"], split["test"]
    )
    print(f"  {report['split_leakage']}")
    report["split_leakage_note"] = (
        "Fraction of TEST images with a near-duplicate in TRAIN. Values at or "
        "below the clustering threshold should be ~0. Residual values at looser "
        "thresholds are expected and are reported rather than hidden."
    )

    split_path = RESULTS / "split.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "seed": int(config["seed"]),
        "classes": list(data_cfg["classes"]),
        "near_duplicate_threshold": data_cfg["near_duplicate_threshold"],
        "indices": split,
        "paths": {k: [samples[i].path for i in v] for k, v in split.items()},
    }, indent=2), encoding="utf-8")
    print(f"  split saved to {split_path}")

    if not args.skip_figures:
        print("Generating figures ...")
        report["figures"]["class_distribution"] = Path(
            analysis.class_distribution_figure(samples, FIGURES / "dataset_class_distribution.png")).name
        report["figures"]["dimensions"] = Path(
            analysis.dimension_scatter_figure(samples, FIGURES / "dataset_dimensions.png")).name
        report["figures"]["intensity"] = Path(
            analysis.intensity_histogram_figure(samples, FIGURES / "dataset_intensity.png")).name
        report["figures"]["samples"] = Path(
            analysis.sample_grid_figure(samples, FIGURES / "dataset_samples.png")).name
        report["figures"]["split"] = Path(
            analysis.split_composition_figure(report["split"], FIGURES / "dataset_split.png")).name
        augmentation_figure = analysis.augmentation_examples_figure(
            samples[0], config, FIGURES / "dataset_augmentation.png")
        if augmentation_figure:
            report["figures"]["augmentation"] = Path(augmentation_figure).name
        for key, name in report["figures"].items():
            print(f"  {key:<20} -> results/figures/{name}")

    if args.legacy_dataset:
        print("Auditing the legacy pre-augmented dataset ...")
        report["legacy_dataset_audit"] = audit_legacy_dataset(
            float(data_cfg["near_duplicate_threshold"]))

    out = RESULTS / "dataset_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    print("Next: python scripts/train.py --config custom_cnn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
