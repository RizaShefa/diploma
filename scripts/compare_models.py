"""Build the cross-model comparison table and figures.

    python scripts/compare_models.py
    python scripts/compare_models.py --models custom_cnn resnet50 efficientnetb0

Reads only what `scripts/evaluate.py` already wrote. It performs no inference and
invents nothing: models that have not been evaluated are listed as missing, not
filled in.

FAIRNESS
--------
Every model compared here was trained on the identical split from
`results/split.json`, with the same augmentation policy, optimiser, learning
rate, batch size, callbacks and seed -- only the architecture and its required
preprocessing profile differ. The comparison is therefore attributable to the
architecture. `verify_comparable` checks this and prints a warning if any
training setting diverged.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.evaluation.metrics import format_metric  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
FIGURES = RESULTS / "figures"

COMPARISON_METRICS = ("sensitivity", "specificity", "precision", "f1",
                      "balanced_accuracy", "roc_auc", "pr_auc")

# Training settings that must match for the comparison to isolate architecture.
CONTROLLED_KEYS = ("epochs", "batch_size", "learning_rate", "optimizer",
                   "loss", "class_weight")


def discover_evaluated() -> List[str]:
    directory = RESULTS / "evaluation"
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def verify_comparable(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check that the compared models really were trained alike."""
    settings: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        metadata_path = MODELS / f"{row['model_name']}.metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = metadata.get("config", {}).get("training", {})
        settings[row["model_name"]] = {
            **{k: training.get(k) for k in CONTROLLED_KEYS},
            "seed": metadata.get("config", {}).get("seed"),
            "augmentation": json.dumps(metadata.get("augmentation", {}), sort_keys=True),
            "split_sizes": json.dumps(metadata.get("split_sizes", {}), sort_keys=True),
        }
    if len(settings) < 2:
        return {"checked": False, "reason": "Fewer than two models with metadata."}

    keys = set().union(*(set(v) for v in settings.values()))
    divergent = {
        key: {name: values.get(key) for name, values in settings.items()}
        for key in sorted(keys)
        if len({json.dumps(values.get(key), sort_keys=True) for values in settings.values()}) > 1
    }
    return {
        "checked": True,
        "comparable": not divergent,
        "divergent_settings": divergent,
        "note": (
            "All listed settings matched across models; differences in results are "
            "attributable to architecture."
            if not divergent else
            "WARNING: the settings below differ across models. Results are NOT a "
            "clean architecture comparison until these match."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--metric", default="sensitivity",
                        help="Metric for the bar figure (default: sensitivity)")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    requested = args.models or discover_evaluated()
    if not requested:
        print("No evaluated models found.", file=sys.stderr)
        print("Run: python scripts/evaluate.py --model <name>", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in requested:
        path = RESULTS / "evaluation" / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    if missing:
        print(f"Not evaluated (skipped): {', '.join(missing)}")
        print(f"  run: python scripts/evaluate.py --model {missing[0]}\n")
    if not rows:
        print("Nothing to compare.", file=sys.stderr)
        return 1

    fairness = verify_comparable(rows)
    if fairness.get("checked") and not fairness["comparable"]:
        print("WARNING: models were NOT trained under identical settings:")
        for key, values in fairness["divergent_settings"].items():
            print(f"  {key}: {values}")
        print()

    # Console table
    header = f"{'model':<22}{'params':>12}{'ms/img':>9}  " + "".join(
        f"{m[:12]:>24}" for m in COMPARISON_METRICS[:4])
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -(r["metrics"].get("sensitivity") or 0)):
        line = (f"{row['model_name']:<22}"
                f"{(row.get('n_parameters') or 0):>12,}"
                f"{row.get('inference', {}).get('per_image_ms', 0):>9.2f}  ")
        for metric in COMPARISON_METRICS[:4]:
            line += f"{format_metric(row['metrics'].get(metric), row['confidence_intervals'].get(metric)):>24}"
        print(line)

    table = []
    for row in rows:
        entry = {
            "model_name": row["model_name"],
            "architecture": row.get("architecture"),
            "n_parameters": row.get("n_parameters"),
            "per_image_ms": row.get("inference", {}).get("per_image_ms"),
            "n_test": row.get("n_samples"),
            "ece": row.get("calibration", {}).get("ece"),
            "metrics": {m: row["metrics"].get(m) for m in COMPARISON_METRICS},
            "confidence_intervals": {
                m: row["confidence_intervals"].get(m) for m in COMPARISON_METRICS
            },
            "formatted": {
                m: format_metric(row["metrics"].get(m), row["confidence_intervals"].get(m))
                for m in COMPARISON_METRICS
            },
        }
        table.append(entry)

    comparison = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": table,
        "metrics": list(COMPARISON_METRICS),
        "fairness_check": fairness,
        "missing_models": missing,
        "interpretation_note": (
            "Confidence intervals are bootstrap percentile intervals over the test "
            "set. Overlapping intervals mean the difference between two models is "
            "not distinguishable at this sample size -- state that plainly rather "
            "than declaring a winner."
        ),
    }

    figures: Dict[str, str] = {}
    if not args.skip_figures and table:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for metric in (args.metric, "specificity", "roc_auc"):
            if metric not in COMPARISON_METRICS:
                continue
            figures[metric] = Path(plots.model_comparison_figure(
                rows, FIGURES / f"comparison_{metric}.png", metric=metric)).name
        comparison["figures"] = figures
        print("\n--- Figures ---")
        for metric, name in figures.items():
            print(f"  {metric:<18} results/figures/{name}")

    out = RESULTS / "comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")

    # Flag indistinguishable leaders rather than crowning one.
    best = max(table, key=lambda e: e["metrics"].get("sensitivity") or 0)
    overlapping = [
        e["model_name"] for e in table
        if e is not best
        and (e["confidence_intervals"].get("sensitivity") or {}).get("upper", 0)
        >= (best["confidence_intervals"].get("sensitivity") or {}).get("lower", 1)
    ]
    print(f"\nHighest sensitivity: {best['model_name']} "
          f"({best['formatted']['sensitivity']})")
    if overlapping:
        print(f"  CI overlaps with: {', '.join(overlapping)} -- "
              "the difference is not statistically distinguishable here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
