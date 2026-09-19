"""Evaluate a trained model on the held-out test split.

    python scripts/evaluate.py --model custom_cnn
    python scripts/evaluate.py --model custom_cnn --bootstrap 5000

Produces, all from real predictions:
    results/evaluation/<model>.json      metrics + bootstrap CIs + calibration
    results/error_analysis/<model>.json  error analysis
    results/figures/<model>_*.png        confusion matrix, ROC, PR, calibration,
                                         threshold sweep
and writes the selected operating threshold back into the model's metadata so
the Flask app serves predictions at that threshold.

This is the ONLY script that touches the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import PROJECT_ROOT, load_config, set_global_seeds  # noqa: E402
from src.data.loader import discover_samples  # noqa: E402
from src.evaluation import errors as error_analysis  # noqa: E402
from src.evaluation import metrics as metric_lib  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.preprocessing import preprocess_batch_paths  # noqa: E402
from src.reporting.results_index import write_index  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
FIGURES = RESULTS / "figures"


def load_model_and_metadata(model_name: str):
    model_path = MODELS / f"{model_name}.keras"
    metadata_path = MODELS / f"{model_name}.metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train it first: python scripts/train.py --config {model_name}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    from keras.models import load_model

    return load_model(model_path), metadata, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None, help="Config (default: inferred from metadata)")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    try:
        model, metadata, metadata_path = load_model_and_metadata(args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    config = load_config(args.config or metadata.get("config", {}).get("_config_name"))
    set_global_seeds(int(config["seed"]))
    evaluation_cfg = config["evaluation"]
    bootstrap = args.bootstrap or int(evaluation_cfg["bootstrap_iterations"])

    data_cfg = config["data"]
    class_names = metadata.get("class_names", data_cfg["classes"])
    profile = metadata.get("preprocessing_profile", config["preprocessing"]["profile"])

    samples = discover_samples(PROJECT_ROOT / data_cfg["root"], class_names)
    split_payload = json.loads((RESULTS / "split.json").read_text(encoding="utf-8"))
    indices = split_payload["indices"][args.split]

    print(f"Model      : {args.model}")
    print(f"Split      : {args.split} (n={len(indices)})")
    print(f"Preprocess : {profile}\n")

    paths = [samples[i].path for i in indices]
    y_true = np.array([samples[i].label for i in indices], dtype=int)
    features = preprocess_batch_paths(paths, profile)

    started = time.perf_counter()
    probabilities = model.predict(features, verbose=0)
    total_seconds = time.perf_counter() - started

    positive_index = class_names.index("malignant") if "malignant" in class_names else 1
    y_prob = probabilities[:, positive_index].astype(float)
    y_true_binary = (y_true == positive_index).astype(int)

    result = metric_lib.evaluate(
        y_true_binary, y_prob,
        threshold=0.5,
        bootstrap_iterations=bootstrap,
        confidence_level=float(evaluation_cfg["confidence_level"]),
        seed=int(config["seed"]),
        target_sensitivity=float(evaluation_cfg["target_sensitivity"]),
    )
    result.update({
        "model_name": args.model,
        "architecture": metadata.get("architecture"),
        "n_parameters": metadata.get("n_parameters"),
        "split": args.split,
        "class_names": class_names,
        "positive_class": class_names[positive_index],
        "preprocessing_profile": profile,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inference": {
            "total_seconds": round(total_seconds, 3),
            "per_image_ms": round(1000.0 * total_seconds / max(1, len(indices)), 3),
        },
    })

    metrics = result["metrics"]
    intervals = result["confidence_intervals"]
    print("--- Test metrics (95% CI) ---")
    for key in ("sensitivity", "specificity", "precision", "npv", "f1",
                "balanced_accuracy", "accuracy", "roc_auc", "pr_auc"):
        print(f"  {key:<18} {metric_lib.format_metric(metrics.get(key), intervals.get(key))}")
    print(f"  {'ECE':<18} {result['calibration']['ece']:.4f}")
    print(f"  counts             TP={metrics['tp']} TN={metrics['tn']} "
          f"FP={metrics['fp']} FN={metrics['fn']}")
    print(f"  inference          {result['inference']['per_image_ms']:.2f} ms/image (CPU)")

    operating = result.get("operating_point")
    if operating:
        print(f"\n--- Operating point (target sensitivity "
              f"{operating['target_sensitivity']:.2f}) ---")
        status = "met" if operating["target_met"] else "NOT met - best available shown"
        print(f"  threshold={operating['threshold']:.3f}  sens={operating['sensitivity']:.3f}  "
              f"spec={operating['specificity']:.3f}  [{status}]")

    print("\n--- Error analysis ---")
    analysis = error_analysis.analyze(samples, indices, y_true_binary, y_prob, 0.5)
    print(f"  {analysis['counts']}")
    separation = analysis["confidence_separation"]
    if separation.get("available"):
        print(f"  mean confidence: correct={separation['correct_mean_confidence']:.3f} "
              f"incorrect={separation['incorrect_mean_confidence']:.3f} "
              f"(gap {separation['gap']:+.3f})")
    comparison = analysis["property_comparison"]
    if comparison.get("available"):
        ranked = sorted(comparison["properties"].items(),
                        key=lambda kv: -abs(kv[1]["cohens_d"]))[:3]
        print("  strongest property differences (correct vs incorrect):")
        for name, stats in ranked:
            print(f"    {name:<20} d={stats['cohens_d']:+.2f} ({stats['interpretation']})")
    else:
        print(f"  {comparison.get('reason', 'unavailable')}")

    (RESULTS / "evaluation").mkdir(parents=True, exist_ok=True)
    (RESULTS / "error_analysis").mkdir(parents=True, exist_ok=True)
    evaluation_path = RESULTS / "evaluation" / f"{args.model}.json"
    evaluation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Keep the stored rows light: drop per-image properties from the saved JSON.
    slim = dict(analysis)
    slim["rows"] = [{k: v for k, v in row.items() if k != "properties"}
                    for row in analysis["rows"]]
    slim["model_name"] = args.model
    (RESULTS / "error_analysis" / f"{args.model}.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8")

    figures = {}
    if not args.skip_figures:
        FIGURES.mkdir(parents=True, exist_ok=True)
        figures["confusion_matrix"] = Path(plots.confusion_matrix_figure(
            metrics, class_names, FIGURES / f"{args.model}_confusion.png",
            title=f"Confusion matrix -- {args.model} ({args.split})")).name
        if "roc" in result:
            figures["roc"] = Path(plots.roc_figure(
                result["roc"], FIGURES / f"{args.model}_roc.png",
                title=f"ROC -- {args.model}", operating_point=operating)).name
            positive_rate = float(y_true_binary.mean())
            figures["pr"] = Path(plots.pr_figure(
                result["pr"], FIGURES / f"{args.model}_pr.png",
                positive_rate=positive_rate, title=f"Precision-recall -- {args.model}")).name
        figures["calibration"] = Path(plots.calibration_figure(
            result["calibration"], FIGURES / f"{args.model}_calibration.png",
            title=f"Reliability -- {args.model}")).name
        figures["threshold"] = Path(plots.threshold_figure(
            metric_lib.threshold_sweep(y_true_binary, y_prob),
            FIGURES / f"{args.model}_threshold.png",
            target_sensitivity=float(evaluation_cfg["target_sensitivity"]),
            title=f"Threshold trade-off -- {args.model}")).name
        result["figures"] = figures
        evaluation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\n--- Figures ---")
        for key, name in figures.items():
            print(f"  {key:<18} results/figures/{name}")

    # Record the validated performance and operating threshold on the model, so
    # the served app reports real numbers instead of "not yet evaluated".
    if args.split == "test" and metadata_path.exists():
        metadata["test_metrics"] = {
            "sensitivity": metric_lib.format_metric(metrics["sensitivity"],
                                                    intervals.get("sensitivity")),
            "specificity": metric_lib.format_metric(metrics["specificity"],
                                                    intervals.get("specificity")),
            "roc_auc": metric_lib.format_metric(metrics.get("roc_auc"),
                                                intervals.get("roc_auc")),
            "n_test": int(len(indices)),
            "evaluated_utc": result["evaluated_utc"],
        }
        if operating and operating["target_met"]:
            metadata["operating_threshold"] = float(operating["threshold"])
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\n  metadata updated: {metadata_path.name}")

    existing = []
    index_path = RESULTS / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8")).get("evaluated_models", [])
    write_index(RESULTS, sorted(set(existing) | {args.model}))

    print(f"\nSaved: {evaluation_path}")
    print("Next: python scripts/compare_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
