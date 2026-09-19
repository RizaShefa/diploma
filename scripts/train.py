"""Train one model under the corrected protocol.

    python scripts/train.py --config custom_cnn
    python scripts/train.py --config resnet50 --epochs 30
    python scripts/train.py --config custom_cnn --no-augmentation   # ablation arm

Every model reads the SAME split from `results/split.json`, written by
`analyze_dataset.py`. That is what makes the comparison fair: differences in
results come from the architecture, not from a different shuffle.

The test split is loaded only to record its size. It is never passed to `fit`
and never used for model selection -- early stopping monitors the VALIDATION
split. Test metrics come from `scripts/evaluate.py`, run afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.data.loader import discover_samples  # noqa: E402
from src.training.trainer import train  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"


def load_split(samples) -> dict:
    """Load the shared split, verifying it matches the current dataset."""
    split_path = RESULTS / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found. Run `python scripts/analyze_dataset.py` first -- "
            "it creates the split that all models share."
        )
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    indices = payload["indices"]

    # Guard against a stale split: verify the recorded paths still line up.
    for name, paths in payload.get("paths", {}).items():
        for path, index in zip(paths, indices[name]):
            if samples[index].path != path:
                raise ValueError(
                    f"results/split.json is stale (mismatch in '{name}' split). "
                    "Re-run `python scripts/analyze_dataset.py`."
                )
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Config name, e.g. custom_cnn")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable augmentation (for the augmentation ablation)")
    parser.add_argument("--name", default=None, help="Override the output model name")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_augmentation:
        config["augmentation"]["enabled"] = False
        config["model"]["name"] = args.name or f"{config['model']['name']}_noaug"
    if args.name:
        config["model"]["name"] = args.name

    data_cfg = config["data"]
    root = PROJECT_ROOT / data_cfg["root"]
    try:
        samples = discover_samples(root, data_cfg["classes"])
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        split = load_split(samples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Model        : {config['model']['name']} ({config['model']['architecture']})")
    print(f"Preprocessing: {config['preprocessing']['profile']}")
    print(f"Augmentation : {config['augmentation']['enabled']}")
    print(f"Split        : train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} (test held back)")
    print(f"Seed         : {config['seed']}\n")

    artifacts = train(
        config, samples, split,
        output_dir=MODELS,
        class_names=data_cfg["classes"],
        verbose=0 if args.quiet else 1,
    )

    metadata = artifacts.metadata
    print(f"\nTrained in {metadata['train_seconds']}s "
          f"({metadata['epochs_run']}/{metadata['epochs_requested']} epochs)")
    print(f"  model    : {artifacts.model_path}")
    print(f"  metadata : {artifacts.metadata_path}")

    history = artifacts.history
    if history.get("val_loss"):
        best = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
        print(f"  best epoch {best + 1}: val_loss={history['val_loss'][best]:.4f} "
              f"val_accuracy={history.get('val_accuracy', [float('nan')])[best]:.4f}")
        print("\n  NOTE: these are VALIDATION figures used for model selection. "
              "They are not test results.")

    try:
        from src.evaluation.plots import training_history_figure

        figure = training_history_figure(
            history, RESULTS / "figures" / f"{metadata['model_name']}_history.png",
            title=f"Training history -- {metadata['model_name']}")
        print(f"  history figure: {figure}")
    except Exception as exc:  # noqa: BLE001 - a missing figure must not fail training
        print(f"  (history figure skipped: {exc})")

    print(f"\nNext: python scripts/evaluate.py --model {metadata['model_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
