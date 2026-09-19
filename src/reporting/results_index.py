"""Loads saved experiment output for the research dashboard.

HONESTY RULE
------------
These loaders never fabricate. When an experiment has not been run, the returned
structure carries `available: False` and the command that would produce it, and
the template renders "Experiment not yet executed". There are no placeholder
numbers, no example charts and no defaults that could be mistaken for results.

Expected layout, all written by `scripts/`:

    results/
      index.json                     manifest of completed runs
      dataset_report.json            dataset analysis + integrity audit
      evaluation/<model>.json        per-model test metrics
      comparison.json                cross-model comparison table
      error_analysis/<model>.json    error analysis
      figures/*.png                  generated figures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _unavailable(what: str, command: str) -> Dict[str, Any]:
    return {
        "available": False,
        "message": f"{what} not yet executed.",
        "how_to_run": command,
    }


def load_dataset_report(results_dir: Path) -> Dict[str, Any]:
    data = _read_json(Path(results_dir) / "dataset_report.json")
    if data is None:
        return _unavailable("Dataset analysis", "python scripts/analyze_dataset.py")
    return {"available": True, **data}


def load_evaluation(results_dir: Path, model_name: str) -> Dict[str, Any]:
    data = _read_json(Path(results_dir) / "evaluation" / f"{model_name}.json")
    if data is None:
        return _unavailable(
            f"Evaluation for '{model_name}'",
            f"python scripts/evaluate.py --model {model_name}",
        )
    return {"available": True, **data}


def load_comparison(results_dir: Path) -> Dict[str, Any]:
    data = _read_json(Path(results_dir) / "comparison.json")
    if data is None:
        return _unavailable("Model comparison", "python scripts/compare_models.py")
    return {"available": True, **data}


def load_error_analysis(results_dir: Path, model_name: str) -> Dict[str, Any]:
    data = _read_json(Path(results_dir) / "error_analysis" / f"{model_name}.json")
    if data is None:
        return _unavailable(
            f"Error analysis for '{model_name}'",
            f"python scripts/evaluate.py --model {model_name}",
        )
    return {"available": True, **data}


def list_figures(results_dir: Path) -> List[str]:
    figures_dir = Path(results_dir) / "figures"
    if not figures_dir.exists():
        return []
    return sorted(p.name for p in figures_dir.glob("*.png"))


def load_results_index(results_dir: Path) -> Dict[str, Any]:
    """Everything the research dashboard needs, in one call."""
    results_dir = Path(results_dir)
    index = _read_json(results_dir / "index.json") or {}
    evaluated: List[str] = index.get("evaluated_models", [])

    evaluations = {name: load_evaluation(results_dir, name) for name in evaluated}
    errors = {name: load_error_analysis(results_dir, name) for name in evaluated}

    return {
        "any_results": bool(evaluated),
        "generated_utc": index.get("generated_utc"),
        "evaluated_models": evaluated,
        "dataset": load_dataset_report(results_dir),
        "comparison": load_comparison(results_dir),
        "evaluations": evaluations,
        "error_analysis": errors,
        "figures": list_figures(results_dir),
        "next_steps": [
            "python scripts/prepare_data.py --source <path-to-BUSI>",
            "python scripts/analyze_dataset.py",
            "python scripts/train.py --config custom_cnn",
            "python scripts/evaluate.py --model custom_cnn",
            "python scripts/compare_models.py",
        ],
    }


def write_index(results_dir: Path, evaluated_models: List[str]) -> Path:
    """Refresh results/index.json after an evaluation run."""
    from datetime import datetime, timezone

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "index.json"
    existing = _read_json(path) or {}
    known = set(existing.get("evaluated_models", [])) | set(evaluated_models)
    path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "evaluated_models": sorted(known),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
