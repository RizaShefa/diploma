"""Prediction history (SQLite, stdlib only).

PRIVACY DESIGN -- READ BEFORE CHANGING
--------------------------------------
This stores prediction METADATA, not medical images. `store_images` defaults to
FALSE, and when enabled it keeps only a small JPEG thumbnail, never the original
upload.

The reasoning: the original application saved every upload permanently to
`pred/` under a user-influenced filename, with no retention limit and no way to
delete. For a system that accepts medical images that is a liability the project
does not need -- and the thesis claims GDPR/HIPAA compliance (Ch. 11.6) that the
code has never implemented. Keeping images out of storage by default means the
prototype can honestly say it does not retain patient data.

There is no authentication, so history is a single shared local log. It is
suitable for a local research prototype and a defense demo; it is NOT a
multi-user clinical record, and the README says so.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc         TEXT    NOT NULL,
    model_name          TEXT    NOT NULL,
    model_legacy        INTEGER NOT NULL DEFAULT 0,
    predicted_label     TEXT    NOT NULL,
    probability_positive REAL   NOT NULL,
    confidence          REAL    NOT NULL,
    confidence_band     TEXT    NOT NULL,
    threshold           REAL    NOT NULL,
    original_filename   TEXT,
    image_width         INTEGER,
    image_height        INTEGER,
    inference_ms        REAL,
    had_explanation     INTEGER NOT NULL DEFAULT 0,
    thumbnail_jpeg      BLOB,
    extra_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_utc DESC);
"""


class PredictionHistory:
    """Append-only local log of predictions."""

    def __init__(self, db_path: str | Path, store_images: bool = False,
                 thumbnail_size: int = 128, max_rows: int = 500):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_images = bool(store_images)
        self.thumbnail_size = int(thumbnail_size)
        self.max_rows = int(max_rows)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _thumbnail(self, image_bgr) -> Optional[bytes]:
        if not self.store_images or image_bgr is None:
            return None
        import cv2

        height, width = image_bgr.shape[:2]
        scale = self.thumbnail_size / max(height, width)
        if scale < 1.0:
            image_bgr = cv2.resize(
                image_bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buffer.tobytes() if ok else None

    def record(self, result: Dict[str, Any], *, original_filename: Optional[str] = None,
               image_width: Optional[int] = None, image_height: Optional[int] = None,
               image_bgr=None) -> int:
        """Append one prediction. Returns the new row id."""
        prediction = result["prediction"]
        model = result["model"]
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model.get("name", "unknown"),
            1 if model.get("legacy") else 0,
            prediction["label"],
            float(prediction["probability_positive"]),
            float(prediction["confidence"]),
            prediction["confidence_band"],
            float(prediction["threshold"]),
            original_filename,
            image_width,
            image_height,
            float(result.get("timing", {}).get("inference_ms", 0.0)),
            1 if result.get("explanation") else 0,
            self._thumbnail(image_bgr),
            json.dumps({"probabilities": prediction.get("probabilities", {})}),
        )
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO predictions (created_utc, model_name, model_legacy,"
                " predicted_label, probability_positive, confidence, confidence_band,"
                " threshold, original_filename, image_width, image_height, inference_ms,"
                " had_explanation, thumbnail_jpeg, extra_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            new_id = int(cursor.lastrowid)
            # Bounded retention: drop the oldest rows beyond max_rows.
            conn.execute(
                "DELETE FROM predictions WHERE id NOT IN "
                "(SELECT id FROM predictions ORDER BY id DESC LIMIT ?)",
                (self.max_rows,),
            )
        return new_id

    def list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_utc, model_name, model_legacy, predicted_label,"
                " probability_positive, confidence, confidence_band, threshold,"
                " original_filename, image_width, image_height, inference_ms,"
                " had_explanation FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        return [dict(row) for row in rows]

    def thumbnail(self, row_id: int) -> Optional[bytes]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT thumbnail_jpeg FROM predictions WHERE id = ?", (int(row_id),)
            ).fetchone()
        return bytes(row["thumbnail_jpeg"]) if row and row["thumbnail_jpeg"] else None

    def summary(self) -> Dict[str, Any]:
        """Aggregate counts for the history view.

        NOTE: these describe what the MODEL OUTPUT on user-supplied images. They
        are not accuracy figures -- uploaded images have no ground-truth label,
        so no correctness can be derived from this table.
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            by_label = conn.execute(
                "SELECT predicted_label, COUNT(*) AS n FROM predictions GROUP BY predicted_label"
            ).fetchall()
            by_band = conn.execute(
                "SELECT confidence_band, COUNT(*) AS n FROM predictions GROUP BY confidence_band"
            ).fetchall()
            latency = conn.execute(
                "SELECT AVG(inference_ms) AS avg_ms FROM predictions"
            ).fetchone()["avg_ms"]
        return {
            "total": int(total),
            "by_predicted_label": {r["predicted_label"]: r["n"] for r in by_label},
            "by_confidence_band": {r["confidence_band"]: r["n"] for r in by_band},
            "mean_inference_ms": round(float(latency), 2) if latency else None,
            "note": (
                "Counts of model OUTPUTS on user-supplied images. Uploaded images "
                "carry no ground truth, so no accuracy can be computed from this log."
            ),
        }

    def clear(self) -> int:
        """Delete all history. Returns the number of rows removed."""
        with self._lock, self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            conn.execute("DELETE FROM predictions")
        return int(count)
