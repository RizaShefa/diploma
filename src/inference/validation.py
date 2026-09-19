"""Upload validation.

ORIGINAL BEHAVIOUR
------------------
`app.py` accepted `request.files['file']` and saved it straight to disk. The only
restriction was the HTML `accept=".png,.jpg,.jpeg"` attribute, which is a client
-side hint and trivially bypassed by any direct POST. There was no size limit, no
content-type check, no verification that the bytes were an image at all, and
uploads were persisted permanently under user-influenced names.

WHAT IS CHECKED HERE
--------------------
1. A filename is present.
2. The extension is allow-listed.
3. The payload is non-empty and within the size cap.
4. The MAGIC BYTES match a supported image format -- extensions lie.
5. OpenCV can actually decode the bytes.
6. Dimensions are sane (not a decompression bomb, not 1x1).

This is defence in depth for a research prototype. It is NOT a claim of
regulatory compliance: the app has no authentication, no transport encryption
and no audit trail, and the README says so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024      # 10 MB
MIN_DIMENSION = 16
MAX_DIMENSION = 10000                    # guards against decompression bombs

# Leading magic bytes for the formats we accept.
MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"BM", "bmp"),
)


class ValidationError(ValueError):
    """Raised when an upload is rejected. The message is safe to show a user."""


@dataclass
class ValidatedImage:
    """A decoded, accepted upload."""

    image_bgr: np.ndarray
    detected_format: str
    width: int
    height: int
    size_bytes: int
    original_filename: str


def _extension_of(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def detect_format(payload: bytes) -> Optional[str]:
    """Identify the image format from magic bytes, or None if unrecognised."""
    for signature, name in MAGIC_SIGNATURES:
        if payload.startswith(signature):
            return name
    return None


def validate_upload(
    payload: bytes,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> ValidatedImage:
    """Validate and decode an uploaded image, or raise ValidationError.

    Error messages describe what the user should do differently and never leak
    paths, stack traces or internal state.
    """
    if not filename or not filename.strip():
        raise ValidationError("No file was selected.")

    extension = _extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(f"Unsupported file type '{extension or 'unknown'}'. Allowed: {allowed}.")

    if not payload:
        raise ValidationError("The uploaded file is empty.")

    if len(payload) > max_bytes:
        raise ValidationError(
            f"File is too large ({len(payload) / 1e6:.1f} MB). Maximum is {max_bytes / 1e6:.0f} MB."
        )

    detected = detect_format(payload)
    if detected is None:
        raise ValidationError(
            "The file does not appear to be a PNG, JPEG or BMP image, "
            "regardless of its extension."
        )

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError("The image could not be decoded. It may be corrupted.")

    height, width = image.shape[:2]
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ValidationError(
            f"Image is too small ({width}x{height}). Minimum is "
            f"{MIN_DIMENSION}x{MIN_DIMENSION} pixels."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValidationError(f"Image dimensions are too large ({width}x{height}).")

    return ValidatedImage(
        image_bgr=image,
        detected_format=detected,
        width=int(width),
        height=int(height),
        size_bytes=len(payload),
        original_filename=filename,
    )
